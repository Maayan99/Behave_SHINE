"""
BehaveSHINE Training Script

Trains the SHINE hypernetwork (metanetwork + metalora) using the complementation
objective: the system prompt is fed to BOTH the hypernetwork (as evidence) and the
forward-pass chat template. Loss targets 235B teacher model responses.

Run from the SHINE repo root:
    python BehaveSHINE_experiment/training/scripts/train_behaveshire.py \
        --config BehaveSHINE_experiment/training/configs/train_config.yaml
"""

import argparse
import math
import os
import sys

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Ensure SHINE repo root is on the path when the script is run from elsewhere
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from metanetwork_family import Metanetwork
from utils.myinit import _import_class, _resolve_device
from utils.mylogging import get_logger
from utils.myloradict import iter_learnable_tensors
from utils.mysaveload import load_checkpoint, save_checkpoint
from utils.myseed import set_seed
from transformers import AutoConfig

from BehaveSHINE_experiment.training.scripts.dataset import (
    BehaveSHINECollator,
    BehaveSHINEDataset,
)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(metanetwork, metalora, eval_loader, device, tokenizer, logger):
    metanetwork.eval()
    total_loss = 0
    num_batches = 0

    for batch in eval_loader:
        evidence_ids = batch["evidence_ids"].to(device)
        evidence_mask = batch["evidence_attention_mask"].to(device)

        # ── Generation preview: first batch, first item only ─────────────────
        if num_batches == 0:
            lora_dict = metanetwork.generate_lora_dict(
                evidence_ids[:1],
                evidence_mask[:1],
                metalora,
            )
            prompt_ids = batch["prompt_only_ids"][:1].to(device)

            # Strip right-padding — generate() requires left-padding (or no padding).
            mask = prompt_ids[0] != tokenizer.pad_token_id
            clean_prompt_ids = prompt_ids[0][mask].unsqueeze(0)
            clean_prompt_attn = torch.ones_like(clean_prompt_ids)

            gen_ids = metanetwork.metamodel.generate(
                input_ids=clean_prompt_ids,
                attention_mask=clean_prompt_attn,
                loradict=lora_dict,
                ignore_mem_token=True,
                max_new_tokens=200,
                do_sample=False,
            )

            new_tokens = gen_ids[0][clean_prompt_ids.shape[1]:]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)
            target = batch["raw_answer"][0]

            sep = "=" * 60
            dash = "-" * 60
            logger.info("\n" + sep)
            logger.info("TARGET (235B MoE):")
            logger.info(target)
            logger.info(dash)
            logger.info("PREDICTION (8B + BehaveSHINE):")
            logger.info(prediction)
            logger.info(sep + "\n")

            print("\n" + sep)
            print("TARGET (235B MoE):")
            print(target)
            print(dash)
            print("PREDICTION (8B + BehaveSHINE):")
            print(prediction)
            print(sep + "\n")

            # Free generation tensors to prevent OOM on the subsequent dense forward pass
            del lora_dict, clean_prompt_ids, clean_prompt_attn, gen_ids
            torch.cuda.empty_cache()

        # ── Standard loss eval ───────────────────────────────────────────────
        lora_dict = metanetwork.generate_lora_dict(
            evidence_ids,
            evidence_mask,
            metalora,
        )
        outputs = metanetwork.metamodel(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
            loradict=lora_dict,
            ignore_mem_token=True,
        )
        total_loss += outputs.loss.item()
        num_batches += 1

    metanetwork.train()
    return total_loss / max(num_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BehaveSHINE training")
    parser.add_argument("--config", required=True, help="Path to train_config.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # ── Setup ─────────────────────────────────────────────────────────────────
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    set_seed(cfg.run.seed)
    device = _resolve_device(cfg.run.device)

    log_file = os.path.join(cfg.paths.output_dir, "train.log")
    logger = get_logger("BehaveSHINE", filename=log_file)
    writer = SummaryWriter(os.path.join(cfg.paths.output_dir, "tensorboard"))

    logger.info("BehaveSHINE training starting")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    MetaModelCls = _import_class(cfg.model.metamodel_class_path)
    ConfigCls = _import_class(cfg.model.config_class_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_from)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Initializing configuration...")
    config = ConfigCls.from_pretrained(cfg.model.model_from)

    # 1. First, we must inject a dummy value so the model __init__ doesn't crash
    config.num_mem_token = 64
    config.lora_r = cfg.model.lora_r
    config.metalora_r = cfg.model.metalora_r

    logger.info(f"Loading base model from {cfg.model.model_from} ...")
    metamodel = MetaModelCls.from_pretrained(
        cfg.model.model_from,
        config=config,
        ignore_mismatched_sizes=True
    )

    # 2. NOW we get the real, exact numel from the loaded model
    real_lora_numel = metamodel.lora_params_numel(cfg.model.lora_r)

    # 3. Recalculate and update the config with the mathematically perfect token count
    computed_mem_tokens = real_lora_numel * cfg.metanetwork.transformer_cfg.mean_pool_size // (
                config.hidden_size * config.num_hidden_layers)

    metamodel.config.num_mem_token = computed_mem_tokens
    cfg.num_mem_token = computed_mem_tokens
    cfg.hidden_size = config.hidden_size
    cfg.num_layers = config.num_hidden_layers

    # Re-init mem tokens with the correct size
    metamodel.reset_mem_tokens()
    metamodel.resize_token_embeddings(len(tokenizer))

    logger.info(f"Perfectly aligned num_mem_token set to {computed_mem_tokens}")

    # 4. Initialize Metanetwork with the EXACT numel the model reports
    metanetwork = Metanetwork(
        metamodel,
        cfg,
        real_lora_numel,
    )
    metanetwork.to(device)


    # ── Checkpoint (warm start) ────────────────────────────────────────────────
    # load_checkpoint internally calls freeze(metamodel) after loading weights
    logger.info(f"Loading pretrained checkpoint from {cfg.paths.pretrained_checkpoint} ...")
    metanetwork, metalora, _ = load_checkpoint(
        metanetwork,
        cfg.paths.pretrained_checkpoint,
        device,
    )

    trainable = sum(p.numel() for p in metanetwork.metanetwork.parameters() if p.requires_grad)
    metalora_params = sum(t.numel() for t in iter_learnable_tensors(metalora))
    logger.info(f"Trainable params — metanetwork: {trainable:,}  metalora: {metalora_params:,}")

    # ── Data ──────────────────────────────────────────────────────────────────
    collator = BehaveSHINECollator(
        tokenizer=tokenizer,
        context_max_length=cfg.data.context_max_length,
        conversation_max_length=cfg.data.conversation_max_length,
    )

    train_dataset = BehaveSHINEDataset(cfg.paths.train_data)
    eval_dataset = BehaveSHINEDataset(cfg.paths.eval_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.data.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    logger.info(f"Train examples: {len(train_dataset)}  Eval examples: {len(eval_dataset)}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(metanetwork.metanetwork.parameters()),
                "weight_decay": cfg.training.weight_decay,
            },
            {
                "params": list(iter_learnable_tensors(metalora)),
                "weight_decay": cfg.training.weight_decay,
            },
        ],
        lr=cfg.training.learning_rate,
    )

    steps_per_epoch = math.ceil(len(train_loader) / cfg.run.gradient_accumulation_steps)
    total_steps = cfg.training.num_epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(f"Steps per epoch: {steps_per_epoch}  Total steps: {total_steps}")

    # ── Training Loop ─────────────────────────────────────────────────────────
    global_step = 0

    for epoch in range(cfg.training.num_epochs):
        metanetwork.train()
        epoch_loss = 0.0
        epoch_tokens = 0

        for step, batch in enumerate(train_loader):
            evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
            evidence_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # ── Forward (explicit two-step) ───────────────────────────────────
            # We bypass metanetwork.forward() to avoid any internal logic from the
            # original replacement objective that could strip or modify input_ids.
            # Step 1: generate LoRA weights from the system prompt (evidence)
            lora_dict = metanetwork.generate_lora_dict(
                evidence_ids,
                evidence_mask,
                metalora,
                use_gradient_checkpoint=cfg.run.use_gradient_checkpoint,
            )
            # Step 2: forward pass with the generated LoRA + system prompt in context
            outputs = metanetwork.metamodel(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                loradict=lora_dict,
                ignore_mem_token=True,
                use_gradient_checkpoint=cfg.run.use_gradient_checkpoint,
            )

            loss = outputs.loss
            backward_loss = loss / cfg.run.gradient_accumulation_steps

            # ── Backward ─────────────────────────────────────────────────────
            valid_tokens = (labels != -100).sum().item()
            if not math.isinf(loss.item()) and not math.isnan(loss.item()) and valid_tokens > 0:
                backward_loss.backward()
                epoch_loss += loss.item() * valid_tokens
                epoch_tokens += valid_tokens

            # ── Optimizer step on accumulation boundary ───────────────────────
            is_accum_step = (step + 1) % cfg.run.gradient_accumulation_steps == 0
            is_last_step = (step + 1) == len(train_loader)

            if is_accum_step or is_last_step:
                torch.nn.utils.clip_grad_norm_(
                    list(metanetwork.metanetwork.parameters())
                    + list(iter_learnable_tensors(metalora)),
                    cfg.training.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                # ── Logging ──────────────────────────────────────────────────
                if global_step % cfg.training.log_every_n_steps == 0:
                    avg_loss = epoch_loss / max(epoch_tokens, 1)
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Epoch {epoch}  Step {step}  GlobalStep {global_step}  "
                        f"Loss {loss.item():.4f}  AvgLoss {avg_loss:.4f}  LR {lr:.2e}"
                    )
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/lr", lr, global_step)

                # ── Mid-epoch checkpoint & eval ───────────────────────────────
                if global_step % cfg.training.save_every_n_steps == 0:
                    ckpt_path = os.path.join(cfg.paths.output_dir, f"checkpoint-step-{global_step}")
                    save_checkpoint(metanetwork, ckpt_path, metalora)
                    logger.info(f"Checkpoint saved → {ckpt_path}")

                if global_step % cfg.training.eval_every_n_steps == 0:
                    eval_loss = run_eval(metanetwork, metalora, eval_loader, device, tokenizer, logger)
                    logger.info(f"[Eval] GlobalStep {global_step}  EvalLoss {eval_loss:.4f}")
                    writer.add_scalar("eval/loss", eval_loss, global_step)

        # ── End-of-epoch checkpoint ───────────────────────────────────────────
        epoch_avg_loss = epoch_loss / max(epoch_tokens, 1)
        logger.info(f"Epoch {epoch} complete.  AvgLoss {epoch_avg_loss:.4f}")
        writer.add_scalar("train/epoch_loss", epoch_avg_loss, epoch)

        ckpt_path = os.path.join(cfg.paths.output_dir, f"checkpoint-epoch-{epoch}")
        save_checkpoint(metanetwork, ckpt_path, metalora)
        logger.info(f"End-of-epoch checkpoint saved → {ckpt_path}")

    writer.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
