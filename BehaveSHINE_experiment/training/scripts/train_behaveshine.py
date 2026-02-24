"""
BehaveSHINE v2 Training Script

Trains the SHINE hypernetwork (metanetwork + metalora) using the complementation
objective: the system prompt is fed to BOTH the hypernetwork (as evidence) and the
forward-pass chat template. Loss targets 32B teacher model responses.

v2 additions:
- Difficulty-aware loss weighting (examples where 8B diverged most get higher weight)
- Comprehensive monitoring (windowed loss, gradient norms, LoRA diversity)
- Two-tier evaluation (quick every 100 steps, full every 500 with generation previews)
- Checkpointing every 250 steps, keep all

Run from the SHINE repo root:
    python BehaveSHINE_experiment/training/scripts/train_behaveshine.py \
        --config BehaveSHINE_experiment/training/configs/train_config.yaml
"""

import argparse
import math
import os
import sys
import time
from collections import deque

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
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_grad_norm(params):
    """Compute total L2 gradient norm for a list of parameters."""
    total_norm_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    return total_norm_sq ** 0.5


def _iter_lora_tensors(d):
    """Recursively yield all tensors from a nested lora dict."""
    for v in d.values():
        if isinstance(v, dict):
            yield from _iter_lora_tensors(v)
        elif isinstance(v, torch.Tensor):
            yield v


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(metanetwork, metalora, eval_loader, device, tokenizer, logger,
             num_generations=0, eval_name="eval"):
    """
    Run evaluation.

    Args:
        num_generations: Number of generation previews to produce. 0 = loss only.
        eval_name: Prefix for log messages ("quick_eval" or "full_eval").
    """
    metanetwork.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    generations_done = 0

    for batch in eval_loader:
        evidence_ids = batch["evidence_ids"].to(device)
        evidence_mask = batch["evidence_attention_mask"].to(device)

        # ── Generation previews ──────────────────────────────────────────
        if generations_done < num_generations:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                lora_dict = metanetwork.generate_lora_dict(
                    evidence_ids[:1], evidence_mask[:1], metalora,
                )
                prompt_ids = batch["prompt_only_ids"][:1].to(device)
                mask = prompt_ids[0] != tokenizer.pad_token_id
                clean_prompt = prompt_ids[0][mask].unsqueeze(0)
                clean_attn = torch.ones_like(clean_prompt)

                gen_ids = metanetwork.metamodel.generate(
                    input_ids=clean_prompt,
                    attention_mask=clean_attn,
                    loradict=lora_dict,
                    ignore_mem_token=True,
                    max_new_tokens=200,
                    do_sample=False,
                )
            new_tokens = gen_ids[0][clean_prompt.shape[1]:]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)
            target = batch["raw_answer"][0]

            sep = "=" * 60
            dash = "-" * 60
            logger.info(f"\n{sep}")
            logger.info(f"[{eval_name}] Generation preview {generations_done + 1}:")
            logger.info(f"TARGET (32B teacher):\n{target[:500]}")
            logger.info(dash)
            logger.info(f"PREDICTION (8B + LoRA):\n{prediction[:500]}")
            logger.info(f"{sep}\n")

            print(f"\n{sep}")
            print(f"[{eval_name}] Generation preview {generations_done + 1}:")
            print(f"TARGET (32B teacher):\n{target[:500]}")
            print(dash)
            print(f"PREDICTION (8B + LoRA):\n{prediction[:500]}")
            print(f"{sep}\n")

            generations_done += 1
            del lora_dict, clean_prompt, clean_attn, gen_ids
            torch.cuda.empty_cache()

            # ── Loss computation ─────────────────────────────────────────────
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                lora_dict = metanetwork.generate_lora_dict(
                    evidence_ids, evidence_mask, metalora,
                )
                outputs = metanetwork.metamodel(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                    loradict=lora_dict,
                    ignore_mem_token=True,
                )

        valid_tokens = (batch["labels"] != -100).sum().item()
        total_loss += outputs.loss.item() * valid_tokens
        total_tokens += valid_tokens
        num_batches += 1

    metanetwork.train()
    return total_loss / max(total_tokens, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BehaveSHINE v2 training")
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

    logger.info("BehaveSHINE v2 training starting")
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
    config.num_mem_token = 148
    config.lora_r = cfg.model.lora_r
    config.metalora_r = cfg.model.metalora_r

    logger.info(f"Loading base model from {cfg.model.model_from} ...")
    metamodel = MetaModelCls.from_pretrained(
        cfg.model.model_from,
        config=config,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.bfloat16,
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

    # Cast trainable metanetwork encoder to float32 for stable training
    metanetwork.metanetwork.float()

    # Verify dtypes
    base_param = next(metanetwork.metamodel.parameters())
    encoder_param = next(metanetwork.metanetwork.parameters())
    logger.info(f"Base model dtype: {base_param.dtype}  Metanetwork encoder dtype: {encoder_param.dtype}")

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

    # ── Monitoring state ──────────────────────────────────────────────────────
    loss_window = deque(maxlen=cfg.training.loss_window_size)
    loss_weighted_window = deque(maxlen=cfg.training.loss_window_size)
    lora_analysis_buffer = deque(maxlen=cfg.training.lora_buffer_size)

    # ── Quick eval subset ─────────────────────────────────────────────────────
    quick_eval_size = min(
        cfg.training.get("quick_eval_num_examples", 50),
        len(eval_dataset),
    )
    quick_eval_subset = torch.utils.data.Subset(eval_dataset, list(range(quick_eval_size)))
    quick_eval_loader = DataLoader(
        quick_eval_subset,
        batch_size=cfg.data.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    global_step = 0
    accum_start_time = time.time()

    for epoch in range(cfg.training.num_epochs):
        metanetwork.train()
        epoch_loss = 0.0
        epoch_tokens = 0

        for step, batch in enumerate(train_loader):
            if step % cfg.run.gradient_accumulation_steps == 0:
                accum_start_time = time.time()

            evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
            evidence_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            difficulty_w = batch["difficulty_weight"].to(device, non_blocking=True)

            # ── Forward ───────────────────────────────────────────────────────
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                lora_dict = metanetwork.generate_lora_dict(
                    evidence_ids, evidence_mask, metalora,
                    use_gradient_checkpoint=cfg.run.use_gradient_checkpoint,
                )

                outputs = metanetwork.metamodel(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    loradict=lora_dict,
                    ignore_mem_token=True,
                    use_gradient_checkpoint=cfg.run.use_gradient_checkpoint,
                )

            # Store LoRA snapshot for diversity analysis (on CPU, no grad)
            if global_step % cfg.training.lora_analysis_every_n_steps == 0:
                with torch.no_grad():
                    lora_tensors = list(_iter_lora_tensors(lora_dict))
                    flat_lora = torch.cat([t.detach().cpu().reshape(-1) for t in lora_tensors])
                    lora_analysis_buffer.append(flat_lora)


            raw_loss = outputs.loss
            reg_loss = getattr(outputs, 'reg_loss', None)

            # ── Difficulty-aware weighting ────────────────────────────────────
            if cfg.training.get("use_difficulty_weighting", True):
                weighted_loss = raw_loss * difficulty_w.mean()
            else:
                weighted_loss = raw_loss

            backward_loss = weighted_loss / cfg.run.gradient_accumulation_steps

            # ── Backward ──────────────────────────────────────────────────────
            valid_tokens = (labels != -100).sum().item()
            if not math.isinf(raw_loss.item()) and not math.isnan(raw_loss.item()) and valid_tokens > 0:
                backward_loss.backward()

                # Track losses in windows
                loss_window.append(raw_loss.item())
                loss_weighted_window.append(weighted_loss.item())

                epoch_loss += raw_loss.item() * valid_tokens
                epoch_tokens += valid_tokens

            # ── Optimizer step on accumulation boundary ───────────────────────
            is_accum_step = (step + 1) % cfg.run.gradient_accumulation_steps == 0
            is_last_step = (step + 1) == len(train_loader)

            if is_accum_step or is_last_step:
                # Compute per-component gradient norms BEFORE clipping
                metanet_params = list(metanetwork.metanetwork.parameters())
                metalora_params_list = list(iter_learnable_tensors(metalora))
                all_params = metanet_params + metalora_params_list

                metanet_grad_norm = compute_grad_norm(metanet_params)
                metalora_grad_norm = compute_grad_norm(metalora_params_list)
                total_grad_norm_pre_clip = (metanet_grad_norm**2 + metalora_grad_norm**2) ** 0.5

                # Clip
                torch.nn.utils.clip_grad_norm_(all_params, cfg.training.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                step_elapsed = time.time() - accum_start_time

                # ── Logging ───────────────────────────────────────────────────
                if global_step % cfg.training.log_every_n_steps == 0:
                    lr = scheduler.get_last_lr()[0]

                    # Windowed stats
                    wl = list(loss_window)
                    w_mean = sum(wl) / len(wl) if wl else 0
                    w_min = min(wl) if wl else 0
                    w_max = max(wl) if wl else 0

                    logger.info(
                        f"Epoch {epoch}  Step {step}  GStep {global_step}  "
                        f"Loss {raw_loss.item():.4f}  WLoss {weighted_loss.item():.4f}  "
                        f"WinAvg {w_mean:.4f}  WinMin {w_min:.4f}  WinMax {w_max:.4f}  "
                        f"GradNorm {total_grad_norm_pre_clip:.4f}  "
                        f"LR {lr:.2e}  DW {difficulty_w.mean().item():.2f}  "
                        f"Tok {valid_tokens}  {step_elapsed:.1f}s"
                    )

                    # Tensorboard — losses
                    writer.add_scalar("train/loss_raw", raw_loss.item(), global_step)
                    writer.add_scalar("train/loss_weighted", weighted_loss.item(), global_step)
                    writer.add_scalar("train/loss_window_mean", w_mean, global_step)
                    writer.add_scalar("train/loss_window_min", w_min, global_step)
                    writer.add_scalar("train/loss_window_max", w_max, global_step)
                    if len(wl) > 1:
                        writer.add_scalar("train/loss_window_std", torch.tensor(wl).std().item(), global_step)

                    # Tensorboard — loss breakdown
                    if reg_loss is not None:
                        writer.add_scalar("train/reg_loss", reg_loss.item(), global_step)
                        writer.add_scalar("train/task_loss", (raw_loss - reg_loss).item(), global_step)

                    # Tensorboard — gradient norms
                    writer.add_scalar("grad/total_norm_pre_clip", total_grad_norm_pre_clip, global_step)
                    writer.add_scalar("grad/metanetwork_norm", metanet_grad_norm, global_step)
                    writer.add_scalar("grad/metalora_norm", metalora_grad_norm, global_step)
                    clip_ratio = min(1.0, cfg.training.max_grad_norm / max(total_grad_norm_pre_clip, 1e-8))
                    writer.add_scalar("grad/clip_ratio", clip_ratio, global_step)

                    # Tensorboard — token counts and difficulty
                    writer.add_scalar("train/valid_tokens", valid_tokens, global_step)
                    writer.add_scalar("train/difficulty_weight", difficulty_w.mean().item(), global_step)
                    writer.add_scalar("train/lr", lr, global_step)

                    # Tensorboard — speed
                    writer.add_scalar("perf/step_time_seconds", step_elapsed, global_step)
                    writer.add_scalar("perf/tokens_per_second", valid_tokens / max(step_elapsed, 1e-6), global_step)

                    writer.flush()

                # ── LoRA diversity analysis ───────────────────────────────────
                if global_step % cfg.training.lora_analysis_every_n_steps == 0 and len(lora_analysis_buffer) >= 10:
                    lora_vectors = torch.stack(list(lora_analysis_buffer))  # (N, D)

                    # L2 norm statistics
                    norms = lora_vectors.norm(dim=1)

                    # Pairwise cosine similarity (sample if buffer is large)
                    normalized = lora_vectors / norms.unsqueeze(1).clamp(min=1e-8)
                    if len(normalized) > 20:
                        indices = torch.randperm(len(normalized))[:20]
                        normalized = normalized[indices]

                    cos_sim_matrix = normalized @ normalized.T
                    triu_mask = torch.triu(torch.ones_like(cos_sim_matrix, dtype=torch.bool), diagonal=1)
                    pairwise_sims = cos_sim_matrix[triu_mask]

                    writer.add_scalar("lora/mean_l2_norm", norms.mean().item(), global_step)
                    writer.add_scalar("lora/std_l2_norm", norms.std().item(), global_step)
                    writer.add_scalar("lora/mean_pairwise_cos_sim", pairwise_sims.mean().item(), global_step)
                    writer.add_scalar("lora/std_pairwise_cos_sim", pairwise_sims.std().item(), global_step)
                    writer.add_scalar("lora/min_pairwise_cos_sim", pairwise_sims.min().item(), global_step)
                    writer.add_scalar("lora/max_pairwise_cos_sim", pairwise_sims.max().item(), global_step)

                    logger.info(
                        f"[LoRA] Step {global_step}: "
                        f"Norm={norms.mean():.4f}\u00b1{norms.std():.4f}  "
                        f"CosSim={pairwise_sims.mean():.4f}\u00b1{pairwise_sims.std():.4f}  "
                        f"Range=[{pairwise_sims.min():.4f}, {pairwise_sims.max():.4f}]"
                    )

                # ── Quick eval ────────────────────────────────────────────────
                if global_step % cfg.training.quick_eval_every_n_steps == 0:
                    qe_loss = run_eval(
                        metanetwork, metalora, quick_eval_loader, device, tokenizer, logger,
                        num_generations=0, eval_name="quick_eval",
                    )
                    logger.info(f"[Quick Eval] GStep {global_step}  Loss {qe_loss:.4f}")
                    writer.add_scalar("eval/quick_loss", qe_loss, global_step)
                    writer.flush()

                # ── Full eval ─────────────────────────────────────────────────
                if global_step % cfg.training.full_eval_every_n_steps == 0:
                    fe_loss = run_eval(
                        metanetwork, metalora, eval_loader, device, tokenizer, logger,
                        num_generations=cfg.training.full_eval_num_generations,
                        eval_name="full_eval",
                    )
                    logger.info(f"[Full Eval] GStep {global_step}  Loss {fe_loss:.4f}")
                    writer.add_scalar("eval/full_loss", fe_loss, global_step)
                    writer.flush()

                # ── Checkpoint ────────────────────────────────────────────────
                if global_step % cfg.training.save_every_n_steps == 0:
                    ckpt_path = os.path.join(cfg.paths.output_dir, f"checkpoint-step-{global_step}")
                    save_checkpoint(metanetwork, ckpt_path, metalora)
                    logger.info(f"Checkpoint saved \u2192 {ckpt_path}")

        # ── End-of-epoch checkpoint ───────────────────────────────────────────
        epoch_avg_loss = epoch_loss / max(epoch_tokens, 1)
        logger.info(f"Epoch {epoch} complete.  AvgLoss {epoch_avg_loss:.4f}")
        writer.add_scalar("train/epoch_loss", epoch_avg_loss, epoch)

        ckpt_path = os.path.join(cfg.paths.output_dir, f"checkpoint-epoch-{epoch}")
        save_checkpoint(metanetwork, ckpt_path, metalora)
        logger.info(f"End-of-epoch checkpoint saved \u2192 {ckpt_path}")

    writer.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
