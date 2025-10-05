#!/usr/bin/env python
# -*- coding: utf-8 -*-

from csv import writer
import os
import math
import time
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from functools import partial
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    Qwen3ForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from datasets import Dataset as HFDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import importlib
from omegaconf import DictConfig, OmegaConf
import hydra
from datasets import load_dataset
import logging
from torch.utils.tensorboard import SummaryWriter
from metanetwork_family import Metanetwork

from utils.mydataset import TextDataset, CausalLMDataCollator, create_mock_dataset
from utils.myseed import set_seed
from utils.mylogging import get_logger
from utils.mysaveload import save_checkpoint, load_checkpoint, save_training_state, load_training_state, get_latest_checkpoint
from utils.myfreeze import freeze
from utils.myoptmize import init_optimize

from collections import OrderedDict


logger = get_logger("metalora")

@torch.no_grad()
def evaluate(model, metanetwork, dataloader, device, use_amp: bool = True) -> Dict[str, float]:
    model.eval()
    metanetwork.eval()
    total_loss = 0.0
    n_tokens = 0
    if use_amp and device.type == "cuda":
        scaler_ctx = partial(torch.amp.autocast, device_type=str(device))
    else:
        # On CPU autocast is a no-op on many PyTorch versions; using a dummy ctx manager
        from contextlib import nullcontext
        scaler_ctx = nullcontext
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with scaler_ctx():
            loradict = metanetwork(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            new_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, loradict=loradict)
            loss = new_outputs.loss

        valid_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * valid_tokens
        n_tokens += valid_tokens

    avg_loss = total_loss / max(n_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    model.train()
    metanetwork.train()
    return {"eval_loss": avg_loss, "perplexity": ppl}


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg in ("cuda", "cpu"):
        return torch.device(device_cfg)
    raise ValueError(f"Unsupported device setting: {device_cfg}")


def _import_class(path: str):
    """
    Import a class from a 'module.ClassName' path.
    """
    if "." not in path:
        raise ValueError("model.class_path must be 'module.ClassName'")
    mod_name, cls_name = path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)



@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):
    logger.info("Resolved config:")
    logger.info(f"\n\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # Seed & device
    set_seed(cfg.run.seed)
    device = _resolve_device(cfg.run.device)

    # Load model/tokenizer (supports your local LoRA-wrapped Qwen class)
    logger.info("Loading model & tokenizer...")
    ModelCls = _import_class(cfg.model.model_class_path)
    MetaModelCls = _import_class(cfg.model.meta_model_class_path)
    ConfigCls = _import_class(cfg.model.config_class_path)
    config = ConfigCls.from_pretrained(cfg.model.model_from)
    config.num_mem_token = -1
    cfg.hidden_size = config.hidden_size
    cfg.num_layers = config.num_hidden_layers
    model = ModelCls.from_pretrained(cfg.model.model_from, config=config)
    model.train()
    model.to(device)
    if cfg.metanetwork.type == "transformer":
        assert model.lora_params_numel(cfg.model.lora_r) % (cfg.hidden_size * cfg.num_layers) == 0, "For transformer metanetwork, num_mem_token must be set to model.lora_params_numel(lora_r) / (hidden_size * num_layers)"
        config.num_mem_token = model.lora_params_numel(cfg.model.lora_r) // (cfg.hidden_size * cfg.num_layers)
        cfg.num_mem_token = config.num_mem_token
        logger.info(f"Using transformer metanetwork, set num_mem_token to {config.num_mem_token}")
    else:
        config.num_mem_token = cfg.model.num_mem_token
    metamodel = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
    metamodel.reset_mem_tokens()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_from)
    metanetwork = Metanetwork(metamodel, cfg, model.lora_params_numel(cfg.model.lora_r))
    metanetwork.train()
    metanetwork.to(device)
    freeze(model, metamodel)

    # Data
    logger.info("Preparing data...")
    if cfg.data.source == "mock":
        train_texts, val_texts = create_mock_dataset()
    elif cfg.data.source == "transmla":
        dataset = load_dataset(os.path.join("data", "transmla_pretrain_6B_tokens"), split="train")
        split_dataset = dataset.train_test_split(test_size=0.001, seed=42)
        train_texts = split_dataset["train"]
        val_texts = split_dataset["test"]
        logger.info(f"Train len: {len(train_texts)}")
        logger.info(f"Val len: {len(val_texts)}")
    else:
        raise ValueError(f"Unknown data source: {cfg.data.source}")

    train_ds = TextDataset(train_texts, tokenizer, max_length=cfg.data.max_length)
    val_ds = TextDataset(val_texts, tokenizer, max_length=cfg.data.max_length)

    collator = CausalLMDataCollator(tokenizer=tokenizer, max_length=cfg.data.max_length)

    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.train_batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=pin,    
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=pin,
    )

    # Optimizer & Scheduler
    logger.info("Setting up optimizer & scheduler...")
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight", "norm1", "norm2"]
    grouped_params = [
        {
            "params": [p for n, p in metanetwork.named_parameters() if (not any(nd in n for nd in no_decay) and not n.startswith("metamodel"))],
            "weight_decay": cfg.optim.weight_decay,
        },
        {
            "params": [p for n, p in metanetwork.named_parameters() if (any(nd in n for nd in no_decay) and not n.startswith("metamodel"))],
            "weight_decay": 0.0,
        },
        {
            "params": [metamodel.mem_tokens],
            "weight_decay": cfg.optim.weight_decay,
        }
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=cfg.optim.learning_rate)

    optimizer, lr_scheduler, scaler = init_optimize(grouped_params, train_loader, cfg, device)

    # Training loop scaffolding
    hydra_run_dir = os.getcwd()
    tb_log_dir = os.path.join(hydra_run_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_log_dir)
    logger.info(f"TensorBoard logs will be written to: {tb_log_dir}")
    logger.info("Starting training loop...")

    # Checkpoint root
    ckpt_root = os.path.join(hydra_run_dir, "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)
    
    global_step = 0
    best_eval_loss = float("inf")

    # # ---- Resume
    # if cfg.resume_global_step == -1:
    #     resume_dir = None
    # elif cfg.resume_global_step == "latest":
    #     resume_dir = get_latest_checkpoint(ckpt_root)
    # elif isinstance(cfg.resume_global_step, int) and cfg.resume_global_step > 0:
    #     resume_dir = os.path.join(ckpt_root, f"checkpoint-{cfg.resume_global_step}")
    #     if not os.path.isdir(resume_dir):
    #         raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    # else:
    #     raise ValueError(f"Invalid resume_global_step: {cfg.resume_global_step}")


    # global_step = 0
    # start_epoch = 1
    # resume_step_in_epoch = 0  # 1-based; 0 means start at beginning
    # best_eval_loss = float("inf")

    # if resume_dir:
    #     logger.info(f"Resuming from checkpoint: {resume_dir}")

    #     # 1) reload weights/tokenizer
    #     model, metanetwork, tokenizer = load_checkpoint(model, metanetwork, tokenizer, resume_dir)
        
    #     # ensure device placement
    #     model.train()
    #     model.to(device)
    #     metanetwork.train()
    #     metanetwork.to(device)
    #     # important: keep local reference to metamodel in sync
    #     metamodel = metanetwork.metamodel

    #     # 2) load training state (optimizer, scheduler, scaler, counters, RNG)
    #     state = load_training_state(resume_dir, optimizer, lr_scheduler, scaler)
    #     train_loader = state["dataloader"] if state["dataloader"] is not None else train_loader
    #     if state is not None:
    #         global_step = state["global_step"]
    #         start_epoch = state["epoch"]
    #         resume_step_in_epoch = state["step_in_epoch"]
    #         best_eval_loss = state["best_eval_loss"]
    #         logger.info(f"Restored: global_step={global_step}, epoch={start_epoch}, step_in_epoch={resume_step_in_epoch}, best_eval_loss={best_eval_loss}")
    #     else:
    #         logger.warning("No trainer_state.pt found—weights loaded, but optimizer/scheduler/scaler/counters not restored.")

    # def one_train_epoch(epoch, resume_step_in_epoch=None):
    #     nonlocal global_step, best_eval_loss
    #     epoch_loss = 0.0
    #     epoch_tokens = 0
    #     pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.optim.num_epochs}")

    #     for step, batch in enumerate(pbar, start=1):
    #         # If we are resuming within this epoch, skip seen steps
    #         if resume_step_in_epoch and step <= resume_step_in_epoch:
    #             continue

    #         input_ids = batch["input_ids"].to(device, non_blocking=True)
    #         attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    #         labels = batch["labels"].to(device, non_blocking=True)

    #         with torch.amp.autocast(enabled=(cfg.run.use_fp16 and device.type == "cuda"), device_type=str(device)):
    #             loradict = metanetwork(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #             new_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, loradict=loradict)
    #             loss = new_outputs.loss / max(1, cfg.run.gradient_accumulation_steps)

    #         writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)

    #         scaler.scale(loss).backward()

    #         valid_tokens = (labels != -100).sum().item()
    #         epoch_loss += loss.item() * valid_tokens * max(1, cfg.run.gradient_accumulation_steps)
    #         epoch_tokens += valid_tokens

    #         if step % max(1, cfg.run.gradient_accumulation_steps) == 0:
    #             if cfg.optim.grad_clip_norm and cfg.optim.grad_clip_norm > 0:
    #                 scaler.unscale_(optimizer)
    #                 for group in optimizer.param_groups:
    #                     torch.nn.utils.clip_grad_norm_(
    #                         group["params"], cfg.optim.grad_clip_norm
    #                     )

    #             scaler.step(optimizer)
    #             scaler.update()
    #             optimizer.zero_grad(set_to_none=True)
    #             lr_scheduler.step()
    #             global_step += 1

    #             if cfg.logging.logging_steps and global_step % cfg.logging.logging_steps == 0:
    #                 avg_loss = (epoch_loss / max(epoch_tokens, 1))
    #                 ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    #                 writer.add_scalar("train/loss", avg_loss, global_step)
    #                 writer.add_scalar("train/ppl", ppl, global_step)
    #                 pbar.set_postfix({"lr": lr_scheduler.get_last_lr()[0], "loss": f"{avg_loss:.4f}", "ppl": f"{ppl:.2f}"})

    #             # ---- Periodic checkpoint (model + training state) ----
    #             if getattr(cfg.save, "save_steps", 0) and global_step % cfg.save.save_steps == 0:
    #                 ckpt_dir = os.path.join(ckpt_root, f"checkpoint-{global_step}")
    #                 logger.info(f"Saving checkpoint to {ckpt_dir}")
    #                 save_checkpoint(
    #                     model, metanetwork, tokenizer, ckpt_dir,
    #                     extra_state={"global_step": global_step}
    #                 )
    #                 save_training_state(
    #                     ckpt_dir, optimizer, lr_scheduler, scaler,
    #                     global_step=global_step, epoch=epoch,
    #                     step_in_epoch=step,
    #                     best_eval_loss=best_eval_loss, cfg=cfg, dataloader=train_loader
    #                 )

    #             # ---- Eval + best checkpoint (model + training state) ----
    #             if getattr(cfg.eval, "eval_steps", 0) and global_step % cfg.eval.eval_steps == 0:
    #                 eval_metrics = evaluate(model, metanetwork, val_loader, device, use_amp=cfg.run.use_fp16)
    #                 writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
    #                 writer.add_scalar("eval/ppl", eval_metrics["perplexity"], global_step)
    #                 logger.info(f"[Eval @ step {global_step}] loss={eval_metrics['eval_loss']:.4f} ppl={eval_metrics['perplexity']:.2f}")

    #                 if getattr(cfg.save, "save_best", True) and eval_metrics["eval_loss"] < best_eval_loss:
    #                     best_eval_loss = eval_metrics["eval_loss"]
    #                     best_dir = os.path.join(ckpt_root, "best")
    #                     logger.info(f"New best model! Saving to {best_dir}")
    #                     save_checkpoint(
    #                         model, metanetwork, tokenizer, best_dir,
    #                         extra_state={"global_step": global_step, "best_eval_loss": best_eval_loss}
    #                     )
    #                     save_training_state(
    #                         best_dir, optimizer, lr_scheduler, scaler,
    #                         global_step=global_step, epoch=epoch,
    #                         step_in_epoch=step,
    #                         best_eval_loss=best_eval_loss, cfg=cfg, dataloader=train_loader
    #                     )

    #     epoch_avg = (epoch_loss / max(epoch_tokens, 1))
    #     epoch_ppl = math.exp(epoch_avg) if epoch_avg < 20 else float("inf")
    #     logger.info(f"Epoch {epoch} done. train_loss={epoch_avg:.4f} train_ppl={epoch_ppl:.2f}")

    #     eval_metrics = evaluate(model, metanetwork, val_loader, device, use_amp=cfg.run.use_fp16)
    #     writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
    #     writer.add_scalar("eval/ppl", eval_metrics["perplexity"], global_step)
    #     logger.info(f"[Epoch {epoch} Eval] loss={eval_metrics['eval_loss']:.4f} ppl={eval_metrics['perplexity']:.2f}")
    #     if getattr(cfg.save, "save_best", True) and eval_metrics["eval_loss"] < best_eval_loss:
    #         best_eval_loss = eval_metrics["eval_loss"]
    #         best_dir = os.path.join(ckpt_root, "best")
    #         logger.info(f"New best model! Saving to {best_dir}")
    #         save_checkpoint(
    #             model, metanetwork, tokenizer, best_dir,
    #             extra_state={"global_step": global_step, "best_eval_loss": best_eval_loss}
    #         )
    #         save_training_state(
    #             best_dir, optimizer, lr_scheduler, scaler,
    #             global_step=global_step, epoch=epoch,
    #             step_in_epoch=0,  # end of epoch
    #             best_eval_loss=best_eval_loss, cfg=cfg, dataloader=train_loader
    #         )

    def one_train_epoch(epoch):
        nonlocal global_step, best_eval_loss
        epoch_loss = 0.0
        epoch_tokens = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.optim.num_epochs}")

        for step, batch in enumerate(pbar, start=1):
            # If we are resuming within this epoch, skip seen steps
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(enabled=(cfg.run.use_fp16 and device.type == "cuda"), device_type=str(device)):
                loradict = metanetwork(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                new_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, loradict=loradict)
                loss = new_outputs.loss / max(1, cfg.run.gradient_accumulation_steps)

            writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)

            scaler.scale(loss).backward()

            valid_tokens = (labels != -100).sum().item()
            epoch_loss += loss.item() * valid_tokens * max(1, cfg.run.gradient_accumulation_steps)
            epoch_tokens += valid_tokens

            if step % max(1, cfg.run.gradient_accumulation_steps) == 0:
                if cfg.optim.grad_clip_norm and cfg.optim.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    for group in optimizer.param_groups:
                        torch.nn.utils.clip_grad_norm_(
                            group["params"], cfg.optim.grad_clip_norm
                        )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                global_step += 1

                if cfg.logging.logging_steps and global_step % cfg.logging.logging_steps == 0:
                    avg_loss = (epoch_loss / max(epoch_tokens, 1))
                    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/ppl", ppl, global_step)
                    pbar.set_postfix({"lr": lr_scheduler.get_last_lr()[0], "loss": f"{avg_loss:.4f}", "ppl": f"{ppl:.2f}"})

                # ---- Periodic checkpoint (model + training state) ----
                if getattr(cfg.save, "save_steps", 0) and global_step % cfg.save.save_steps == 0:
                    ckpt_dir = os.path.join(ckpt_root, f"checkpoint-{global_step}")
                    logger.info(f"Saving checkpoint to {ckpt_dir}")
                    save_checkpoint(
                        model, metanetwork, tokenizer, ckpt_dir,
                        extra_state={"global_step": global_step}
                    )

                # ---- Eval + best checkpoint (model + training state) ----
                if getattr(cfg.eval, "eval_steps", 0) and global_step % cfg.eval.eval_steps == 0:
                    eval_metrics = evaluate(model, metanetwork, val_loader, device, use_amp=cfg.run.use_fp16)
                    writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
                    writer.add_scalar("eval/ppl", eval_metrics["perplexity"], global_step)
                    logger.info(f"[Eval @ step {global_step}] loss={eval_metrics['eval_loss']:.4f} ppl={eval_metrics['perplexity']:.2f}")

                    if getattr(cfg.save, "save_best", True) and eval_metrics["eval_loss"] < best_eval_loss:
                        best_eval_loss = eval_metrics["eval_loss"]
                        best_dir = os.path.join(ckpt_root, "best")
                        logger.info(f"New best model! Saving to {best_dir}")
                        save_checkpoint(
                            model, metanetwork, tokenizer, best_dir,
                            extra_state={"global_step": global_step, "best_eval_loss": best_eval_loss}
                        )

        epoch_avg = (epoch_loss / max(epoch_tokens, 1))
        epoch_ppl = math.exp(epoch_avg) if epoch_avg < 20 else float("inf")
        logger.info(f"Epoch {epoch} done. train_loss={epoch_avg:.4f} train_ppl={epoch_ppl:.2f}")

        eval_metrics = evaluate(model, metanetwork, val_loader, device, use_amp=cfg.run.use_fp16)
        writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
        writer.add_scalar("eval/ppl", eval_metrics["perplexity"], global_step)
        logger.info(f"[Epoch {epoch} Eval] loss={eval_metrics['eval_loss']:.4f} ppl={eval_metrics['perplexity']:.2f}")
        if getattr(cfg.save, "save_best", True) and eval_metrics["eval_loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["eval_loss"]
            best_dir = os.path.join(ckpt_root, "best")
            logger.info(f"New best model! Saving to {best_dir}")
            save_checkpoint(
                model, metanetwork, tokenizer, best_dir,
                extra_state={"global_step": global_step, "best_eval_loss": best_eval_loss}
            )

    # Initial eval (optional quick check before training)
    eval_metrics = evaluate(model, metanetwork, val_loader, device, use_amp=cfg.run.use_fp16)
    writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
    writer.add_scalar("eval/ppl", eval_metrics["perplexity"], global_step)
    logger.info(f"[Eval @ step {global_step}] loss={eval_metrics['eval_loss']:.4f} ppl={eval_metrics['perplexity']:.2f}")

    # Main training epochs (respect resume point)
    for epoch in range(1, cfg.optim.num_epochs + 1):
        one_train_epoch(epoch)

    # Final save (both to Hydra run dir and an optional stable output_dir)
    logger.info("Saving final model...")
    final_dir = os.path.join(ckpt_root, "final")
    save_checkpoint(model, metanetwork, tokenizer, final_dir, extra_state={"global_step": global_step})

    if cfg.paths.output_dir:
        stable_out = cfg.paths.output_dir
        os.makedirs(stable_out, exist_ok=True)
        save_checkpoint(model, metanetwork, tokenizer, stable_out, extra_state={"global_step": global_step})
        logger.info(f"Model saved to {stable_out}")

    logger.info(f"All artifacts in Hydra run dir: {hydra_run_dir}")


if __name__ == "__main__":
    main()
