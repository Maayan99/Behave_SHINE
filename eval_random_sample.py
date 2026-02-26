#!/usr/bin/env python3
# coding: utf-8
"""
eval_random_sample.py — BehaveSHINE evaluation with multi-checkpoint comparison.

Modes:
  1. Random sample from test set (default):
     python eval_random_sample.py \\
       --checkpoints step-1000:./ckpts/checkpoint-step-1000 \\
       --n 10 --seed 42 --scale 0.001

  2. Adversarial stress-test (--adversarial):
     python eval_random_sample.py \\
       --checkpoints step-1000:./ckpts/checkpoint-step-1000 \\
       --adversarial adversarial_test_cases.json \\
       --scale 0.001

     Runs ALL cases in the adversarial file (ignores --n and --seed for sampling).
     The adversarial JSON is a list of objects with at minimum:
       { "system_prompt_id", "category", "question_type", "difficulty_weight",
         "context", "question" }
     "notes" field is preserved in output for human review but not fed to models.

  3. Specific indices from test set (--indices):
     python eval_random_sample.py \\
       --checkpoints step-1000:./ckpts/checkpoint-step-1000 \\
       --indices 0 5 12 99 \\
       --scale 0.001
"""

import os
import gc
import json
import re
import sys
import argparse
import random
import logging
from collections import OrderedDict
from datetime import datetime
from typing import Tuple, List, Dict, Any

import torch
from torch.utils.data import DataLoader, Subset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import OmegaConf
from tqdm.auto import tqdm

# Project imports
try:
    from metanetwork_family import Metanetwork
    from utils.myseed import set_seed
    from utils.mysaveload import load_checkpoint
    from utils.myfreeze import freeze
    from utils.myinit import _import_class
    from BehaveSHINE_experiment.training.scripts.dataset import BehaveSHINEDataset, BehaveSHINECollator
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_random_sample")

EVAL_DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Adversarial dataset wrapper
# ---------------------------------------------------------------------------

class AdversarialDataset(Dataset):
    """Wraps a list of adversarial test cases to look like BehaveSHINEDataset."""

    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.data = []
        for i, item in enumerate(raw):
            row = {
                "context": item["context"],
                "question": item["question"],
                # No teacher answer — we're generating all responses
                "answer": item.get("teacher_answer", "(adversarial — no teacher answer)"),
                "answer_8b": item.get("answer_8b", ""),
                "question_id": item.get("question_id", f"adv_{i+1:03d}"),
                "system_prompt_id": item.get("system_prompt_id", f"sp_ADV{i+1:02d}"),
                "category": item.get("category", "adversarial"),
                "question_type": item.get("question_type", "on_topic"),
                "difficulty_weight": item.get("difficulty_weight", 3.0),
                # Preserve notes for output
                "_notes": item.get("notes", ""),
            }
            self.data.append(row)

        logger.info(f"Loaded {len(self.data)} adversarial test cases from {json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_think_and_answer(text: str) -> Tuple[str, str]:
    think, answer = "", text
    if "<think>" in text:
        parts = text.split("<think>", 1)
        rest = parts[1]
        if "</think>" in rest:
            think_part, answer_part = rest.split("</think>", 1)
            think, answer = think_part.strip(), answer_part.strip()
        else:
            think, answer = rest.strip(), ""
    else:
        answer = text.strip()
    answer = re.sub(r"^(final answer|answer)\s*:\s*", "", answer, flags=re.IGNORECASE).strip()
    return think, answer


def cast_lora_dict_dtype(obj, device, dtype):
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, dtype=dtype)
    if isinstance(obj, dict):
        return {k: cast_lora_dict_dtype(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cast_lora_dict_dtype(v, device, dtype) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cast_lora_dict_dtype(v, device, dtype) for v in obj)
    return obj


def parse_checkpoint_arg(raw: str) -> Tuple[str, str]:
    """Parse 'label:path' or just 'path' (label = directory basename)."""
    if ":" in raw and not raw.startswith("/") and not raw.startswith("./"):
        label, path = raw.split(":", 1)
        return label.strip(), path.strip()
    if ":" in raw:
        parts = raw.split(":", 1)
        if os.path.exists(parts[1].strip()) or parts[1].strip().startswith("./") or parts[1].strip().startswith("/"):
            return parts[0].strip(), parts[1].strip()
    basename = os.path.basename(raw.rstrip("/"))
    return basename, raw


def make_dataloader(subset_or_dataset, collator, batch_size):
    return DataLoader(
        subset_or_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Init / load
# ---------------------------------------------------------------------------

def init_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def build_cfg(args):
    conf_dict = {
        "run": {"seed": args.seed, "device": "cuda"},
        "paths": {"model_path": args.model_path},
        "model": {
            "lora_r": 8,
            "metalora_r": 128,
            "num_mem_token": 4,
            "metamodel_class_path": "LoraQwen.LoraQwen3ForCausalLM",
            "config_class_path": "LoraQwen.Qwen3Config",
            "tokenizer_from": args.model_path,
            "model_from": args.model_path,
        },
        "metanetwork": {
            "type": "transformer",
            "method": "rl",
            "transformer_cfg": {
                "encoder_cfg": {
                    "d_model": 4096, "nhead": 32, "dim_feedforward": 8192,
                    "dropout": 0, "activation": "gelu", "layer_norm_eps": 1e-5,
                    "batch_first": True, "norm_first": False, "bias": True,
                },
                "couple_encoder_cfg": {
                    "d_model": 4096, "nhead": 32, "dim_feedforward": 8192,
                    "dropout": 0, "activation": "gelu", "layer_norm_eps": 1e-5,
                    "batch_first": True, "norm_first": False, "bias": True,
                },
                "layer_transformer_first": True,
                "mean_pool_size": 1,
                "num_layers": 4,
                "couple_num_layers": 0,
                "scale": args.scale,
            },
        },
    }
    return OmegaConf.create(conf_dict)


def init_metanetwork(cfg, tokenizer, device):
    MetaModelCls = _import_class(cfg.model.metamodel_class_path)
    ConfigCls = _import_class(cfg.model.config_class_path)
    config = ConfigCls.from_pretrained(cfg.model.model_from)

    config.num_mem_token = -1
    cfg.hidden_size = config.hidden_size
    cfg.num_layers = config.num_hidden_layers

    tmp_model = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
    lora_params = tmp_model.lora_params_numel(cfg.model.lora_r)
    base_params = cfg.hidden_size * cfg.num_layers
    config.num_mem_token = lora_params // base_params
    cfg.num_mem_token = config.num_mem_token
    del tmp_model
    gc.collect()
    torch.cuda.empty_cache()

    metamodel = MetaModelCls.from_pretrained(
        cfg.model.model_from, config=config, torch_dtype=EVAL_DTYPE
    )
    metamodel.reset_mem_tokens()
    metamodel.resize_token_embeddings(len(tokenizer))

    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
    metanetwork.to(device=device, dtype=EVAL_DTYPE)
    freeze(metamodel)
    return metanetwork


def load_ckpt(metanetwork, ckpt_path, device):
    logger.info(f"Loading checkpoint: {ckpt_path}")
    metanetwork, metalora_ckpt, _ = load_checkpoint(metanetwork, ckpt_path, device)
    metanetwork = metanetwork.to(device=device, dtype=EVAL_DTYPE)
    metanetwork.metanetwork.float()
    metalora_ckpt = cast_lora_dict_dtype(metalora_ckpt, device=device, dtype=torch.float32)
    return metanetwork, metalora_ckpt


def init_vanilla_model(model_path, tokenizer, device):
    logger.info("Loading vanilla baseline model...")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=EVAL_DTYPE)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def add_no_think_prefix(input_ids, tokenizer, device):
    """Prepend <think>\n</think>\n to force model to skip thinking."""
    prefix = tokenizer.encode("<think>\n</think>\n", add_special_tokens=False)
    prefix_t = torch.tensor([prefix], device=device).expand(input_ids.shape[0], -1)
    return torch.cat([input_ids, prefix_t], dim=1)


@torch.no_grad()
def run_baseline(vanilla_model, dataloader, tokenizer, device, max_new_tokens):
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    results = []

    for batch in tqdm(dataloader, desc="Baseline"):
        prompt_only_ids = batch["prompt_only_ids"].to(device)
        prompt_only_ids = add_no_think_prefix(prompt_only_ids, tokenizer, device)
        attention_mask = (prompt_only_ids != pad_id).long()

        outputs = vanilla_model.generate(
            input_ids=prompt_only_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            do_sample=False,
        )

        for i in range(outputs.shape[0]):
            in_len = int(attention_mask[i].sum().item())
            new_tokens = outputs[i, in_len:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
            think, ans = extract_think_and_answer(raw)
            results.append({"think": think, "answer": ans})

    return results


@torch.no_grad()
def run_shine(metanetwork, metalora_ckpt, dataloader, tokenizer, device, max_new_tokens):
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    metanetwork.eval()
    results = []

    for batch in tqdm(dataloader, desc="SHINE"):
        evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
        evidence_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)

        prompt_only_ids = batch["prompt_only_ids"].to(device)
        prompt_only_ids = add_no_think_prefix(prompt_only_ids, tokenizer, device)
        prompt_mask = (prompt_only_ids != pad_id).long()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            lora_dict = metanetwork.generate_lora_dict(evidence_ids, evidence_mask, metalora_ckpt)

            outputs = metanetwork.metamodel.generate(
                input_ids=prompt_only_ids,
                attention_mask=prompt_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                do_sample=False,
                ignore_mem_token=True,
                loradict=lora_dict,
            )

        for i in range(outputs.shape[0]):
            in_len = int(prompt_mask[i].sum().item())
            new_tokens = outputs[i, in_len:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
            think, ans = extract_think_and_answer(raw)
            results.append({"think": think, "answer": ans})

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_outputs(samples, baseline_results, shine_results_by_label, args, checkpoint_labels, is_adversarial=False):
    out_dir = "./BehaveSHINE_experiment/eval_outputs"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    prefix = "eval_adversarial" if is_adversarial else "eval_compare"
    json_path = os.path.join(out_dir, f"{prefix}_{ts}.json")
    txt_path = os.path.join(out_dir, f"{prefix}_{ts}.txt")

    rows = []
    for idx, ex in enumerate(samples):
        row = {
            "question_id": ex.get("question_id"),
            "system_prompt_id": ex.get("system_prompt_id"),
            "category": ex.get("category"),
            "question_type": ex.get("question_type"),
            "difficulty_weight": ex.get("difficulty_weight"),
            "context": ex["context"],
            "question": ex["question"],
            "teacher_answer": ex.get("answer", ""),
            "answer_8b": ex.get("answer_8b", ""),
            "baseline": baseline_results[idx],
        }
        # Include notes for adversarial cases
        if ex.get("_notes"):
            row["_notes"] = ex["_notes"]
        for label in checkpoint_labels:
            row[f"shine_{label}"] = shine_results_by_label[label][idx]
        rows.append(row)

    meta = {
        "mode": "adversarial" if is_adversarial else "random_sample",
        "checkpoints": {label: label for label in checkpoint_labels},
        "n": len(rows),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "context_max_length": args.context_max_length,
        "conversation_max_length": args.conversation_max_length,
        "metanet_scale": args.scale,
        "eval_dtype": str(EVAL_DTYPE),
    }
    if not is_adversarial:
        meta["dataset"] = args.dataset

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "results": rows}, f, ensure_ascii=False, indent=2)

    # Human-readable text output
    sep = "=" * 100
    thin_sep = "-" * 80

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"BehaveSHINE {'Adversarial Stress Test' if is_adversarial else 'Multi-Checkpoint Comparison'}\n")
        f.write(f"Generated: {ts}\n")
        f.write(f"Scale: {args.scale}  Seed: {args.seed}  N: {len(rows)}\n")
        f.write(f"Checkpoints: {', '.join(checkpoint_labels)}\n")
        f.write(sep + "\n\n")

        for i, row in enumerate(rows, 1):
            f.write(sep + "\n")
            f.write(f"[Sample {i}]  qid={row.get('question_id')}  "
                    f"category={row.get('category')}  "
                    f"type={row.get('question_type')}  "
                    f"difficulty={row.get('difficulty_weight')}\n")

            # Show notes for adversarial cases
            if row.get("_notes"):
                f.write(f"\n>>> EXPECTED FAILURE MODES:\n{row['_notes']}\n")

            f.write(thin_sep + "\n")
            f.write(f"CONTEXT:\n{row['context'][:500]}{'...' if len(row['context']) > 500 else ''}\n\n")
            f.write(f"QUESTION:\n{row['question']}\n\n")

            if row.get("teacher_answer") and row["teacher_answer"] != "(adversarial — no teacher answer)":
                f.write(f"TEACHER (32B):\n{row['teacher_answer']}\n\n")

            f.write(thin_sep + "\n")
            f.write(f"BASELINE (8B, no LoRA):\n{row['baseline']['answer']}\n\n")

            for label in checkpoint_labels:
                f.write(thin_sep + "\n")
                f.write(f"SHINE [{label}]:\n{row[f'shine_{label}']['answer']}\n\n")

            f.write("\n")

    logger.info(f"Saved JSON: {json_path}")
    logger.info(f"Saved TXT:  {txt_path}")

    # Print to stdout
    print(f"\n{'=' * 100}")
    title = "ADVERSARIAL STRESS TEST" if is_adversarial else "COMPARISON SUMMARY"
    print(f"  {title}  (scale={args.scale}, n={len(rows)})")
    print(f"{'=' * 100}\n")

    for i, row in enumerate(rows, 1):
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(rows)}]  {row.get('question_id', '')}  |  {row.get('category', '')}  |  diff={row.get('difficulty_weight', '')}")

        if row.get("_notes"):
            print(f"  >>> EXPECTED: {row['_notes'][:200]}{'...' if len(row.get('_notes','')) > 200 else ''}")

        print(f"  Q: {row['question'][:200]}{'...' if len(row['question']) > 200 else ''}\n")

        print(f"  BASELINE (8B):")
        print(f"    {row['baseline']['answer'][:400]}{'...' if len(row['baseline']['answer']) > 400 else ''}\n")

        for label in checkpoint_labels:
            ans = row[f"shine_{label}"]["answer"]
            print(f"  SHINE [{label}]:")
            print(f"    {ans[:400]}{'...' if len(ans) > 400 else ''}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BehaveSHINE eval with multi-checkpoint comparison and adversarial mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Checkpoint args
    parser.add_argument("--checkpoint", default=None,
                        help="Single checkpoint path (backward-compatible)")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="Multiple checkpoints as 'label:path' or just 'path'")

    # Mode selection
    parser.add_argument("--adversarial", default=None, metavar="JSON_PATH",
                        help="Path to adversarial test cases JSON. Runs ALL cases in the file.")
    parser.add_argument("--indices", nargs="+", type=int, default=None,
                        help="Specific indices to evaluate from the dataset (instead of random)")

    # Dataset / sampling
    parser.add_argument("--dataset", default="./BehaveSHINE_experiment/v2/data/splits/test.jsonl")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # Model / generation
    parser.add_argument("--model_path", default="./models/Qwen3-8B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=500)

    parser.add_argument("--context_max_length", type=int, default=1550)
    parser.add_argument("--conversation_max_length", type=int, default=2800)

    parser.add_argument("--scale", type=float, default=0.001)

    parser.add_argument("--no_baseline", action="store_true",
                        help="Skip baseline generation")

    args = parser.parse_args()

    # ---- Resolve checkpoints ----
    ckpt_list: List[Tuple[str, str]] = []
    if args.checkpoints:
        for raw in args.checkpoints:
            label, path = parse_checkpoint_arg(raw)
            ckpt_list.append((label, path))
    elif args.checkpoint:
        label, path = parse_checkpoint_arg(args.checkpoint)
        ckpt_list.append((label, path))
    else:
        parser.error("Provide --checkpoint or --checkpoints")

    for label, path in ckpt_list:
        if not os.path.exists(path):
            parser.error(f"Checkpoint not found: {path} (label={label})")

    device = torch.device("cuda")
    set_seed(args.seed)

    is_adversarial = args.adversarial is not None

    logger.info(f"Eval dtype: {EVAL_DTYPE}")
    logger.info(f"Scale: {args.scale}")
    logger.info(f"Mode: {'ADVERSARIAL' if is_adversarial else 'indices' if args.indices else 'random sample'}")
    logger.info(f"Checkpoints ({len(ckpt_list)}):")
    for label, path in ckpt_list:
        logger.info(f"  [{label}] -> {path}")

    tokenizer = init_tokenizer(args.model_path)
    cfg = build_cfg(args)

    # ---- Load data ----
    if is_adversarial:
        # Adversarial mode: load custom test cases
        if not os.path.exists(args.adversarial):
            parser.error(f"Adversarial file not found: {args.adversarial}")

        adv_dataset = AdversarialDataset(args.adversarial)
        raw_rows = adv_dataset.data
        eval_dataset = adv_dataset
        n = len(adv_dataset)
        logger.info(f"Adversarial mode: {n} test cases loaded")

    elif args.indices:
        # Specific indices mode
        dataset = BehaveSHINEDataset(args.dataset)
        for idx in args.indices:
            if idx >= len(dataset):
                parser.error(f"Index {idx} out of range (dataset has {len(dataset)} items)")
        eval_dataset = Subset(dataset, args.indices)
        raw_rows = [dataset.data[i] for i in args.indices]
        n = len(args.indices)
        logger.info(f"Index mode: {n} specific examples")

    else:
        # Random sample mode (default)
        dataset = BehaveSHINEDataset(args.dataset)
        total = len(dataset)
        n = min(args.n, total)
        rng = random.Random(args.seed)
        sampled_indices = rng.sample(list(range(total)), n)
        logger.info(f"Sampled {n} examples, indices: {sampled_indices}")
        eval_dataset = Subset(dataset, sampled_indices)
        raw_rows = [dataset.data[i] for i in sampled_indices]

    collator = BehaveSHINECollator(
        tokenizer=tokenizer,
        context_max_length=args.context_max_length,
        conversation_max_length=args.conversation_max_length,
    )

    # ---- Baseline ----
    if not args.no_baseline:
        vanilla = init_vanilla_model(args.model_path, tokenizer, device)
        loader = make_dataloader(eval_dataset, collator, args.batch_size)
        baseline_results = run_baseline(vanilla, loader, tokenizer, device, args.max_new_tokens)
        del vanilla
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Baseline complete.")
    else:
        baseline_results = [{"think": "", "answer": "(skipped)"} for _ in range(n)]
        logger.info("Baseline skipped (--no_baseline).")

    # ---- SHINE for each checkpoint ----
    metanetwork = init_metanetwork(cfg, tokenizer, device)

    shine_results_by_label: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
    checkpoint_labels = []

    for ckpt_idx, (label, path) in enumerate(ckpt_list):
        logger.info(f"\n{'='*60}")
        logger.info(f"Checkpoint {ckpt_idx+1}/{len(ckpt_list)}: [{label}]")
        logger.info(f"{'='*60}")

        metanetwork, metalora_ckpt = load_ckpt(metanetwork, path, device)

        loader = make_dataloader(eval_dataset, collator, args.batch_size)
        results = run_shine(metanetwork, metalora_ckpt, loader, tokenizer, device, args.max_new_tokens)

        shine_results_by_label[label] = results
        checkpoint_labels.append(label)

        del metalora_ckpt
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"[{label}] complete — {len(results)} samples generated.")

    # ---- Save & print ----
    save_outputs(raw_rows, baseline_results, shine_results_by_label, args, checkpoint_labels, is_adversarial=is_adversarial)

    logger.info("Done.")


if __name__ == "__main__":
    main()