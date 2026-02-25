#!/usr/bin/env python3
# coding: utf-8

import os
import gc
import json
import re
import sys
import argparse
import random
import logging
from datetime import datetime
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Subset
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

    # Your actual dataset/collator module path may differ:
    # e.g. from BehaveSHINE_experiment.training.utils.behaveshine_data import ...
    from BehaveSHINE_experiment.training.scripts.dataset import BehaveSHINEDataset, BehaveSHINECollator
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_random_sample")

# ---------------------------------------------------------------------------
# Must match training dtype. Training used bfloat16, so eval must too.
# ---------------------------------------------------------------------------
EVAL_DTYPE = torch.bfloat16


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


def init_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def init_metanetwork(cfg, tokenizer, device):
    MetaModelCls = _import_class(cfg.model.metamodel_class_path)
    ConfigCls = _import_class(cfg.model.config_class_path)
    config = ConfigCls.from_pretrained(cfg.model.model_from)

    # Compute num_mem_token same as training
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

    # --- FIX: load in bfloat16 to match training ---
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
    # --- FIX: keep bfloat16, not float32 ---
    metanetwork = metanetwork.to(device=device, dtype=EVAL_DTYPE)
    metalora_ckpt = cast_lora_dict_dtype(metalora_ckpt, device=device, dtype=EVAL_DTYPE)
    return metanetwork, metalora_ckpt


def init_vanilla_model(model_path, tokenizer, device):
    logger.info("Loading vanilla baseline model...")
    # --- FIX: load baseline in bfloat16 too for fair comparison ---
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=EVAL_DTYPE)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_baseline(vanilla_model, dataloader, tokenizer, device, max_new_tokens):
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    results = []

    for batch in tqdm(dataloader, desc="Baseline"):
        prompt_only_ids = batch["prompt_only_ids"].to(device)
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
        prompt_mask = (prompt_only_ids != pad_id).long()

        lora_dict = metanetwork.generate_lora_dict(evidence_ids, evidence_mask, metalora_ckpt)
        # --- FIX: keep bfloat16, not float32 ---
        lora_dict = cast_lora_dict_dtype(lora_dict, device=device, dtype=EVAL_DTYPE)

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


def save_outputs(samples, baseline_results, shine_results, args):
    out_dir = "./BehaveSHINE_experiment/eval_outputs"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(out_dir, f"eval_random_sample_{ts}.json")
    txt_path = os.path.join(out_dir, f"eval_random_sample_{ts}.txt")

    rows = []
    for ex, b, s in zip(samples, baseline_results, shine_results):
        rows.append({
            "question_id": ex.get("question_id"),
            "system_prompt_id": ex.get("system_prompt_id"),
            "category": ex.get("category"),
            "question_type": ex.get("question_type"),
            "difficulty_weight": ex.get("difficulty_weight"),
            "context": ex["context"],
            "question": ex["question"],
            "teacher_answer": ex["answer"],
            "answer_8b": ex.get("answer_8b", ""),
            "baseline": b,
            "shine": s,
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "checkpoint": args.checkpoint,
                "dataset": args.dataset,
                "n": args.n,
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
                "context_max_length": args.context_max_length,
                "conversation_max_length": args.conversation_max_length,
                "metanet_scale": args.scale,
                "eval_dtype": str(EVAL_DTYPE),
            },
            "results": rows
        }, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            f.write("=" * 100 + "\n")
            f.write(f"[Sample {i}] qid={row.get('question_id')} category={row.get('category')}\n")
            f.write(f"Question: {row['question']}\n\n")
            f.write("[Teacher]\n")
            f.write(row["teacher_answer"] + "\n\n")
            f.write("[Baseline]\n")
            f.write(row["baseline"]["answer"] + "\n\n")
            f.write("[SHINE]\n")
            f.write(row["shine"]["answer"] + "\n\n")

    logger.info(f"Saved JSON: {json_path}")
    logger.info(f"Saved TXT:  {txt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="./BehaveSHINE_experiment/v2/data/splits/test.jsonl")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_path", default="./models/Qwen3-8B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=500)

    parser.add_argument("--context_max_length", type=int, default=1550)
    parser.add_argument("--conversation_max_length", type=int, default=2800)

    # training-consistent scale
    parser.add_argument("--scale", type=float, default=0.008)

    args = parser.parse_args()

    device = torch.device("cuda")
    set_seed(args.seed)

    logger.info(f"Eval dtype: {EVAL_DTYPE}")
    logger.info(f"Using metanetwork scale from training config: {args.scale}")

    tokenizer = init_tokenizer(args.model_path)

    # Build cfg matching training
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
                "scale": args.scale,  # <-- important
            },
        },
    }
    cfg = OmegaConf.create(conf_dict)

    # Load dataset raw for sampling metadata + collator pipeline
    dataset = BehaveSHINEDataset(args.dataset)
    total = len(dataset)
    n = min(args.n, total)

    rng = random.Random(args.seed)
    sampled_indices = rng.sample(list(range(total)), n)
    logger.info(f"Sampled indices: {sampled_indices}")
    subset = Subset(dataset, sampled_indices)

    # For saving metadata we also want raw rows
    raw_rows = [dataset.data[i] for i in sampled_indices]

    collator = BehaveSHINECollator(
        tokenizer=tokenizer,
        context_max_length=args.context_max_length,
        conversation_max_length=args.conversation_max_length,
    )
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    # SHINE
    metanetwork = init_metanetwork(cfg, tokenizer, device)
    metanetwork, metalora_ckpt = load_ckpt(metanetwork, args.checkpoint, device)

    # Baseline
    vanilla = init_vanilla_model(args.model_path, tokenizer, device)
    baseline_results = run_baseline(vanilla, loader, tokenizer, device, args.max_new_tokens)

    del vanilla
    gc.collect()
    torch.cuda.empty_cache()

    # Rebuild loader because iterators are exhausted
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    shine_results = run_shine(metanetwork, metalora_ckpt, loader, tokenizer, device, args.max_new_tokens)

    # Print quick view
    for i, (ex, b, s) in enumerate(zip(raw_rows, baseline_results, shine_results), 1):
        print("\n" + "=" * 80)
        print(f"[Sample {i}] {ex.get('question_id', '')}")
        print(f"Q: {ex['question']}\n")
        print(f"TEACHER:\n{ex['answer']}\n")
        print(f"BASELINE:\n{b['answer']}\n")
        print(f"SHINE:\n{s['answer']}\n")

    save_outputs(raw_rows, baseline_results, shine_results, args)


if __name__ == "__main__":
    main()