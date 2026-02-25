#!/usr/bin/env python
# coding: utf-8
"""
Random-sample multi-turn eval for BehaveSHINE.

Runs a side-by-side eval on a random subset of conversations from test.jsonl:
  [1] Base Qwen3-8B + System Prompt
  [2] SHINE + System Prompt

Key point:
- Loads the SAME training config YAML so metanetwork scale (and related params)
  stay consistent with training.
"""

import argparse
import gc
import json
import logging
import os
import random
import re
import sys
from copy import deepcopy
from datetime import datetime
from typing import Tuple, Optional, List

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

assert torch.cuda.is_available(), "CUDA not available!"

# ---------------- project-specific imports ----------------
try:
    from metanetwork_family import Metanetwork
    from utils.mydataset import HumanDataset, HumanCollator
    from utils.myseed import set_seed
    from utils.mysaveload import load_checkpoint
    from utils.myfreeze import freeze
    from utils.myinit import _import_class
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_random_sample")

USE_VANILLA_FOR_BASELINE = True


# =========================================================================
# HELPERS
# =========================================================================

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


def tokenize_messages(tokenizer, messages, max_length):
    # Must match training behavior (esp. enable_thinking=False)
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        return_dict=True,
        enable_thinking=False,
        # no padding
    )
    return enc


def load_jsonl_dataset(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_idx}: {e}")

            # Minimal schema check
            for k in ("context", "questions"):
                if k not in obj:
                    raise ValueError(f"Missing key '{k}' on line {line_idx}")

            # ground_truths optional for some test sets
            if "ground_truths" not in obj:
                obj["ground_truths"] = [""] * len(obj["questions"])

            data.append(obj)

    if not data:
        raise ValueError(f"No conversations found in {path}")
    return data


def sample_conversations(data: List[dict], n: int, seed: int) -> List[dict]:
    if n <= 0:
        raise ValueError("--n must be >= 1")
    if n > len(data):
        logger.warning(f"Requested n={n}, but dataset has only {len(data)} rows. Using all rows.")
        n = len(data)

    rng = random.Random(seed)
    indices = rng.sample(range(len(data)), n)
    logger.info(f"Sampled indices: {indices}")
    return [data[i] for i in indices]


# =========================================================================
# GENERATION
# =========================================================================

@torch.no_grad()
def generate_multiturn_vanilla(
    model,
    dataloader,
    tokenizer,
    device,
    max_new_tokens: int,
    max_conversation_length: int,
):
    model.eval()
    results = []

    for _i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Vanilla baseline"):
        questions = batch["questions"][0]
        initial_msg = batch["initial_messages"][0]
        messages = [initial_msg] if isinstance(initial_msg, dict) and initial_msg else []

        conversation_log = [{"initial message": deepcopy(messages)}]

        for q_idx, question in enumerate(questions):
            messages.append({"role": "user", "content": question})

            enc = tokenize_messages(tokenizer, messages, max_conversation_length)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

            new_tokens = outputs[0, input_ids.shape[1]:]
            raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            think_text, answer_text = extract_think_and_answer(raw_text)

            messages.append({"role": "assistant", "content": answer_text})
            conversation_log.append({
                "turn": q_idx + 1,
                "question": question,
                "think": think_text,
                "answer": answer_text,
            })

        results.append(conversation_log)
    return results


@torch.no_grad()
def generate_multiturn_custom(
    metanetwork,
    dataloader,
    tokenizer,
    device,
    use_metanet: bool = True,
    metalora: Optional[torch.Tensor] = None,
    max_new_tokens: int = 500,
    max_conversation_length: int = 3000,
    external_lora_dicts: Optional[List] = None,
):
    metanetwork.eval()
    results = []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Custom model generation"):
        questions = batch["questions"][0]
        initial_msg = batch["initial_messages"][0]
        messages = [initial_msg] if isinstance(initial_msg, dict) and initial_msg else []

        evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
        evidence_attention_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)

        lora_dict = None
        if external_lora_dicts is not None:
            lora_dict = external_lora_dicts[i]
        elif use_metanet:
            lora_dict = metanetwork.generate_lora_dict(evidence_ids, evidence_attention_mask, metalora)
            lora_dict = cast_lora_dict_dtype(lora_dict, device=device, dtype=torch.float32)

        conversation_log = [{"initial message": deepcopy(messages)}]

        for q_idx, question in enumerate(questions):
            messages.append({"role": "user", "content": question})

            enc = tokenize_messages(tokenizer, messages, max_conversation_length)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = metanetwork.metamodel.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                ignore_mem_token=True,
                loradict=lora_dict,
            )

            new_tokens = outputs[0, input_ids.shape[1]:]
            raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            think_text, answer_text = extract_think_and_answer(raw_text)

            messages.append({"role": "assistant", "content": answer_text})
            conversation_log.append({
                "turn": q_idx + 1,
                "question": question,
                "think": think_text,
                "answer": answer_text,
            })

        results.append(conversation_log)

    return results


@torch.no_grad()
def precompute_lora_dicts(metanetwork, dataloader, metalora, device):
    metanetwork.eval()
    lora_dicts = []
    for batch in tqdm(dataloader, desc="Pre-computing LoRA dicts"):
        evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
        evidence_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)
        lora_dict = metanetwork.generate_lora_dict(evidence_ids, evidence_mask, metalora)
        lora_dict = cast_lora_dict_dtype(lora_dict, device=device, dtype=torch.float32)
        lora_dicts.append(lora_dict)
    return lora_dicts


# =========================================================================
# MODEL INIT
# =========================================================================

def init_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def init_metanetwork(cfg, tokenizer, device):
    MetaModelCls = _import_class(cfg.model.metamodel_class_path)
    ConfigCls = _import_class(cfg.model.config_class_path)
    config = ConfigCls.from_pretrained(cfg.model.model_from)

    # Compute num_mem_token the same way as training
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

    metamodel = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
    metamodel.reset_mem_tokens()
    metamodel.resize_token_embeddings(len(tokenizer))

    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
    metanetwork.to(device)
    freeze(metamodel)
    return metanetwork, config


def load_ckpt(metanetwork, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info(f"Loading checkpoint from {ckpt_path}")
    metanetwork, metalora_ckpt, _ = load_checkpoint(metanetwork, ckpt_path, device)

    metanetwork = metanetwork.to(device=device, dtype=torch.float32)
    metalora_ckpt = cast_lora_dict_dtype(metalora_ckpt, device=device, dtype=torch.float32)
    return metanetwork, metalora_ckpt


def init_vanilla_model(model_path, tokenizer, device):
    logger.info("Loading vanilla Qwen3 model for baseline...")
    vanilla_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )
    vanilla_model.resize_token_embeddings(len(tokenizer))
    vanilla_model.to(device)
    vanilla_model.eval()
    return vanilla_model


# =========================================================================
# OUTPUT
# =========================================================================

def print_results(data, results_baseline, results_shine):
    print("\n" + "=" * 80)
    print("RESULTS — RANDOM SAMPLE MULTI-TURN COMPARISON")
    print("=" * 80 + "\n")

    for i in range(len(data)):
        print(f"\n{'=' * 60}")
        print(f"--- Conversation {i + 1} ---")
        ctx_preview = data[i]["context"][:200].replace("\n", " ") + "..."
        print(f"System Prompt: {ctx_preview}\n")
        print("=" * 60)

        for j in range(len(data[i]["questions"])):
            gt = data[i].get("ground_truths", [""] * len(data[i]["questions"]))[j]
            print(f"\n[Turn {j + 1}] User: {data[i]['questions'][j]}\n")
            print(f"  [Ground Truth]     : {gt}\n")
            print(f"  [Base + SysPrompt] : {results_baseline[i][j + 1]['answer']}\n")
            print(f"  [SHINE + SysPrompt]: {results_shine[i][j + 1]['answer']}\n")
            print("-" * 40)


def save_results(data, results_baseline, results_shine, cfg, ckpt_path, args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f"random_eval_{timestamp}.json")
    txt_path = os.path.join(out_dir, f"random_eval_{timestamp}.txt")

    full_results = {
        "metadata": {
            "timestamp": timestamp,
            "dataset_path": args.dataset,
            "checkpoint_path": ckpt_path,
            "config_path": args.config,
            "model_from": cfg.model.model_from,
            "tokenizer_from": cfg.model.tokenizer_from,
            "max_new_tokens": int(cfg.test.max_new_tokens),
            "conversation_max_length": int(cfg.test.conversation_max_length),
            "context_max_length": int(cfg.test.context_max_length),
            "num_sampled_conversations": len(data),
            "sample_seed": args.sample_seed,
            "use_vanilla_baseline": USE_VANILLA_FOR_BASELINE,
            "metanet_scale": float(cfg.metanetwork.transformer_cfg.scale),
        },
        "conversations": [],
    }

    for i in range(len(data)):
        conv_obj = {
            "conversation_index_in_sample": i,
            "context": data[i]["context"],
            "turns": [],
        }
        for j in range(len(data[i]["questions"])):
            base_turn = results_baseline[i][j + 1]
            shine_turn = results_shine[i][j + 1]
            turn_obj = {
                "turn_index": j + 1,
                "question": data[i]["questions"][j],
                "ground_truth": data[i].get("ground_truths", [""] * len(data[i]["questions"]))[j],
                "base_sysprompt": {
                    "think": base_turn.get("think", ""),
                    "answer": base_turn.get("answer", ""),
                },
                "shine_sysprompt": {
                    "think": shine_turn.get("think", ""),
                    "answer": shine_turn.get("answer", ""),
                },
            }
            conv_obj["turns"].append(turn_obj)
        full_results["conversations"].append(conv_obj)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RESULTS — RANDOM SAMPLE MULTI-TURN COMPARISON\n")
        f.write(f"Vanilla baseline: {USE_VANILLA_FOR_BASELINE}\n")
        f.write(f"Scale (from training config): {float(cfg.metanetwork.transformer_cfg.scale)}\n")
        f.write("=" * 80 + "\n\n")

        for i in range(len(data)):
            f.write("=" * 60 + "\n")
            f.write(f"--- Conversation {i + 1} ---\n")
            f.write(f"Context: {data[i]['context'][:300].replace(chr(10), ' ')}...\n")
            f.write("=" * 60 + "\n")

            for j in range(len(data[i]["questions"])):
                base_turn = results_baseline[i][j + 1]
                shine_turn = results_shine[i][j + 1]
                gt = data[i].get("ground_truths", [""] * len(data[i]["questions"]))[j]

                f.write(f"\n[Turn {j + 1}] User: {data[i]['questions'][j]}\n\n")
                f.write(f"[Ground Truth]\n{gt}\n\n")

                if base_turn.get("think", ""):
                    f.write(f"[Base THINK]\n{base_turn['think']}\n\n")
                f.write(f"[Base ANSWER]\n{base_turn.get('answer', '')}\n\n")

                if shine_turn.get("think", ""):
                    f.write(f"[SHINE THINK]\n{shine_turn['think']}\n\n")
                f.write(f"[SHINE ANSWER]\n{shine_turn.get('answer', '')}\n\n")

                f.write("-" * 40 + "\n")

    logger.info(f"Saved JSON: {json_path}")
    logger.info(f"Saved TXT:  {txt_path}")
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved TXT:  {txt_path}")


# =========================================================================
# ARGPARSE + CONFIG
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="./BehaveSHINE_experiment/training/data/test.jsonl",
        help="Path to test.jsonl",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="Number of random conversations to sample (default: 2)",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed for random sampling",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./BehaveSHINE_experiment/training/configs/train_config.yaml",
        help="Training config YAML (used to keep scale/settings consistent with training)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path (e.g. ./BehaveSHINE_experiment/training/checkpoints/checkpoint-step-2000)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./BehaveSHINE_experiment/eval_outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override cfg.test.max_new_tokens",
    )
    parser.add_argument(
        "--conversation-max-length",
        type=int,
        default=None,
        help="Override cfg.test.conversation_max_length",
    )
    parser.add_argument(
        "--context-max-length",
        type=int,
        default=None,
        help="Override cfg.test.context_max_length",
    )
    return parser.parse_args()


def load_eval_cfg_from_train_config(config_path: str):
    """
    Load the SAME training config so scale and metanetwork settings match training.
    """
    cfg = OmegaConf.load(config_path)

    # Make sure test section exists (some train configs may not have it)
    if "test" not in cfg:
        cfg.test = OmegaConf.create({
            "context_max_length": 1550,
            "conversation_max_length": 3000,
            "max_new_tokens": 500,
        })
    else:
        if "context_max_length" not in cfg.test:
            cfg.test.context_max_length = 1550
        if "conversation_max_length" not in cfg.test:
            cfg.test.conversation_max_length = 3000
        if "max_new_tokens" not in cfg.test:
            cfg.test.max_new_tokens = 500

    # These are set later dynamically
    cfg.hidden_size = -1
    cfg.num_layers = -1
    if "num_mem_token" not in cfg:
        cfg.num_mem_token = 4

    # Make sure paths exist in expected places
    # (Your training config probably already has these)
    assert "model" in cfg, "Training config missing 'model' section"
    assert "model_from" in cfg.model, "Training config missing model.model_from"
    assert "tokenizer_from" in cfg.model, "Training config missing model.tokenizer_from"

    # This is the important bit the user asked about:
    logger.info(f"Using metanetwork scale from training config: {cfg.metanetwork.transformer_cfg.scale}")

    return cfg


# =========================================================================
# MAIN
# =========================================================================

def main():
    args = parse_args()
    device = torch.device("cuda")

    cfg = load_eval_cfg_from_train_config(args.config)

    # Optional overrides
    if args.max_new_tokens is not None:
        cfg.test.max_new_tokens = args.max_new_tokens
    if args.conversation_max_length is not None:
        cfg.test.conversation_max_length = args.conversation_max_length
    if args.context_max_length is not None:
        cfg.test.context_max_length = args.context_max_length

    set_seed(int(cfg.run.seed) if "run" in cfg and "seed" in cfg.run else 42)

    # Load dataset + sample
    all_data = load_jsonl_dataset(args.dataset)
    sampled_data = sample_conversations(all_data, args.n, args.sample_seed)
    logger.info(f"Loaded {len(all_data)} conversations, evaluating {len(sampled_data)} sampled convos")

    # Tokenizer
    tokenizer = init_tokenizer(cfg.model.tokenizer_from)
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    # SHINE model + ckpt
    metanetwork, _ = init_metanetwork(cfg, tokenizer, device)
    metanetwork, metalora_ckpt = load_ckpt(metanetwork, args.checkpoint, device)

    # DataLoader
    human_dataset = HumanDataset(sampled_data)
    test_collator = HumanCollator(
        tokenizer,
        context_max_length=cfg.test.context_max_length,
        conversation_max_length=cfg.test.conversation_max_length,
        cfg=cfg,
        sys_msg=True,
    )
    test_loader = DataLoader(
        human_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=test_collator,
        num_workers=0,
        pin_memory=True,
    )

    # Precompute SHINE LoRAs
    logger.info("Pre-computing LoRA dicts for SHINE...")
    shine_lora_dicts = precompute_lora_dicts(metanetwork, test_loader, metalora_ckpt, device)

    # Baseline
    if USE_VANILLA_FOR_BASELINE:
        logger.info("[1/2] Running VANILLA baseline...")
        vanilla_model = init_vanilla_model(cfg.model.model_from, tokenizer, device)
        results_baseline = generate_multiturn_vanilla(
            vanilla_model,
            test_loader,
            tokenizer,
            device,
            max_new_tokens=int(cfg.test.max_new_tokens),
            max_conversation_length=int(cfg.test.conversation_max_length),
        )
        del vanilla_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        logger.info("[1/2] Running custom baseline (loradict=None)...")
        results_baseline = generate_multiturn_custom(
            metanetwork,
            test_loader,
            tokenizer,
            device,
            use_metanet=False,
            max_new_tokens=int(cfg.test.max_new_tokens),
            max_conversation_length=int(cfg.test.conversation_max_length),
        )

    # SHINE
    logger.info("[2/2] Running SHINE...")
    results_shine = generate_multiturn_custom(
        metanetwork,
        test_loader,
        tokenizer,
        device,
        use_metanet=False,
        max_new_tokens=int(cfg.test.max_new_tokens),
        max_conversation_length=int(cfg.test.conversation_max_length),
        external_lora_dicts=shine_lora_dicts,
    )

    print_results(sampled_data, results_baseline, results_shine)
    save_results(sampled_data, results_baseline, results_shine, cfg, args.checkpoint, args)


if __name__ == "__main__":
    main()