#!/usr/bin/env python3
# coding: utf-8
"""
eval_gpt4tools.py — BehaveSHINE evaluation on GPT4Tools benchmark.

Parses GPT4Tools test data, extracts system prompts + conversation history + user queries,
generates LoRAs via SHINE, and compares against baseline (no LoRA) responses.

Scoring follows the GPT4Tools paper (Yang et al., 2024a):
  - SRt:    Successful rate of Thought (did it decide tool use correctly?)
  - SRact:  Successful rate of Action (correct tool name?)
  - SRargs: Successful rate of Arguments (correct action input?)
  - SR:     Overall success rate (all three correct)

Usage:
  python eval_gpt4tools.py \\
    --checkpoints step-1000:./ckpts/checkpoint-step-1000 \\
    --gpt4tools_path ./data/gpt4tools_test_unseen.json \\
    --n 300 --seed 42 --scale 0.001
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
from typing import Tuple, List, Dict, Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset
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
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_gpt4tools")

EVAL_DTYPE = torch.bfloat16


# ===========================================================================
# GPT4Tools Parsing
# ===========================================================================

def parse_gpt4tools_entry(entry: dict) -> dict:
    """
    Parse a GPT4Tools JSON entry into structured components.

    The 'instruction' field contains:
      1. System prompt (tool descriptions + formatting rules)
      2. "Previous conversation:" section (may be empty)
      3. "New input:" with user query + CoT suffix
      4. Possibly partial assistant trace (multi-step tool chains)

    Returns dict with keys:
      system_prompt, conversation_history (list of role/content),
      user_query, partial_trace, expected_output, raw_instruction, id
    """
    instruction = entry["instruction"]
    output = entry.get("output", "")
    entry_id = entry.get("id", None)

    # --- Split at "Previous conversation:\n" ---
    prev_conv_marker = "Previous conversation:\n"
    if prev_conv_marker in instruction:
        system_part, rest = instruction.split(prev_conv_marker, 1)
    else:
        system_part = instruction
        rest = ""

    system_prompt = system_part.strip()

    # --- Split at "New input: " ---
    new_input_marker = "\nNew input: "
    conversation_history = []
    user_query = ""

    if new_input_marker in rest:
        conv_part, query_part = rest.split(new_input_marker, 1)

        # Parse conversation history
        conv_part = conv_part.strip()
        if conv_part:
            conversation_history = _parse_conversation_turns(conv_part)

        # Split user query from the CoT suffix
        cot_marker = "\nGPT4Tools needs to use tools"
        if cot_marker in query_part:
            user_query, after_cot = query_part.split(cot_marker, 1)
            user_query = user_query.strip()
        else:
            user_query = query_part.strip()
            after_cot = ""
    else:
        # Fallback — try simpler split
        user_query = rest.strip()
        after_cot = ""

    # --- Extract partial trace (multi-step continuations) ---
    partial_trace = ""
    step_marker = "Let's think step by step.\n"
    if step_marker in (after_cot or ""):
        after_step = after_cot.split(step_marker, 1)[1].strip()
        if after_step and after_step.startswith("Thought:"):
            partial_trace = after_step

    return {
        "id": entry_id,
        "system_prompt": system_prompt,
        "conversation_history": conversation_history,
        "user_query": user_query,
        "partial_trace": partial_trace,
        "expected_output": output.strip(),
        "raw_instruction": instruction,
    }


def _parse_conversation_turns(conv_text: str) -> List[Dict[str, str]]:
    """Parse 'Human: .../AI: ...' blocks into list of role/content dicts."""
    messages = []
    current_role = None
    current_content = []

    for line in conv_text.strip().split("\n"):
        if line.startswith("Human: "):
            if current_role is not None:
                messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
            current_role = "user"
            current_content = [line[len("Human: "):]]
        elif line.startswith("AI: "):
            if current_role is not None:
                messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
            current_role = "assistant"
            current_content = [line[len("AI: "):]]
        else:
            if current_role is not None:
                current_content.append(line)

    if current_role is not None:
        messages.append({"role": current_role, "content": "\n".join(current_content).strip()})

    return messages


# ===========================================================================
# GPT4Tools Scoring (SRt / SRact / SRargs / SR)
# ===========================================================================

def extract_thought_action(text: str) -> Dict[str, Optional[str]]:
    """
    Extract structured fields from a Thought/Action/Action Input response.
    Returns dict with keys: thought_yes (bool or None), action, action_input, ai_response.
    """
    result = {"thought_yes": None, "action": None, "action_input": None, "ai_response": None}

    # Check for "Thought: Do I need to use a tool? Yes/No"
    thought_match = re.search(
        r"Thought:\s*Do I need to use a tool\?\s*(Yes|No)",
        text, re.IGNORECASE
    )
    if thought_match:
        result["thought_yes"] = thought_match.group(1).strip().lower() == "yes"

    # Extract Action
    action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
    if action_match:
        result["action"] = action_match.group(1).strip()

    # Extract Action Input
    input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text)
    if input_match:
        result["action_input"] = input_match.group(1).strip()

    # Extract AI response (for no-tool cases)
    ai_match = re.search(r"AI:\s*(.+?)(?:\n|$)", text, re.DOTALL)
    if ai_match:
        result["ai_response"] = ai_match.group(1).strip()

    return result


def score_entry(expected_output: str, model_output: str) -> Dict[str, Any]:
    """
    Score a single entry following GPT4Tools metrics.

    Returns dict with:
      thought_correct (bool), action_correct (bool), args_correct (bool),
      full_correct (bool), expected_parsed, model_parsed
    """
    exp = extract_thought_action(expected_output)
    mod = extract_thought_action(model_output)

    # SRt: Did the model make the right tool-use decision?
    thought_correct = False
    if exp["thought_yes"] is not None and mod["thought_yes"] is not None:
        thought_correct = exp["thought_yes"] == mod["thought_yes"]

    # SRact: Did the model pick the right tool?
    action_correct = False
    if exp["action"] is not None and mod["action"] is not None:
        # Normalize whitespace and case for comparison
        action_correct = exp["action"].strip().lower() == mod["action"].strip().lower()
    elif exp["action"] is None and mod["action"] is None:
        # Both correctly decided no tool needed
        action_correct = True

    # SRargs: Did the model provide the right arguments?
    args_correct = False
    if exp["action_input"] is not None and mod["action_input"] is not None:
        # For action inputs, we do case-insensitive comparison
        # and also handle path variations (the model might use a different path format)
        exp_input = exp["action_input"].strip().lower()
        mod_input = mod["action_input"].strip().lower()
        args_correct = exp_input == mod_input

        # Relaxed: if the core content matches (ignoring path prefixes)
        if not args_correct:
            # Strip common path prefixes for comparison
            exp_base = exp_input.split("/")[-1] if "/" in exp_input else exp_input
            mod_base = mod_input.split("/")[-1] if "/" in mod_input else mod_input
            args_correct = exp_base == mod_base
    elif exp["action_input"] is None and mod["action_input"] is None:
        # No tool needed — no args expected
        args_correct = True

    # For no-tool cases, check if AI response is reasonable
    if exp["thought_yes"] is False and mod["thought_yes"] is False:
        # Both said no tool needed — that's a full success on the structural level
        # (content quality of the AI response is harder to auto-grade)
        action_correct = True
        args_correct = True

    full_correct = thought_correct and action_correct and args_correct

    return {
        "thought_correct": thought_correct,
        "action_correct": action_correct,
        "args_correct": args_correct,
        "full_correct": full_correct,
        "expected_parsed": exp,
        "model_parsed": mod,
    }


def compute_aggregate_scores(scores: List[Dict]) -> Dict[str, float]:
    """Compute SRt, SRact, SRargs, SR from a list of per-entry scores."""
    n = len(scores)
    if n == 0:
        return {"SRt": 0.0, "SRact": 0.0, "SRargs": 0.0, "SR": 0.0, "n": 0}

    return {
        "SRt": sum(1 for s in scores if s["thought_correct"]) / n,
        "SRact": sum(1 for s in scores if s["action_correct"]) / n,
        "SRargs": sum(1 for s in scores if s["args_correct"]) / n,
        "SR": sum(1 for s in scores if s["full_correct"]) / n,
        "n": n,
    }


# ===========================================================================
# Dataset wrapper
# ===========================================================================

class GPT4ToolsDataset(Dataset):
    """
    Wraps parsed GPT4Tools entries to work with BehaveSHINE's collation.
    Each __getitem__ returns a dict with system_prompt, question, answer
    matching what BehaveSHINECollator expects.
    """

    def __init__(self, parsed_entries: List[dict], tokenizer):
        self.entries = parsed_entries
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Build the full conversation prompt for the model.
        # system_prompt -> evidence for the hypernetwork
        # conversation_history + user_query -> the actual prompt to generate from
        messages = []

        # Add system prompt as system message
        messages.append({"role": "system", "content": entry["system_prompt"]})

        # Add previous conversation turns
        for turn in entry["conversation_history"]:
            messages.append(turn)

        # Add the current user query
        user_content = entry["user_query"]
        # If there's a partial trace, append it as context
        # (the model is expected to continue from here)
        if entry["partial_trace"]:
            user_content += "\n" + entry["partial_trace"]

        messages.append({"role": "user", "content": user_content})

        # Build prompt string using tokenizer chat template
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return {
            "system_prompt": entry["system_prompt"],
            "question": user_content,
            "answer": entry["expected_output"],
            "prompt_text": prompt_text,
            "difficulty_weight": 1.0,
        }


class GPT4ToolsCollator:
    """
    Collates GPT4Tools entries for both the hypernetwork (evidence) and generation (prompt).
    """

    def __init__(self, tokenizer, context_max_length=1550, conversation_max_length=2800):
        self.tokenizer = tokenizer
        self.context_max_length = context_max_length
        self.conversation_max_length = conversation_max_length

    def __call__(self, batch):
        # Evidence = system prompts (for hypernetwork to generate LoRA from)
        evidence_texts = [item["system_prompt"] for item in batch]
        evidence_enc = self.tokenizer(
            evidence_texts,
            max_length=self.context_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Prompt = full chat-formatted prompt (for generation)
        prompt_texts = [item["prompt_text"] for item in batch]
        prompt_enc = self.tokenizer(
            prompt_texts,
            max_length=self.conversation_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "evidence_ids": evidence_enc["input_ids"],
            "evidence_attention_mask": evidence_enc["attention_mask"],
            "prompt_only_ids": prompt_enc["input_ids"],
        }


# ===========================================================================
# Helpers (reused from eval_random_sample.py)
# ===========================================================================

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
    if ":" in raw and not raw.startswith("/") and not raw.startswith("./"):
        label, path = raw.split(":", 1)
        return label.strip(), path.strip()
    if ":" in raw:
        parts = raw.split(":", 1)
        if os.path.exists(parts[1].strip()) or parts[1].strip().startswith(("./", "/")):
            return parts[0].strip(), parts[1].strip()
    return os.path.basename(raw.rstrip("/")), raw


def make_dataloader(dataset, collator, batch_size):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )


# ===========================================================================
# Init / Load (reused from eval_random_sample.py)
# ===========================================================================

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
            "lora_r": 8, "metalora_r": 128, "num_mem_token": 4,
            "metamodel_class_path": "LoraQwen.LoraQwen3ForCausalLM",
            "config_class_path": "LoraQwen.Qwen3Config",
            "tokenizer_from": args.model_path, "model_from": args.model_path,
        },
        "metanetwork": {
            "type": "transformer", "method": "rl",
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
                "layer_transformer_first": True, "mean_pool_size": 1,
                "num_layers": 4, "couple_num_layers": 0, "scale": args.scale,
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
    gc.collect(); torch.cuda.empty_cache()

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
    model.to(device); model.eval()
    return model


# ===========================================================================
# Generation
# ===========================================================================

def add_no_think_prefix(input_ids, tokenizer, device):
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
            input_ids=prompt_only_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, pad_token_id=pad_id,
            eos_token_id=eos_id, do_sample=False,
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
                input_ids=prompt_only_ids, attention_mask=prompt_mask,
                max_new_tokens=max_new_tokens, pad_token_id=pad_id,
                eos_token_id=eos_id, do_sample=False,
                ignore_mem_token=True, loradict=lora_dict,
            )

        for i in range(outputs.shape[0]):
            in_len = int(prompt_mask[i].sum().item())
            new_tokens = outputs[i, in_len:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
            think, ans = extract_think_and_answer(raw)
            results.append({"think": think, "answer": ans})

    return results


# ===========================================================================
# Output
# ===========================================================================

def save_outputs(parsed_entries, baseline_results, baseline_scores,
                 shine_results_by_label, shine_scores_by_label,
                 args, checkpoint_labels):
    out_dir = "./BehaveSHINE_experiment/eval_outputs"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(out_dir, f"eval_gpt4tools_{ts}.json")
    txt_path = os.path.join(out_dir, f"eval_gpt4tools_{ts}.txt")

    # --- Build per-entry rows ---
    rows = []
    for idx, entry in enumerate(parsed_entries):
        row = {
            "id": entry["id"],
            "system_prompt": entry["system_prompt"][:300] + "..." if len(entry["system_prompt"]) > 300 else entry["system_prompt"],
            "conversation_history": entry["conversation_history"],
            "user_query": entry["user_query"],
            "partial_trace": entry["partial_trace"],
            "expected_output": entry["expected_output"],
            "baseline": {
                "response": baseline_results[idx],
                "scores": baseline_scores[idx],
            },
        }
        for label in checkpoint_labels:
            row[f"shine_{label}"] = {
                "response": shine_results_by_label[label][idx],
                "scores": shine_scores_by_label[label][idx],
            }
        rows.append(row)

    # --- Aggregate scores ---
    baseline_agg = compute_aggregate_scores(baseline_scores)
    shine_aggs = {}
    for label in checkpoint_labels:
        shine_aggs[label] = compute_aggregate_scores(shine_scores_by_label[label])

    meta = {
        "benchmark": "GPT4Tools",
        "gpt4tools_path": args.gpt4tools_path,
        "n": len(rows),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "metanet_scale": args.scale,
        "eval_dtype": str(EVAL_DTYPE),
        "aggregate_scores": {
            "baseline": baseline_agg,
            **{f"shine_{label}": shine_aggs[label] for label in checkpoint_labels},
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "results": rows}, f, ensure_ascii=False, indent=2)

    # --- Human-readable output ---
    sep = "=" * 100
    thin_sep = "-" * 80

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"BehaveSHINE — GPT4Tools Benchmark Evaluation\n")
        f.write(f"Generated: {ts}\n")
        f.write(f"Scale: {args.scale}  Seed: {args.seed}  N: {len(rows)}\n")
        f.write(f"Checkpoints: {', '.join(checkpoint_labels)}\n")
        f.write(sep + "\n\n")

        # Aggregate scores summary
        f.write("AGGREGATE SCORES\n")
        f.write(thin_sep + "\n")
        f.write(f"{'Method':<25} {'SRt':>8} {'SRact':>8} {'SRargs':>8} {'SR':>8}\n")
        f.write(thin_sep + "\n")
        f.write(f"{'Baseline (no LoRA)':<25} {baseline_agg['SRt']:>8.1%} {baseline_agg['SRact']:>8.1%} {baseline_agg['SRargs']:>8.1%} {baseline_agg['SR']:>8.1%}\n")
        for label in checkpoint_labels:
            agg = shine_aggs[label]
            f.write(f"{'SHINE [' + label + ']':<25} {agg['SRt']:>8.1%} {agg['SRact']:>8.1%} {agg['SRargs']:>8.1%} {agg['SR']:>8.1%}\n")
        f.write(sep + "\n\n")

        # Per-entry details
        for i, row in enumerate(rows, 1):
            f.write(sep + "\n")
            f.write(f"[Sample {i}]  id={row['id']}\n")
            f.write(thin_sep + "\n")

            # Show user query
            f.write(f"USER QUERY: {row['user_query']}\n")
            if row['partial_trace']:
                f.write(f"PARTIAL TRACE: {row['partial_trace'][:200]}...\n")
            f.write(f"\nEXPECTED:\n  {row['expected_output']}\n")

            # Baseline
            b = row["baseline"]
            f.write(f"\nBASELINE:\n  {b['response']['answer']}\n")
            f.write(f"  Scores: thought={b['scores']['thought_correct']}  "
                    f"action={b['scores']['action_correct']}  "
                    f"args={b['scores']['args_correct']}  "
                    f"full={b['scores']['full_correct']}\n")

            # SHINE checkpoints
            for label in checkpoint_labels:
                s = row[f"shine_{label}"]
                f.write(f"\nSHINE [{label}]:\n  {s['response']['answer']}\n")
                f.write(f"  Scores: thought={s['scores']['thought_correct']}  "
                        f"action={s['scores']['action_correct']}  "
                        f"args={s['scores']['args_correct']}  "
                        f"full={s['scores']['full_correct']}\n")

            f.write("\n")

    logger.info(f"Saved JSON: {json_path}")
    logger.info(f"Saved TXT:  {txt_path}")

    # --- Print summary to stdout ---
    print(f"\n{sep}")
    print(f"  GPT4Tools EVALUATION SUMMARY  (scale={args.scale}, n={len(rows)})")
    print(f"{sep}\n")
    print(f"{'Method':<25} {'SRt':>8} {'SRact':>8} {'SRargs':>8} {'SR':>8}")
    print(thin_sep)
    print(f"{'Baseline (no LoRA)':<25} {baseline_agg['SRt']:>8.1%} {baseline_agg['SRact']:>8.1%} {baseline_agg['SRargs']:>8.1%} {baseline_agg['SR']:>8.1%}")
    for label in checkpoint_labels:
        agg = shine_aggs[label]
        print(f"{'SHINE [' + label + ']':<25} {agg['SRt']:>8.1%} {agg['SRact']:>8.1%} {agg['SRargs']:>8.1%} {agg['SR']:>8.1%}")
    print()

    # Show a few example comparisons
    n_show = min(5, len(rows))
    print(f"\n--- First {n_show} examples ---\n")
    for i, row in enumerate(rows[:n_show], 1):
        print(f"[{i}] Q: {row['user_query'][:120]}...")
        print(f"    Expected: {row['expected_output'][:120]}...")
        b = row["baseline"]
        print(f"    Baseline: {b['response']['answer'][:120]}...  [SR={'✓' if b['scores']['full_correct'] else '✗'}]")
        for label in checkpoint_labels:
            s = row[f"shine_{label}"]
            print(f"    SHINE[{label}]: {s['response']['answer'][:120]}...  [SR={'✓' if s['scores']['full_correct'] else '✗'}]")
        print()


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BehaveSHINE eval on GPT4Tools benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data
    parser.add_argument("--gpt4tools_path", required=True,
                        help="Path to GPT4Tools JSON file (e.g. gpt4tools_test_unseen.json)")
    parser.add_argument("--n", type=int, default=300,
                        help="Number of random samples to evaluate")
    parser.add_argument("--seed", type=int, default=42)

    # Checkpoints
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="Multiple checkpoints as 'label:path' or 'path'")

    # Model / generation
    parser.add_argument("--model_path", default="./models/Qwen3-8B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--context_max_length", type=int, default=1550)
    parser.add_argument("--conversation_max_length", type=int, default=2800)
    parser.add_argument("--scale", type=float, default=0.001)
    parser.add_argument("--no_baseline", action="store_true")

    args = parser.parse_args()

    # ---- Resolve checkpoints ----
    ckpt_list: List[Tuple[str, str]] = []
    if args.checkpoints:
        for raw in args.checkpoints:
            ckpt_list.append(parse_checkpoint_arg(raw))
    elif args.checkpoint:
        ckpt_list.append(parse_checkpoint_arg(args.checkpoint))
    else:
        parser.error("Provide --checkpoint or --checkpoints")

    for label, path in ckpt_list:
        if not os.path.exists(path):
            parser.error(f"Checkpoint not found: {path} (label={label})")

    device = torch.device("cuda")
    set_seed(args.seed)

    logger.info(f"GPT4Tools eval — {args.n} samples from {args.gpt4tools_path}")
    logger.info(f"Scale: {args.scale}")
    logger.info(f"Checkpoints ({len(ckpt_list)}):")
    for label, path in ckpt_list:
        logger.info(f"  [{label}] -> {path}")

    # ---- Load and parse GPT4Tools data ----
    if not os.path.exists(args.gpt4tools_path):
        parser.error(f"GPT4Tools file not found: {args.gpt4tools_path}")

    with open(args.gpt4tools_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    logger.info(f"Loaded {len(raw_data)} entries from GPT4Tools file")

    # Random sample
    rng = random.Random(args.seed)
    n = min(args.n, len(raw_data))
    sampled = rng.sample(raw_data, n)
    logger.info(f"Sampled {n} entries")

    # Parse all entries
    parsed_entries = [parse_gpt4tools_entry(entry) for entry in sampled]

    # Quick sanity check
    n_with_tools = sum(1 for e in parsed_entries if "Action:" in e["expected_output"])
    n_no_tools = n - n_with_tools
    logger.info(f"  Tool-use entries: {n_with_tools}, No-tool entries: {n_no_tools}")

    # ---- Build dataset and collator ----
    tokenizer = init_tokenizer(args.model_path)
    cfg = build_cfg(args)

    eval_dataset = GPT4ToolsDataset(parsed_entries, tokenizer)
    collator = GPT4ToolsCollator(
        tokenizer=tokenizer,
        context_max_length=args.context_max_length,
        conversation_max_length=args.conversation_max_length,
    )

    # ---- Baseline ----
    if not args.no_baseline:
        vanilla = init_vanilla_model(args.model_path, tokenizer, device)
        loader = make_dataloader(eval_dataset, collator, args.batch_size)
        baseline_results = run_baseline(vanilla, loader, tokenizer, device, args.max_new_tokens)
        del vanilla; gc.collect(); torch.cuda.empty_cache()
        logger.info("Baseline complete.")
    else:
        baseline_results = [{"think": "", "answer": "(skipped)"} for _ in range(n)]

    # Score baseline
    baseline_scores = []
    for idx, entry in enumerate(parsed_entries):
        score = score_entry(entry["expected_output"], baseline_results[idx]["answer"])
        baseline_scores.append(score)

    baseline_agg = compute_aggregate_scores(baseline_scores)
    logger.info(f"Baseline:  SRt={baseline_agg['SRt']:.1%}  SRact={baseline_agg['SRact']:.1%}  "
                f"SRargs={baseline_agg['SRargs']:.1%}  SR={baseline_agg['SR']:.1%}")

    # ---- SHINE for each checkpoint ----
    metanetwork = init_metanetwork(cfg, tokenizer, device)

    shine_results_by_label = OrderedDict()
    shine_scores_by_label = OrderedDict()
    checkpoint_labels = []

    for ckpt_idx, (label, path) in enumerate(ckpt_list):
        logger.info(f"\n{'='*60}")
        logger.info(f"Checkpoint {ckpt_idx+1}/{len(ckpt_list)}: [{label}]")
        logger.info(f"{'='*60}")

        metanetwork, metalora_ckpt = load_ckpt(metanetwork, path, device)

        loader = make_dataloader(eval_dataset, collator, args.batch_size)
        results = run_shine(metanetwork, metalora_ckpt, loader, tokenizer, device, args.max_new_tokens)

        # Score
        scores = []
        for idx, entry in enumerate(parsed_entries):
            score = score_entry(entry["expected_output"], results[idx]["answer"])
            scores.append(score)

        shine_results_by_label[label] = results
        shine_scores_by_label[label] = scores
        checkpoint_labels.append(label)

        agg = compute_aggregate_scores(scores)
        logger.info(f"[{label}]:  SRt={agg['SRt']:.1%}  SRact={agg['SRact']:.1%}  "
                    f"SRargs={agg['SRargs']:.1%}  SR={agg['SR']:.1%}")

        del metalora_ckpt; gc.collect(); torch.cuda.empty_cache()

    # ---- Save ----
    save_outputs(
        parsed_entries, baseline_results, baseline_scores,
        shine_results_by_label, shine_scores_by_label,
        args, checkpoint_labels,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()