#!/usr/bin/env python3
"""Generate ground-truth responses for BehaveSHINE using Qwen models.

For each (system_prompt, question) pair, generates responses under three conditions:
  1. 72b_with_sp  — Qwen-72B with system prompt (ceiling target)
  2. 8b_with_sp   — Qwen-8B with system prompt (replication target)
  3. 8b_only_q    — Qwen-8B without system prompt (baseline)

Requires: 2x H200 GPUs, system_prompts.jsonl, questions.jsonl from scripts 01/02.

Usage:
    python 03_generate_responses.py --model-72b /path/to/Qwen3-72B --model-8b /path/to/Qwen3-8B
    python 03_generate_responses.py --resume                          # resume after interruption
    python 03_generate_responses.py --only-72b --model-72b /path      # only 72B condition
    python 03_generate_responses.py --only-8b --model-8b /path        # only 8B conditions
    python 03_generate_responses.py --limit 50                        # test with 50 items
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add scripts dir to path for utils import
sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_think_and_answer(text: str) -> tuple[str, str]:
    """Split Qwen model output into (think_part, answer_part).

    Handles <think>...</think> blocks produced by Qwen3 reasoning mode.
    Returns ("", answer) when no think block is present.
    Returns ("[error]", raw_text) for malformed tags.
    """
    lower = text.lower()
    has_start = "<think>" in lower
    has_end = "</think>" in lower

    # Neither tag — normal non-thinking response
    if not has_start and not has_end:
        return "", text.strip()

    # Only opening tag (common: model starts <think> but never closes)
    if has_start and not has_end:
        if text.lstrip().lower().startswith("<think>"):
            after_tag = text[text.lower().find("<think>") + len("<think>"):]
            return "", after_tag.strip()
        return "[error]", text

    # Only closing tag — malformed
    if not has_start and has_end:
        return "[error]", text

    # Both tags — check order
    start_idx = lower.find("<think>")
    end_idx = lower.find("</think>")
    if end_idx < start_idx:
        return "[error]", text

    think = text[start_idx + len("<think>"):end_idx].strip()
    answer = text[end_idx + len("</think>"):].strip()
    return think, answer


def get_model_input_device(model) -> torch.device:
    """Get the device for model inputs (handles multi-GPU sharded models)."""
    if hasattr(model, "hf_device_map"):
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, int):
            return torch.device(f"cuda:{first_device}")
        return torch.device(first_device)
    return model.device


def load_model_and_tokenizer(model_path: str, device_map, logger):
    """Load a Qwen model and tokenizer with left-padding for batched generation."""
    logger.info(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="left", use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad_token = eos_token")

    logger.info(f"Loading model from {model_path} with device_map={device_map}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()

    input_device = get_model_input_device(model)
    logger.info(f"Model loaded. input_device={input_device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            logger.info(f"  GPU {i} memory allocated: {alloc:.2f} GB")

    return model, tokenizer


def unload_model(model, logger):
    """Delete model and free GPU memory."""
    logger.info("Unloading model...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            logger.info(f"  GPU {i} memory after unload: {alloc:.2f} GB")


def build_chat_messages(system_prompt: str, question: str, include_sp: bool) -> list[dict]:
    """Construct chat messages for tokenizer.apply_chat_template()."""
    messages = []
    if include_sp and system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return messages


def tokenize_and_pad(tokenizer, all_messages: list[list[dict]], device: torch.device):
    """Tokenize each message list individually, then left-pad into a batch.

    This avoids compatibility issues with batched apply_chat_template across
    different transformers versions.

    Returns (input_ids, attention_mask, input_lengths) where input_lengths is
    the list of actual (non-padded) token counts per item.
    """
    all_input_ids = []
    for msgs in all_messages:
        ids = tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=True,
        )
        all_input_ids.append(ids)

    input_lengths = [len(ids) for ids in all_input_ids]
    max_len = max(input_lengths)

    padded_ids = []
    padded_mask = []
    for ids in all_input_ids:
        pad_len = max_len - len(ids)
        padded_ids.append([tokenizer.pad_token_id] * pad_len + ids)
        padded_mask.append([0] * pad_len + [1] * len(ids))

    input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(padded_mask, dtype=torch.long, device=device)
    return input_ids, attention_mask, input_lengths


def generate_batch(
    model,
    tokenizer,
    batch_items: list[dict],
    condition: str,
    max_new_tokens: int,
    logger,
) -> list[dict]:
    """Generate responses for a batch of items. Returns list of result dicts."""
    device = get_model_input_device(model)

    # Build chat messages
    all_messages = [
        build_chat_messages(item["system_prompt"], item["question"], item["include_sp"])
        for item in batch_items
    ]

    # Tokenize and left-pad
    input_ids, attention_mask, input_lengths = tokenize_and_pad(
        tokenizer, all_messages, device,
    )

    # Generate
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.perf_counter() - t0

    # Decode each response
    padded_input_len = input_ids.shape[1]
    results = []
    for i, item in enumerate(batch_items):
        new_tokens = outputs[i, padded_input_len:]
        # Count actual output tokens (non-padding, non-zero for left-padded outputs)
        output_token_count = int((new_tokens != tokenizer.pad_token_id).sum().item())
        truncated = output_token_count >= max_new_tokens

        raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        think, answer = extract_think_and_answer(raw_response)

        result = {
            "system_prompt_id": item["system_prompt_id"],
            "question_id": item["question_id"],
            "question": item["question"],
            "condition": condition,
            "think": think,
            "answer": answer,
            "raw_response": raw_response,
            "generation_time_seconds": round(gen_time / len(batch_items), 3),
            "input_tokens": input_lengths[i],
            "output_tokens": output_token_count,
            "truncated": truncated,
        }

        if not answer.strip():
            logger.warning(f"Empty answer for {item['question_id']} condition={condition}")
        if truncated:
            logger.warning(f"Truncated response for {item['question_id']} condition={condition}")

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Track generation progress with periodic summary stats."""

    def __init__(self, condition: str, total: int, logger):
        self.condition = condition
        self.total = total
        self.logger = logger
        self.done = 0
        self.start_time = time.time()
        self.batch_count = 0
        self.total_output_tokens = 0
        self.truncated_count = 0
        self.empty_count = 0
        self.error_count = 0

    def update_batch(self, results: list[dict]):
        self.batch_count += 1
        for r in results:
            self.done += 1
            self.total_output_tokens += r["output_tokens"]
            if r["truncated"]:
                self.truncated_count += 1
            if not r["answer"].strip():
                self.empty_count += 1
            if r["think"] == "[error]":
                self.error_count += 1

    def log_batch(self, batch_num: int, total_batches: int):
        elapsed = time.time() - self.start_time
        avg_per_item = elapsed / max(self.done, 1)
        remaining = (self.total - self.done) * avg_per_item
        eta_h = remaining / 3600
        self.logger.info(
            f"[{self.condition}] Batch {batch_num}/{total_batches} | "
            f"{self.done}/{self.total} ({100 * self.done / self.total:.1f}%) | "
            f"avg {avg_per_item:.2f}s/item | ETA {eta_h:.1f}h"
        )

    def log_summary(self):
        avg_tokens = self.total_output_tokens / max(self.done, 1)
        elapsed = time.time() - self.start_time
        self.logger.info(
            f"=== [{self.condition}] Summary at {self.done}/{self.total} ===\n"
            f"  Avg output tokens: {avg_tokens:.1f}\n"
            f"  Truncated: {self.truncated_count} ({100 * self.truncated_count / max(self.done, 1):.2f}%)\n"
            f"  Empty: {self.empty_count}\n"
            f"  Think errors: {self.error_count}\n"
            f"  Elapsed: {elapsed / 60:.1f}min"
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_join_data(config: dict, base_dir: Path, limit: int | None, logger) -> list[dict]:
    """Load system prompts + questions and flatten to (sp, question) pairs.

    Returns list sorted by system_prompt length descending to minimize padding.
    """
    sp_path = str(base_dir / config["system_prompts"]["output_path"])
    q_path = str(base_dir / config["questions"]["output_path"])

    system_prompts = utils.load_jsonl(sp_path)
    questions_data = utils.load_jsonl(q_path)

    if not system_prompts:
        logger.error(f"No system prompts found at {sp_path}")
        sys.exit(1)
    if not questions_data:
        logger.error(f"No questions found at {q_path}")
        sys.exit(1)

    sp_lookup = {sp["id"]: sp["system_prompt"] for sp in system_prompts}

    items = []
    for q_record in questions_data:
        sp_id = q_record["system_prompt_id"]
        sp_text = sp_lookup.get(sp_id)
        if sp_text is None:
            logger.warning(f"System prompt {sp_id} not found, skipping its questions")
            continue
        for q in q_record["questions"]:
            items.append({
                "system_prompt_id": sp_id,
                "system_prompt": sp_text,
                "question_id": q["id"],
                "question": q["question"],
            })

    logger.info(f"Loaded {len(items)} (system_prompt, question) pairs "
                f"from {len(system_prompts)} system prompts")

    if limit:
        items = items[:limit]
        logger.info(f"Limited to {limit} items")

    # Sort by system_prompt length descending — items with similar-length prompts
    # cluster into the same batch, reducing padding waste.
    items.sort(key=lambda x: len(x["system_prompt"]), reverse=True)

    return items


# ---------------------------------------------------------------------------
# Per-condition generation loop
# ---------------------------------------------------------------------------

def run_condition(
    model,
    tokenizer,
    items: list[dict],
    condition: str,
    include_sp: bool,
    output_path: str,
    batch_size: int,
    max_new_tokens: int,
    logger,
    resume: bool = False,
):
    """Run generation for one condition, saving results incrementally."""
    # Create working copies with the include_sp flag (don't mutate originals)
    work_items = [dict(it, include_sp=include_sp) for it in items]

    # Resume: skip already-completed question_ids
    if resume:
        done_ids = utils.get_existing_ids(output_path, id_field="question_id")
        before = len(work_items)
        work_items = [it for it in work_items if it["question_id"] not in done_ids]
        logger.info(f"[{condition}] Resume: {before - len(work_items)} already done, "
                     f"{len(work_items)} remaining")

    if not work_items:
        logger.info(f"[{condition}] Nothing to do.")
        return

    total_batches = (len(work_items) + batch_size - 1) // batch_size
    tracker = ProgressTracker(condition, len(work_items), logger)
    consecutive_failures = 0
    current_batch_size = batch_size

    logger.info(f"[{condition}] Starting: {len(work_items)} items, "
                f"batch_size={batch_size}, total_batches={total_batches}")

    idx = 0
    batch_num = 0
    while idx < len(work_items):
        batch = work_items[idx:idx + current_batch_size]
        batch_num += 1

        try:
            results = generate_batch(
                model, tokenizer, batch, condition, max_new_tokens, logger,
            )
            consecutive_failures = 0
        except torch.cuda.OutOfMemoryError:
            # OOM: halve batch size and retry
            new_bs = max(1, current_batch_size // 2)
            logger.warning(
                f"[{condition}] OOM at batch {batch_num}, "
                f"reducing batch size {current_batch_size} -> {new_bs}"
            )
            torch.cuda.empty_cache()
            current_batch_size = new_bs
            continue  # retry same idx with smaller batch
        except Exception as e:
            logger.error(f"[{condition}] Batch {batch_num} failed: {e}")
            consecutive_failures += len(batch)
            if consecutive_failures >= 10:
                logger.critical(
                    f"[{condition}] {consecutive_failures} consecutive failures, aborting."
                )
                return
            idx += len(batch)
            continue

        # Save each result
        for r in results:
            utils.append_jsonl(output_path, r)

        tracker.update_batch(results)
        tracker.log_batch(batch_num, total_batches)

        # Summary every 500 items
        if tracker.done % 500 < len(results):
            tracker.log_summary()

        idx += len(batch)

    # Final summary
    tracker.log_summary()
    total_time = time.time() - tracker.start_time
    logger.info(f"[{condition}] COMPLETE: {tracker.done} items in {total_time / 3600:.1f}h "
                f"-> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BehaveSHINE ground-truth responses using Qwen models",
    )
    parser.add_argument("--model-72b", type=str, default=None,
                        help="Path to Qwen3-72B model (overrides config)")
    parser.add_argument("--model-8b", type=str, default=None,
                        help="Path to Qwen3-8B model (overrides config)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output files")
    parser.add_argument("--only-72b", action="store_true",
                        help="Run only the 72B condition")
    parser.add_argument("--only-8b", action="store_true",
                        help="Run only the 8B conditions")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N (system_prompt, question) pairs")
    parser.add_argument("--batch-size-72b", type=int, default=None,
                        help="Override 72B batch size")
    parser.add_argument("--batch-size-8b", type=int, default=None,
                        help="Override 8B batch size")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve paths (same pattern as scripts 01/02)
    base_dir = Path(__file__).resolve().parent.parent
    config_path = args.config or str(base_dir / "configs" / "generation_config.yaml")
    config = utils.load_config(config_path)

    logger = utils.setup_logging("generate_responses")
    resp_cfg = config["responses"]

    # Resolve output paths
    output_72b = str(base_dir / resp_cfg["output_72b_path"])
    output_8b_sp = str(base_dir / resp_cfg["output_8b_with_sp_path"])
    output_8b_q = str(base_dir / resp_cfg["output_8b_only_q_path"])

    # Model paths (CLI overrides config)
    model_72b_path = args.model_72b or resp_cfg["model_72b_path"]
    model_8b_path = args.model_8b or resp_cfg["model_8b_path"]

    # Batch sizes (CLI overrides config)
    bs_72b = args.batch_size_72b or resp_cfg["batch_size_72b"]
    bs_8b = args.batch_size_8b or resp_cfg["batch_size_8b"]
    max_new_tokens = resp_cfg["max_new_tokens"]

    # Validate flags
    if args.only_72b and args.only_8b:
        logger.error("Cannot specify both --only-72b and --only-8b")
        sys.exit(1)

    # Load and join data
    items = load_and_join_data(config, base_dir, args.limit, logger)

    # Log GPU info
    if torch.cuda.is_available():
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            total = torch.cuda.get_device_properties(i).total_mem / 1e9
            logger.info(f"  GPU {i}: {name}, {total:.1f} GB")
    else:
        logger.warning("No CUDA devices available!")

    # ===== Phase 1: Qwen-72B =====
    if not args.only_8b:
        logger.info("=" * 60)
        logger.info("PHASE 1: Qwen-72B with system prompt")
        logger.info("=" * 60)

        model, tokenizer = load_model_and_tokenizer(
            model_72b_path, device_map="auto", logger=logger,
        )

        run_condition(
            model=model,
            tokenizer=tokenizer,
            items=items,
            condition="72b_with_sp",
            include_sp=True,
            output_path=output_72b,
            batch_size=bs_72b,
            max_new_tokens=max_new_tokens,
            logger=logger,
            resume=args.resume,
        )

        unload_model(model, logger)
        del tokenizer

    # ===== Phase 2: Qwen-8B =====
    if not args.only_72b:
        logger.info("=" * 60)
        logger.info("PHASE 2: Qwen-8B with and without system prompt")
        logger.info("=" * 60)

        model, tokenizer = load_model_and_tokenizer(
            model_8b_path, device_map="cuda:0", logger=logger,
        )

        # Condition 2a: 8B with system prompt
        run_condition(
            model=model,
            tokenizer=tokenizer,
            items=items,
            condition="8b_with_sp",
            include_sp=True,
            output_path=output_8b_sp,
            batch_size=bs_8b,
            max_new_tokens=max_new_tokens,
            logger=logger,
            resume=args.resume,
        )

        # Condition 2b: 8B without system prompt
        run_condition(
            model=model,
            tokenizer=tokenizer,
            items=items,
            condition="8b_only_q",
            include_sp=False,
            output_path=output_8b_q,
            batch_size=bs_8b,
            max_new_tokens=max_new_tokens,
            logger=logger,
            resume=args.resume,
        )

        unload_model(model, logger)
        del tokenizer

    # ===== Final Summary =====
    logger.info("=" * 60)
    logger.info("ALL CONDITIONS COMPLETE")
    for label, path in [
        ("72b_with_sp", output_72b),
        ("8b_with_sp", output_8b_sp),
        ("8b_only_q", output_8b_q),
    ]:
        count = len(utils.load_jsonl(path))
        logger.info(f"  {label}: {count} responses -> {path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
