#!/usr/bin/env python3
"""Script 03: Generate teacher responses via Nebius Token Factory API (async).

Calls Qwen3-235B-A22B-Instruct-2507 on each (system_prompt, question) pair
and writes results to data/raw/responses_teacher.jsonl.

Usage:
    python 03_generate_responses.py
    python 03_generate_responses.py --resume
    python 03_generate_responses.py --limit 20
    python 03_generate_responses.py --concurrency 10
    python 03_generate_responses.py --config configs/generation_config.yaml
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

# Allow importing utils from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from utils import append_jsonl, get_existing_ids, load_config, load_jsonl, setup_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_think_and_answer(text: str) -> tuple[str, str]:
    """Split a Qwen3 response into (think_part, answer_part).

    Returns ("", answer) when no <think> block is present.
    """
    lower = text.lower()
    has_start = "<think>" in lower
    has_end = "</think>" in lower

    if not has_start and not has_end:
        return "", text.strip()

    if has_start and not has_end:
        # Opening tag without closing — treat everything after tag as answer
        after_tag = text[lower.find("<think>") + len("<think>"):]
        return "", after_tag.strip()

    if not has_start and has_end:
        return "[error]", text

    start_idx = lower.find("<think>")
    end_idx = lower.find("</think>")
    if end_idx < start_idx:
        return "[error]", text

    think = text[start_idx + len("<think>"): end_idx].strip()
    answer = text[end_idx + len("</think>"):].strip()
    return think, answer


def build_work_items(
    system_prompts: list[dict],
    questions_records: list[dict],
    completed_ids: set[str],
    limit: int | None,
) -> list[dict]:
    """Flatten questions.jsonl into per-question work items, skipping completed ones."""
    sp_lookup = {sp["id"]: sp["system_prompt"] for sp in system_prompts}

    items = []
    for record in questions_records:
        sp_id = record["system_prompt_id"]
        sp_text = sp_lookup.get(sp_id)
        if sp_text is None:
            continue
        for q in record.get("questions", []):
            q_id = q["id"]
            if q_id in completed_ids:
                continue
            items.append({
                "sp_id": sp_id,
                "question_id": q_id,
                "system_prompt_text": sp_text,
                "question_text": q["question"],
            })
            if limit is not None and len(items) >= limit:
                return items
    return items


# ---------------------------------------------------------------------------
# Async generation
# ---------------------------------------------------------------------------

async def generate_one(
    item: dict,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    teacher_model: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    logger,
) -> dict | None:
    """Generate one response with retries and exponential backoff. Returns None on failure."""
    async with semaphore:
        last_exc = None
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=teacher_model,
                    messages=[
                        {"role": "system", "content": item["system_prompt_text"]},
                        {"role": "user", "content": item["question_text"]},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body={"enable_thinking": False},
                )
                content = response.choices[0].message.content or ""
                think, answer = extract_think_and_answer(content)
                return {
                    "system_prompt_id": item["sp_id"],
                    "question_id": item["question_id"],
                    "model": teacher_model,
                    "raw_response": content,
                    "think": think,
                    "answer": answer,
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "truncated": response.choices[0].finish_reason == "length",
                }
            except Exception as e:
                last_exc = e
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    retry_after = e.response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else 2 ** attempt  # 1s, 2s, 4s
                status = getattr(e, "status_code", "?")
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for {item['question_id']} "
                    f"(status={status}): {e}. Waiting {wait:.1f}s."
                )
                await asyncio.sleep(wait)

        logger.error(f"All retries exhausted for {item['question_id']}: {last_exc}")
        return None


async def run_generation(
    items: list[dict],
    client: AsyncOpenAI,
    teacher_model: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    max_retries: int,
    batch_save_interval: int,
    output_path: str,
    failed_path: str,
    logger,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    total = len(items)
    saved = 0
    failed = 0
    start = time.time()

    for chunk_start in range(0, total, batch_save_interval):
        chunk = items[chunk_start: chunk_start + batch_save_interval]
        tasks = [
            generate_one(
                item, client, semaphore, teacher_model, max_tokens,
                temperature, max_retries, logger,
            )
            for item in chunk
        ]
        results = await asyncio.gather(*tasks)

        for result, item in zip(results, chunk):
            if result is not None:
                append_jsonl(output_path, result)
                saved += 1
            else:
                append_jsonl(failed_path, {
                    "question_id": item["question_id"],
                    "sp_id": item["sp_id"],
                })
                failed += 1

        done = chunk_start + len(chunk)
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        logger.info(
            f"Progress: {done}/{total} | saved={saved} failed={failed} | "
            f"{rate:.1f} items/s | ETA {eta/60:.1f}min"
        )

    logger.info(f"Done. Total: saved={saved}, failed={failed}.")


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate teacher responses via Nebius API.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed question IDs.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N items (for testing).")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Override max concurrent API requests.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "configs" / "generation_config.yaml"),
        help="Path to generation_config.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("03_generate_responses")

    base_dir = Path(__file__).parent.parent
    config = load_config(args.config)
    resp_cfg = config["responses"]

    teacher_model: str = resp_cfg["teacher_model"]
    max_tokens: int = resp_cfg["max_tokens"]
    temperature: float = resp_cfg["temperature"]
    concurrency: int = args.concurrency if args.concurrency is not None else resp_cfg["concurrency"]
    batch_save_interval: int = resp_cfg["batch_save_interval"]
    max_retries: int = resp_cfg["max_retries"]
    output_path = str(base_dir / resp_cfg["output_path"])
    failed_path = str(base_dir / "data/raw/responses_teacher_failed.jsonl")

    sp_path = str(base_dir / config["system_prompts"]["output_path"])
    q_path = str(base_dir / config["questions"]["output_path"])

    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        logger.error("NEBIUS_API_KEY environment variable is not set.")
        sys.exit(1)

    logger.info("Loading system prompts and questions...")
    system_prompts = load_jsonl(sp_path)
    questions_records = load_jsonl(q_path)
    logger.info(f"  {len(system_prompts)} system prompts, {len(questions_records)} question records")

    completed_ids: set[str] = set()
    if args.resume:
        completed_ids = get_existing_ids(output_path, "question_id")
        logger.info(f"Resume mode: {len(completed_ids)} already completed.")

    items = build_work_items(system_prompts, questions_records, completed_ids, args.limit)
    logger.info(f"Items to generate: {len(items)}")
    if not items:
        logger.info("Nothing to do.")
        return

    avg_input, avg_output = 350, 200
    est_cost = (
        len(items) * avg_input / 1_000_000 * 0.20
        + len(items) * avg_output / 1_000_000 * 0.60
    )
    logger.info(f"Estimated cost: ~${est_cost:.2f} (rough, at $0.20/M input + $0.60/M output)")

    client = AsyncOpenAI(
        base_url=resp_cfg["nebius_base_url"],
        api_key=api_key,
    )

    total_start = time.time()
    asyncio.run(
        run_generation(
            items=items,
            client=client,
            teacher_model=teacher_model,
            max_tokens=max_tokens,
            temperature=temperature,
            concurrency=concurrency,
            max_retries=max_retries,
            batch_save_interval=batch_save_interval,
            output_path=output_path,
            failed_path=failed_path,
            logger=logger,
        )
    )
    elapsed = time.time() - total_start
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
