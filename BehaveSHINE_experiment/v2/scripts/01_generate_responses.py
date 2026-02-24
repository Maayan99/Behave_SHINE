#!/usr/bin/env python3
"""Script 01: Generate dual responses via Qwen3-32B (Nebius) and Qwen3-8B (HuggingFace).

For each (system_prompt, question) pair, calls both models concurrently and writes
results to v2/data/raw/responses_{32b,8b}.jsonl.

Architecture: queue-based worker pool with buffered I/O.
  - N async workers pull items from a work queue and call both APIs concurrently
  - Per-provider semaphores cap concurrent HTTP requests independently
  - A single writer task drains a result queue, buffers records, and flushes to disk
  - Graceful Ctrl+C: stops accepting work, lets in-flight requests finish, flushes

Resume-safe: on startup, loads existing question_ids from output files and skips them.

Usage:
    python 01_generate_responses.py
    python 01_generate_responses.py --limit 20
    python 01_generate_responses.py --concurrency 15
    python 01_generate_responses.py --only-32b
    python 01_generate_responses.py --only-8b
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_existing_ids, load_jsonl, setup_logging


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = {
    "32b": {
        "name": "Qwen/Qwen3-32B",
        "env_key": "NEBIUS_API_KEY",
        "base_url": "https://api.studio.nebius.com/v1/",
    },
    "8b": {
        "name": "Qwen/Qwen3-8B:nscale",
        "env_key": "HF_TOKEN",
        "base_url": "https://router.huggingface.co/v1",
    },
}

# Greedy decoding for reproducibility
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 2048
MAX_RETRIES = 3


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


def flush_buffer(path: str, buffer: list[dict]) -> None:
    """Append all buffered records to a JSONL file in one write, then clear buffer."""
    if not buffer:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in buffer)
    buffer.clear()


# ---------------------------------------------------------------------------
# Async generation
# ---------------------------------------------------------------------------

async def generate_one(
    item: dict,
    client: AsyncOpenAI,
    model_key: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
    logger,
) -> dict | None:
    """Generate one response with retries and exponential backoff."""
    async with semaphore:
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": item["system_prompt_text"]},
                        {"role": "user", "content": item["question_text"]},
                    ],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    extra_body={"enable_thinking": False},
                )
                content = response.choices[0].message.content or ""
                _, answer = extract_think_and_answer(content)
                return {
                    "system_prompt_id": item["sp_id"],
                    "question_id": item["question_id"],
                    "model": model_name,
                    "raw_response": content,
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
                wait = float(retry_after) if retry_after else (1 * 4 ** attempt)  # 1s, 4s, 16s
                status = getattr(e, "status_code", "?")
                logger.warning(
                    f"[{model_key}] Attempt {attempt + 1}/{MAX_RETRIES} failed for "
                    f"{item['question_id']} (status={status}): {e}. Waiting {wait:.1f}s."
                )
                await asyncio.sleep(wait)

        logger.error(f"[{model_key}] All retries exhausted for {item['question_id']}: {last_exc}")
        return None


async def generate_dual(
    item: dict,
    clients: dict[str, AsyncOpenAI],
    semaphores: dict[str, asyncio.Semaphore],
    active_models: list[str],
    logger,
) -> dict[str, dict | None]:
    """Generate responses from all active models concurrently for one item."""
    tasks = {
        key: generate_one(item, clients[key], key, MODELS[key]["name"], semaphores[key], logger)
        for key in active_models
    }
    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))


# ---------------------------------------------------------------------------
# Pipeline: queue-based workers + buffered writer
# ---------------------------------------------------------------------------

async def run_generation(
    items: list[dict],
    clients: dict[str, AsyncOpenAI],
    active_models: list[str],
    concurrency: int,
    flush_threshold: int,
    output_paths: dict[str, str],
    failed_paths: dict[str, str],
    shutdown: asyncio.Event,
    logger,
) -> None:
    semaphores = {key: asyncio.Semaphore(concurrency) for key in active_models}
    total = len(items)

    # Shared mutable state (single-threaded async — no lock needed)
    buffers: dict[str, dict[str, list]] = {
        key: {"ok": [], "fail": []} for key in active_models
    }
    counters = {key: {"saved": 0, "failed": 0} for key in active_models}
    done_count = 0
    start = time.time()

    work_queue: asyncio.Queue = asyncio.Queue()
    result_queue: asyncio.Queue = asyncio.Queue()

    for item in items:
        work_queue.put_nowait(item)

    # -- Workers: pull items, generate, push results -----------------------

    async def worker():
        while not shutdown.is_set():
            try:
                item = work_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            result_dict = await generate_dual(item, clients, semaphores, active_models, logger)
            await result_queue.put((item, result_dict))
            work_queue.task_done()

    # -- Writer: drain results, buffer, flush, log progress ----------------

    async def writer():
        nonlocal done_count
        last_log_time = start

        def flush_all():
            for key in active_models:
                flush_buffer(output_paths[key], buffers[key]["ok"])
                flush_buffer(failed_paths[key], buffers[key]["fail"])

        def log_progress():
            nonlocal last_log_time
            elapsed = time.time() - start
            rate = done_count / elapsed if elapsed > 0 else 0
            eta = (total - done_count) / rate if rate > 0 else 0
            stats_str = " | ".join(
                f"{k}: saved={counters[k]['saved']} failed={counters[k]['failed']}"
                for k in active_models
            )
            logger.info(
                f"Progress: {done_count}/{total} | {stats_str} | "
                f"{rate:.1f} items/s | ETA {eta / 60:.1f}min"
            )
            last_log_time = time.time()

        try:
            while done_count < total:
                # Drain with timeout so we can do periodic flushes / progress logs
                try:
                    item, result_dict = await asyncio.wait_for(
                        result_queue.get(), timeout=10.0,
                    )
                except asyncio.TimeoutError:
                    flush_all()
                    if time.time() - last_log_time >= 10:
                        log_progress()
                    continue

                # Buffer results
                for key in active_models:
                    result = result_dict[key]
                    if result is not None:
                        buffers[key]["ok"].append(result)
                        counters[key]["saved"] += 1
                    else:
                        buffers[key]["fail"].append({
                            "question_id": item["question_id"],
                            "sp_id": item["sp_id"],
                        })
                        counters[key]["failed"] += 1
                done_count += 1

                # Flush when buffer hits threshold
                for key in active_models:
                    if len(buffers[key]["ok"]) >= flush_threshold:
                        flush_buffer(output_paths[key], buffers[key]["ok"])
                    if len(buffers[key]["fail"]) >= flush_threshold:
                        flush_buffer(failed_paths[key], buffers[key]["fail"])

                # Log progress every ~10s
                if time.time() - last_log_time >= 10:
                    log_progress()
        finally:
            # Always flush remaining records
            flush_all()

        log_progress()

    # -- Launch pipeline ---------------------------------------------------

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    writer_task = asyncio.create_task(writer())

    try:
        await asyncio.gather(*workers)
        # Workers done — wait for writer to drain remaining results
        await writer_task
    finally:
        # Safety net: flush on any exception path
        for key in active_models:
            flush_buffer(output_paths[key], buffers[key]["ok"])
            flush_buffer(failed_paths[key], buffers[key]["fail"])

    stats_str = " | ".join(
        f"{k}: saved={counters[k]['saved']} failed={counters[k]['failed']}"
        for k in active_models
    )
    logger.info(f"Done. {stats_str}")


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dual model responses (32B + 8B).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N items (for testing).")
    parser.add_argument("--concurrency", type=int, default=15,
                        help="Max concurrent requests per provider (default 15).")
    parser.add_argument("--only-32b", action="store_true",
                        help="Only generate 32B responses.")
    parser.add_argument("--only-8b", action="store_true",
                        help="Only generate 8B responses.")
    parser.add_argument("--batch-save-interval", type=int, default=50,
                        help="Buffer flush threshold: flush to disk every N records (default 50).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("01_generate_responses")

    # Quiet httpx "HTTP Request ... 200 OK" spam
    logging.getLogger("httpx").setLevel(logging.WARNING)

    base_dir = Path(__file__).parent.parent
    data_dir = base_dir.parent / "data"
    output_dir = base_dir / "data" / "raw"

    # Determine active models
    if args.only_32b and args.only_8b:
        logger.error("Cannot use --only-32b and --only-8b together.")
        sys.exit(1)

    if args.only_32b:
        active_models = ["32b"]
    elif args.only_8b:
        active_models = ["8b"]
    else:
        active_models = ["32b", "8b"]

    # Validate API keys and create clients
    clients: dict[str, AsyncOpenAI] = {}
    for key in active_models:
        cfg = MODELS[key]
        api_key = os.environ.get(cfg["env_key"])
        if not api_key:
            logger.error(f"{cfg['env_key']} environment variable is not set.")
            sys.exit(1)
        clients[key] = AsyncOpenAI(base_url=cfg["base_url"], api_key=api_key)

    # Paths
    output_paths = {key: str(output_dir / f"responses_{key}.jsonl") for key in active_models}
    failed_paths = {key: str(output_dir / f"responses_{key}_failed.jsonl") for key in active_models}

    # Load data
    logger.info("Loading system prompts and questions...")
    sp_path = str(data_dir / "system_prompts.jsonl")
    q_path = str(data_dir / "questions.jsonl")
    system_prompts = load_jsonl(sp_path)
    questions_records = load_jsonl(q_path)
    logger.info(f"  {len(system_prompts)} system prompts, {len(questions_records)} question records")

    # Resume: skip items completed by ALL active models
    completed_per_model = {
        key: get_existing_ids(output_paths[key], "question_id") for key in active_models
    }
    if len(active_models) == 1:
        completed_ids = completed_per_model[active_models[0]]
    else:
        completed_ids = set.intersection(*completed_per_model.values()) if all(completed_per_model.values()) else set()

    if completed_ids:
        logger.info(f"Resume: {len(completed_ids)} items already completed by all active models.")

    items = build_work_items(system_prompts, questions_records, completed_ids, args.limit)
    logger.info(f"Items to generate: {len(items)}")
    if not items:
        logger.info("Nothing to do.")
        return

    logger.info(f"Active models: {', '.join(MODELS[k]['name'] for k in active_models)}")
    logger.info(f"Concurrency per provider: {args.concurrency}")
    logger.info(f"Buffer flush threshold: {args.batch_save_interval} records")

    # Run with graceful shutdown on Ctrl+C
    shutdown = asyncio.Event()
    loop = asyncio.new_event_loop()

    def handle_sigint():
        logger.info("Ctrl+C received — finishing in-flight requests and flushing buffers...")
        shutdown.set()

    loop.add_signal_handler(signal.SIGINT, handle_sigint)

    total_start = time.time()
    try:
        loop.run_until_complete(
            run_generation(
                items=items,
                clients=clients,
                active_models=active_models,
                concurrency=args.concurrency,
                flush_threshold=args.batch_save_interval,
                output_paths=output_paths,
                failed_paths=failed_paths,
                shutdown=shutdown,
                logger=logger,
            )
        )
    finally:
        loop.close()

    elapsed = time.time() - total_start
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    for key in active_models:
        logger.info(f"Output [{key}]: {output_paths[key]}")


if __name__ == "__main__":
    main()
