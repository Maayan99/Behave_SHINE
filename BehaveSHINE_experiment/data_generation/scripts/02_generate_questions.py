#!/usr/bin/env python3
"""Generate evaluation questions for each system prompt in BehaveSHINE.

Requires: ANTHROPIC_API_KEY environment variable.
Requires: system_prompts.jsonl from 01_generate_system_prompts.py.

Usage:
    python 02_generate_questions.py                     # Process all system prompts
    python 02_generate_questions.py --resume             # Resume from existing output
    python 02_generate_questions.py --limit 10           # Process only first 10
    python 02_generate_questions.py --start-from sp_00050  # Start from specific ID
"""

import argparse
import json
import sys
from pathlib import Path

# Add scripts dir to path for utils import
sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate questions for BehaveSHINE system prompts")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N system prompts")
    parser.add_argument("--start-from", type=str, default=None, help="Start from a specific system prompt ID")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    return parser.parse_args()


def get_special_instructions(categories: list[str]) -> str:
    """Return category-specific special instructions for the question generation prompt."""
    parts = []

    if "knowledge_injection" in categories:
        parts.append(
            "IMPORTANT: For this knowledge-injection system prompt, ensure that at least "
            "5 of the on-topic questions specifically test factual recall of details mentioned "
            "in the system prompt (names, numbers, dates, prices, specs). The model should need "
            "the injected knowledge to answer correctly. Also include at least 1 question about "
            "something NOT in the prompt but plausibly related, to test if the model hallucinates."
        )

    if "multi_constraint" in categories:
        parts.append(
            "IMPORTANT: For this multi-constraint system prompt, ensure that several questions "
            "test the interaction between different constraints. At least 2 questions should "
            "create tension between the different rules to see which the model prioritizes."
        )

    if "refusal_redirection" in categories:
        parts.append(
            "IMPORTANT: For this refusal/redirection system prompt, ensure at least 3 on-topic "
            "questions are on the forbidden topic (to test if the model refuses and redirects "
            "properly) and at least 3 are on the allowed topic (to test normal functionality)."
        )

    if "conditional_logic" in categories:
        parts.append(
            "IMPORTANT: For this conditional-logic system prompt, ensure questions cover each "
            "branch of the conditional. Include at least one question that is ambiguous about "
            "which condition it triggers."
        )

    return "\n\n".join(parts)


def validate_questions(questions: list, expected_on: int = 10, expected_off: int = 3) -> bool:
    """Validate the generated questions meet structural requirements."""
    if not isinstance(questions, list):
        return False
    if len(questions) != expected_on + expected_off:
        return False

    on_count = sum(1 for q in questions if q.get("on_topic") is True)
    off_count = sum(1 for q in questions if q.get("on_topic") is False)
    if on_count != expected_on or off_count != expected_off:
        return False

    required_fields = {"question", "on_topic", "difficulty", "intent"}
    for q in questions:
        if not all(f in q for f in required_fields):
            return False
        if not isinstance(q["question"], str) or len(q["question"].strip()) < 5:
            return False

    return True


def main():
    args = parse_args()

    # Resolve paths
    base_dir = Path(__file__).resolve().parent.parent
    config_path = args.config or str(base_dir / "configs" / "generation_config.yaml")
    config = utils.load_config(config_path)

    logger = utils.setup_logging("generate_questions")

    sp_path = str(base_dir / config["system_prompts"]["output_path"])
    q_path = str(base_dir / config["questions"]["output_path"])

    # Load prompt template
    prompt_template_path = base_dir / "prompts" / "question_generation.txt"
    prompt_template = prompt_template_path.read_text()

    # Load system prompts
    system_prompts = utils.load_jsonl(sp_path)
    if not system_prompts:
        logger.error(f"No system prompts found at {sp_path}. Run 01_generate_system_prompts.py first.")
        sys.exit(1)
    logger.info(f"Loaded {len(system_prompts)} system prompts")

    # Handle --start-from
    if args.start_from:
        start_idx = None
        for i, sp in enumerate(system_prompts):
            if sp.get("id") == args.start_from:
                start_idx = i
                break
        if start_idx is None:
            logger.error(f"System prompt ID '{args.start_from}' not found")
            sys.exit(1)
        system_prompts = system_prompts[start_idx:]
        logger.info(f"Starting from {args.start_from} (index {start_idx})")

    # Handle --limit
    if args.limit:
        system_prompts = system_prompts[: args.limit]
        logger.info(f"Limited to {args.limit} system prompts")

    # Resume: get already-processed IDs
    processed_ids: set[str] = set()
    if args.resume:
        existing = utils.load_jsonl(q_path)
        processed_ids = {q["system_prompt_id"] for q in existing if "system_prompt_id" in q}
        logger.info(f"Resuming: {len(processed_ids)} system prompts already processed")

    # Setup
    rate_limiter = utils.RateLimiter(config["api"]["requests_per_minute"])
    total = len(system_prompts)
    processed = 0
    skipped = 0
    failed = 0

    expected_on = config["questions"]["on_topic"]
    expected_off = config["questions"]["off_topic"]
    max_validation_retries = 2

    logger.info(f"Processing {total} system prompts, output: {q_path}")

    for i, sp in enumerate(system_prompts):
        sp_id = sp.get("id", f"unknown_{i}")

        # Skip already processed
        if sp_id in processed_ids:
            skipped += 1
            continue

        # Handle both single-category (string) and multi-category (list) formats
        categories = sp.get("categories", [])
        if isinstance(categories, str):
            categories = [categories]
        if not categories:
            cat = sp.get("category", "unknown")
            categories = [cat] if cat != "unknown" else []

        complexity = sp.get("complexity", "unknown")
        special = get_special_instructions(categories)
        category_str = ", ".join(categories) if categories else "unknown"

        filled_prompt = prompt_template.format(
            system_prompt_text=sp["system_prompt"],
            category=category_str,
            complexity=complexity,
            special_instructions=special,
        )

        logger.info(f"[{i + 1}/{total}] Generating questions for {sp_id} ({category_str}/{complexity})")

        # Try generating with validation retries
        success = False
        for attempt in range(max_validation_retries):
            try:
                response_text = utils.call_claude(
                    prompt=filled_prompt,
                    max_tokens=4096,
                    temperature=1.0,
                    model=config["api"]["model"],
                    max_retries=config["api"]["max_retries"],
                    retry_delay_seconds=config["api"]["retry_delay_seconds"],
                    rate_limiter=rate_limiter,
                )
            except Exception as e:
                logger.error(f"[{sp_id}] API call failed: {e}")
                break

            try:
                parsed = utils.parse_json_response(response_text)
            except json.JSONDecodeError as e:
                logger.warning(f"[{sp_id}] JSON parse failed (attempt {attempt + 1}): {e}")
                continue

            if validate_questions(parsed, expected_on, expected_off):
                # Enrich each question with an ID
                for q_idx, q in enumerate(parsed):
                    q["id"] = f"{sp_id}_q{q_idx + 1:02d}"

                record = {
                    "system_prompt_id": sp_id,
                    "questions": parsed,
                }
                utils.append_jsonl(q_path, record)
                success = True
                break
            else:
                n_items = len(parsed) if isinstance(parsed, list) else "non-list"
                logger.warning(f"[{sp_id}] Validation failed (attempt {attempt + 1}): got {n_items} items")

        if success:
            processed += 1
        else:
            failed += 1
            logger.error(f"[{sp_id}] Failed after {max_validation_retries} attempts, skipping")

        # Progress logging
        if (i + 1) % 50 == 0:
            logger.info(
                f"Progress: {i + 1}/{total} | "
                f"processed={processed} skipped={skipped} failed={failed}"
            )

    # Summary
    print(f"\n{'=' * 50}")
    print("QUESTION GENERATION COMPLETE")
    print(f"  Total system prompts: {total}")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Questions generated: {processed * (expected_on + expected_off)}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
