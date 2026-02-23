#!/usr/bin/env python3
"""Generate diverse system prompts for BehaveSHINE training.

Requires: ANTHROPIC_API_KEY environment variable.

Usage:
    python 01_generate_system_prompts.py                     # Generate all 2000
    python 01_generate_system_prompts.py --resume             # Resume from existing output
    python 01_generate_system_prompts.py --target-count 20    # Generate only 20 (for testing)
    python 01_generate_system_prompts.py --batch-size 5       # Override batch size
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Add scripts dir to path for utils import
sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate system prompts for BehaveSHINE")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument("--target-count", type=int, default=None, help="Override total target count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    return parser.parse_args()


def compute_targets(config: dict, target_count: int) -> dict[tuple[str, str], int]:
    """Compute target counts for each (category, complexity) bucket via joint probability."""
    targets = {}
    for cat, cat_weight in config["categories"].items():
        for comp, comp_weight in config["complexity"].items():
            targets[(cat, comp)] = round(target_count * cat_weight * comp_weight)
    return targets


def count_distribution(prompts: list[dict]) -> dict[tuple[str, str], int]:
    """Count prompts per (category, complexity) bucket."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for p in prompts:
        counts[(p.get("category", "unknown"), p.get("complexity", "unknown"))] += 1
    return dict(counts)


def find_most_underrepresented(
    targets: dict[tuple[str, str], int],
    current: dict[tuple[str, str], int],
) -> tuple[str, str] | None:
    """Return the (category, complexity) bucket with the largest deficit."""
    best_bucket = None
    max_deficit = -float("inf")
    for bucket, target in targets.items():
        deficit = target - current.get(bucket, 0)
        if deficit > max_deficit:
            max_deficit = deficit
            best_bucket = bucket
    if max_deficit <= 0:
        return None
    return best_bucket


def format_anti_repetition_list(domains: list[str], max_recent: int = 100) -> str:
    """Format covered domains as a bullet list for the prompt template."""
    if not domains:
        return "None yet (this is the first batch)."
    recent = domains[-max_recent:]
    return "\n".join(f"- {d}" for d in recent)


def validate_prompt(obj: dict, expected_category: str, expected_complexity: str) -> bool:
    """Validate a single generated system prompt object."""
    required = {"system_prompt", "category", "complexity", "domain"}
    if not all(f in obj for f in required):
        return False
    if not isinstance(obj["system_prompt"], str) or len(obj["system_prompt"].strip()) < 20:
        return False
    if obj["category"].strip().lower() != expected_category.strip().lower():
        return False
    if obj["complexity"].strip().lower() != expected_complexity.strip().lower():
        return False
    if not isinstance(obj["domain"], str) or len(obj["domain"].strip()) < 2:
        return False
    return True


def print_summary(targets: dict, current: dict, total: int) -> None:
    """Print distribution summary table."""
    print(f"\n{'=' * 72}")
    print(f"GENERATION COMPLETE: {total} system prompts")
    print(f"{'=' * 72}")
    print(f"{'Category':<25} {'Complexity':<12} {'Target':>8} {'Actual':>8} {'Delta':>8}")
    print(f"{'-' * 72}")

    cat_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"target": 0, "actual": 0})
    for (cat, comp), target in sorted(targets.items()):
        actual = current.get((cat, comp), 0)
        delta = actual - target
        sign = "+" if delta >= 0 else ""
        print(f"{cat:<25} {comp:<12} {target:>8} {actual:>8} {sign}{delta:>7}")
        cat_totals[cat]["target"] += target
        cat_totals[cat]["actual"] += actual

    print(f"{'-' * 72}")
    print(f"\n{'Category Totals':<25} {'Target':>8} {'Actual':>8}")
    print(f"{'-' * 45}")
    for cat in sorted(cat_totals):
        t = cat_totals[cat]["target"]
        a = cat_totals[cat]["actual"]
        print(f"{cat:<25} {t:>8} {a:>8}")
    print(f"{'=' * 72}")


def main():
    args = parse_args()

    # Resolve paths
    base_dir = Path(__file__).resolve().parent.parent
    config_path = args.config or str(base_dir / "configs" / "generation_config.yaml")
    config = utils.load_config(config_path)

    logger = utils.setup_logging("generate_system_prompts")

    target_count = args.target_count or config["system_prompts"]["target_count"]
    batch_size = args.batch_size or config["system_prompts"]["batch_size"]
    output_path = str(base_dir / config["system_prompts"]["output_path"])

    # Load prompt template
    prompt_template_path = base_dir / "prompts" / "system_prompt_generation.txt"
    prompt_template = prompt_template_path.read_text()

    # Resume support
    existing_prompts: list[dict] = []
    covered_domains: list[str] = []
    if args.resume:
        existing_prompts = utils.load_jsonl(output_path)
        covered_domains = [p["domain"] for p in existing_prompts if "domain" in p]
        logger.info(f"Resuming: found {len(existing_prompts)} existing prompts")

    # Distribution tracking
    targets = compute_targets(config, target_count)
    current_counts = count_distribution(existing_prompts)
    total_generated = len(existing_prompts)

    # Setup
    rate_limiter = utils.RateLimiter(config["api"]["requests_per_minute"])
    batch_num = 0

    logger.info(f"Target: {target_count} system prompts, batch size: {batch_size}")
    logger.info(f"Output: {output_path}")

    # Main generation loop
    while total_generated < target_count:
        batch_num += 1

        # Find most underrepresented bucket
        bucket = find_most_underrepresented(targets, current_counts)
        if bucket is None:
            logger.info("All category x complexity buckets at target. Stopping.")
            break
        target_category, target_complexity = bucket

        remaining_for_bucket = targets[bucket] - current_counts.get(bucket, 0)
        effective_batch_size = min(batch_size, remaining_for_bucket, target_count - total_generated)

        # Build prompt
        anti_rep = format_anti_repetition_list(covered_domains)
        filled_prompt = prompt_template.format(
            batch_size=effective_batch_size,
            target_category=target_category,
            target_complexity=target_complexity,
            anti_repetition_list=anti_rep,
        )

        logger.info(
            f"Batch {batch_num}: requesting {effective_batch_size} "
            f"{target_category}/{target_complexity} prompts "
            f"(total so far: {total_generated}/{target_count})"
        )

        # Call Claude
        try:
            response_text = utils.call_claude(
                prompt=filled_prompt,
                max_tokens=8192,
                temperature=1.0,
                model=config["api"]["model"],
                max_retries=config["api"]["max_retries"],
                retry_delay_seconds=config["api"]["retry_delay_seconds"],
                rate_limiter=rate_limiter,
            )
        except Exception as e:
            logger.error(f"Batch {batch_num}: API call failed after retries: {e}")
            continue

        # Parse response
        try:
            parsed = utils.parse_json_response(response_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Batch {batch_num}: failed to parse JSON: {e}")
            logger.debug(f"Raw response (first 500 chars): {response_text[:500]}")
            continue

        if not isinstance(parsed, list):
            logger.warning(f"Batch {batch_num}: expected list, got {type(parsed).__name__}")
            continue

        # Validate and save each prompt
        valid_count = 0
        for item in parsed:
            if not validate_prompt(item, target_category, target_complexity):
                logger.warning(f"Batch {batch_num}: invalid item skipped: {item.get('domain', 'unknown')}")
                continue

            # Normalize category/complexity to canonical casing and assign metadata
            item["category"] = target_category
            item["complexity"] = target_complexity
            item["id"] = f"sp_{total_generated:05d}"
            item["estimated_tokens"] = utils.count_tokens_approx(item["system_prompt"])
            item["generation_batch"] = batch_num

            utils.append_jsonl(output_path, item)

            # Update tracking
            covered_domains.append(item["domain"])
            current_counts[bucket] = current_counts.get(bucket, 0) + 1
            total_generated += 1
            valid_count += 1

        logger.info(
            f"Batch {batch_num} complete: {valid_count}/{len(parsed)} valid, "
            f"total: {total_generated}/{target_count}"
        )

        if valid_count == 0 and len(parsed) > 0:
            logger.warning(f"Batch {batch_num}: zero valid items — will retry this bucket.")

    # Final summary
    print_summary(targets, current_counts, total_generated)


if __name__ == "__main__":
    main()
