#!/usr/bin/env python3
"""Script 04: Prepare training dataset from generated system prompts, questions, and responses.

Joins data/raw/{system_prompts,questions,responses_teacher}.jsonl and produces
80/10/10 train/val/test splits (split by system_prompt_id, not by question).

Usage:
    python 04_prepare_dataset.py
    python 04_prepare_dataset.py --val-ratio 0.1 --test-ratio 0.1
    python 04_prepare_dataset.py --seed 42
    python 04_prepare_dataset.py --config configs/generation_config.yaml
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, load_jsonl, setup_logging


# ---------------------------------------------------------------------------
# Data loading and joining
# ---------------------------------------------------------------------------

def load_and_join(
    sp_path: str,
    q_path: str,
    resp_path: str,
    logger,
) -> tuple[list[dict], int]:
    """Load and join all three raw files into training examples.

    Returns (examples, missing_count) where missing_count is the number of
    (sp_id, question_id) pairs for which no response was found.
    """
    logger.info("Loading raw files...")
    system_prompts = load_jsonl(sp_path)
    questions_records = load_jsonl(q_path)
    responses = load_jsonl(resp_path)

    logger.info(
        f"  {len(system_prompts)} system prompts, "
        f"{len(questions_records)} question records, "
        f"{len(responses)} responses"
    )

    # Build lookup dicts
    sp_lookup: dict[str, dict] = {sp["id"]: sp for sp in system_prompts}
    resp_lookup: dict[str, dict] = {r["question_id"]: r for r in responses}

    # Flatten questions into a lookup: question_id -> {question_text, type, difficulty, tags, sp_id}
    q_lookup: dict[str, dict] = {}
    for record in questions_records:
        sp_id = record["system_prompt_id"]
        for q in record.get("questions", []):
            q_lookup[q["id"]] = {
                "question_text": q["question"],
                "question_type": q.get("type", "on_topic"),
                "difficulty": q.get("difficulty", ""),
                "tags": q.get("tags", []),
                "sp_id": sp_id,
            }

    # Join: iterate responses and look up matching question + system prompt
    examples = []
    missing = 0
    for q_id, q_info in q_lookup.items():
        sp_id = q_info["sp_id"]
        sp = sp_lookup.get(sp_id)
        if sp is None:
            missing += 1
            continue
        resp = resp_lookup.get(q_id)
        if resp is None:
            missing += 1
            continue

        sp_text = sp["system_prompt"]
        examples.append({
            "system_prompt_id": sp_id,
            "question_id": q_id,
            "context": sp_text,
            "question": q_info["question_text"],
            "answer": resp["answer"],
            "messages_with_context": [
                {"role": "system", "content": sp_text},
                {"role": "user", "content": q_info["question_text"]},
            ],
            "category": sp.get("category", ""),
            "complexity": sp.get("complexity", ""),
            "question_type": q_info["question_type"],
            "difficulty": q_info["difficulty"],
            "tags": q_info["tags"],
        })

    logger.info(f"  {len(examples)} complete examples, {missing} missing responses skipped")
    return examples, missing


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_by_sp_id(
    examples: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split examples into train/val/test by system_prompt_id (all questions for a given
    system prompt go into the same split)."""
    all_sp_ids = sorted({ex["system_prompt_id"] for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(all_sp_ids)

    n = len(all_sp_ids)
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))

    test_ids = set(all_sp_ids[:n_test])
    val_ids = set(all_sp_ids[n_test: n_test + n_val])

    train, val, test = [], [], []
    for ex in examples:
        sp_id = ex["system_prompt_id"]
        if sp_id in test_ids:
            test.append(ex)
        elif sp_id in val_ids:
            val.append(ex)
        else:
            train.append(ex)

    return train, val, test


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(
    all_examples: list[dict],
    train: list[dict],
    val: list[dict],
    test: list[dict],
    missing: int,
) -> dict:
    def avg_len(examples: list[dict], field: str) -> float:
        vals = [len(ex[field]) for ex in examples if ex.get(field)]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    def count_field(examples: list[dict], field: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ex in examples:
            v = ex.get(field, "unknown")
            counts[v] = counts.get(v, 0) + 1
        return dict(sorted(counts.items()))

    return {
        "total": len(all_examples),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "missing_responses": missing,
        "avg_context_chars": avg_len(all_examples, "context"),
        "avg_answer_chars": avg_len(all_examples, "answer"),
        "category_distribution": count_field(all_examples, "category"),
        "complexity_distribution": count_field(all_examples, "complexity"),
        "question_type_distribution": count_field(all_examples, "question_type"),
    }


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_jsonl(path: str, examples: list[dict], logger) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"  Wrote {len(examples)} examples -> {path}")


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BehaveSHINE training dataset.")
    parser.add_argument("--val-ratio", type=float, default=None,
                        help="Fraction of system prompts for validation (default: from config).")
    parser.add_argument("--test-ratio", type=float, default=None,
                        help="Fraction of system prompts for test (default: from config).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for splitting (default: from config).")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "configs" / "generation_config.yaml"),
        help="Path to generation_config.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("04_prepare_dataset")

    base_dir = Path(__file__).parent.parent
    config = load_config(args.config)
    ds_cfg = config["dataset"]
    resp_cfg = config["responses"]

    val_ratio: float = args.val_ratio if args.val_ratio is not None else ds_cfg["val_ratio"]
    test_ratio: float = args.test_ratio if args.test_ratio is not None else ds_cfg["test_ratio"]
    seed: int = args.seed if args.seed is not None else ds_cfg["seed"]

    sp_path = str(base_dir / config["system_prompts"]["output_path"])
    q_path = str(base_dir / config["questions"]["output_path"])
    resp_path = str(base_dir / resp_cfg["output_path"])

    train_path = str(base_dir / ds_cfg["train_path"])
    val_path = str(base_dir / ds_cfg["val_path"])
    test_path = str(base_dir / ds_cfg["test_path"])
    test_off_topic_path = str(base_dir / ds_cfg["test_off_topic_path"])
    stats_path = str(base_dir / ds_cfg["stats_path"])

    # Load and join
    examples, missing = load_and_join(sp_path, q_path, resp_path, logger)
    if not examples:
        logger.error("No complete examples found. Run script 03 first.")
        sys.exit(1)

    # Split
    train, val, test = split_by_sp_id(examples, val_ratio, test_ratio, seed)
    logger.info(
        f"Split: train={len(train)}, val={len(val)}, test={len(test)} "
        f"(ratios: val={val_ratio}, test={test_ratio}, seed={seed})"
    )

    # Write splits
    logger.info("Writing output files...")
    write_jsonl(train_path, train, logger)
    write_jsonl(val_path, val, logger)
    write_jsonl(test_path, test, logger)

    test_off_topic = [ex for ex in test if ex["question_type"] == "off_topic"]
    write_jsonl(test_off_topic_path, test_off_topic, logger)

    # Write stats
    stats = compute_stats(examples, train, val, test, missing)
    stats_p = Path(stats_path)
    stats_p.parent.mkdir(parents=True, exist_ok=True)
    stats_p.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    logger.info(f"  Wrote stats -> {stats_path}")

    # Print summary
    logger.info("=" * 50)
    logger.info(f"Dataset summary:")
    logger.info(f"  Total examples   : {stats['total']}")
    logger.info(f"  Train            : {stats['train']}")
    logger.info(f"  Val              : {stats['val']}")
    logger.info(f"  Test             : {stats['test']}")
    logger.info(f"  Test (off-topic) : {len(test_off_topic)}")
    logger.info(f"  Missing responses: {stats['missing_responses']}")
    logger.info(f"  Avg context chars: {stats['avg_context_chars']}")
    logger.info(f"  Avg answer chars : {stats['avg_answer_chars']}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
