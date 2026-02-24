#!/usr/bin/env python3
"""Script 03: Prepare train/eval/test splits from filtered response pairs.

Joins kept pairs with system prompt text and question metadata. Splits by
system_prompt_id (85/10/5). Off-topic questions are excluded from train but
included in eval/test for capability preservation measurement.

Usage:
    python 03_prepare_splits.py
    python 03_prepare_splits.py --train-ratio 0.85 --eval-ratio 0.10 --test-ratio 0.05
    python 03_prepare_splits.py --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_jsonl, setup_logging, write_jsonl


# ---------------------------------------------------------------------------
# Difficulty weight computation
# ---------------------------------------------------------------------------

def compute_difficulty_weight(filter_metrics: dict) -> float:
    """
    Weight training examples by how badly 8B diverged from 32B.

    Higher weight = 8B failed more = more useful training signal.
    Returns float in [0.5, 2.0].

    - Identical responses -> 0.5 (low priority but still trained on)
    - Moderately different -> ~1.25 (normal priority)
    - Completely different -> 2.0 (high priority)
    """
    norm_edit = filter_metrics.get("norm_edit_distance", 0.5)
    jaccard = filter_metrics.get("jaccard_similarity", 0.5)

    edit_score = norm_edit                    # 0=identical, 1=completely different
    content_score = 1.0 - jaccard            # 0=identical content, 1=no overlap
    raw_difficulty = 0.6 * edit_score + 0.4 * content_score   # weighted average
    weight = 0.5 + 1.5 * raw_difficulty      # map to [0.5, 2.0]
    return round(weight, 4)


# ---------------------------------------------------------------------------
# Data loading and joining
# ---------------------------------------------------------------------------

def load_and_join(
    sp_path: str,
    q_path: str,
    resp_32b_path: str,
    resp_8b_path: str,
    filter_path: str,
    logger,
) -> list[dict]:
    """Load all data sources, join kept pairs into training examples."""
    logger.info("Loading data files...")
    system_prompts = load_jsonl(sp_path)
    questions_records = load_jsonl(q_path)
    responses_32b = load_jsonl(resp_32b_path)
    responses_8b = load_jsonl(resp_8b_path)
    filter_records = load_jsonl(filter_path)

    logger.info(
        f"  {len(system_prompts)} SPs, {len(questions_records)} Q records, "
        f"{len(responses_32b)} 32B responses, {len(responses_8b)} 8B responses, "
        f"{len(filter_records)} filter records"
    )

    # Build lookups
    sp_lookup = {sp["id"]: sp for sp in system_prompts}
    resp_32b_lookup = {r["question_id"]: r for r in responses_32b}
    resp_8b_lookup = {r["question_id"]: r for r in responses_8b}

    # Question metadata lookup
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

    # Filter record lookup (for metrics)
    filter_lookup = {}
    for r in filter_records:
        filter_lookup[r["question_id"]] = {
            "edit_distance": r.get("edit_distance"),
            "trigram_jaccard": r.get("trigram_jaccard"),
        }

    # Get kept question_ids from filter
    kept_ids = {r["question_id"] for r in filter_records if r.get("kept", False)}
    logger.info(f"  Kept pairs from filter: {len(kept_ids)}")

    # Join
    examples = []
    missing = 0
    for q_id in sorted(kept_ids):
        q_info = q_lookup.get(q_id)
        if q_info is None:
            missing += 1
            continue
        sp = sp_lookup.get(q_info["sp_id"])
        if sp is None:
            missing += 1
            continue
        resp_32b = resp_32b_lookup.get(q_id)
        resp_8b = resp_8b_lookup.get(q_id)
        if resp_32b is None:
            missing += 1
            continue

        # Build filter_metrics and difficulty_weight
        fr = filter_lookup.get(q_id, {})
        answer_8b_text = resp_8b["answer"] if resp_8b else ""
        answer_32b_text = resp_32b["answer"]

        filter_metrics = {
            "norm_edit_distance": fr.get("edit_distance", 0.5),
            "jaccard_similarity": fr.get("trigram_jaccard", 0.5),
            "length_ratio": len(answer_32b_text) / max(len(answer_8b_text), 1),
        }
        difficulty_weight = compute_difficulty_weight(filter_metrics)

        examples.append({
            "context": sp["system_prompt"],
            "question": q_info["question_text"],
            "answer": answer_32b_text,
            "answer_8b": answer_8b_text,
            "difficulty_weight": difficulty_weight,
            "filter_metrics": filter_metrics,
            "system_prompt_id": q_info["sp_id"],
            "question_id": q_id,
            "category": sp.get("category", ""),
            "question_type": q_info["question_type"],
            "question_difficulty": q_info["difficulty"],
        })

    logger.info(f"  {len(examples)} joined examples, {missing} missing data skipped")
    return examples


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_by_sp_id(
    examples: list[dict],
    eval_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split examples into train/eval/test by system_prompt_id."""
    all_sp_ids = sorted({ex["system_prompt_id"] for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(all_sp_ids)

    n = len(all_sp_ids)
    n_test = max(1, round(n * test_ratio))
    n_eval = max(1, round(n * eval_ratio))

    test_ids = set(all_sp_ids[:n_test])
    eval_ids = set(all_sp_ids[n_test: n_test + n_eval])

    train, eval_set, test = [], [], []
    for ex in examples:
        sp_id = ex["system_prompt_id"]
        if sp_id in test_ids:
            test.append(ex)
        elif sp_id in eval_ids:
            eval_set.append(ex)
        else:
            train.append(ex)

    return train, eval_set, test


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(
    all_examples: list[dict],
    train: list[dict],
    eval_set: list[dict],
    test: list[dict],
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

    def count_off_topic(examples: list[dict]) -> int:
        return sum(1 for ex in examples if ex.get("question_type") == "off_topic")

    return {
        "total": len(all_examples),
        "train": len(train),
        "eval": len(eval_set),
        "test": len(test),
        "train_off_topic": count_off_topic(train),
        "eval_off_topic": count_off_topic(eval_set),
        "test_off_topic": count_off_topic(test),
        "unique_sps": len({ex["system_prompt_id"] for ex in all_examples}),
        "avg_context_chars": avg_len(all_examples, "context"),
        "avg_answer_chars": avg_len(all_examples, "answer"),
        "category_distribution": count_field(all_examples, "category"),
        "question_type_distribution": count_field(all_examples, "question_type"),
        "difficulty_distribution": count_field(all_examples, "question_difficulty"),
    }


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BehaveSHINE v2 train/eval/test splits.")
    parser.add_argument("--train-ratio", type=float, default=0.85,
                        help="Fraction of SPs for train (default 0.85).")
    parser.add_argument("--eval-ratio", type=float, default=0.10,
                        help="Fraction of SPs for eval (default 0.10).")
    parser.add_argument("--test-ratio", type=float, default=0.05,
                        help="Fraction of SPs for test (default 0.05).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splitting (default 42).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("03_prepare_splits")

    base_dir = Path(__file__).parent.parent
    data_dir = base_dir.parent / "data"
    raw_dir = base_dir / "data" / "raw"
    filtered_dir = base_dir / "data" / "filtered"
    splits_dir = base_dir / "data" / "splits"

    # Load and join
    examples = load_and_join(
        sp_path=str(data_dir / "system_prompts.jsonl"),
        q_path=str(data_dir / "questions.jsonl"),
        resp_32b_path=str(raw_dir / "responses_32b.jsonl"),
        resp_8b_path=str(raw_dir / "responses_8b.jsonl"),
        filter_path=str(filtered_dir / "filtered_pairs.jsonl"),
        logger=logger,
    )
    if not examples:
        logger.error("No examples found. Run scripts 01 and 02 first.")
        sys.exit(1)

    # Split by SP ID
    train_all, eval_set, test = split_by_sp_id(
        examples, args.eval_ratio, args.test_ratio, args.seed,
    )
    logger.info(
        f"Split (before off-topic removal): train={len(train_all)}, "
        f"eval={len(eval_set)}, test={len(test)}"
    )

    # Remove off-topic from train
    train_off_topic = [ex for ex in train_all if ex.get("question_type") == "off_topic"]
    train = [ex for ex in train_all if ex.get("question_type") != "off_topic"]
    logger.info(
        f"Removed {len(train_off_topic)} off-topic from train. "
        f"Final train: {len(train)}"
    )

    # Write splits
    logger.info("Writing output files...")
    write_jsonl(str(splits_dir / "train.jsonl"), train)
    logger.info(f"  train.jsonl: {len(train)} examples")
    write_jsonl(str(splits_dir / "eval.jsonl"), eval_set)
    logger.info(f"  eval.jsonl: {len(eval_set)} examples")
    write_jsonl(str(splits_dir / "test.jsonl"), test)
    logger.info(f"  test.jsonl: {len(test)} examples")

    # Write stats
    stats = compute_stats(examples, train, eval_set, test)
    stats["ratios"] = {
        "train": args.train_ratio,
        "eval": args.eval_ratio,
        "test": args.test_ratio,
    }
    stats["seed"] = args.seed
    stats_path = splits_dir / "split_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    logger.info(f"  Wrote stats -> {stats_path}")

    # Print summary
    logger.info("=" * 50)
    logger.info(f"Dataset summary:")
    logger.info(f"  Total kept examples : {stats['total']}")
    logger.info(f"  Unique SPs          : {stats['unique_sps']}")
    logger.info(f"  Train               : {stats['train']} (off-topic removed: {len(train_off_topic)})")
    logger.info(f"  Eval                : {stats['eval']} (off-topic: {stats['eval_off_topic']})")
    logger.info(f"  Test                : {stats['test']} (off-topic: {stats['test_off_topic']})")
    logger.info(f"  Avg context chars   : {stats['avg_context_chars']}")
    logger.info(f"  Avg answer chars    : {stats['avg_answer_chars']}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
