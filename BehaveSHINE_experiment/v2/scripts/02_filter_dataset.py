#!/usr/bin/env python3
"""Script 02: Filter dataset by comparing 8B and 32B responses.

Drops pairs where responses are nearly identical (no learning signal).
Never drops off-topic questions (q11, q12).

Uses normalized edit distance (difflib.SequenceMatcher) and character trigram
Jaccard similarity. A pair is dropped only if BOTH thresholds are exceeded.

Usage:
    python 02_filter_dataset.py
    python 02_filter_dataset.py --dry-run
    python 02_filter_dataset.py --edit-threshold 0.15 --jaccard-threshold 0.85
    python 02_filter_dataset.py --show-dropped 10
"""

import argparse
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_jsonl, setup_logging, write_jsonl


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def normalized_edit_distance(a: str, b: str) -> float:
    """Return 1 - SequenceMatcher.ratio(). 0 = identical, 1 = completely different."""
    if not a and not b:
        return 0.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def char_trigrams(text: str) -> set[str]:
    """Return set of character trigrams from text."""
    if len(text) < 3:
        return {text} if text else set()
    return {text[i:i+3] for i in range(len(text) - 2)}


def trigram_jaccard(a: str, b: str) -> float:
    """Jaccard similarity of character trigram sets. 1.0 = identical, 0.0 = no overlap."""
    tris_a = char_trigrams(a)
    tris_b = char_trigrams(b)
    if not tris_a and not tris_b:
        return 1.0
    if not tris_a or not tris_b:
        return 0.0
    intersection = len(tris_a & tris_b)
    union = len(tris_a | tris_b)
    return intersection / union


# ---------------------------------------------------------------------------
# Filtering logic
# ---------------------------------------------------------------------------

def is_off_topic(question_id: str, question_type: str | None) -> bool:
    """Check if a question is off-topic. Uses question_type if available, else question_id."""
    if question_type and question_type == "off_topic":
        return True
    # Fallback: q11 and q12 are off-topic by convention
    try:
        q_num = int(question_id.split("_q")[-1])
        return q_num >= 11
    except (ValueError, IndexError):
        return False


def filter_pair(
    resp_32b: dict,
    resp_8b: dict,
    question_type: str | None,
    edit_threshold: float,
    jaccard_threshold: float,
) -> dict:
    """Evaluate one pair. Returns a record with kept/dropped flag and metrics."""
    q_id = resp_32b["question_id"]
    answer_32b = resp_32b.get("answer", "")
    answer_8b = resp_8b.get("answer", "")

    # Always drop broken/empty responses
    if not answer_32b.strip() or not answer_8b.strip():
        return {
            "question_id": q_id,
            "system_prompt_id": resp_32b["system_prompt_id"],
            "kept": False,
            "reason": "empty_response",
            "edit_distance": None,
            "trigram_jaccard": None,
            "off_topic": is_off_topic(q_id, question_type),
        }

    edit_dist = normalized_edit_distance(answer_32b, answer_8b)
    jaccard = trigram_jaccard(answer_32b, answer_8b)
    off_topic = is_off_topic(q_id, question_type)

    # Never drop off-topic
    if off_topic:
        kept = True
        reason = "off_topic_preserved"
    elif edit_dist < edit_threshold and jaccard > jaccard_threshold:
        kept = False
        reason = "too_similar"
    else:
        kept = True
        reason = "kept"

    return {
        "question_id": q_id,
        "system_prompt_id": resp_32b["system_prompt_id"],
        "kept": kept,
        "reason": reason,
        "edit_distance": round(edit_dist, 4),
        "trigram_jaccard": round(jaccard, 4),
        "off_topic": off_topic,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter 8B vs 32B response pairs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing output files.")
    parser.add_argument("--edit-threshold", type=float, default=0.15,
                        help="Normalized edit distance threshold (default 0.15).")
    parser.add_argument("--jaccard-threshold", type=float, default=0.85,
                        help="Trigram Jaccard threshold (default 0.85).")
    parser.add_argument("--show-dropped", type=int, default=0,
                        help="Print N dropped examples for inspection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("02_filter_dataset")

    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw"
    filtered_dir = base_dir / "data" / "filtered"

    # Load responses
    logger.info("Loading responses...")
    responses_32b = load_jsonl(str(raw_dir / "responses_32b.jsonl"))
    responses_8b = load_jsonl(str(raw_dir / "responses_8b.jsonl"))
    logger.info(f"  32B: {len(responses_32b)} responses, 8B: {len(responses_8b)} responses")

    # Build lookup by question_id
    resp_32b_lookup = {r["question_id"]: r for r in responses_32b}
    resp_8b_lookup = {r["question_id"]: r for r in responses_8b}

    # Load questions for authoritative question_type
    data_dir = base_dir.parent / "data"
    questions_records = load_jsonl(str(data_dir / "questions.jsonl"))
    q_type_lookup: dict[str, str] = {}
    for record in questions_records:
        for q in record.get("questions", []):
            q_type_lookup[q["id"]] = q.get("type", "on_topic")

    # Find common question_ids
    common_ids = sorted(set(resp_32b_lookup) & set(resp_8b_lookup))
    only_32b = set(resp_32b_lookup) - set(resp_8b_lookup)
    only_8b = set(resp_8b_lookup) - set(resp_32b_lookup)
    if only_32b or only_8b:
        logger.warning(f"  Only in 32B: {len(only_32b)}, only in 8B: {len(only_8b)}")
    logger.info(f"  Common pairs: {len(common_ids)}")

    # Filter
    logger.info(f"Filtering with edit_threshold={args.edit_threshold}, jaccard_threshold={args.jaccard_threshold}...")
    results = []
    for q_id in common_ids:
        result = filter_pair(
            resp_32b_lookup[q_id],
            resp_8b_lookup[q_id],
            q_type_lookup.get(q_id),
            args.edit_threshold,
            args.jaccard_threshold,
        )
        results.append(result)

    # Compute stats
    kept = [r for r in results if r["kept"]]
    dropped = [r for r in results if not r["kept"]]
    reason_counts = Counter(r["reason"] for r in results)

    kept_on_topic = [r for r in kept if not r["off_topic"]]
    kept_off_topic = [r for r in kept if r["off_topic"]]
    dropped_off_topic = [r for r in dropped if r["off_topic"]]

    edit_dists = [r["edit_distance"] for r in results if r["edit_distance"] is not None]
    jaccards = [r["trigram_jaccard"] for r in results if r["trigram_jaccard"] is not None]

    stats = {
        "total_pairs": len(results),
        "kept": len(kept),
        "dropped": len(dropped),
        "kept_pct": round(100 * len(kept) / len(results), 1) if results else 0,
        "kept_on_topic": len(kept_on_topic),
        "kept_off_topic": len(kept_off_topic),
        "dropped_off_topic": len(dropped_off_topic),
        "reason_counts": dict(reason_counts),
        "edit_distance_mean": round(sum(edit_dists) / len(edit_dists), 4) if edit_dists else None,
        "trigram_jaccard_mean": round(sum(jaccards) / len(jaccards), 4) if jaccards else None,
        "thresholds": {
            "edit_distance": args.edit_threshold,
            "trigram_jaccard": args.jaccard_threshold,
        },
    }

    # Print summary
    logger.info(f"Results: {len(kept)} kept ({stats['kept_pct']}%), {len(dropped)} dropped")
    logger.info(f"  On-topic kept: {len(kept_on_topic)}, Off-topic kept: {len(kept_off_topic)}")
    logger.info(f"  Off-topic dropped: {len(dropped_off_topic)} (should be 0)")
    for reason, count in sorted(reason_counts.items()):
        logger.info(f"  {reason}: {count}")
    if edit_dists:
        logger.info(f"  Mean edit distance: {stats['edit_distance_mean']}")
        logger.info(f"  Mean trigram Jaccard: {stats['trigram_jaccard_mean']}")

    if args.show_dropped > 0:
        logger.info(f"\n--- Dropped examples (first {args.show_dropped}) ---")
        for r in dropped[:args.show_dropped]:
            q_id = r["question_id"]
            logger.info(f"  {q_id}: edit={r['edit_distance']}, jaccard={r['trigram_jaccard']}, reason={r['reason']}")
            a32 = resp_32b_lookup[q_id].get("answer", "")[:100]
            a8 = resp_8b_lookup[q_id].get("answer", "")[:100]
            logger.info(f"    32B: {a32}...")
            logger.info(f"    8B:  {a8}...")

    if args.dry_run:
        logger.info("Dry run — no files written.")
        return

    # Write outputs
    logger.info("Writing output files...")
    write_jsonl(str(filtered_dir / "filtered_pairs.jsonl"), results)
    logger.info(f"  Wrote {len(results)} filter records -> {filtered_dir / 'filtered_pairs.jsonl'}")

    stats_path = filtered_dir / "filter_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    logger.info(f"  Wrote stats -> {stats_path}")


if __name__ == "__main__":
    main()
