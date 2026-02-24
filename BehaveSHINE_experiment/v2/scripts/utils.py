#!/usr/bin/env python3
"""Shared utilities for BehaveSHINE v2 data pipeline."""

import json
import logging
from pathlib import Path


def setup_logging(name: str) -> logging.Logger:
    """Configure and return a named logger with timestamp format."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger(name)


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts. Returns [] if file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return []
    results = []
    for i, line in enumerate(p.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            logging.getLogger("utils").warning(f"Skipping corrupt line {i+1} in {path}")
    return results


def append_jsonl(path: str, obj: dict) -> None:
    """Append one JSON object as a line to a JSONL file. Creates file and dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def get_existing_ids(path: str, id_field: str = "id") -> set:
    """Get set of IDs already present in a JSONL file, for resume support."""
    return {obj[id_field] for obj in load_jsonl(path) if id_field in obj}


def write_jsonl(path: str, records: list[dict]) -> None:
    """Write a list of dicts to a JSONL file (overwrite mode). Creates dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
