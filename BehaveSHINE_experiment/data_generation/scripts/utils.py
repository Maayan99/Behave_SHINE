#!/usr/bin/env python3
"""Shared utilities for BehaveSHINE data generation scripts."""

import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Optional

import anthropic
import yaml


# Lazy-initialized Anthropic client singleton
_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    """Get or create the Anthropic API client singleton."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def setup_logging(name: str) -> logging.Logger:
    """Configure and return a named logger with timestamp format."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger(name)


def load_config(config_path: str) -> dict:
    """Load a YAML config file and return as dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return yaml.safe_load(path.read_text())


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


def count_tokens_approx(text: str) -> int:
    """Rough token count estimate: word_count * 1.3, rounded up."""
    return math.ceil(len(text.split()) * 1.3)


def parse_json_response(text: str) -> Any:
    """Parse JSON from Claude's response, stripping markdown fences if present."""
    text = text.strip()

    # Try extracting from markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try parsing the full text directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the outermost JSON structure
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise json.JSONDecodeError("No valid JSON found in response", text, 0)


class RateLimiter:
    """Simple sleep-based rate limiter."""

    def __init__(self, requests_per_minute: int):
        self._min_interval = 60.0 / requests_per_minute
        self._last_request_time = 0.0

    def wait(self) -> None:
        """Block until it's safe to make the next request."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()


def call_claude(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    model: str = "claude-sonnet-4-6-20250929",
    max_retries: int = 3,
    retry_delay_seconds: float = 5.0,
    rate_limiter: Optional[RateLimiter] = None,
) -> str:
    """Call Claude API with retries, exponential backoff, and rate limiting.

    Returns the text content of Claude's response.
    Raises the last exception after max_retries exhausted.
    """
    logger = logging.getLogger("utils.call_claude")
    client = _get_client()

    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system is not None:
        kwargs["system"] = system

    last_exception = None
    for attempt in range(max_retries):
        if rate_limiter is not None:
            rate_limiter.wait()

        try:
            response = client.messages.create(**kwargs)
            return response.content[0].text
        except anthropic.RateLimitError as e:
            last_exception = e
            wait_time = retry_delay_seconds * (2**attempt)
            logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                last_exception = e
                wait_time = retry_delay_seconds * (2**attempt)
                logger.warning(f"Server error {e.status_code} (attempt {attempt+1}/{max_retries}), waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            else:
                raise
        except anthropic.APIConnectionError as e:
            last_exception = e
            wait_time = retry_delay_seconds * (2**attempt)
            logger.warning(f"Connection error (attempt {attempt+1}/{max_retries}), waiting {wait_time:.1f}s")
            time.sleep(wait_time)

    raise last_exception
