import hydra
from omegaconf import DictConfig
import asyncio
import os
import json
from openai import AsyncOpenAI
from openai import APIError, RateLimitError, APITimeoutError
from typing import Any, Dict, List
from tqdm import tqdm  # <-- fixed import

PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "groundtruth = {reference}\n"
    "predict_answer = {pred}"
)

SYS_MSG = (
    "Given one question, there is a groundtruth and a predict_answer. "
    "Decide whether they are semantically the same. "
    "Only output True or False."
)

async def safe_chat_complete(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    sem: asyncio.Semaphore,
    max_retries: int = 6,
) -> str:
    backoff = 0.5
    attempt = 0
    async with sem:
        while True:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=5,
                )
                return resp.choices[0].message.content or ""
            except (RateLimitError, APITimeoutError, APIError) as e:
                attempt += 1
                status = getattr(e, "status_code", None) or getattr(e, "status", None)
                # For non-retryable 4xx except 429, re-raise.
                if isinstance(e, APIError) and status and 400 <= status < 500 and status != 429:
                    raise
                if attempt > max_retries:
                    raise
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
            except Exception:
                attempt += 1
                if attempt > max_retries:
                    raise
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 8.0)

def parse_bool(text: str) -> bool:
    return "true" in (text or "").strip().lower()

async def score_one_qa(
    client: AsyncOpenAI,
    model: str,
    question: str,
    answer: str,
    ground_truth: str,
    sem: asyncio.Semaphore,
    max_retries: int,
) -> bool:
    messages = [
        {"role": "system", "content": SYS_MSG},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(
                question=question,
                reference=ground_truth,
                pred=answer,
            ),
        },
    ]
    reply = await safe_chat_complete(
        client=client,
        model=model,
        messages=messages,
        sem=sem,
        max_retries=max_retries,
    )
    return parse_bool(reply)

async def process_sample(
    client: AsyncOpenAI,
    model: str,
    sample: List[Dict[str, Any]],  # <-- fixed type
    sem: asyncio.Semaphore,
    max_retries: int,
) -> List[Dict[str, Any]]:
    tasks = []
    for i in range(len(sample)):
        tasks.append(
            score_one_qa(
                client=client,
                model=model,
                question=sample[i]["question"],
                answer=sample[i]["answer"],
                ground_truth=sample[i]["ground_truth"],
                sem=sem,
                max_retries=max_retries,
            )
        )
    scores: List[bool] = await asyncio.gather(*tasks, return_exceptions=False)
    for i, s in enumerate(scores):
        sample[i]["score"] = bool(s)
    return sample

async def judge(
    client: AsyncOpenAI,
    model: str,
    res: List[Dict[str, Any]],
    out_json_path: str,
    sem: asyncio.Semaphore,
    max_concurrency: int,
    max_retries: int,
    desc: str,
) -> None:
    all_scores: List[bool] = []
    # Process per batch (each batch evaluated concurrently)
    for i in tqdm(range(0, len(res), max_concurrency), desc=desc):
        batch = res[i : i + max_concurrency]
        try:
            updated = await process_sample(
                client=client,
                model=model,
                sample=batch,
                sem=sem,
                max_retries=max_retries,
            )
        except Exception as e:
            # Attach an error marker to each item in the batch
            updated = []
            for item in batch:
                item = dict(item)
                item["_error"] = f"{type(e).__name__}: {e}"
                updated.append(item)

        # Collect scores
        for t in updated:
            if isinstance(t.get("score"), bool):
                all_scores.append(t["score"])

        # Append results
        with open(out_json_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(updated, ensure_ascii=False) + "\n")

    # Summary
    if all_scores:
        avg = sum(all_scores) / len(all_scores)
        print(f"Average = {avg:.6f}; Total Samples = {len(all_scores)}")
        with open(out_json_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"Average": avg}, ensure_ascii=False) + "\n")

@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig):
    asyncio.run(amain(cfg))

async def amain(cfg: DictConfig):
    if cfg.test.source == "loogle":
        names = ["shortdep_qa", "shortdep_cloze", "longdep_qa", "summarization"]
        load_dir = os.path.join(cfg.test.save_path, cfg.test.source)
    elif cfg.test.source == "squad":
        names = ["squad"]
        load_dir = os.path.join(cfg.test.save_path, cfg.test.source)
    else:
        raise ValueError(f"Unknown data source: {cfg.test.source}")

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(max(1, cfg.test.max_concurrency))

    for name in names:
        json_path = os.path.join(load_dir, f"{name}.json")
        json_path_no_metanet = json_path.replace(".json", "_no_metanet.json")

        with open(json_path, "r", encoding="utf-8") as f:
            res = json.load(f)
        with open(json_path_no_metanet, "r", encoding="utf-8") as f:
            res_no_metanet = json.load(f)

        out_with = json_path.replace(".json", "_results.json")
        out_without = json_path_no_metanet.replace(".json", "_results.json")

        # Truncate old results for a clean run (optional)
        open(out_with, "w", encoding="utf-8").close()
        open(out_without, "w", encoding="utf-8").close()

        # Run both sets (sequentially here; you can gather if you prefer)
        await judge(
            client=client,
            model=cfg.test.model,
            res=res,
            out_json_path=out_with,
            sem=sem,
            max_concurrency=cfg.test.max_concurrency,
            max_retries=cfg.test.max_retries,
            desc="Evaluating with metanetwork",
        )
        await judge(
            client=client,
            model=cfg.test.model,
            res=res_no_metanet,
            out_json_path=out_without,
            sem=sem,
            max_concurrency=cfg.test.max_concurrency,
            max_retries=cfg.test.max_retries,
            desc="Evaluating without metanetwork",
        )

if __name__ == "__main__":
    main()
