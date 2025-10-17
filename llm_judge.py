import hydra
from omegaconf import DictConfig
import asyncio
import os
import json
from openai import AsyncOpenAI
from openai import APIError, RateLimitError, APITimeoutError
from typing import Any, Dict, List
import tqdm

PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "groundtruth = {reference}\n"
    "predict_answer = {pred}"
)

SYS_MSG = (
    "Given one question, there is a groundtruth and a predict_answer."
    "Please decide whether they are the same or not in semantic. "
    "Please only output 'True' or 'False'."
)

async def safe_chat_complete(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    sem: asyncio.Semaphore,
    max_retries: int = 6,
) -> str:
    """
    Call chat.completions.create with concurrency control and retries.
    """
    backoff = 0.5
    attempt = 0
    async with sem:
        while True:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                return resp.choices[0].message.content or ""
            except (RateLimitError, APITimeoutError, APIError) as e:
                attempt += 1
                # For non-retryable 4xx (except 429), bubble up quickly
                status = getattr(e, "status_code", None)
                if isinstance(e, APIError) and status and 400 <= status < 500 and status != 429:
                    raise
                if attempt > max_retries:
                    raise
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 8.0)  # cap the backoff
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
    sample: Dict[str, Any],
    sem: asyncio.Semaphore,
    max_retries: int,
) -> Dict[str, Any]:

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
    scores: List[bool] = await asyncio.gather(*tasks)
    for i, s in enumerate(scores):
        sample[i]["score"] = s
    return sample

@hydra.main(version_base=None, config_path="configs", config_name="base")
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

        async def judge(res, json_path):
            all_scores: List[bool] = []
            # Process each sample (article) sequentially so we can append per-line safely,
            # but do each sample's QAs concurrently.
            pbar = tqdm(range(0, len(res), cfg.test.max_concurrency), desc="Evaluating with metanetwork")
            for i in pbar:
                batch = res[i:i + cfg.test.max_concurrency]
                try:
                    updated = await process_sample(
                        client=client,
                        model=cfg.test.model,
                        sample=batch,
                        sem=sem,
                        max_retries=cfg.test.max_retries,
                    )
                except Exception as e:
                    # If a sample fails entirely, keep a trace and continue
                    updated = batch
                    updated["_error"] = f"{type(e).__name__}: {e}"

                # Collect scores for average in this run
                for t in updated:
                    if "score" in t and isinstance(t["score"], bool):
                        all_scores.append(t["score"])

                # Write out
                with open(json_path.replace(".json", "_results.json"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(updated, ensure_ascii=True) + "\n")

            # Summary
            if all_scores:
                avg = sum(all_scores) / len(all_scores)
                print(f"Average = {avg:.6f}")
                with open(json_path.replace(".json", "_results.json"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"Average": avg}, ensure_ascii=True) + "\n")
        
        judge(res, json_path)
        judge(res_no_metanet, json_path_no_metanet)
        
        
if __name__ == "__main__":
    main()