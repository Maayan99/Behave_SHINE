import json
from dataclasses import dataclass
from typing import Any, List, Dict

import torch
from torch.utils.data import Dataset


def _load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class BehaveSHINEDataset(Dataset):
    """
    Loads the processed BehaveSHINE training/eval data.
    Each record has: context (system prompt), question, answer (235B teacher response).
    """

    def __init__(self, data_path: str):
        self.data = _load_jsonl(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "system_prompt": item["context"],
            "question": item["question"],
            "answer": item["answer"],
        }


@dataclass
class BehaveSHINECollator:
    """
    Tokenizes and prepares batches for the BehaveSHINE complementation objective.

    Produces per batch:
    - evidence_ids / evidence_attention_mask : system prompt for hypernetwork input
    - input_ids / attention_mask / labels    : full chat template for forward pass
    - prompt_only_ids                        : chat template without assistant response (for generation)
    - raw_answer                             : raw teacher response strings (for printing)
    """

    tokenizer: Any
    context_max_length: int
    conversation_max_length: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ── 1. Evidence encoding (hypernetwork input) ────────────────────────
        evidence_texts = [item["system_prompt"] for item in batch]
        evidence_enc = self.tokenizer(
            evidence_texts,
            max_length=self.context_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # ── 2. Per-item: build full tokens + labels + prompt-only tokens ─────
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        all_prompt_only_ids = []
        all_prompt_only_masks = []

        pad_id = self.tokenizer.pad_token_id
        L = self.conversation_max_length

        for item in batch:
            prompt_msgs = [
                {"role": "system", "content": item["system_prompt"]},
                {"role": "user", "content": item["question"]},
            ]

            # Prompt-only tokens (system + user + assistant generation prompt)
            prompt_only_tokens = self.tokenizer.apply_chat_template(
                prompt_msgs,
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=False,
            )

            # Full tokens (system + user + assistant response)
            full_msgs = prompt_msgs + [{"role": "assistant", "content": item["answer"]}]
            full_tokens = self.tokenizer.apply_chat_template(
                full_msgs,
                tokenize=True,
                enable_thinking=False,
            )

            # Labels: backward-counting method — avoids chat template boundary mismatch.
            # Tokenize only the raw answer text to get the response length, then mask
            # everything before it. The +1 accounts for the <|im_end|> appended by the
            # chat template after the assistant turn.
            answer_tokens = self.tokenizer.encode(
                item["answer"], add_special_tokens=False
            )
            target_len = len(answer_tokens)
            mask_len = len(full_tokens) - target_len - 1
            labels = [-100] * mask_len + full_tokens[-(target_len + 1):]

            # Truncate to conversation_max_length
            full_tokens = full_tokens[:L]
            labels = labels[:L]
            prompt_only_tokens = prompt_only_tokens[:L]

            # Pad to conversation_max_length
            full_len = len(full_tokens)
            prompt_only_len = len(prompt_only_tokens)

            full_pad = L - full_len
            prompt_pad = L - prompt_only_len

            input_ids = full_tokens + [pad_id] * full_pad
            attn_mask = [1] * full_len + [0] * full_pad

            # Labels for padding positions are also -100
            labels = labels + [-100] * (L - len(labels))

            prompt_ids = prompt_only_tokens + [pad_id] * prompt_pad
            prompt_mask = [1] * prompt_only_len + [0] * prompt_pad

            all_input_ids.append(input_ids)
            all_attention_masks.append(attn_mask)
            all_labels.append(labels)
            all_prompt_only_ids.append(prompt_ids)
            all_prompt_only_masks.append(prompt_mask)

        return {
            "evidence_ids": evidence_enc["input_ids"],
            "evidence_attention_mask": evidence_enc["attention_mask"],
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_attention_masks, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long),
            "prompt_only_ids": torch.tensor(all_prompt_only_ids, dtype=torch.long),
            "raw_answer": [item["answer"] for item in batch],
        }
