from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import torch
from torch.utils.data import Sampler
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter
from metanetwork_family import Metanetwork


# ---------------------------
# Mock dataset for demo
# ---------------------------
def create_mock_dataset() -> Tuple[List[str], List[str]]:
    texts = [
        "1231",
        "2342",
        "3453",
        "4564",
        "5675",
        "6786",
        "7897",
        "8908",
        "9019",
        "0120",
    ] * 50
    df = pd.DataFrame({'text': texts})
    train_texts, val_texts = train_test_split(df['text'], test_size=0.1, random_state=42)
    return train_texts.tolist(), val_texts.tolist()


# ---------------------------
# Dataset
# ---------------------------
class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {"text": str(self.texts[idx])}

class LoogleDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {"question": str(self.data[idx]['question']), "evidence": str(self.data[idx]['evidence']), "answer": str(self.data[idx]['answer'])}

class SquadDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {"evidence": str(self.data[idx]['context']), "question": str(self.data[idx]['question']), "answer": str(self.data[idx]['answers']['text'][0])}

# ---------------------------
# Collator with dynamic padding and label masking
# ---------------------------
@dataclass
class CausalLMDataCollator:
    tokenizer: Any
    max_length: int = 512

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex["text"] for ex in batch]
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()

        # Ensure a pad token exists
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        pad_id = self.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

@dataclass
class LoogleCollator:
    tokenizer: Any
    max_length: int = 1024
    use_reference: bool = True

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        questions = [ex["question"] for ex in batch]
        evidences = [ex["evidence"] for ex in batch]
        answers = [ex["answer"] for ex in batch]
           
        evidence_enc = self.tokenizer(
            evidences,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        evidence_ids = evidence_enc["input_ids"]
        evidence_attention_mask = evidence_enc["attention_mask"]
        
        if self.use_reference:
            messages = [[
                {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
                {"role": "user", "content": "Please review the following reference materials."},
                {"role": "user", "content": f"{evidence}"},
                {"role": "user", "content": f"Based on the above, answer this question: {question}"}
            ] for evidence, question in zip(evidences, questions)]
        else:
            messages = [[
                {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
                {"role": "user", "content": f"Please answer the following question: {question}"}
            ] for question in questions]

        input_enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
                return_dict=True,
            )
        input_ids = input_enc["input_ids"]
        input_attention_mask = input_enc["attention_mask"]
        return {
            "evidence": evidences,
            "evidence_ids": evidence_ids,
            "evidence_attention_mask": evidence_attention_mask,
            "input_ids": input_ids,
            "input_attention_mask": input_attention_mask,
            "answers": answers,
        }

@dataclass
class SquadCollator:
    tokenizer: Any
    max_length: int = 1024
    use_reference: bool = True
    metatrain: bool = False

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        questions = [ex["question"] for ex in batch]
        evidences = [ex["evidence"] for ex in batch]
        answers = [ex["answer"] for ex in batch]
           
        evidence_enc = self.tokenizer(
            evidences,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        evidence_ids = evidence_enc["input_ids"]
        evidence_attention_mask = evidence_enc["attention_mask"]
        
        answer_enc = self.tokenizer(
            answers,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        answer_ids = answer_enc["input_ids"]
        answer_attention_mask = answer_enc["attention_mask"]

        if self.metatrain:
            messages = [[
                {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
                {"role": "user", "content": f"Please answer the following question: {question}"},
                {"role": "assistant", "content": f"{answer}"}
            ] for question, answer in zip(questions, answers)]
        elif self.use_reference:
            messages = [[
                {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
                {"role": "user", "content": "Please review the following reference materials."},
                {"role": "user", "content": f"{evidence}"},
                {"role": "user", "content": f"Based on the above, answer this question: {question}"}
            ] for evidence, question in zip(evidences, questions)]
        else:
            messages = [[
                {"role": "system", "content": "You are a concise assistant. Only output the final answer with no extra words."},
                {"role": "user", "content": f"Please answer the following question: {question}"}
            ] for question in questions]

        input_enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,   # adds the assistant turn start
                tokenize=True,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
                return_dict=True,
            )
        input_ids = input_enc["input_ids"]
        input_attention_mask = input_enc["attention_mask"]
        labels = input_ids.clone()
        return {
            "evidence": evidences,
            "evidence_ids": evidence_ids,
            "evidence_attention_mask": evidence_attention_mask,
            "input_ids": input_ids,
            "labels": labels,
            "input_attention_mask": input_attention_mask,
            "answers": answers,
            "answer_ids": answer_ids,
            "answer_attention_mask": answer_attention_mask,
        }