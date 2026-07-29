from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass
class TextBatch:
    token_states: torch.Tensor
    attention_mask: torch.Tensor
    pooled: torch.Tensor


class BaseTextEncoder(nn.Module):
    hidden_dim: int

    def encode(self, texts: Sequence[str], device: torch.device | str | None = None) -> TextBatch:
        raise NotImplementedError


class HashingTextEncoder(BaseTextEncoder):
    """Offline trainable encoder with deterministic hashing tokenization."""

    def __init__(
        self,
        hidden_dim: int = 128,
        vocab_size: int = 32768,
        max_length: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position = nn.Embedding(max_length, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, hidden_dim // 32),
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_\-]+", text.lower())

    def _token_id(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return 1 + int.from_bytes(digest, "little") % (self.vocab_size - 1)

    def encode(self, texts: Sequence[str], device: torch.device | str | None = None) -> TextBatch:
        if not texts:
            raise ValueError("At least one text is required.")
        target_device = torch.device(device) if device is not None else self.embedding.weight.device
        rows: list[list[int]] = []
        for text in texts:
            ids = [self._token_id(token) for token in self._tokens(text)[: self.max_length]]
            rows.append(ids or [1])
        max_len = min(max(len(row) for row in rows), self.max_length)
        input_ids = torch.zeros((len(rows), max_len), dtype=torch.long, device=target_device)
        mask = torch.zeros((len(rows), max_len), dtype=torch.bool, device=target_device)
        for idx, row in enumerate(rows):
            row = row[:max_len]
            input_ids[idx, : len(row)] = torch.tensor(row, device=target_device)
            mask[idx, : len(row)] = True
        positions = torch.arange(max_len, device=target_device).unsqueeze(0)
        hidden = self.embedding(input_ids) + self.position(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        hidden = self.norm(hidden)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
        return TextBatch(token_states=hidden, attention_mask=mask, pooled=pooled)


class HuggingFaceTextEncoder(BaseTextEncoder):
    """Thin wrapper around an AutoModel hidden-state encoder."""

    def __init__(
        self,
        model_name: str,
        max_length: int = 384,
        torch_dtype: str = "bfloat16",
        freeze: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install ReSemReason with the 'hf' extra.") from exc

        dtype = getattr(torch, torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        self.max_length = max_length
        self.hidden_dim = int(self.model.config.hidden_size)
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def encode(self, texts: Sequence[str], device: torch.device | str | None = None) -> TextBatch:
        target_device = torch.device(device) if device is not None else next(self.model.parameters()).device
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(target_device)
        outputs = self.model(**batch)
        hidden = outputs.last_hidden_state
        mask = batch["attention_mask"].bool()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
        return TextBatch(token_states=hidden, attention_mask=mask, pooled=pooled)


def build_text_encoder(config: dict) -> BaseTextEncoder:
    backend = config.get("backend", "hashing")
    if backend == "hashing":
        return HashingTextEncoder(
            hidden_dim=int(config.get("hidden_dim", 128)),
            vocab_size=int(config.get("vocab_size", 32768)),
            max_length=int(config.get("max_length", 96)),
            dropout=float(config.get("dropout", 0.1)),
        )
    if backend == "huggingface":
        return HuggingFaceTextEncoder(
            model_name=config["model_name"],
            max_length=int(config.get("max_length", 384)),
            torch_dtype=config.get("torch_dtype", "bfloat16"),
            freeze=bool(config.get("freeze", False)),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
    raise ValueError(f"Unsupported encoder backend: {backend}")
