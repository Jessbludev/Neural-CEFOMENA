"""Carga de datos conversacionales (JSONL) y empaquetado en tokens."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cefomena.chat.template import Message, render, render_ids
from cefomena.tokenizer.bpe import BPETokenizer


def read_jsonl(path: str | Path) -> Iterator[dict[str, object]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_conversations(paths: Iterable[str | Path]) -> Iterator[list[Message]]:
    """Admite filas ``{"messages": [...]}`` o ``{"text": "..."}`` (texto plano de usuario)."""
    for path in paths:
        for row in read_jsonl(path):
            if "messages" in row:
                msgs = row["messages"]
                if not isinstance(msgs, list):
                    raise ValueError(f"{path}: 'messages' debe ser una lista")
                yield [Message.from_dict(m) for m in msgs]
            elif "text" in row:
                yield [Message("assistant", str(row["text"]))]


def iter_texts(paths: Iterable[str | Path]) -> Iterator[str]:
    for conv in iter_conversations(paths):
        yield render(conv)


@dataclass
class TokenDataset:
    tokens: np.ndarray
    mask: np.ndarray

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    @classmethod
    def from_conversations(
        cls, tokenizer: BPETokenizer, conversations: Iterable[list[Message]]
    ) -> TokenDataset:
        ids: list[int] = []
        mask: list[int] = []
        for conv in conversations:
            i, m = render_ids(tokenizer, conv)
            ids.extend(i)
            mask.extend(m)
        return cls(np.asarray(ids, dtype=np.int32), np.asarray(mask, dtype=np.uint8))

    def split(self, val_fraction: float = 0.05) -> tuple[TokenDataset, TokenDataset]:
        cut = int(len(self) * (1 - val_fraction))
        return (
            TokenDataset(self.tokens[:cut], self.mask[:cut]),
            TokenDataset(self.tokens[cut:], self.mask[cut:]),
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, tokens=self.tokens, mask=self.mask)

    @classmethod
    def load(cls, path: str | Path) -> TokenDataset:
        data = np.load(path, allow_pickle=False)
        return cls(data["tokens"], data["mask"])

    def get_batch(
        self,
        batch_size: int,
        context: int,
        generator: torch.Generator | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self) <= context + 1:
            raise ValueError("dataset más pequeño que la ventana de contexto")
        starts = torch.randint(len(self) - context - 1, (batch_size,), generator=generator)
        x = np.stack([self.tokens[s : s + context] for s in starts.tolist()])
        y = np.stack([self.tokens[s + 1 : s + 1 + context] for s in starts.tolist()])
        m = np.stack([self.mask[s + 1 : s + 1 + context] for s in starts.tolist()])
        return (
            torch.from_numpy(x.astype(np.int64)).to(device),
            torch.from_numpy(y.astype(np.int64)).to(device),
            torch.from_numpy(m.astype(np.float32)).to(device),
        )
