"""Tokenizador BPE byte-level propio, sin dependencias externas."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path

SPECIAL_TOKENS: tuple[str, ...] = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|tool_call|>",
    "<|end|>",
)

_PRETOKENIZE = re.compile(r" ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+")
_FORMAT = "cefomena-bpe"


class BPETokenizer:
    """BPE a nivel de bytes: ids 0-255 = bytes, luego tokens especiales, luego fusiones."""

    def __init__(
        self, merges: list[tuple[int, int]] | None = None, specials: Iterable[str] = SPECIAL_TOKENS
    ) -> None:
        self.specials: dict[str, int] = {s: 256 + i for i, s in enumerate(specials)}
        self._special_ids = set(self.specials.values())
        self.merges: list[tuple[int, int]] = list(merges or [])
        self._offset = 256 + len(self.specials)
        self.ranks: dict[tuple[int, int], int] = {p: i for i, p in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for s, i in self.specials.items():
            self.vocab[i] = s.encode("utf-8")
        for i, (a, b) in enumerate(self.merges):
            self.vocab[self._offset + i] = self.vocab[a] + self.vocab[b]
        self._special_re = re.compile(
            "(" + "|".join(re.escape(s) for s in sorted(self.specials, key=len, reverse=True)) + ")"
        )
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return self._offset + len(self.merges)

    def token_id(self, special: str) -> int:
        return self.specials[special]

    @property
    def pad_id(self) -> int:
        return self.specials["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.specials["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.specials["<|eos|>"]

    @property
    def end_id(self) -> int:
        return self.specials["<|end|>"]

    @classmethod
    def train(
        cls, texts: Iterable[str], vocab_size: int, specials: Iterable[str] = SPECIAL_TOKENS
    ) -> BPETokenizer:
        """Entrena fusiones BPE con conteo incremental de pares."""
        specials = tuple(specials)
        n_merges = vocab_size - 256 - len(specials)
        if n_merges < 0:
            raise ValueError("vocab_size demasiado pequeño para bytes + especiales")
        special_re = re.compile("|".join(re.escape(s) for s in specials))
        word_counts: Counter[bytes] = Counter()
        for text in texts:
            for chunk in special_re.split(text):
                word_counts.update(w.encode("utf-8") for w in _PRETOKENIZE.findall(chunk))

        words: list[list[int]] = [list(w) for w in word_counts]
        freqs: list[int] = list(word_counts.values())
        pair_counts: Counter[tuple[int, int]] = Counter()
        where: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, w in enumerate(words):
            for pair in pairwise(w):
                pair_counts[pair] += freqs[wi]
                where[pair].add(wi)

        offset = 256 + len(specials)
        merges: list[tuple[int, int]] = []
        for m in range(n_merges):
            if not pair_counts:
                break
            best, count = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if count < 2:
                break
            new_id = offset + m
            merges.append(best)
            for wi in list(where.pop(best, ())):
                w, f = words[wi], freqs[wi]
                for pair in pairwise(w):
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                merged = _merge(w, best, new_id)
                words[wi] = merged
                for pair in pairwise(merged):
                    pair_counts[pair] += f
                    where[pair].add(wi)
        return cls(merges, specials)

    def _encode_word(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = list(word.encode("utf-8"))
        while len(ids) >= 2:
            ranked = [(self.ranks.get(p, len(self.ranks)), i) for i, p in enumerate(pairwise(ids))]
            rank, _ = min(ranked)
            if rank == len(self.ranks):
                break
            ids = _merge(ids, self.merges[rank], self._offset + rank)
        if len(self._cache) < 200_000:
            self._cache[word] = ids
        return ids

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        out: list[int] = []
        chunks = self._special_re.split(text) if allow_special else [text]
        for chunk in chunks:
            if not chunk:
                continue
            if allow_special and chunk in self.specials:
                out.append(self.specials[chunk])
                continue
            for w in _PRETOKENIZE.findall(chunk):
                out.extend(self._encode_word(w))
        return out

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        parts = [
            self.vocab[i]
            for i in ids
            if i in self.vocab and not (skip_special and i in self._special_ids)
        ]
        return b"".join(parts).decode("utf-8", errors="replace")

    def to_dict(self) -> dict[str, object]:
        return {
            "format": _FORMAT,
            "version": 1,
            "specials": list(self.specials),
            "merges": [list(m) for m in self.merges],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        data = json.loads(Path(path).read_text())
        if data.get("format") != _FORMAT:
            raise ValueError(f"formato de tokenizador no reconocido: {data.get('format')!r}")
        return cls([(int(a), int(b)) for a, b in data["merges"]], data["specials"])


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
