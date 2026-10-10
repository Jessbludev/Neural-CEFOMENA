"""Configuración del modelo CefomenaLM."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class CefomenaConfig:
    """Hiperparámetros del modelo.

    ``layer_pattern`` define el orden de bloques: ``N`` = bloque NerveNet (medusa, mezcla local),
    ``A`` = bloque de atención con memoria fúngica y MoE de osciladores (coral).
    """

    vocab_size: int = 8192
    d_model: int = 384
    n_heads: int = 6
    context_length: int = 1024
    layer_pattern: str = "NNANNANNA"
    conv_kernel: int = 7
    conv_dilation: int = 4
    mlp_ratio: float = 4.0
    sparse_fraction: float = 0.25
    n_experts: int = 4
    expert_ratio: float = 2.0
    oscillator_coupling: float = 1.0
    oscillator_sync_steps: int = 2
    memory_slots: int = 32
    dual_cache_rank: int = 8
    dropout: float = 0.0
    rope_base: float = 10000.0
    tie_embeddings: bool = True
    model_type: str = "cefomena"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model debe ser divisible por n_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("la dimensión por cabeza debe ser par (RoPE)")
        if not self.layer_pattern or set(self.layer_pattern) - {"N", "A"}:
            raise ValueError("layer_pattern solo admite 'N' y 'A'")
        if not 0.0 < self.sparse_fraction <= 1.0:
            raise ValueError("sparse_fraction debe estar en (0, 1]")

    @property
    def n_layers(self) -> int:
        return len(self.layer_pattern)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CefomenaConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> CefomenaConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))


PRESETS: dict[str, CefomenaConfig] = {
    "tiny": CefomenaConfig(),
    "micro": CefomenaConfig(
        vocab_size=4096, d_model=256, n_heads=4, context_length=512, layer_pattern="NNANNA"
    ),
    "test": CefomenaConfig(
        vocab_size=300,
        d_model=32,
        n_heads=2,
        context_length=64,
        layer_pattern="NAN",
        memory_slots=4,
        n_experts=2,
        dual_cache_rank=2,
    ),
}
