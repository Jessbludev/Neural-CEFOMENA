"""Delfín — codificación temporal escasa modulada por recompensa.

Un MLP con compuerta multiplicativa ("unidades tipo parvalbúmina") y k-WTA: por token solo
una fracción pequeña de unidades ocultas queda activa. Los parámetros de compuerta se exponen
por separado para que un canal de recompensa los ajuste sin tocar el resto de pesos.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def k_winners(h: torch.Tensor, k: int) -> torch.Tensor:
    if k >= h.shape[-1]:
        return h
    threshold = h.topk(k, dim=-1).values[..., -1:]
    return h * (h >= threshold).to(h.dtype)


class SparseGateMLP(nn.Module):
    def __init__(self, d_model: int, hidden: int, sparse_fraction: float = 0.25) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.k = max(1, math.ceil(hidden * sparse_fraction))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.up(x)) * torch.sigmoid(self.gate(x))
        return self.down(k_winners(h, self.k))

    def reward_parameters(self) -> list[nn.Parameter]:
        """Parámetros que modula el canal de recompensa (compuertas PV-like)."""
        return list(self.gate.parameters())
