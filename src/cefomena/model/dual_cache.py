"""Mariposa — dual-cache sensorial–central.

Sobre los embeddings "centrales" (lentos) se aplica una modulación periférica de bajo rango
(ganancia y sesgo por token). Al inicio es la identidad. Para adaptar el modelo a un usuario o
tarea se entrenan solo estos parámetros, dejando intactos los pesos profundos.
"""

from __future__ import annotations

import torch
from torch import nn


class DualCache(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, rank: int) -> None:
        super().__init__()
        self.gain_a = nn.Embedding(vocab_size, rank)
        self.gain_b = nn.Parameter(torch.zeros(rank, d_model))
        self.bias_a = nn.Embedding(vocab_size, rank)
        self.bias_b = nn.Parameter(torch.zeros(rank, d_model))
        nn.init.normal_(self.gain_a.weight, std=0.02)
        nn.init.normal_(self.bias_a.weight, std=0.02)

    def forward(self, idx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gain = self.gain_a(idx) @ self.gain_b
        bias = self.bias_a(idx) @ self.bias_b
        return x * (1 + gain) + bias
