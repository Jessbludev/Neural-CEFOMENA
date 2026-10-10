"""Medusa — cómputo topológico distribuido.

Cada posición solo interactúa con sus vecinas inmediatas (convoluciones causales depthwise),
sin una unidad central que integre toda la secuencia. Apilar bloques propaga la señal como
una onda a lo largo de la "malla" de tokens. Coste O(n) por capa.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CausalDepthwiseConv(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel, dilation=dilation, groups=channels, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        y = F.pad(x.transpose(1, 2), (self.pad, 0))
        return self.conv(y).transpose(1, 2)


class NerveNetMixer(nn.Module):
    """Mezclador local con excitabilidad: vecindad corta + vecindad dilatada + compuerta."""

    def __init__(self, d_model: int, kernel: int = 7, dilation: int = 4) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.local = CausalDepthwiseConv(d_model, kernel, 1)
        self.wide = CausalDepthwiseConv(d_model, kernel, dilation)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v, gate = self.in_proj(x).chunk(2, dim=-1)
        v = self.local(v)
        v = v + self.wide(F.silu(v))
        return self.out_proj(F.silu(gate) * v)
