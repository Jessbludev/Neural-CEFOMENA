"""Coral — consenso mediante osciladores acoplados.

Cada experto es un oscilador con frecuencia natural ω_i y fase φ_i. La entrada perturba las
fases; unas pocas iteraciones de acoplamiento tipo Kuramoto acercan cada fase al campo medio ψ.
El peso de cada experto es su coherencia con ψ: no existe un router central que decida.
"""

from __future__ import annotations

import torch
from torch import nn

from cefomena.model.sparse_gate import SparseGateMLP


class OscillatorMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        hidden: int,
        sparse_fraction: float,
        coupling: float = 1.0,
        sync_steps: int = 2,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            SparseGateMLP(d_model, hidden, sparse_fraction) for _ in range(n_experts)
        )
        self.omega = nn.Parameter(torch.linspace(0.01, 0.1, n_experts))
        self.phi = nn.Parameter(torch.linspace(0, 2 * torch.pi, n_experts + 1)[:-1].clone())
        self.drive = nn.Linear(d_model, n_experts, bias=False)
        self.sharpness = nn.Parameter(torch.tensor(2.0))
        self.coupling = coupling
        self.sync_steps = sync_steps
        self.register_buffer("last_load", torch.zeros(n_experts), persistent=False)

    def phases(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).view(1, -1, 1)
        theta = self.omega * t + self.phi + torch.pi * torch.tanh(self.drive(x))
        for _ in range(self.sync_steps):
            psi = torch.atan2(
                theta.sin().mean(-1, keepdim=True), theta.cos().mean(-1, keepdim=True)
            )
            theta = theta + self.coupling * torch.sin(psi - theta) / self.sync_steps
        return theta

    def routing(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.phases(x)
        psi = torch.atan2(theta.sin().mean(-1, keepdim=True), theta.cos().mean(-1, keepdim=True))
        coherence = torch.cos(theta - psi)
        return torch.softmax(self.sharpness * coherence + self.drive(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.routing(x)
        self.last_load = w.detach().float().mean(dim=(0, 1))
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out = out + w[..., i : i + 1] * expert(x)
        return out
