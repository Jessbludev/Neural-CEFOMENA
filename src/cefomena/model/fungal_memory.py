"""Hongos — memoria no paramétrica con reglas locales (habituación y poda).

La memoria es un *estado* (slots), no pesos: se escribe sin gradiente con reglas locales
inspiradas en Physarum/Schizophyllum:
- refuerzo: el slot más parecido absorbe la nueva traza y aumenta su conductancia;
- habituación: todas las conductancias decaen con el tiempo;
- poda: si no hay slot parecido, se reemplaza el de menor conductancia;
- difusión de autómata celular: cada slot se suaviza con sus vecinos del anillo.
El modelo lee la memoria por atención cruzada. Sin estado externo se usan slots innatos
(aprendidos), lo que entrena la ruta de lectura.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class FungalMemoryState:
    slots: torch.Tensor  # (S, D)
    conductance: torch.Tensor  # (S,)

    def clone(self) -> FungalMemoryState:
        return FungalMemoryState(self.slots.clone(), self.conductance.clone())


class FungalMemory(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_slots: int,
        decay: float = 0.95,
        write_rate: float = 0.5,
        novelty_threshold: float = 0.5,
        diffusion: float = 0.05,
    ) -> None:
        super().__init__()
        self.innate = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.decay = decay
        self.write_rate = write_rate
        self.novelty_threshold = novelty_threshold
        self.diffusion = diffusion

    def initial_state(self) -> FungalMemoryState:
        slots = self.innate.detach().clone()
        return FungalMemoryState(slots, torch.ones(slots.shape[0], device=slots.device))

    def forward(self, x: torch.Tensor, state: FungalMemoryState | None = None) -> torch.Tensor:
        mem = self.innate if state is None else state.slots.to(x.dtype)
        q = self.q(x)
        k = self.k(mem)
        v = self.v(mem)
        att = torch.softmax(q @ k.t() / k.shape[-1] ** 0.5, dim=-1)
        return self.out(att @ v)

    @torch.no_grad()
    def write(self, state: FungalMemoryState, traces: torch.Tensor) -> FungalMemoryState:
        """Escribe trazas (N, D) en la memoria usando solo reglas locales."""
        slots, cond = state.slots.clone(), state.conductance.clone()
        for trace in traces.to(slots.dtype):
            cond = cond * self.decay
            sim = F.cosine_similarity(slots, trace.unsqueeze(0), dim=-1)
            best = int(sim.argmax())
            if float(sim[best]) >= self.novelty_threshold:
                slots[best] = (1 - self.write_rate) * slots[best] + self.write_rate * trace
                cond[best] = cond[best] + 1.0
            else:
                weakest = int(cond.argmin())
                slots[weakest] = trace
                cond[weakest] = 1.0
            neighbors = slots.roll(1, 0) + slots.roll(-1, 0) - 2 * slots
            slots = slots + self.diffusion * neighbors
        return FungalMemoryState(slots, cond)
