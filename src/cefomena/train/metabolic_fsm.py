"""Murciélago — máquina de estados metabólicos acoplada al aprendizaje.

ACTIVE  → aprendizaje normal (LR completa).
REST    → sin actualizaciones; solo evaluación (vaciado de ruido).
SLEEP   → consolidación: replay de los lotes con mayor pérdida, LR reducida, EMA.
WAKE    → recalentamiento progresivo de la LR antes de volver a ACTIVE.

En inferencia, el estado determina el presupuesto de bloques (early-exit).
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from enum import StrEnum

import torch


class MetabolicState(StrEnum):
    ACTIVE = "active"
    REST = "rest"
    SLEEP = "sleep"
    WAKE = "wake"


_ORDER = (MetabolicState.ACTIVE, MetabolicState.REST, MetabolicState.SLEEP, MetabolicState.WAKE)


@dataclass
class MetabolicFSM:
    active_steps: int = 800
    rest_steps: int = 20
    sleep_steps: int = 100
    wake_steps: int = 50
    sleep_lr_factor: float = 0.3
    state: MetabolicState = MetabolicState.ACTIVE
    _t: int = field(default=0, repr=False)

    def duration(self, state: MetabolicState) -> int:
        return {
            MetabolicState.ACTIVE: self.active_steps,
            MetabolicState.REST: self.rest_steps,
            MetabolicState.SLEEP: self.sleep_steps,
            MetabolicState.WAKE: self.wake_steps,
        }[state]

    def step(self) -> MetabolicState:
        """Avanza un paso y devuelve el estado vigente para el siguiente paso."""
        self._t += 1
        while self._t >= self.duration(self.state):
            self._t = 0
            self.state = _ORDER[(_ORDER.index(self.state) + 1) % len(_ORDER)]
            if self.duration(self.state) > 0:
                break
        return self.state

    def should_update(self) -> bool:
        return self.state is not MetabolicState.REST

    def use_replay(self) -> bool:
        return self.state is MetabolicState.SLEEP

    def lr_multiplier(self) -> float:
        if self.state is MetabolicState.REST:
            return 0.0
        if self.state is MetabolicState.SLEEP:
            return self.sleep_lr_factor
        if self.state is MetabolicState.WAKE:
            return (self._t + 1) / max(1, self.wake_steps)
        return 1.0

    @staticmethod
    def inference_budget(state: MetabolicState, n_layers: int) -> int:
        frac = {
            MetabolicState.ACTIVE: 1.0,
            MetabolicState.WAKE: 1.0,
            MetabolicState.REST: 0.67,
            MetabolicState.SLEEP: 0.34,
        }[state]
        return max(1, round(n_layers * frac))


class ReplayBuffer:
    """Conserva los lotes de mayor pérdida para la fase SLEEP."""

    def __init__(self, capacity: int = 64) -> None:
        self.capacity = capacity
        self._heap: list[tuple[float, int, tuple[torch.Tensor, ...]]] = []
        self._counter = itertools.count()

    def __len__(self) -> int:
        return len(self._heap)

    def add(self, loss: float, batch: tuple[torch.Tensor, ...]) -> None:
        item = (loss, next(self._counter), tuple(t.detach().cpu() for t in batch))
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, item)
        elif loss > self._heap[0][0]:
            heapq.heapreplace(self._heap, item)

    def sample(self, generator: torch.Generator | None = None) -> tuple[torch.Tensor, ...]:
        if not self._heap:
            raise IndexError("replay buffer vacío")
        i = int(torch.randint(len(self._heap), (1,), generator=generator))
        return self._heap[i][2]
