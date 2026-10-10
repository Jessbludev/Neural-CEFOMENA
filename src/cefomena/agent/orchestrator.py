"""Orquestador del agente con máquina de estados fúngica (tolerante a errores).

IDLE → PLAN → ACT → VERIFY → PLAN … → DONE
                └→ RECOVER (formato inválido / error de herramienta) → PLAN | FAILED

Cada episodio se registra como trayectoria para aprendizaje posterior (adaptación periférica
o ajuste de compuertas por recompensa).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cefomena.agent.safety import SafetyPolicy
from cefomena.agent.tools import ToolRegistry
from cefomena.chat.template import Message

if TYPE_CHECKING:
    from cefomena.chat.generate import ChatSession

Policy = Callable[[list[Message]], Message]

AGENT_SYSTEM_PROMPT = (
    "Eres CEFOMENA, un asistente en español que puede usar herramientas. "
    "Para usar una herramienta responde solo con <|tool_call|> y un JSON "
    '{{"name": ..., "arguments": {{...}}}}. Cuando termines, responde con texto breve.\n'
    "Herramientas disponibles:\n{tools}"
)


class AgentState(StrEnum):
    IDLE = "idle"
    PLAN = "plan"
    ACT = "act"
    VERIFY = "verify"
    RECOVER = "recover"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Step:
    state: str
    detail: dict[str, Any]


@dataclass
class Episode:
    task: str
    steps: list[Step] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    final_answer: str = ""
    success: bool = False
    reward: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "success": self.success,
            "reward": self.reward,
            "final_answer": self.final_answer,
            "steps": [asdict(s) for s in self.steps],
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "tool_call": asdict(m.tool_call) if m.tool_call else None,
                }
                for m in self.messages
            ],
        }


class Orchestrator:
    def __init__(
        self,
        policy: Policy,
        registry: ToolRegistry,
        safety: SafetyPolicy | None = None,
        max_retries: int = 2,
        episode_log: Path | None = None,
    ) -> None:
        self.policy = policy
        self.registry = registry
        self.safety = safety or SafetyPolicy()
        self.max_retries = max_retries
        self.episode_log = episode_log
        self.state = AgentState.IDLE

    def _transition(self, ep: Episode, state: AgentState, **detail: Any) -> None:
        self.state = state
        ep.steps.append(Step(state.value, detail))
        self.safety.audit({"state": state.value, **detail})

    def run(self, task: str) -> Episode:
        ep = Episode(task)
        system = AGENT_SYSTEM_PROMPT.format(tools=self.registry.describe())
        ep.messages = [Message("system", system), Message("user", task)]
        retries = 0
        actions = 0
        self._transition(ep, AgentState.PLAN)
        while self.state not in (AgentState.DONE, AgentState.FAILED):
            if actions >= self.safety.max_steps:
                self._transition(ep, AgentState.FAILED, reason="límite de pasos alcanzado")
                break
            reply = self.policy(ep.messages)
            if reply.tool_call is not None:
                call = reply.tool_call
                ep.messages.append(reply)
                actions += 1
                self._transition(ep, AgentState.ACT, tool=call.name, arguments=call.arguments)
                decision = self.safety.check(call, self.registry.get(call.name))
                if not decision.allowed:
                    observation = f"denegado: {decision.reason}"
                else:
                    try:
                        observation = self.registry.call(call.name, call.arguments)
                    except Exception as exc:
                        observation = f"error: {exc}"
                ep.messages.append(Message("tool", observation))
                self._transition(ep, AgentState.VERIFY, observation=observation)
                if observation.startswith("error:"):
                    retries += 1
                    self._transition(ep, AgentState.RECOVER, retries=retries)
                    if retries > self.max_retries:
                        self._transition(ep, AgentState.FAILED, reason="demasiados errores")
                        break
                self._transition(ep, AgentState.PLAN)
            elif reply.content.strip():
                ep.messages.append(reply)
                ep.final_answer = reply.content.strip()
                ep.success = True
                self._transition(ep, AgentState.DONE, answer=ep.final_answer)
            else:
                retries += 1
                self._transition(ep, AgentState.RECOVER, retries=retries, reason="salida vacía")
                if retries > self.max_retries:
                    self._transition(ep, AgentState.FAILED, reason="formato inválido")
                    break
                ep.messages.append(
                    Message("system", "Formato inválido. Responde con texto o con <|tool_call|>.")
                )
                self._transition(ep, AgentState.PLAN)
        self._log(ep)
        return ep

    def _log(self, ep: Episode) -> None:
        if self.episode_log is None:
            return
        self.episode_log.parent.mkdir(parents=True, exist_ok=True)
        with self.episode_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")


def model_policy(session: ChatSession) -> Policy:
    """Usa un ``ChatSession`` (CefomenaLM) como política del agente."""
    return lambda messages: session.complete(messages)


def episodes_to_conversations(
    episodes: list[Episode], min_reward: float = 0.0
) -> list[list[Message]]:
    """Convierte episodios exitosos en conversaciones de entrenamiento (SFT)."""
    return [
        ep.messages
        for ep in episodes
        if ep.success and (ep.reward is None or ep.reward >= min_reward)
    ]
