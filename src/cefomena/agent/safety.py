"""Política de seguridad del agente: allowlist, confirmación humana, límites y auditoría."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cefomena.agent.tools import Tool
from cefomena.chat.template import ToolCall

DANGEROUS_TEXT = (
    r"rm\s+-rf",
    r"\bsudo\b",
    r"\bmkfs\b",
    r":\(\)\s*\{",
    r"\bdd\s+if=",
    r"curl[^|]*\|\s*(ba)?sh",
    r"wget[^|]*\|\s*(ba)?sh",
    r"\bshutdown\b",
    r"\breboot\b",
)
DANGEROUS_KEYS = ("ctrl+alt+delete", "ctrl+alt+backspace", "alt+f4", "ctrl+q", "ctrl+w")


@dataclass
class SafetyDecision:
    allowed: bool
    reason: str = ""


@dataclass
class SafetyPolicy:
    allowed_tools: set[str] | None = None
    max_steps: int = 12
    confirm: Callable[[ToolCall], bool] | None = None
    audit_path: Path | None = None
    dangerous_text: tuple[str, ...] = DANGEROUS_TEXT
    dangerous_keys: tuple[str, ...] = DANGEROUS_KEYS
    _patterns: list[re.Pattern[str]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.dangerous_text]

    def _is_risky(self, call: ToolCall) -> str | None:
        for value in call.arguments.values():
            if isinstance(value, str) and any(p.search(value) for p in self._patterns):
                return "texto potencialmente peligroso"
        combo = str(call.arguments.get("combinacion", "")).lower().replace(" ", "")
        if combo in self.dangerous_keys:
            return "combinación de teclas peligrosa"
        return None

    def check(self, call: ToolCall, tool: Tool | None) -> SafetyDecision:
        if tool is None:
            return SafetyDecision(False, f"herramienta desconocida: {call.name}")
        if self.allowed_tools is not None and call.name not in self.allowed_tools:
            return SafetyDecision(False, f"herramienta no permitida: {call.name}")
        risk = self._is_risky(call)
        if tool.destructive or risk:
            reason = risk or "acción marcada como destructiva"
            if self.confirm is None or not self.confirm(call):
                return SafetyDecision(False, f"requiere confirmación humana ({reason})")
        return SafetyDecision(True)

    def audit(self, event: dict[str, Any]) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.time(), **event}
        with self.audit_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
