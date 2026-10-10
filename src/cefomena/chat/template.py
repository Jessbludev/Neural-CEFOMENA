"""Plantilla de chat en español con soporte de llamadas a herramientas."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from cefomena.tokenizer.bpe import BPETokenizer

Role = Literal["system", "user", "assistant", "tool"]

DEFAULT_SYSTEM_PROMPT = (
    "Eres CEFOMENA, un asistente conversacional sencillo que habla en español. "
    "Respondes de forma breve, amable y honesta. Si no sabes algo, lo dices. "
    "Cuando una tarea requiere una herramienta, respondes solo con la llamada en JSON."
)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "arguments": self.arguments}, ensure_ascii=False)


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_call: ToolCall | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        tc = data.get("tool_call")
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            tool_call=ToolCall(tc["name"], tc.get("arguments", {})) if tc else None,
        )


def _segments(m: Message) -> tuple[str, str]:
    """Devuelve (prefijo, cuerpo) del mensaje; el cuerpo es lo que aprende el asistente."""
    prefix = f"<|{m.role}|>"
    body = f"<|tool_call|>{m.tool_call.to_json()}" if m.tool_call else m.content
    return prefix, body + "<|end|>"


def render(messages: list[Message], add_generation_prompt: bool = False) -> str:
    out = ["<|bos|>"]
    for m in messages:
        prefix, body = _segments(m)
        out.append(prefix + body)
    if add_generation_prompt:
        out.append("<|assistant|>")
    return "".join(out)


def render_ids(
    tokenizer: BPETokenizer, messages: list[Message], train_on: tuple[Role, ...] = ("assistant",)
) -> tuple[list[int], list[int]]:
    """Tokeniza la conversación y devuelve (ids, máscara de pérdida)."""
    ids = [tokenizer.bos_id]
    mask = [0]
    for m in messages:
        prefix, body = _segments(m)
        p_ids = tokenizer.encode(prefix)
        b_ids = tokenizer.encode(body)
        ids += p_ids + b_ids
        learn = 1 if m.role in train_on else 0
        mask += [0] * len(p_ids) + [learn] * len(b_ids)
    ids.append(tokenizer.eos_id)
    mask.append(0)
    return ids, mask


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Corrección sintáctica tolerante: extrae el primer objeto JSON balanceado."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    return obj if isinstance(obj, dict) else None
        start = text.find("{", start + 1)
    return None


def parse_assistant(text: str) -> Message:
    """Convierte el texto generado por el asistente en un ``Message``."""
    body = text.split("<|end|>", 1)[0].replace("<|eos|>", "").strip()
    if "<|tool_call|>" in body:
        obj = extract_json_object(body.split("<|tool_call|>", 1)[1])
        if obj and isinstance(obj.get("name"), str):
            args = obj.get("arguments")
            return Message(
                "assistant", tool_call=ToolCall(obj["name"], args if isinstance(args, dict) else {})
            )
        return Message("assistant", content="")
    return Message("assistant", content=body)
