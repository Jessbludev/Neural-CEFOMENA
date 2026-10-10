"""Fuentes de datos conversacionales en español.

- OpenAssistant oasst2 (Apache-2.0): se extraen hilos en español (``lang == "es"``) siguiendo
  la respuesta mejor clasificada en cada nivel.
- Diálogos sintéticos propios (ver ``synth.py``).
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cefomena.chat.template import Message

OASST2_REPO = "OpenAssistant/oasst2"
OASST2_FILE = "2023-11-05_oasst2_ready.trees.jsonl.gz"
_ROLE = {"prompter": "user", "assistant": "assistant"}


def download_oasst2() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(OASST2_REPO, OASST2_FILE, repo_type="dataset"))


def _best_child(node: dict[str, Any]) -> dict[str, Any] | None:
    replies = [r for r in node.get("replies") or [] if not r.get("deleted")]
    if not replies:
        return None
    return min(replies, key=lambda r: r.get("rank") if r.get("rank") is not None else 1e9)


def oasst2_spanish(path: str | Path, max_turns: int = 8) -> Iterator[list[Message]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            tree = json.loads(line)
            node: dict[str, Any] | None = tree.get("prompt")
            if not node or node.get("lang") != "es":
                continue
            conv: list[Message] = []
            while node is not None and len(conv) < max_turns:
                role = _ROLE.get(node.get("role", ""))
                text = (node.get("text") or "").strip()
                if role is None or not text:
                    break
                conv.append(Message(role, text))  # type: ignore[arg-type]
                node = _best_child(node)
            while conv and conv[-1].role != "assistant":
                conv.pop()
            if len(conv) >= 2:
                yield conv


def write_jsonl(
    conversations: Iterator[list[Message]] | list[list[Message]], path: str | Path
) -> int:
    n = 0
    with Path(path).open("w", encoding="utf-8") as fh:
        for conv in conversations:
            rows = []
            for m in conv:
                row: dict[str, Any] = {"role": m.role, "content": m.content}
                if m.tool_call:
                    row["tool_call"] = {
                        "name": m.tool_call.name,
                        "arguments": m.tool_call.arguments,
                    }
                rows.append(row)
            fh.write(json.dumps({"messages": rows}, ensure_ascii=False) + "\n")
            n += 1
    return n
