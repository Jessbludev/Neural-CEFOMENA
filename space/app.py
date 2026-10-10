"""Hugging Face Space (Gradio) para CEFOMENA."""

from __future__ import annotations

import os
import re
from typing import Any

import gradio as gr

from cefomena.chat.generate import ChatSession, GenerationConfig
from cefomena.chat.template import Message
from cefomena.export.hub import download_from_hub
from cefomena.export.safetensors_io import load_pretrained

SPECIAL_TOKEN = re.compile(r"<\|[a-z_]+\|>")
MODEL_REPO = os.environ.get("MODEL_REPO", "Jessbludev/cefomena-tiny")

model, tokenizer = load_pretrained(download_from_hub(MODEL_REPO))


def _text(content: Any) -> str:
    """Gradio 6 entrega ``content`` como lista de partes ``{"type": "text", "text": ...}``."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", "")) if content.get("type", "text") == "text" else ""
    if isinstance(content, list | tuple):
        return "".join(_text(part) for part in content)
    return ""


def respond(message: Any, history: list[dict[str, Any]], temperature: float) -> str:
    session = ChatSession(
        model,
        tokenizer,
        generation=GenerationConfig(temperature=temperature, max_new_tokens=160),
    )
    for turn in history[-12:]:
        text = _text(turn.get("content")).strip()
        if turn.get("role") in ("user", "assistant") and text:
            session.add(Message(turn["role"], text))
    user_text = _text(message).strip()
    if not user_text:
        return "Escribe un mensaje, por favor."
    reply = session.reply(user_text[:2000])
    if reply.tool_call is not None:
        return (
            f"(quiere usar la herramienta `{reply.tool_call.name}` con {reply.tool_call.arguments})"
        )
    return SPECIAL_TOKEN.sub("", reply.content).strip() or "(sin respuesta)"


demo = gr.ChatInterface(
    respond,
    title="CEFOMENA 🪼",
    description="Asistente pequeño, solo en español. Modelo experimental: puede equivocarse.",
    additional_inputs=[gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperatura")],
    examples=[["Hola, ¿qué tal?", 0.7], ["Ayúdame a organizar mi día", 0.7]],
)

if __name__ == "__main__":
    demo.launch()
