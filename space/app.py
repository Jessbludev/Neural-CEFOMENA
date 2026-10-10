"""Hugging Face Space (Gradio) para CEFOMENA."""

from __future__ import annotations

import os

import gradio as gr

from cefomena.chat.generate import ChatSession, GenerationConfig
from cefomena.chat.template import Message
from cefomena.export.hub import download_from_hub
from cefomena.export.safetensors_io import load_pretrained

MODEL_REPO = os.environ.get("MODEL_REPO", "Jessbludev/cefomena-tiny")

model, tokenizer = load_pretrained(download_from_hub(MODEL_REPO))


def respond(message: str, history: list[dict[str, str]], temperature: float) -> str:
    session = ChatSession(
        model,
        tokenizer,
        generation=GenerationConfig(temperature=temperature, max_new_tokens=160),
    )
    for turn in history[-12:]:
        if turn.get("role") in ("user", "assistant") and isinstance(turn.get("content"), str):
            session.add(Message(turn["role"], turn["content"]))  # type: ignore[arg-type]
    reply = session.reply(message)
    if reply.tool_call is not None:
        return (
            f"(quiere usar la herramienta `{reply.tool_call.name}` con {reply.tool_call.arguments})"
        )
    return reply.content


demo = gr.ChatInterface(
    respond,
    type="messages",
    title="CEFOMENA 🪼",
    description="Asistente pequeño, solo en español. Modelo experimental: puede equivocarse.",
    additional_inputs=[gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperatura")],
    examples=[["Hola, ¿qué tal?", 0.7], ["Ayúdame a organizar mi día", 0.7]],
)

if __name__ == "__main__":
    demo.launch()
