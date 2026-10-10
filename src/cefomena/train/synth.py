"""Generador de diálogos sintéticos en español: charla básica y uso de herramientas."""

from __future__ import annotations

import random

from cefomena.chat.template import DEFAULT_SYSTEM_PROMPT, Message, ToolCall

_SALUDOS = ["Hola", "Buenas", "Hola, ¿qué tal?", "Buenos días", "Buenas tardes", "Hey"]
_RESP_SALUDO = [
    "¡Hola! ¿En qué te puedo ayudar hoy?",
    "¡Hola! Estoy aquí para ayudarte. ¿Qué necesitas?",
    "¡Buenas! Cuéntame, ¿qué quieres hacer?",
]
_DESPEDIDAS = ["Gracias, eso es todo", "Adiós", "Muchas gracias", "Hasta luego"]
_RESP_DESPEDIDA = [
    "¡De nada! Que tengas un buen día.",
    "¡Con gusto! Hasta pronto.",
    "¡A ti! Aquí estaré.",
]
_DESCONOCIDO = [
    "¿Cuál es la fórmula exacta de la proteína más rara del océano?",
    "Explícame la física cuántica de campos en detalle",
    "¿Quién ganó la liga de fútbol de 1931 en Bolivia?",
]
_RESP_DESCONOCIDO = [
    "No tengo información fiable sobre eso. Soy un asistente básico; te sugiero consultar una fuente especializada.",
    "Eso está fuera de lo que sé hacer bien. Puedo ayudarte con tareas sencillas o a organizarte.",
]
_TAREAS = [
    (
        "organizar mi día",
        "Claro. Empieza por lo más importante, agrupa tareas parecidas y deja pausas cortas. ¿Qué tienes pendiente hoy?",
    ),
    (
        "escribir un mensaje para felicitar a un amigo",
        "¡Claro! Por ejemplo: «¡Feliz cumpleaños! Que este año te traiga mucha alegría y buenos momentos.»",
    ),
    (
        "hacer una lista de compras",
        "Perfecto. Dime qué comidas planeas y armamos la lista por secciones: frutas, verduras, despensa y limpieza.",
    ),
    (
        "concentrarme mejor",
        "Prueba bloques de 25 minutos de trabajo con 5 de descanso y deja el móvil lejos. ¿Quieres que te ayude a planear el primer bloque?",
    ),
]


def _calc_dialog(rng: random.Random) -> list[Message]:
    a, b = rng.randint(2, 999), rng.randint(2, 99)
    op = rng.choice(["+", "-", "*"])
    expr = f"{a} {op} {b}"
    result = {"+": a + b, "-": a - b, "*": a * b}[op]
    pedido = rng.choice(["¿Cuánto es", "Calcula", "Me ayudas con"])
    return [
        Message("user", f"{pedido} {expr}?"),
        Message("assistant", tool_call=ToolCall("calculadora", {"expresion": expr})),
        Message("tool", str(result)),
        Message("assistant", f"El resultado de {expr} es {result}."),
    ]


def _hora_dialog(rng: random.Random) -> list[Message]:
    h, m = rng.randint(0, 23), rng.randint(0, 59)
    return [
        Message(
            "user", rng.choice(["¿Qué hora es?", "Dime la hora", "¿Me dices la fecha y hora?"])
        ),
        Message("assistant", tool_call=ToolCall("hora_actual", {})),
        Message("tool", f"2026-03-{rng.randint(1, 28):02d} {h:02d}:{m:02d}"),
        Message("assistant", f"Son las {h:02d}:{m:02d}."),
    ]


def _computer_dialog(rng: random.Random) -> list[Message]:
    texto = rng.choice(["lista de tareas", "hola mundo", "reunión a las 10", "comprar pan"])
    return [
        Message("user", f"Escribe «{texto}» en la ventana abierta"),
        Message("assistant", tool_call=ToolCall("observar", {})),
        Message("tool", "Ventana activa: Editor de texto - sin título\nRatón: x:400 y:300"),
        Message("assistant", tool_call=ToolCall("escribir", {"texto": texto})),
        Message("tool", "ok"),
        Message("assistant", f"Listo, escribí «{texto}» en el editor."),
    ]


def _guardar_dialog(rng: random.Random) -> list[Message]:
    return [
        Message("user", rng.choice(["Guarda el documento", "Guarda el archivo, por favor"])),
        Message("assistant", tool_call=ToolCall("tecla", {"combinacion": "ctrl+s"})),
        Message("tool", "ok"),
        Message("assistant", "Hecho, pulsé Ctrl+S para guardar."),
    ]


def _chat_dialog(rng: random.Random) -> list[Message]:
    conv = [Message("user", rng.choice(_SALUDOS)), Message("assistant", rng.choice(_RESP_SALUDO))]
    kind = rng.random()
    if kind < 0.6:
        tarea, resp = rng.choice(_TAREAS)
        conv += [Message("user", f"Quiero {tarea}"), Message("assistant", resp)]
    elif kind < 0.8:
        conv += [
            Message("user", rng.choice(_DESCONOCIDO)),
            Message("assistant", rng.choice(_RESP_DESCONOCIDO)),
        ]
    conv += [
        Message("user", rng.choice(_DESPEDIDAS)),
        Message("assistant", rng.choice(_RESP_DESPEDIDA)),
    ]
    return conv


_GENERATORS = (
    _chat_dialog,
    _chat_dialog,
    _calc_dialog,
    _hora_dialog,
    _computer_dialog,
    _guardar_dialog,
)


def synthetic_conversations(n: int, seed: int = 0, with_system: bool = True) -> list[list[Message]]:
    rng = random.Random(seed)  # noqa: S311 - datos sintéticos, no criptografía
    out = []
    for _ in range(n):
        conv = rng.choice(_GENERATORS)(rng)
        if with_system:
            conv = [Message("system", DEFAULT_SYSTEM_PROMPT), *conv]
        out.append(conv)
    return out
