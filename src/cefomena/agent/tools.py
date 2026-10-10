"""Registro de herramientas del agente."""

from __future__ import annotations

import ast
import datetime as dt
import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cefomena.agent.computer import ComputerBackend


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., str]
    parameters: dict[str, str] = field(default_factory=dict)
    destructive: bool = False

    def signature(self) -> str:
        params = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        flag = " [requiere confirmación]" if self.destructive else ""
        return f"- {self.name}({params}): {self.description}{flag}"


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"herramienta duplicada: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> str:
        return "\n".join(self._tools[n].signature() for n in self.names())

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"herramienta desconocida: {name}")
        unknown = set(arguments) - set(tool.parameters)
        if unknown:
            raise TypeError(f"argumentos no válidos para {name}: {sorted(unknown)}")
        return str(tool.func(**arguments))


_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_calc(expresion: str) -> str:
    """Evalúa aritmética básica sin ``eval``."""

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(ev(node.right)) > 100:
                raise ValueError("exponente demasiado grande")
            return float(_OPS[type(node.op)](ev(node.left), ev(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return float(_OPS[type(node.op)](ev(node.operand)))
        raise ValueError("expresión no permitida")

    if len(expresion) > 200:
        raise ValueError("expresión demasiado larga")
    result = ev(ast.parse(expresion.replace(",", "."), mode="eval").body)
    return f"{result:g}"


def default_tools() -> list[Tool]:
    return [
        Tool(
            "hora_actual",
            "Devuelve la fecha y hora local.",
            lambda: dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        ),
        Tool(
            "calculadora",
            "Calcula una expresión aritmética simple.",
            safe_calc,
            {"expresion": "texto"},
        ),
    ]


def computer_tools(backend: ComputerBackend) -> list[Tool]:
    return [
        Tool("observar", "Describe la pantalla actual en texto.", backend.observe),
        Tool("clic", "Hace clic en coordenadas.", backend.click, {"x": "entero", "y": "entero"}),
        Tool(
            "escribir",
            "Escribe texto con el teclado.",
            lambda texto: backend.type_text(texto),
            {"texto": "texto"},
        ),
        Tool(
            "tecla",
            "Pulsa una tecla o combinación (p. ej. ctrl+s).",
            lambda combinacion: backend.key(combinacion),
            {"combinacion": "texto"},
        ),
        Tool(
            "desplazar",
            "Desplaza la vista (positivo = abajo).",
            lambda cantidad: backend.scroll(cantidad),
            {"cantidad": "entero"},
        ),
    ]
