"""Backends de uso de computadora: el modelo actúa sobre observaciones textuales."""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 - se invoca solo xdotool con argumentos en lista, sin shell
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ComputerBackend(ABC):
    name: str = "base"

    @abstractmethod
    def observe(self) -> str: ...

    @abstractmethod
    def click(self, x: int, y: int) -> str: ...

    @abstractmethod
    def type_text(self, text: str) -> str: ...

    @abstractmethod
    def key(self, combo: str) -> str: ...

    @abstractmethod
    def scroll(self, amount: int) -> str: ...


@dataclass
class DryRunComputer(ComputerBackend):
    """Backend simulado (por defecto): registra acciones y devuelve una pantalla textual."""

    screen: str = "Escritorio vacío. Ventanas: ninguna."
    actions: list[dict[str, Any]] = field(default_factory=list)
    name: str = "dryrun"

    def _record(self, action: str, **kwargs: Any) -> str:
        self.actions.append({"action": action, **kwargs})
        return f"ok: {action} {kwargs}" if kwargs else f"ok: {action}"

    def observe(self) -> str:
        last = self.actions[-1] if self.actions else None
        return f"{self.screen}\nÚltima acción: {last}"

    def click(self, x: int, y: int) -> str:
        return self._record("click", x=int(x), y=int(y))

    def type_text(self, text: str) -> str:
        return self._record("type", text=str(text))

    def key(self, combo: str) -> str:
        return self._record("key", combo=str(combo))

    def scroll(self, amount: int) -> str:
        return self._record("scroll", amount=int(amount))


class XdotoolComputer(ComputerBackend):
    """Backend Linux/X11 vía ``xdotool`` (sin shell, con timeout)."""

    name = "xdotool"

    def __init__(self, timeout: float = 10.0) -> None:
        exe = shutil.which("xdotool")
        if exe is None:
            raise RuntimeError("xdotool no está instalado")
        self.exe = exe
        self.timeout = timeout

    def _run(self, *args: str) -> str:
        proc = subprocess.run(  # noqa: S603  # nosec B603
            [self.exe, *args], capture_output=True, text=True, timeout=self.timeout, check=False
        )
        if proc.returncode != 0:
            return f"error: {proc.stderr.strip()}"
        return proc.stdout.strip() or "ok"

    def observe(self) -> str:
        window = self._run("getactivewindow", "getwindowname")
        mouse = self._run("getmouselocation")
        return f"Ventana activa: {window}\nRatón: {mouse}"

    def click(self, x: int, y: int) -> str:
        return self._run("mousemove", str(int(x)), str(int(y)), "click", "1")

    def type_text(self, text: str) -> str:
        return self._run("type", "--delay", "20", "--", str(text))

    def key(self, combo: str) -> str:
        return self._run("key", "--", str(combo))

    def scroll(self, amount: int) -> str:
        button = "5" if amount > 0 else "4"
        return self._run("click", "--repeat", str(abs(int(amount))), button)


class PyAutoGUIComputer(ComputerBackend):
    """Backend multiplataforma opcional (requiere ``pyautogui``)."""

    name = "pyautogui"

    def __init__(self) -> None:
        try:
            import pyautogui
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("instala pyautogui para usar este backend") from exc
        pyautogui.FAILSAFE = True
        self.gui = pyautogui

    def observe(self) -> str:
        w, h = self.gui.size()
        x, y = self.gui.position()
        return f"Pantalla {w}x{h}. Ratón en ({x}, {y})."

    def click(self, x: int, y: int) -> str:
        self.gui.click(int(x), int(y))
        return "ok"

    def type_text(self, text: str) -> str:
        self.gui.write(str(text), interval=0.02)
        return "ok"

    def key(self, combo: str) -> str:
        self.gui.hotkey(*str(combo).split("+"))
        return "ok"

    def scroll(self, amount: int) -> str:
        self.gui.scroll(-int(amount))
        return "ok"


def make_backend(name: str = "dryrun") -> ComputerBackend:
    backends: dict[str, type[ComputerBackend]] = {
        "dryrun": DryRunComputer,
        "xdotool": XdotoolComputer,
        "pyautogui": PyAutoGUIComputer,
    }
    if name not in backends:
        raise ValueError(f"backend desconocido: {name}. Opciones: {sorted(backends)}")
    return backends[name]()
