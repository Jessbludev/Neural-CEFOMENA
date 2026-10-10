"""Exportación/carga en formato safetensors con verificación de integridad SHA256."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_model, save_model

from cefomena.config import CefomenaConfig
from cefomena.model.model import CefomenaLM
from cefomena.tokenizer.bpe import BPETokenizer

WEIGHTS = "model.safetensors"
CONFIG = "config.json"
TOKENIZER = "tokenizer.json"
GENERATION = "generation_config.json"
CHECKSUMS = "SHA256SUMS"
_HASHED = (WEIGHTS, CONFIG, TOKENIZER, GENERATION)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_pretrained(
    model: CefomenaLM,
    tokenizer: BPETokenizer,
    out_dir: str | Path,
    generation: dict[str, object] | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_model(model, str(out / WEIGHTS), metadata={"format": "pt", "model_type": "cefomena"})
    model.config.save(out / CONFIG)
    tokenizer.save(out / TOKENIZER)
    gen = generation or {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "max_new_tokens": 200}
    (out / GENERATION).write_text(json.dumps(gen, indent=2) + "\n")
    lines = [f"{sha256_file(out / name)}  {name}" for name in _HASHED]
    (out / CHECKSUMS).write_text("\n".join(lines) + "\n")
    return out


def verify_checksums(model_dir: str | Path) -> None:
    root = Path(model_dir)
    sums = root / CHECKSUMS
    if not sums.exists():
        raise FileNotFoundError(f"falta {CHECKSUMS} en {root}")
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if Path(name).name != name:
            raise ValueError(f"ruta no permitida en {CHECKSUMS}: {name!r}")
        if sha256_file(root / name) != digest:
            raise ValueError(f"checksum inválido para {name}")


def load_pretrained(
    model_dir: str | Path, device: str | torch.device = "cpu", verify: bool = True
) -> tuple[CefomenaLM, BPETokenizer]:
    root = Path(model_dir)
    if verify:
        verify_checksums(root)
    cfg = CefomenaConfig.load(root / CONFIG)
    model = CefomenaLM(cfg)
    load_model(model, str(root / WEIGHTS), device=str(device))
    tokenizer = BPETokenizer.load(root / TOKENIZER)
    return model.to(device).eval(), tokenizer
