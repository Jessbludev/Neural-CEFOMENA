"""Publicación en Hugging Face Hub (requiere el extra ``hub``)."""

from __future__ import annotations

import os
from pathlib import Path

from cefomena.export.safetensors_io import verify_checksums


def push_to_hub(
    model_dir: str | Path, repo_id: str, token: str | None = None, private: bool = False
) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("instala el extra: uv sync --extra hub") from exc
    verify_checksums(model_dir)
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    info = api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Subir pesos CEFOMENA (safetensors)",
    )
    return str(info)


def download_from_hub(repo_id: str, revision: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, revision=revision))
