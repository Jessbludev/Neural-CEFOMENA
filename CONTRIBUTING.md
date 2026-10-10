# Contribuir

```bash
uv sync                     # entorno (Python 3.11, torch CPU)
uv tool install pre-commit && pre-commit install
uv run ruff check src tests && uv run mypy && uv run pytest
```

- Ramas: `feat/…`, `fix/…`; PR obligatorio hacia `main` (historial lineal, checks requeridos).
- Cada PR enlaza su issue de Linear (`KER-…`).
- Datos, prompts y ejemplos: **solo en español**.
- No subas pesos, datasets ni secretos (ver `.gitignore`).
