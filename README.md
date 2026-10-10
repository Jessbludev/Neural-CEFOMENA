# Neural-CEFOMENA

Modelo conversacional **pequeño y solo en español** (≈30 M parámetros) con arquitectura híbrida
**bioinspirada** y un **orquestador de agente** con uso de computadora adaptable. Pensado para
asistencia básica, no como modelo de conocimiento general ni de frontera.

| Organismo | Módulo |
|---|---|
| 🪼 Medusa | `NerveNetMixer`: mezcla causal local (convoluciones) |
| 🐬 Delfín | `SparseGateMLP`: activación escasa k-WTA + canal de recompensa |
| 🪸 Coral | `OscillatorMoE`: expertos ruteados por osciladores acoplados |
| 🍄 Hongos | `FungalMemory` + orquestador FSM |
| 🦋 Mariposa | `DualCache`: adaptación periférica por usuario |
| 🦇 Murciélago | `MetabolicFSM`: activo → descanso → sueño (replay) → despertar |

Detalles: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Plan: [docs/PLAN.md](docs/PLAN.md) ·
Datos: [docs/DATA.md](docs/DATA.md) · Seguridad: [SECURITY.md](SECURITY.md)

## Requisitos

[uv](https://docs.astral.sh/uv/) y Python 3.11. Dependencias de ejecución: `torch`, `numpy`, `safetensors`.

```bash
uv sync
uv run pytest
```

## Uso rápido (CPU)

```bash
uv run cefomena info --preset tiny
uv run cefomena synth --n 5000 --out data/synth.jsonl
uv run --extra hub cefomena oasst-es --out data/oasst_es.jsonl      # opcional
uv run cefomena train-tokenizer --data data/*.jsonl --vocab-size 8192 --out data/tok.json
uv run cefomena prepare --data data/*.jsonl --tokenizer data/tok.json --out data/train.npz
uv run cefomena train --data data/train.npz --tokenizer data/tok.json --preset micro --steps 2000 --out runs/micro
uv run cefomena chat --model runs/micro
uv run cefomena agent --model runs/micro --task "Escribe hola en el editor" --backend dryrun
```

El entrenamiento real se hace en Colab: [notebooks/colab_train.ipynb](notebooks/colab_train.ipynb).

## Exportación y despliegue

`save_pretrained` genera `model.safetensors`, `config.json`, `tokenizer.json`,
`generation_config.json` y `SHA256SUMS`. `load_pretrained` verifica los checksums antes de cargar.

```bash
HF_TOKEN=... uv run --extra hub cefomena push --model runs/tiny --repo-id <usuario>/cefomena-tiny
```

Demo en Hugging Face Spaces (Gradio): [space/](space/).

## Licencia

Apache-2.0.
