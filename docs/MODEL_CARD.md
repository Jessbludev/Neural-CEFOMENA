---
language: es
license: apache-2.0
library_name: cefomena
pipeline_tag: text-generation
tags: [bio-inspired, small-language-model, spanish, safetensors, agent]
datasets: [OpenAssistant/oasst2]
---

# CEFOMENA-tiny

Modelo conversacional pequeño (≈30,8 M parámetros) **solo en español**, con arquitectura híbrida
bioinspirada (medusa, delfín, coral, hongos, mariposa, murciélago). Código:
https://github.com/Jessbludev/Neural-CEFOMENA

## Uso previsto
Asistencia básica en español (saludos, organización, redacción corta) y llamadas a herramientas en
JSON para un orquestador de agente. **No** es una fuente de conocimiento: puede inventar datos.

## Datos
Diálogos sintéticos propios y el subconjunto en español de OpenAssistant oasst2 (Apache-2.0).
Ver `docs/DATA.md`.

## Entrenamiento
Google Colab (GPU), decoder causal, `MetabolicFSM` (activo/descanso/sueño con replay/despertar),
EMA y pérdida solo sobre los turnos del asistente. Hiperparámetros: `config.json` y el notebook.

## Evaluación
_Pendiente de completar tras el entrenamiento: pérdida de validación, perplejidad y tasa de llamadas
a herramientas con JSON válido._

## Archivos e integridad
`model.safetensors`, `config.json`, `tokenizer.json`, `generation_config.json`, `SHA256SUMS`.

```python
from cefomena.export.hub import download_from_hub
from cefomena.export.safetensors_io import load_pretrained

model, tok = load_pretrained(download_from_hub("Jessbludev/cefomena-tiny"))  # verifica SHA256SUMS
```

## Limitaciones y riesgos
Modelo muy pequeño: puede dar respuestas incoherentes o incorrectas, y sesgos heredados de los datos.
El uso de computadora debe hacerse con el backend `dryrun` o con confirmación humana.
