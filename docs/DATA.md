# Datos — solo español

| Fuente | Licencia | Uso | Cómo |
|---|---|---|---|
| Diálogos sintéticos (`cefomena synth`) | Apache-2.0 (propio) | charla básica, honestidad, tool-calling, uso de computadora | `train/synth.py` |
| OpenAssistant oasst2, `lang == "es"` | Apache-2.0 | conversación general en español | `cefomena oasst-es` (`train/sources.py`) |

Reglas:
- Solo conversaciones con raíz `lang == "es"`; se sigue la respuesta mejor clasificada en cada turno
  y se descartan los mensajes eliminados.
- No se usan datasets con licencias no comerciales ni contenido enciclopédico masivo: el modelo es un
  asistente básico, no una base de conocimiento.
- Los datos y artefactos generados no se versionan (`data/`, `*.npz` en `.gitignore`).

Formato JSONL:

```json
{"messages": [{"role": "user", "content": "¿Cuánto es 2 + 2?"},
              {"role": "assistant", "tool_call": {"name": "calculadora", "arguments": {"expresion": "2 + 2"}}},
              {"role": "tool", "content": "4"},
              {"role": "assistant", "content": "El resultado es 4."}]}
```
