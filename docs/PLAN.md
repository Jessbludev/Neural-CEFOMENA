# Neural-CEFOMENA — Plan de seguimiento (v0)

> Modelo conversacional **pequeño** (≈20–30 M parámetros), dedicado a asistencia básica y a orquestar un agente con uso de computadora adaptable. **No** es un modelo frontera ni de conocimiento enciclopédico. Arquitectura híbrida bioinspirada según el reporte `research.md` (6 organismos → 6 principios computacionales).

## 0. Principios de ingeniería
- **uv** como único gestor (`pyproject.toml` + `uv.lock`, `uv run`, `uv build`). Python 3.11.
- **Dependencias mínimas**: `torch`, `numpy`, `safetensors`. Opcionales por grupo: `hub` (`huggingface_hub`), `space` (`gradio`), `computer` (`pyautogui`/`xdotool` vía subprocess), `dev` (`ruff`, `mypy`, `pytest`, `pip-audit`). Sin Transformers/Lightning/Accelerate/LangChain.
- Tokenizador BPE byte-level **propio** (~200 líneas, sin dependencia externa), vocab 8 192.
- Pesos solo en **safetensors** (nunca pickle; `torch.load(weights_only=True)` si fuera necesario).
- Las metáforas biológicas son **heurísticas de diseño**, no afirmaciones de fidelidad biológica (regla "no antropomorfizar" del reporte).

## 1. Diseño híbrido (mapa reporte → módulo)

| Organismo | Principio (reporte §4) | Módulo en `cefomena` | Dónde actúa |
|---|---|---|---|
| Medusa | Cómputo topológico distribuido (vecindad local, sin centro) | `NerveNetMixer`: convolución causal depthwise + compuerta (mezcla local de tokens, O(n)) | 2 de cada 3 bloques (reemplaza atención) |
| Delfín | Codificación temporal escasa + refuerzo | `SparseGate` (k-WTA "PV-like") en el MLP; canal de recompensa que en RL/feedback solo ajusta parámetros de compuerta | MLP de cada bloque + fine-tune por feedback |
| Coral | Consenso por osciladores acoplados | `OscillatorRouter`: cada experto tiene fase θᵢ; acoplamiento tipo Kuramoto decide su peso por paso (router sin red central) | MoE ligero (4 expertos) en bloques de atención |
| Hongos | Memoria no paramétrica + autómata celular robusto | `FungalMemory`: buffer de slots con reglas locales de habituación/decay (estado, no pesos) leído por atención cruzada; además FSM robusta del orquestador | Memoria conversacional + planner del agente |
| Mariposa | Dual-cache sensorial–central | `DualCache`: ganancia/sesgo de bajo rango "periférico" (fast weights, adaptación online por usuario/tarea) sobre pesos centrales congelados | Embedding/entrada; personalización sin reentrenar |
| Murciélago | Máquina de estados metabólicos | `MetabolicFSM` (ACTIVE→REST→SLEEP→WAKE) que controla LR, replay/consolidación, EMA; en inferencia controla presupuesto (early-exit) | Trainer + generación |

Núcleo base (config `cefomena-tiny`): decoder causal, d=384, 9 bloques (6 NerveNet + 3 atención causal con RoPE y OscillatorRouter-MoE), contexto 1 024, vocab 8 192, embeddings atados. Objetivo ≈ 25 M params, entrenable en Colab T4/L4.

```
tokens → Embedding + DualCache ─┬─ [NerveNet+SparseGate] ×2 ─ [Attn+OscMoE+FungalMemory read] ─┐ ×3
                                 └────────────────────────────────────────────────────────────────┘
        → RMSNorm → LM head (atado)          MetabolicFSM controla entrenamiento / presupuesto de inferencia
```

### Validaciones computacionales (de §7.2 del reporte)
1. Aprende tarea asociativa/sintética con pocos parámetros (test de copy/recall).
2. Memoria vía estado (FungalMemory) mejora recall multi-turno vs. ablación.
3. Router de osciladores sincroniza/especializa expertos sin router aprendido central (métrica: entropía de carga).
4. Ablaciones por módulo documentadas en `docs/ABLATIONS.md`.

## 2. Comportamiento conversacional y agente
- **Datos** (solo conversación/asistencia básica, licencias permisivas, a verificar en `docs/DATA.md`): **solo español**: p. ej. OpenAssistant oasst2 (subset español), everyday-conversations, smol-smoltalk subset, + diálogos sintéticos de tool-calling propios. Filtro: sin dominios especializados, sin PII.
- **Plantilla de chat** con roles `system/user/assistant/tool` y llamadas a herramientas en JSON (`<|tool_call|>{...}<|end|>`).
- **Orquestador** (`cefomena.agent`): bucle observar→planear→actuar con FSM fúngica (estados IDLE/PLAN/ACT/VERIFY/RECOVER, corrección de errores), registro de herramientas, memoria de episodio.
- **Use-computer adaptable**: interfaz `ComputerBackend` (`observe()` texto/árbol de accesibilidad, `click`, `type`, `key`, `scroll`); backends `DryRun` (por defecto, tests), `Xdotool` (Linux), `PyAutoGUI` (opcional). El modelo actúa sobre observaciones **textuales** (no visión).
- **Seguridad del agente**: allowlist de acciones, confirmación humana para acciones destructivas, límite de pasos, logs auditables.
- **Aprendizaje del agente**: trayectorias registradas → (a) adaptación rápida en `DualCache`, (b) fine-tune offline con recompensa sobre `SparseGate`.

## 3. Estructura del repo
```
pyproject.toml  uv.lock  README.md  LICENSE  SECURITY.md  CONTRIBUTING.md
src/cefomena/
  config.py  cli.py
  tokenizer/bpe.py
  model/{nervenet,sparse_gate,oscillator_router,fungal_memory,dual_cache,attention,model}.py
  train/{data,metabolic_fsm,trainer}.py
  chat/{template,generate}.py
  agent/{orchestrator,tools,computer,safety}.py
  export/{safetensors_io,hub}.py
tests/                       # unit + smoke (CPU, <2 min)
notebooks/colab_train.ipynb  # entrenamiento en Colab (uv dentro de Colab)
space/{app.py,requirements.txt,README.md}   # Hugging Face Space (Gradio)
docs/{PLAN,ARCHITECTURE,DATA,MODEL_CARD,ABLATIONS}.md  docs/adr/
.github/{workflows/*,dependabot.yml,CODEOWNERS}  .pre-commit-config.yaml
```

## 4. GitHub: publicación, seguridad e integridad
- **CI** (`ci.yml`): `uv sync --locked`, ruff (incl. reglas `S` bandit), mypy, pytest + cobertura, smoke-train CPU.
- **Seguridad** (`security.yml`): `pip-audit` sobre `uv export`, gitleaks (secretos), CodeQL (Python + Actions), OpenSSF Scorecard, Dependabot (pip/uv + actions).
- **Integridad**: acciones fijadas por SHA, `permissions:` mínimos, SBOM CycloneDX, `actions/attest-build-provenance` para wheel y pesos, `SHA256SUMS` de los `.safetensors`.
- **Rulesets** en `main`: PR obligatorio + 1 aprobación, checks requeridos (CI/seguridad), bloqueo de force-push y borrado, historial lineal. (Firmas de commit obligatorias: opcional, ver decisiones.)
- `SECURITY.md`, `CODEOWNERS`, release `v0.1.0` con tag.

## 5. Entrenamiento (Colab) → Export → Deploy
1. Notebook Colab: instala uv, `uv sync --group train`, descarga/filtra datos, entrena el tokenizador, preentrena (ciclo MetabolicFSM), SFT conversacional + tool-calling, evalúa, exporta.
2. **Export**: `model.safetensors` + `config.json` + `tokenizer.json` + `generation_config.json` + model card. Se descarta exportar como modelo de *diffusion*: para texto conversacional pequeño un decoder autoregresivo en safetensors es más simple y estable; diffusion-LM queda como experimento futuro (ADR).
3. **Hugging Face**: repo de modelo `<usuario>/cefomena-tiny` + Space Gradio (CPU básico) que carga los safetensors con el propio paquete. Requiere `HF_TOKEN` (secreto).

## 6. Linear (equipo **Kerna Studio / KER**)
Proyecto **Neural-CEFOMENA** con hitos e issues:
- **M0 Plan y diseño**: PLAN, ARCHITECTURE, ADRs, revisión de diseño.
- **M1 Núcleo del modelo**: tokenizer, 6 módulos, modelo, tests, ablaciones sintéticas.
- **M2 Chat + agente**: plantilla, generación, orquestador, ComputerBackend, safety.
- **M3 Publicación GitHub**: CI, seguridad, rulesets, SBOM/attestations, release v0.1.0.
- **M4 Entrenamiento Colab**: datos/licencias, notebook, run de entrenamiento, evaluación.
- **M5 Export y deploy**: safetensors + checksums, HF model repo, HF Space.

## 7. Flujo de trabajo y entregables por etapa
| Etapa | Entregable | Quién |
|---|---|---|
| 1 | Este plan + aprobación | Devin → tú |
| 2 | Proyecto Linear con hitos/issues | Devin |
| 3 | Snapshot/blueprint del entorno (uv, Python 3.11, pre-commit) | Devin (aprobación tuya) |
| 4 | `docs/ARCHITECTURE.md` + ADRs (code design) en PR para revisión | Devin → tú |
| 5 | Implementación M1–M2 en PRs pequeños con tests | Devin |
| 6 | M3: CI/seguridad/rulesets, release v0.1.0 | Devin (rulesets requieren permisos admin del token) |
| 7 | Entrenamiento en Colab con el notebook | Tú (Devin prepara y valida en CPU) |
| 8 | Export safetensors + HF Space | Devin con `HF_TOKEN` |

## 8. Riesgos
- Calidad: un modelo de ~25 M dará conversación básica limitada; se prioriza coherencia en diálogos cortos y formato correcto de tool-calls.
- Cómputo: sin GPU en el VM de Devin → solo smoke-tests CPU; entrenamiento real en Colab.
- Permisos: rulesets/Scorecard requieren permisos de administración del repo.
- Licencias de datos: se documentan y verifican antes de entrenar (no-comercial excluido por defecto).
