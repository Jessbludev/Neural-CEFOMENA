# Arquitectura — Neural-CEFOMENA v0

Modelo conversacional **pequeño** (preset `tiny` ≈ 30,8 M parámetros) solo en español, con un
orquestador de agente y uso de computadora. Implementado en PyTorch puro + safetensors + NumPy;
sin Transformers, Lightning ni frameworks de agentes.

## Vista general

```
texto ──► BPETokenizer (byte-level propio) ──► ids
ids ──► Embedding ──► DualCache (mariposa) ──► [Bloques × 9 según layer_pattern] ──► RMSNorm ──► head (atado)
            N = NerveNetBlock (medusa):  RMSNorm → NerveNetMixer → RMSNorm → SparseGateMLP (delfín)
            A = AttentionBlock:          RMSNorm → CausalSelfAttention(RoPE) + FungalMemory (hongos)
                                         → RMSNorm → OscillatorMoE (coral)
Entrenamiento: MetabolicFSM (murciélago) ACTIVE → REST → SLEEP(replay) → WAKE
Inferencia:    ChatSession (plantilla <|role|> … <|end|>) + presupuesto metabólico (early-exit)
Agente:        Orchestrator FSM (IDLE→PLAN→ACT→VERIFY→RECOVER→DONE/FAILED) + SafetyPolicy + ComputerBackend
Export:        model.safetensors + config.json + tokenizer.json + generation_config.json + SHA256SUMS
```

## Mapa reporte → módulo

| Organismo | Principio (research.md) | Módulo | Archivo |
|---|---|---|---|
| Medusa | Red nerviosa descentralizada, procesamiento local | `NerveNetMixer`: conv. causal depthwise local + dilatada con compuerta; O(T) | `model/nervenet.py` |
| Delfín | Codificación temporal escasa + recompensa | `SparseGateMLP`: k-WTA (25 % activas); `reward_parameters()` expone la compuerta | `model/sparse_gate.py` |
| Coral | Osciladores locales sincronizados sin centro | `OscillatorMoE`: fases Kuramoto por experto; ruteo por coherencia de fase, sin router central | `model/oscillator_router.py` |
| Hongos | Memoria distribuida, habituación, conductancia | `FungalMemory`: slots no paramétricos con decaimiento, refuerzo, reemplazo por novedad y difusión en anillo | `model/fungal_memory.py` |
| Mariposa | Caché periférico vs. central | `DualCache`: modulación de bajo rango de los embeddings, adaptable por usuario sin tocar el núcleo (`adapt_peripheral`) | `model/dual_cache.py` |
| Murciélago | Estados metabólicos | `MetabolicFSM` + `ReplayBuffer`: fases de entrenamiento y presupuesto de inferencia | `train/metabolic_fsm.py` |
| Hongos (FSM) | Robustez por estados | `Orchestrator` con estados explícitos y recuperación | `agent/orchestrator.py` |

## Presets

| Preset | d_model | capas | contexto | vocab | parámetros |
|---|---|---|---|---|---|
| `tiny` | 384 | 9 (`NNANNANNA`) | 1024 | 8192 | ≈30,8 M |
| `micro` | 256 | 6 (`NNANNA`) | 512 | 4096 | ≈9,3 M |
| `test` | 32 | 3 (`NAN`) | 64 | 300 | ≈64 k (solo CI) |

## Datos (solo español)

Ver [DATA.md](DATA.md). Fuentes: diálogos sintéticos propios (`train/synth.py`) y el subconjunto
`lang == "es"` de OpenAssistant oasst2 (Apache-2.0). La pérdida solo se calcula sobre los turnos
del asistente (`render_ids` devuelve la máscara).

## Agente y uso de computadora

- Observación **textual** (ventana activa, posición del ratón); sin visión en v0.
- Backends: `DryRunComputer` (por defecto, registra acciones), `XdotoolComputer` (Linux), `PyAutoGUIComputer` (opcional).
- `SafetyPolicy`: allowlist de herramientas, límite de pasos, detección de texto/teclas peligrosas
  (`sudo`, `rm -rf`, `curl | sh`, `alt+f4`…), confirmación humana para acciones destructivas,
  log de auditoría JSONL.
- Aprendizaje: cada `Episode` se registra; `episodes_to_conversations` convierte episodios exitosos
  en datos SFT, y `adapt_peripheral` adapta solo el `DualCache`.

## Seguridad e integridad

Los pesos se distribuyen solo como safetensors (sin pickle). `load_pretrained` verifica
`SHA256SUMS` antes de cargar y rechaza rutas sospechosas. Ver [../SECURITY.md](../SECURITY.md).

## Decisiones

- [ADR-0001](adr/0001-pytorch-puro-y-uv.md) — PyTorch puro, gestionado con uv.
- [ADR-0002](adr/0002-hibrido-local-atencion.md) — Híbrido local/atención en lugar de Transformer puro.
- [ADR-0003](adr/0003-autoregresivo-vs-diffusion.md) — Autoregresivo en v0; diffusion como experimento.
- [ADR-0004](adr/0004-safetensors-y-checksums.md) — Safetensors + SHA256SUMS.
- [ADR-0005](adr/0005-agente-fsm-y-seguridad.md) — Agente FSM con DryRun por defecto.
