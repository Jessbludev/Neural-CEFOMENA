# ADR-0005: Agente con FSM y DryRun por defecto

- Estado: aceptado
- Fecha: 2026-10-10

## Decisión
El orquestador es una máquina de estados explícita (`PLAN → ACT → VERIFY → RECOVER`) con política
inyectable (el modelo o un guion para tests). Usa observaciones textuales y el backend `dryrun` por
defecto. Las acciones reales requieren `--backend xdotool|pyautogui`, pasan por `SafetyPolicy`
(allowlist, límite de pasos, patrones peligrosos, confirmación humana) y quedan registradas en un
log de auditoría.

## Consecuencias
+ Comportamiento comprobable y reproducible; los episodios se pueden reutilizar como datos.
− Sin visión: el agente depende de lo que el backend pueda describir como texto.
