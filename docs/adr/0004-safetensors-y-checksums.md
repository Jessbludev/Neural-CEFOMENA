# ADR-0004: Safetensors + SHA256SUMS

- Estado: aceptado
- Fecha: 2026-10-10

## Decisión
Los pesos solo se exportan como `model.safetensors` (los pesos atados se guardan una sola vez
con `save_model`). Cada exportación incluye `SHA256SUMS`, que `load_pretrained` verifica antes de
cargar. Nunca se usa `torch.load` sobre archivos que no sean de confianza. Los releases de GitHub
añaden SBOM CycloneDX y attestations de procedencia.

## Consecuencias
+ No se ejecuta código al cargar; se detecta la manipulación de los archivos.
