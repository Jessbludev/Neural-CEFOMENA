# Política de seguridad

## Versiones soportadas

| Versión | Soporte |
|---|---|
| 0.1.x | ✅ |

## Reportar una vulnerabilidad

No abras un issue público. Usa **GitHub → Security → Report a vulnerability** (avisos privados) en
este repositorio. Respondemos en un plazo de 7 días.

## Medidas

- **Pesos**: solo `safetensors` (sin pickle). `load_pretrained` verifica `SHA256SUMS` antes de cargar.
- **Supply chain**: `uv.lock` con hashes y `uv sync --locked` en CI; Dependabot (uv + Actions);
  `pip-audit`; dependency-review en PRs; actions fijadas por SHA.
- **Código**: CodeQL (`security-extended`), Ruff con reglas `S` (bandit), mypy estricto.
- **Secretos**: gitleaks en CI y pre-commit. `HF_TOKEN` solo como variable de entorno o secreto de Colab/GitHub.
- **Integridad de releases**: SBOM CycloneDX, `SHA256SUMS` y attestations de procedencia
  (`gh attestation verify <archivo> -R Jessbludev/Neural-CEFOMENA`).
- **Agente**: backend `dryrun` por defecto; acciones reales con allowlist, límite de pasos,
  confirmación humana para acciones peligrosas y log de auditoría.

## Ruleset de `main`

[`.github/rulesets/main.json`](.github/rulesets/main.json): PR obligatorio, checks requeridos
(`lint-type-test`, `pip-audit`, `gitleaks`, `codeql`, `dependency-review`), historial lineal, sin
force-push ni borrado. Importar en **Settings → Rules → Rulesets → New ruleset → Import a ruleset**.
Aprobaciones requeridas = 0 porque el repo tiene un solo mantenedor (GitHub no permite aprobar el
propio PR); súbelo a 1 cuando haya más revisores.
