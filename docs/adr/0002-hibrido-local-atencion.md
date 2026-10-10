# ADR-0002: Híbrido local/atención

- Estado: aceptado
- Fecha: 2026-10-10

## Contexto
El reporte describe redes descentralizadas (medusa), codificación escasa (delfín) y sincronía sin
centro (coral). El modelo debe ser pequeño y barato en contexto largo.

## Decisión
Patrón `NNANNANNA`: 6 bloques `NerveNet` (convoluciones causales locales y dilatadas, O(T)) y
3 bloques de atención global con RoPE. Las MLP usan k-WTA; los bloques de atención usan un MoE
con ruteo por osciladores acoplados en lugar de un router softmax central.

## Consecuencias
+ Coste por token menor que un Transformer del mismo ancho; inductive bias local adecuado para charla.
− Ruteo por osciladores es experimental: se valida con ablaciones (`docs/ABLATIONS.md`, KER-11).
