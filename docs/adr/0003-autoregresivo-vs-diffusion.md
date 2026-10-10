# ADR-0003: Autoregresivo en v0; diffusion como experimento futuro

- Estado: aceptado
- Fecha: 2026-10-10

## Contexto
Se pidió considerar un modelo de diffusion para la exportación a Hugging Face.

## Decisión
v0 es un decoder causal autoregresivo. Los modelos de diffusion de texto (difusión discreta /
enmascarada) necesitan más datos y cómputo para igualar la calidad conversacional a escala
pequeña, y complican el tool-calling, que requiere JSON exacto. El formato de exportación
(safetensors + config) no depende del tipo de modelo, así que un experimento de diffusion futuro
se podría publicar con el mismo pipeline.

## Consecuencias
+ Entrenamiento y generación simples y estables en Colab.
− Sin generación paralela; se revisará después de v0.1.
