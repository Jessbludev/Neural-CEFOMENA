# ADR-0001: PyTorch puro, gestionado con uv

- Estado: aceptado
- Fecha: 2026-10-10

## Contexto
Se pidió una base con la menor cantidad posible de frameworks pesados, gestionada directamente con `uv`.

## Decisión
Dependencias de ejecución: `torch`, `numpy`, `safetensors`. Extras opcionales: `hub`
(`huggingface_hub`) y `space` (`gradio`). Tokenizador, entrenamiento, plantilla de chat y agente
son código propio. `uv.lock` se versiona y la CI usa `uv sync --locked`. En local se usa la rueda
CPU de PyTorch (`[[tool.uv.index]] pytorch-cpu`); en Colab se usa el torch con CUDA ya instalado.

## Consecuencias
+ Superficie de dependencias pequeña y auditable; código fácil de leer.
− Hay que mantener a mano el tokenizador y el bucle de entrenamiento.
