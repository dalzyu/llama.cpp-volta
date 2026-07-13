# Gemma 4 12B FP16 cuBLAS compute at 150 W

This experiment was rejected before the perplexity gate.

| Setting | pp512 tok/s | tg128 tok/s | PP peak VRAM | TG peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| default | 1496.957 | 69.493 | 7582 MiB | 6966 MiB |
| `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1` | 1502.496 | 69.377 | 7582 MiB | 6966 MiB |

The +0.37% PP and -0.17% TG changes are within run variation. There is no
credible performance benefit to justify reduced accumulation precision.
