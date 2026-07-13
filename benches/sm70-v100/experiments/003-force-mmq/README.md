# Gemma 4 12B forced MMQ at 150 W

This experiment was rejected.

| Setting | pp512 tok/s | tg128 tok/s | PP peak VRAM | TG peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| default | 1496.957 | 69.493 | 7582 MiB | 6966 MiB |
| `GGML_CUDA_FORCE_MMQ=1` | 1489.376 | 69.126 | 7582 MiB | 6966 MiB |

Forced MMQ is about 0.5% slower on both paths and did not reduce observed peak
VRAM for this model.
