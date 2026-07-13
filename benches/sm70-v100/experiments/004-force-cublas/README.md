# Gemma 4 12B forced cuBLAS at 150 W

This experiment was rejected.

| Setting | pp512 tok/s | tg128 tok/s | PP peak VRAM | TG peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| default | 1496.957 | 69.493 | 7582 MiB | 6966 MiB |
| `GGML_CUDA_FORCE_CUBLAS=1` | 1494.640 | 69.440 | 7582 MiB | 6966 MiB |

The forced result is neutral-to-negative and does not change observed VRAM.
