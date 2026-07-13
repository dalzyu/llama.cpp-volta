# Volta Q4_0 dequantization warp packing

This experiment packed independent Q4_0 dequantization work into two and four
warps per thread block on Volta. NVIDIA documents a limit of 64 resident warps
and 32 resident blocks per Volta SM, so the original one-warp block has a 50%
occupancy ceiling. No quantization arithmetic was changed.

The experiment was rejected. The original one-warp implementation was restored.

| Model | Configuration | Repetitions | pp512 tok/s | tg128 tok/s |
| --- | --- | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q4_0 | one warp, pre-control | 10 | 14176.287 | 305.138 |
| Qwen 3.5 0.8B Q4_0 | two warps | 10 | 14099.497 | 304.620 |
| Qwen 3.5 0.8B Q4_0 | four warps | 10 | 14150.164 | 304.869 |
| Qwen 3.5 0.8B Q4_0 | one warp, post-control | 10 | 14205.612 | 305.330 |
| Gemma 4 12B Q4_0 | one warp, short control | 7 | 1495.726 | not run |
| Gemma 4 12B Q4_0 | four warps, short candidate | 7 | 1501.374 | not run |
| Gemma 4 12B Q4_0 | four warps, long candidate | 30 | 1491.879 | not run |
| Gemma 4 12B Q4_0 | one warp, long post-control | 30 | 1495.764 | not run |

The four-warp kernel itself was 1.18% faster in a paired Nsight Systems sample,
but the gain was too small to improve full-model performance. The longer Gemma
comparison regressed by 0.26%, and both Qwen candidates were below the bracketing
controls. VRAM and generation performance were unchanged within measurement
resolution.

Volta occupancy reference:
https://docs.nvidia.com/cuda/archive/12.9.0/volta-tuning-guide/index.html
