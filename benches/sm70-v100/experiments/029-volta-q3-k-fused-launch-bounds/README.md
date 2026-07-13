# Volta fused Q3_K MMVQ launch bounds

Qwen 3.6 27B Q2_K contains 80 Q3_K tensors. The retained two-warp,
single-column, normal-K Q3_K MMVQ kernels used 72 registers unfused and 90
registers fused. The fused kernel was limited to 11 resident blocks by Volta's
65,536-register SM.

Minimum blocks per SM were swept for the Volta fused kernel. Bounds of 12 and
14 emitted the same spill-free 72-register kernel. Bounds of 15 and 16 emitted
64 registers. A bound of 17 introduced a 40-byte stack frame, and a bound of 20
introduced a 72-byte stack frame.

| Variant | Fused registers | Fused stack | Qwen 3.6 27B tg128 tok/s | Change |
| --- | ---: | ---: | ---: | ---: |
| Default | 90 | 0 | 28.780 | baseline |
| Fused, 14 blocks | 72 | 0 | 29.453 | +2.34% |
| Fused, 16 blocks | 64 | 0 | 29.227 | +1.55% |
| Combined, 16 blocks | 64 | 0 | 29.344 | +1.96% |
| Unfused, 16 blocks | 90 | 0 | 28.777 | -0.01% |
| Combined, 17 blocks | 56 | 40 | 28.149 | -2.19% |
| Combined, 20 blocks | 48 | 72 | 27.210 | -5.45% |

The fused-only 14-block bound is retained. Focused Q3_K MUL_MAT tests passed
2/2, and eight-chunk Qwen 3.6 perplexity remained exactly 7.1732.

Reference: https://docs.nvidia.com/cuda/volta-tuning-guide/
