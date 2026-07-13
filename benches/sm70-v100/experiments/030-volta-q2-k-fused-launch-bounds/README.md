# Volta fused Q2_K MMVQ launch bounds

A CUDA decode trace identified fused Q2_K as the largest remaining kernel in
Qwen 3.6 27B, accounting for 27.4% of kernel time. Its retained two-warp form
used 52 registers, allowing 19 resident blocks on a Volta SM.

A fused-only minimum of 20 blocks emitted a 48-register kernel with no stack or
local memory. A 22-block bound emitted 40 registers but introduced a 16-byte
stack frame and regressed substantially.

| Variant | Fused registers | Fused stack | Qwen 3.6 27B tg128 tok/s |
| --- | ---: | ---: | ---: |
| Q3_K bound only | 52 | 0 | 29.453, 29.501 |
| Q3_K plus Q2_K 20 blocks | 48 | 0 | 29.764, 29.659 |
| Q3_K plus Q2_K 22 blocks | 40 | 16 | 29.170 |

The paired 20-block runs improved full-model throughput by 0.53% to 1.06%.
Matching CUDA traces reduced the fused Q2_K average kernel time from 131.158 us
to 126.228 us, a 3.76% kernel-level improvement. The 20-block bound is
retained. Focused K-quant MUL_MAT tests passed 20/20, and eight-chunk Qwen 3.6
perplexity remained exactly 7.1732.

Reference: https://docs.nvidia.com/cuda/volta-tuning-guide/
