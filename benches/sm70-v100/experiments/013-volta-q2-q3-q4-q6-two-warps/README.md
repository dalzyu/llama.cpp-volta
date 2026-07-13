# Volta Q2_K, Q3_K, Q4_K, and Q6_K two-warp MMVQ

This extends the two-warp Volta candidate to the four K-quant kernels observed
in the Qwen 3.6 27B profile.

| Revision | tg128 tok/s | Change |
| --- | ---: | ---: |
| Baseline | 27.721 | - |
| Q2_K and Q3_K | 28.640 | +3.3% |
| Q2_K, Q3_K, Q4_K, and Q6_K | 28.954 | +4.4% |

This is the fastest measured setting and is the retained candidate. A paired
rebuild comparison measured 27.835 tok/s with four warps and 28.826 tok/s with
two warps, a 3.56% gain at the same peak temperature.

## Correctness

Focused `test-backend-ops` coverage passed 20/20 Q2_K, Q3_K, Q4_K, and Q6_K
single-column matrix multiplication cases against the CPU reference. The broad
suite passed the modified cases, then aborted later in an unrelated 320x256
FlashAttention shared-memory configuration; the failure is recorded under
experiment 010.

All final eight-chunk WikiText-2 perplexity estimates exactly match baseline:

| Model | Baseline PPL | Candidate PPL |
| --- | ---: | ---: |
| Gemma 4 12B Q4_0 | 308.3287 | 308.3287 |
| Qwen 3.5 0.8B Q4_0 | 21.8601 | 21.8601 |
| Qwen 3.5 0.8B Q8_0 | 18.3634 | 18.3634 |
| Qwen 3.6 27B Q2_K | 7.1732 | 7.1732 |

Peak Qwen 3.6 benchmark VRAM remains 12326 MiB.
