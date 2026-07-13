# Volta Q8_0 fused MMVQ warps

Normal-K fused Q8_0 MMVQ was isolated from the retained two-warp setting and
tested with one warp. Unfused normal-K remained at two warps and small-K
remained at four.

| Fused normal-K warps | Qwen 3.5 Q8_0 tg128 tok/s | Change |
| ---: | ---: | ---: |
| 1 | 286.680 | -0.82% |
| 2 | 289.048 | baseline |

The global four-warp result was already substantially slower. Two warps remains
the best measured setting, and the one-warp specialization was reverted.
