# Volta Q4_0 fused MMVQ warps

Normal-K fused Q4_0 MMVQ was isolated from the unfused two-warp setting and
tested with one warp. Small-K remained at four warps.

| Fused normal-K warps | Gemma 4 12B tg128 tok/s | Change |
| ---: | ---: | ---: |
| 1 | 69.258 | -1.57% |
| 2 | 70.364 | baseline |

Existing kernel profiles also favor two warps over four: the fused kernel
average falls from 81.988 us with four warps to 81.294 us with two. The
one-warp specialization and its launch-parameter plumbing were reverted.
