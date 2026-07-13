# Volta Q3_K two-warp MMVQ

| Revision | tg128 tok/s |
| --- | ---: |
| Baseline | 27.721 |
| Q3_K two warps | 28.034 |

Q3_K alone improves tg128 by 1.1%, but it is slower than changing Q2_K and Q3_K
together. This isolated variant was rejected in favor of the combined setting.
