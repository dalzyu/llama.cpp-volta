# Volta K-quant one-warp MMVQ

Q2_K, Q3_K, Q4_K, and Q6_K were tested at one warp after the retained
four-to-two-warp improvement. Each type was isolated, and all four were also
tested together.

| One-warp types | Qwen 3.6 27B tg128 tok/s | Change |
| --- | ---: | ---: |
| none | 28.989 | baseline |
| Q2_K | 28.681 | -1.06% |
| Q3_K | 28.847 | -0.49% |
| Q4_K | 28.690 | -1.03% |
| Q6_K | 28.674 | -1.09% |
| Q4_K and Q6_K | 28.660 | -1.14% |
| all four | 28.519 | -1.62% |

Every one-warp variant regressed. Two warps remains the best measured setting
for all four K-quant types, and the source was restored.
