# Volta Q4_0 small-K MMVQ warps

The small-K Q4_0 specialization was tested with one, four, and eight warps per
block while normal-K remained at the retained two-warp setting.

| Small-K warps | Qwen 3.5 Q4_0 tg128 tok/s | Change |
| ---: | ---: | ---: |
| 1 | 308.471 | -0.78% |
| 4 | 310.894 | baseline |
| 8 | 283.826 | -8.70% |

The earlier two-warp all-K experiment was also slower. Four warps remains the
best measured small-K setting and the source was restored unchanged.
