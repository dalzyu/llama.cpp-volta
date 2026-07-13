# Volta gated delta net warp tuning

The Qwen 3.5 prompt path was tested with two, four, and eight warps per gated
delta net block. Four warps remains the baseline. Both alternatives reduced
end-to-end prompt throughput and increased the dominant kernel duration.

| Warps | Qwen 3.5 Q4_0 pp512 tok/s | Qwen 3.5 Q8_0 pp512 tok/s | Kernel average us |
| ---: | ---: | ---: | ---: |
| 2 | 14005.110 | 14917.640 | 390.854 |
| 4 | 14047.727 | 14960.000 | 382.906 |
| 8 | 13956.696 | not run | 399.774 |

The two-warp result was 0.30% slower for Q4_0 and 0.28% slower for Q8_0.
Eight warps was 0.65% slower for Q4_0. The source and binary were restored to
four warps after the sweep.

The four-warp kernel timing comes from the existing retained-build profile and
has 18 instances. The two- and eight-warp profiles each contain 54 instances.
