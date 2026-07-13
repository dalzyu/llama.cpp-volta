# Volta Q8_0 MMVQ occupancy launch bounds

The retained two-warp, single-column, normal-K Q8_0 MMVQ kernel used 64
registers per thread in its unfused form. Volta's 65,536-register SM therefore
limited that kernel to 16 resident blocks, or 32 warps. Minimum launch bounds
of 17 and 18 blocks per SM were tested to trade registers for occupancy.

Both bounds produced the same resource usage: 56 unfused registers and 54
fused registers, with no stack or local memory. The default used 64 and 56
registers respectively. Despite the higher theoretical residency, both
candidates reduced measured generation throughput.

| Minimum blocks per SM | Unfused registers | Fused registers | Qwen 3.5 Q8_0 tg128 tok/s | Change |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 56 | 289.048 | baseline |
| 17 | 56 | 54 | 288.574 | -0.16% |
| 18 | 56 | 54 | 288.587 | -0.16% |

The launch-bound specialization was reverted.

Reference: https://docs.nvidia.com/cuda/volta-tuning-guide/
