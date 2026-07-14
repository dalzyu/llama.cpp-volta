# Volta Q4_0 half2 dequantization and launch packing

This candidate changes only the `dst_t=half` Q4_0 row dequantizer used by
CUDA prompt prefill. Each thread reads four quantized bytes with two aligned
16-bit transfers, subtracts the Q4_0 zero point with packed-byte arithmetic,
and writes four `half2` values. Float and BF16 destinations keep the original
path. Float products are converted to half with the same rounding operation as
the control. The Volta launch packs two independent Q4_0 work units into each
64-thread block; odd block counts are guarded in the kernel.

The comparison used 15 benchmark repetitions and omitted the first cold
sample from the mean and standard deviation. The V100 was pinned by UUID and
held at its existing 150 W power limit.

| Model | PP512 control | PP512 candidate | Change | TG128 control | TG128 candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q4_0 | 14011.421 +/- 427.879 | 14793.329 +/- 412.785 | +5.58% | 310.863 +/- 0.115 | 310.945 +/- 0.116 |
| Gemma 4 12B Q4_0 | 1495.140 +/- 7.104 | 1684.107 +/- 8.760 | +12.64% | 70.207 +/- 0.148 | 70.014 +/- 0.207 |

The prompt measurements use 15 repetitions and the generation measurements use
10 repetitions; the first cold sample is omitted from each mean and standard
deviation. TG is unchanged within run variance because this path is used for
prompt dequantization, not single-token MMVQ. The intermediate one-warp half2
launch measured 14473.107 tok/s on Qwen and 1600.315 tok/s on Gemma. A
four-warp launch measured about 14701 and 1681 tok/s, respectively, and was
rejected as slower than two warps.

The Q4_0 focused backend suite passed 14/14 cases. Qwen 3.5 PPL remained
`21.8601 +/- 1.50861`; Gemma 4 PPL remained `308.3287 +/- 38.71784`.
Compute Sanitizer memcheck on a Qwen 3.5 prompt run reported zero errors.
The sm_70 cubin remains spill-free at 20 registers per thread; the half
kernel reduced global load instructions from 5 to 3 and stores from 8 to 4.

The simpler half2-store-only variant and an FP16 multiply variant were also
tested and rejected as slower or neutral. An unaligned 32-bit load variant was
rejected after full-model CUDA errors.
