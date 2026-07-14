# Volta fused gate output Q8_1 cache

Qwen 3.5 Q8_0 evaluates 24 dense FFN blocks per decode token. Each block used
the dedicated fused up, gate, and SWIGLU kernel from experiment 053, then
launched a separate Q8_1 quantizer before the quantized down projection.

The retained graph path marks a fused GLU output for prequantization only when
a later compatible quantized MUL_MAT directly consumes that tensor. The MMVQ
dispatcher then applies a second exact guard for SM70, Q8_0 up and gate
weights, SWIGLU without bias, scale, or IDs, one contiguous input column of
width 1024, 3584 output rows, stream zero, and the active Q8_1 cache. All
other shapes and cache-disabled execution retain the existing kernels.

One 512-thread CTA owns each 32-row Q8_1 block. Its 16 warps each compute two
independent up and gate rows with the established dot helper and reduction
order. Lane zero stores both F32 results and copies them to shared memory.
After a block barrier, warp zero quantizes the 32 already-rounded F32 values
with the same max, sum, rounding, and half conversion as the standalone
quantizer. The later down projection finds this entry through the existing
tensor-keyed cache.

## Fixed-clock generation result

Candidate and control came from one binary. A temporary environment switch
disabled only gate-output prequantization and was removed from the retained
source. Every run exposed only the V100 UUID and used a 150 W power limit,
fixed 1200 MHz SM clock, and 877 MHz memory clock.

| Run | Path | tg128 tok/s | Standard deviation |
| --- | --- | ---: | ---: |
| A | Fused gate and Q8_1 | 362.254966 | 1.119384 |
| Control | Separate quantizer | 360.370883 | 0.884018 |
| B | Fused gate and Q8_1 | 362.027394 | 0.546416 |
| Candidate mean | Fused gate and Q8_1 | 362.141180 | |

The candidate mean improves Qwen 3.5 0.8B Q8_0 tg128 by 0.491%. The final
switch-free binary measures 361.733397 tok/s, 0.378% above the matched
control. Q4_0 does not enter the Q8_0 gate guard and measures 344.884775
tok/s. Q8_0 pp512 also cannot dispatch and measures 15119.051243 tok/s.

## Launch profile

Node-level traces cover 33 evaluated tokens. The candidate removes 792 Q8_1
launches, exactly 24 per token.

| Path | Launches | GPU time | Average |
| --- | ---: | ---: | ---: |
| Control fused gate | 792 | 12.555806 ms | 15.8533 us |
| Control all remaining Q8_1 quantizers | 1584 | 4.608968 ms | 2.9097 us |
| Fused gate and Q8_1 | 792 | 13.563331 ms | 17.1254 us |
| Candidate remaining Q8_1 quantizers | 792 | 2.553827 ms | 3.2245 us |

The combined measured chain falls from 17.164774 ms to 16.117158 ms, a
6.10% reduction. It saves 31.75 us and 24 launches per token. The fused
kernel uses 64 registers per thread, 128 B shared memory, and no local or
stack memory on SM70.

## Rejected CTA layouts

The quantized output requires all 32 values to meet at one CTA. Four layouts
were measured before selecting the final mapping.

| Layout | Gate average | SM70 registers | End-to-end result |
| --- | ---: | ---: | ---: |
| 32 warps, one row per warp | 18.2838 us | 38 | -0.606% |
| 16 warps, two rows per warp | 17.1028 us | 64 | +0.491% retained |
| 8 warps, four rows per warp | 17.4643 us | 80 | rejected |
| Two rows with explicit activation reuse | 17.4600 us | 62 | rejected |

The 1024-thread layout preserves one warp per row but reduces useful SM70
residency. Four rows per warp loses too much row-level parallelism. Explicitly
loading the shared activation values once lowers registers but schedules worse
than four normal inlined dot calls.

## Correctness and fallback validation

Graph, eager, cache-disabled, and disabled-control Q8_0 execution produce
byte-identical context-128 logits. PPL remains 11.9470 and SHA-256 is
`acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274`.
Q4_0 graph and eager output are also byte-identical, with PPL 12.6381 and
SHA-256 `385e35ba44725c79a84b3ed86297d71de16280492d40086cb04fa0ed819e14a0`.

The focused RMS and matrix-vector fusion suite passes 351/351 cases.
Compute Sanitizer memcheck, initcheck, and synccheck report zero errors. The
architecture graph test passes, and the non-backend CTest suite passes 53/53
tests. Qwen 3.6 27B Q2_K and Gemma 4 12B remain outside the guard and measure
32.563470 and 70.511625 tok/s respectively.
