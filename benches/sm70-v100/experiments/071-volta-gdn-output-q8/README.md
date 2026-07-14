# Volta fused GDN output Q8_1 cache

Qwen 3.5 Q8_0 evaluates 18 GDN blocks per decode token. Experiment 069
fused each gate projection with its SiLU and normalized multiplier epilogue.
The following output projection still launched a standalone Q8_1 quantizer for
the 2048-element result.

The retained SM70 path scans for a later compatible quantized MUL_MAT that
directly consumes the fused result or a contiguous reshape alias of it. The
Q8_1 cache is keyed by that exact consumer tensor. The launcher retains its
existing Q8_0 1024x2048 gate-projection guard, one F32 input column, contiguous
2048-element multiplier and output, stream zero, and active Q8_1 cache. All
other shapes, types, prompt batches, streams, cache-disabled runs, and devices
retain the previous path.

One 512-thread CTA owns each 32-value Q8_1 block. Its 16 warps compute two
projection rows apiece with the established dot helper and reduction order.
Lane zero applies SiLU, multiplies by the normalized value, stores both F32
results, and copies them to shared memory. After a block barrier, warp zero
quantizes the 32 already-rounded values with the same maximum, sum, rounding,
and half conversion as the standalone quantizer.

## Fixed-clock generation result

Candidate and control came from one binary. A temporary environment switch
disabled only GDN-output prequantization and was removed from the retained
source. Every run exposed only the V100 UUID and used a 150 W power limit,
fixed 1200 MHz SM clock, and 877 MHz memory clock.

| Run | Path | tg128 tok/s | Standard deviation |
| --- | --- | ---: | ---: |
| A | Fused epilogue and Q8_1 | 380.415212 | 0.505933 |
| Control | Separate quantizer | 373.704914 | 0.769818 |
| B | Fused epilogue and Q8_1 | 380.288567 | 0.508571 |
| Candidate mean | Fused epilogue and Q8_1 | 380.351889 | |

The bracket improvement is 1.7787%. The final switch-free binary measures
380.159605 tok/s, 1.7272% above the matched control.

## Launch profile

Node-level traces cover 32 decode tokens.

| Path | Launches | GPU time | Average |
| --- | ---: | ---: | ---: |
| Control GDN epilogue | 576 | 4.380905 ms | 7.6057 us |
| Control Q8_1 quantizers | 768 | 2.534210 ms | 3.2998 us |
| Fused GDN epilogue and Q8_1 | 576 | 4.337222 ms | 7.5299 us |
| Remaining Q8_1 quantizers | 192 | 0.553792 ms | 2.8843 us |

The combined measured chain falls from 6.915115 ms to 4.891014 ms, a 29.27%
reduction. It saves 63.25 us and 18 launches per token. The fused SM70 kernel
uses 44 registers per thread, 128 B shared memory, and no local or stack
memory. The original epilogue uses 48 registers and no shared memory.

## Rejected precursors

The first prototype cached Q8_1 under the pre-reshape result tensor. The output
projection consumes an exact contiguous reshape alias, and the graph-local
cache uses tensor identity rather than data address, so the new kernel did not
dispatch. Keying the cache by the consumer alias removes the expected 18
quantizers per token.

A 32-warp, one-row-per-warp CTA increased row parallelism but doubled the
thread footprint. It measured 379.202036 tok/s and 8.0764 us per fused kernel,
versus 380.351889 tok/s and 7.5299 us for the retained 16-warp layout.

## Correctness and fallback validation

Candidate, disabled-control, eager, cache-disabled, and final switch-free Q8_0
runs produce byte-identical context-128 logits. PPL remains 11.9470 and SHA-256
is `acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274`.
Q4_0 graph and eager output are also byte-identical, with PPL 12.6381 and
SHA-256 `385e35ba44725c79a84b3ed86297d71de16280492d40086cb04fa0ed819e14a0`.

The focused RMS and matrix-vector fusion suite passes 351/351 cases. Compute
Sanitizer memcheck, initcheck, and synccheck report zero errors. The architecture
graph test passes, and the non-backend CTest suite passes 53/53 tests. Q4_0
tg128 and Q8_0 pp512 remain outside the guard and measure 344.502257 and
15141.017913 tok/s. Qwen 3.6 27B Q2_K and Gemma 4 12B also remain outside the
guard and measure 32.372567 and 70.531521 tok/s respectively.
