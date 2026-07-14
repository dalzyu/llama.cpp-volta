# Volta attention-gate Q8_1 epilogue

Qwen 3.5 0.8B evaluates six full-attention blocks per decode token. The
attention result is multiplied by a sigmoid gate and then consumed by a
quantized output projection. The existing SM70 `CONT -> SIGMOID -> MUL`
fusion still left a standalone Q8_1 activation quantizer before each output
projection.

The retained path finds a later compatible quantized `MUL_MAT` that consumes
the fused result directly or through an exact contiguous reshape alias. It
keys the Q8_1 cache by that consumer tensor and extends the existing fused
attention-gate kernel to write both the F32 result and its Q8_1 representation.
The launcher requires SM70, stream zero, an active Q8_1 cache, exactly 2048
contiguous F32 output elements, and the existing valid strided gate input.
Other devices, shapes, streams, prompt batches, cache-disabled runs, and
non-quantized consumers retain the previous path.

The 256-thread CTA contains eight warps and covers 256 output elements. Each
thread computes and stores one `sigmoid(gate) * attention` value while
preserving the original strided gate mapping. Each warp then reduces its 32
already-rounded F32 values for maximum magnitude and sum, and writes one Q8_1
block with the same scale, rounding, and half conversion as the standalone
quantizer. Eight CTAs cover the 2048-element vector without shared memory.

## Fixed-clock generation result

Candidate and control came from one binary. A temporary environment switch
disabled only attention-gate prequantization and was removed from the retained
source. Every run exposed only the V100 UUID and used a 150 W power limit,
fixed 1200 MHz SM clock, and 877 MHz memory clock.

| Run | Path | Q8_0 tg128 tok/s | Standard deviation |
| --- | --- | ---: | ---: |
| A | Fused gate and Q8_1 | 383.517946 | 0.547795 |
| Control | Separate quantizer | 380.369144 | 0.575244 |
| B | Fused gate and Q8_1 | 382.997390 | 0.512013 |
| Candidate mean | Fused gate and Q8_1 | 383.257668 | |

The bracket improvement is 0.7594%. The final switch-free binary measures
383.110666 tok/s, 0.7208% above the matched control.

Q4_0 uses the same activation-side Q8_1 cache and also benefits:

| Run | Path | Q4_0 tg128 tok/s | Standard deviation |
| --- | --- | ---: | ---: |
| A | Fused gate and Q8_1 | 346.907777 | 0.579069 |
| Control | Separate quantizer | 344.712021 | 0.537778 |
| B | Fused gate and Q8_1 | 347.199214 | 0.543757 |
| Candidate mean | Fused gate and Q8_1 | 347.053496 | |

The Q4_0 bracket improvement is 0.6793%.

## Launch profile

Node-level traces cover 32 Q8_0 decode tokens.

| Path | Launches | GPU time | Average |
| --- | ---: | ---: | ---: |
| Control attention gate | 192 | 0.550398 ms | 2.8667 us |
| Control Q8_1 quantizers | 192 | 0.554527 ms | 2.8882 us |
| Fused attention gate and Q8_1 | 192 | 0.640541 ms | 3.3362 us |
| Remaining Q8_1 quantizers | 0 | 0 ms | 0 us |

The combined measured chain falls from 1.104925 ms to 0.640541 ms, a 42.03%
reduction. It saves 14.512 us and six launches per token. The fused and
original SM70 kernels both use 20 registers per thread and no shared, local,
or stack memory.

## Rejected precursors

The first prototype extended the generic `UNARY -> MUL` fusion. Qwen's graph
is captured earlier by the dedicated `CONT -> SIGMOID -> MUL` path, so that
kernel never dispatched.

A literal 2048-element output guard was also insufficient. The output
projection consumes a contiguous reshape alias, and the graph-local Q8_1 cache
uses tensor identity rather than data address. Scanning for that exact consumer
alias and keying the cache by it removes the expected six quantizers per token.
The retained CTA layout follows the natural one-warp-per-Q8_1-block mapping, so
no alternate block geometry was carried forward.

## Correctness and fallback validation

Candidate, disabled-control, eager, cache-disabled, and final switch-free Q8_0
runs produce byte-identical context-128 logits. PPL remains 11.9470 and SHA-256
is `acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274`.
Q4_0 graph and eager output are also byte-identical, with PPL 12.6381 and
SHA-256 `385e35ba44725c79a84b3ed86297d71de16280492d40086cb04fa0ed819e14a0`.

The focused RMS and matrix-vector fusion suite passes 351/351 cases. Compute
Sanitizer memcheck, initcheck, and synccheck report zero errors. The architecture
graph test passes, and the non-backend CTest suite passes 53/53 tests. Q8_0
pp512 remains outside the exact decode guard and measures 15127.737108 tok/s.
Qwen 3.6 27B Q2_K and Gemma 4 12B also remain outside the 2048-element guard
and measure 32.243961 and 70.550445 tok/s respectively.
