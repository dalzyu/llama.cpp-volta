# Volta deferred GDN Q/K/V finalize

Qwen 3.5 0.8B Q8_0 evaluates 18 gated delta net blocks per decode token.
The fused projection and convolution path from experiment 061 left raw
convolved Q, K, and V in alias-safe concat scratch, then launched a 32-thread
finalizer to normalize Q and K and copy V into their graph tensors. The GDN
core subsequently loaded those materialized tensors.

The retained SM70 path defers that materialization. The graph walker finds the
later GDN node and records its raw scratch pointer and L2 epsilon in graph-local
CUDA context metadata keyed by the Q tensor. At the GDN node, the specialized
core reads the three contiguous scratch sections directly. Warp zero reproduces
the original Q and K reduction lane mapping and reduction order, publishes two
normalization scales in 8 bytes of shared memory, and all four warps continue
with the existing recurrent update. V needs no transform and is read directly.
Explicit round-to-nearest multiplies preserve the old materialized F32 values.

Deferral requires exact SM70, Q8_0 fused projection and convolution, one token,
one sequence, scalar gating, 128 values per head, 16 heads, equal contiguous
Q/K/V layouts, and matching L2 epsilon. The graph scan also requires GDN to be
the sole consumer of Q, K, and V and proves that no intervening node or input
overlaps the scratch allocation. Other architectures, quantizations, prompt
batches, shapes, and consumer graphs retain the materialized path.

## Fixed-clock generation result

Candidate and control came from one binary. A temporary environment switch
disabled only raw-core deferral and was removed from the retained source. Every
run exposed only the V100 UUID and used a 150 W power limit, fixed 1200 MHz SM
clock, and 877 MHz memory clock.

| Run | Path | Q8_0 tg128 tok/s | Standard deviation |
| --- | --- | ---: | ---: |
| A | Raw Q/K/V GDN core | 395.417258 | 0.466910 |
| Control | Materialized Q/K/V | 383.519424 | 0.479900 |
| B | Raw Q/K/V GDN core | 395.205566 | 0.713873 |
| Candidate mean | Raw Q/K/V GDN core | 395.311412 | |

The bracket improvement is 3.0747%. The final switch-free binary measures
395.061994 tok/s, 3.0096% above the matched control.

Q4_0 does not enter the Q8_0 projection and convolution fusion and remains
unchanged at 346.897895 tok/s.

## Launch profile

Node-level traces cover 32 Q8_0 decode tokens.

| Path | Launches | GPU time | Average |
| --- | ---: | ---: | ---: |
| Control GDN core | 576 | 2.894529 ms | 5.0252 us |
| Control Q/K/V finalizer | 576 | 2.858785 ms | 4.9632 us |
| Raw Q/K/V GDN core | 576 | 3.300831 ms | 5.7306 us |
| Remaining finalizers | 0 | 0 ms | 0 us |

The combined measured chain falls from 5.753314 ms to 3.300831 ms, a
42.6273% reduction. It saves 76.6401 us and 18 launches per token. A final
switch-free trace measures the raw core at 3.288922 ms and again contains no
finalizer launches.

The ordinary SM70 core uses 48 registers per thread and no shared memory. The
raw-input core uses 55 registers and 8 bytes of static shared memory, with no
local or stack memory. The removed finalizer used 32 registers.

## Why the earlier design changed outcome

Experiment 062 tested the same broad launch-removal idea and rejected it. At
that point a host-side CUDA Graph bottleneck masked almost all of the measured
78.86 us per-token GPU saving. Subsequent graph and epilogue work removed that
bottleneck. The current 76.64 us measured saving now closely matches the
end-to-end latency reduction.

The retained kernel is also leaner. Experiment 062 used eight warps and
materialized normalized Q and K in 1024 bytes of shared memory. This version
keeps the established four-warp core and shares only two scales. The graph
lifetime proof is explicit rather than relying on the recognized node pattern
alone.

## Correctness and fallback validation

Candidate, disabled-control, eager, cache-disabled, and final switch-free
Q8_0 runs produce byte-identical context-128 logits. PPL remains 11.9470 and
SHA-256 is
`acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274`.
Q4_0 graph and eager output are also byte-identical, with PPL 12.6381 and
SHA-256 `385e35ba44725c79a84b3ed86297d71de16280492d40086cb04fa0ed819e14a0`.

The focused RMS, matrix-vector fusion, and GDN suite passes 387/387 cases.
Compute Sanitizer memcheck, initcheck, and synccheck report zero errors. The
architecture graph test passes, and the non-backend CTest suite passes 53/53
tests. Q8_0 pp512 remains outside the decode guard and measures 15119.869010
tok/s. Qwen 3.6 27B Q2_K and Gemma 4 12B remain outside the exact fusion and
measure 32.141744 and 70.469449 tok/s respectively.
