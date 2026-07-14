# Rejected Volta raw-QKV GDN fusion

Experiment 061 leaves a small finalize launch between the fused Q8 projection
and convolution kernel and gated delta net. The finalize kernel normalizes Q
and K and copies V from alias-safe concat scratch into their graph outputs.

This experiment removed that launch. The projection kernel left raw convolved
Q/K/V in concat scratch, and the GDN kernel read the scratch directly. One
warp per GDN block reproduced the original 32-thread L2 reduction, while the
other warps waited at a block barrier. The final variant wrote normalized Q
and K to 1 KiB of shared memory and used eight warps per block. It was limited
to exact SM70, Q8_0, one token, one sequence, scalar gating, 128-wide Q/K/V,
one rollback slot, the recognized Qwen graph, and disjoint scratch, state,
output, and cache ranges. All other graphs retained experiment 061.

The design was rejected. It removes 18 graph nodes per generated token and
clearly reduces traced GPU work, but two opposite fixed-clock brackets do not
show a repeatable end-to-end improvement.

## Fixed-clock results

All runs exposed only the V100 UUID and used 1200 MHz SM, 877 MHz memory, and
the 150 W power limit. Candidate and control came from one binary;
`GGML_CUDA_DISABLE_GDN_RAW_FUSION=1` restored experiment 061.

| Order | First | Middle | Last | Bracket result |
| --- | ---: | ---: | ---: | ---: |
| Candidate/control/candidate, 50 each | 342.426828 | 341.543655 | 341.819308 | +0.1696% |
| Control/candidate/control, 30 each | 341.572206 | 341.718610 | 341.954334 | -0.0131% |

An earlier four-warp implementation measured +0.1031% in a 50-run
candidate/control/candidate bracket. With CUDA Graphs disabled, it measured
337.521557 versus 337.076797 tokens/s, or +0.1320%. These small results agree
with the final reversed bracket: the saved GPU work is not on a stable
end-to-end critical path.

## Node-level profile

Nsight Systems node tracing covered 33 token evaluations, or 594 recurrent
layers. The finalizer disappears, while normalization makes GDN itself about
8% slower. Combined affected GPU time falls by 2.602380 ms, which is 78.86 us
per token evaluation.

| Variant | Kernel | Launches | Total | Average |
| --- | --- | ---: | ---: | ---: |
| Control | GDN | 594 | 3.008415 ms | 5.0647 us |
| Control | Q/K/V finalizer | 594 | 2.853350 ms | 4.8036 us |
| Candidate | Raw-input GDN | 594 | 3.259385 ms | 5.4872 us |

The raw-input block-size sweep gave average GDN times of 5.6481 us for four
warps, 5.5037 us for eight, 5.5498 us for sixteen, and 5.7738 us for thirty-two.
The shared-normalization version reduced the eight-warp result to 5.4872 us.
It uses 48 registers and 1024 bytes of static shared memory on SM70, with no
local or stack memory. The direct four-warp version used 56 registers.

## Correctness

The active CUDA Graph path and eager path both produced byte-identical
context-128 all-logit files against experiment 061. PPL remained 11.9470 and
SHA-256 remained
`acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274`.

The source changes were reverted. This directory preserves the negative
result so the same launch-removal design is not repeated without first
removing the host-side graph bottleneck that masks its GPU-time saving.
