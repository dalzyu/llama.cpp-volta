# Volta GDN warp-scalar gate

The scalar Gated DeltaNet gate is identical for every lane in a column warp.
The Volta kernel now evaluates `expf` in lane 0 and broadcasts the result with
a synchronized shuffle. This removes 31 redundant exponential evaluations per
warp without changing the operation or its arithmetic result.

Measurements used the V100 UUID exclusively, a 150 W power limit, and a fixed
1200 MHz SM clock. Each end-to-end value is from 100 llama-bench repetitions.
The candidate was measured on both sides of the control; the table reports the
mean of those two candidate runs.

| Model | Control pp512 | Candidate pp512 | Change |
| --- | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q4_0 | 14197.29 | 14231.11 | +0.24% |
| Qwen 3.5 0.8B Q8_0 | 14548.76 | 14594.87 | +0.32% |

An Nsight Systems comparison reduced the GDN kernel average from 398.780 us to
384.145 us (-3.67%). An attempted follow-up that also computed `delta` in lane
0 and broadcast it increased the kernel average to 414.910 us, so it was
reverted.

All 36 GATED_DELTA_NET backend cases pass. Compute Sanitizer memcheck reports
zero errors. Eight-chunk WikiText-2 perplexity is unchanged at 21.8601 for
Q4_0 and 18.3634 for Q8_0.
