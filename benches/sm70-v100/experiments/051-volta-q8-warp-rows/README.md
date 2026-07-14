# Volta dense Q8_0 warp-row MMVQ

The post-experiment-050 Qwen 3.5 Q8_0 generation profile spends 359.684153 ms
over 41,856 launches in the non-fused Q8_0 MMVQ kernel. This is 34.3% of CUDA
kernel time and remains the largest individual target. The fused Q8_0 MMVQ
path accounts for another 236.414627 ms, but the retained change does not
alter that path.

For a dense single-column Q8_0 matrix-vector product, the generic Volta launch
uses two warps to cooperatively calculate each row and then reduces through
shared memory. The retained kernel instead gives one complete row to each
warp. A 64-thread block still produces two rows, but each result now needs only
a warp shuffle reduction and a lane-zero store.

Dispatch is limited to SM70, Q8_0, normal K, one output column, no fusion, no
IDs, one channel, one sample, exactly two warps, and an even dense row count.
Odd rows and all strided, batched, fused, small-K, or non-Volta cases continue
to use the generic kernel.

## Same-binary generation result

All measurements used only the V100 UUID, a 150 W power limit, fixed 1200 MHz
SM and 877 MHz memory clocks, tg128, and 20 repetitions. A temporary environment
switch disabled only the new dispatch. Alternating control and candidate runs
came from one binary, and the switch was removed from the retained source.

| Run | Generic MMVQ | Warp-row MMVQ |
| --- | ---: | ---: |
| A | 280.164819 | 283.453221 |
| B | 280.112615 | 283.430678 |
| Mean | 280.138717 | 283.441950 |

The retained path improves Qwen 3.5 0.8B Q8_0 tg128 by 1.18%. A final build
without the temporary switch measures 283.535009 tok/s.

The Qwen 3.6 27B Q2_K smoke is 31.779423 tok/s versus the preceding 31.822183
tok/s, a noise-sized -0.13%. Gemma 4 measures 68.930333 tok/s versus 68.851794,
also unchanged within run noise. The final pp512 measurements are
14410.443931 tok/s for Q8_0 and 14094.404656 tok/s for Q4_0; their cold first
samples dominate the reported deviations, and the new single-column dispatch
cannot run on this prompt path.

## Kernel profile

Across three tg128 repetitions, the specialized kernel reduces all non-fused
Q8_0 MMVQ time from 359.684153 ms to 344.809279 ms, or 4.14%. The fused MMVQ
time remains flat at 236.458765 ms. The largest output projection is mostly
bandwidth limited and falls from 313.372 us to 312.340 us per launch. Smaller
matrix averages improve by 2.2% to 19.4% depending on shape.

| Kernel | Registers | Static shared memory | Total time |
| --- | ---: | ---: | ---: |
| Generic non-fused Q8_0 MMVQ | 48 | 256 B | 359.684153 ms |
| Dedicated warp-row Q8_0 MMVQ | 56 | 0 B | 344.809279 ms |

Neither kernel uses local or stack memory. The dedicated kernel uses 384 bytes
of constant memory. Device allocations and model VRAM are unchanged.

## Rejected variants

Increasing only the generic normal-path rows per block from two to four
measures 278.514743 tok/s. It increases total non-fused Q8_0 MMVQ time to
365.197393 ms, so it was reverted.

An initial one-warp-per-row branch inside the generic kernel improves the
20-run bracket mean from 280.152702 to 282.414636 tok/s, or 0.81%, and reduces
profiled kernel time to 350.547803 ms. Moving the exact dense invariant to a
small dedicated kernel removes the unused generic indexing and fusion state
and performs better.

Constraining the dedicated kernel to enough blocks per SM to force register
use from 56 down to 48 measures 280.003242 tok/s and was reverted. Additional
occupancy is counterproductive for this memory-heavy path.

## Validation

The focused Q8_0 single-column suite passes 17/17 cases. It includes a new
17-row case that proves odd dense rows use the generic fallback. Existing
multidimensional cases prove that repeated channel and sample layouts also
fall back. Compute Sanitizer reports zero errors across the same suite.

Eight-chunk WikiText-2 estimates remain exactly 18.3641 for Q8_0 and 21.8604
for Q4_0 at the displayed precision.
