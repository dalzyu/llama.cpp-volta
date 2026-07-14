# Volta dense Q8_0 row-base addressing

The dedicated warp-row kernel from experiment 051 originally passed the full
matrix pointer and `row*stride + k` to the generic Q8_0 dot helper on every
loop iteration. The retained change forms a typed pointer to the selected
Q8_0 row once and passes only the K-block index to the helper.

This arithmetic simplification lowers the SM70 kernel from 56 to 48 registers
per thread without an occupancy directive. Shared, local, and stack memory
remain zero. The dot products, accumulation order, launch geometry, and output
stores are unchanged.

## Same-binary generation result

All measurements used only the V100 UUID, a 150 W power limit, fixed 1200 MHz
SM and 877 MHz memory clocks, tg128, and 30 repetitions. One temporary binary
contained both kernels and selected the old address calculation through an
environment variable. The control kernel and switch were removed afterward.

| Run | Offset per dot | Precomputed row base | Change |
| --- | ---: | ---: | ---: |
| A | 283.669929 | 284.148437 | +0.17% |
| B | 283.617414 | 284.091844 | +0.17% |
| Mean | 283.643672 | 284.120141 | +0.168% |

The final source without the temporary switch measures 283.992210 tok/s for
Qwen 3.5 0.8B Q8_0 and 295.894044 tok/s for Q4_0. Qwen 3.6 27B Q2_K reaches
31.859643 tok/s. Gemma 4 measures 68.757832 tok/s; it does not dispatch the
specialized Q8_0 kernel and remains within normal run variation.

## Profile and resources

Across three tg128 runs, total dense non-fused Q8_0 MMVQ time falls from
344.809279 ms to 343.340241 ms, or 0.43%, over 41,856 launches. The fused Q8_0
kernel is unchanged.

| Variant | Registers | Shared memory | Kernel total |
| --- | ---: | ---: | ---: |
| Per-dot row offset | 56 | 0 B | 344.809279 ms |
| Precomputed row base | 48 | 0 B | 343.340241 ms |

Both variants use 384 bytes of constant memory and no local or stack memory.
Device allocations and model VRAM are unchanged.

## Rejected launch and instruction variants

One warp per output row remains the best work assignment, but two rows per
64-thread block are required to avoid Volta's 32-block-per-SM residency limit.
Packing more independent warps per block also loses latency hiding.

| Variant | Q8_0 tg128 | Q8_0 kernel total | Kernel change | Result |
| --- | ---: | ---: | ---: | --- |
| One row per 32-thread block | 266.749373 | 427.764392 ms | +24.06% | rejected |
| Two rows per 64-thread block | 283.535009 | 344.809279 ms | control | retained geometry |
| Four rows per 128-thread block | 282.757853 | 347.078574 ms | +0.66% | rejected |
| Eight rows per 256-thread block | 281.970379 | 351.015413 ms | +1.80% | rejected |

A two-row-per-warp kernel explicitly reused each Q8_1 activation load across
two weight rows. It dropped to 48 registers but reduced warp-level parallelism,
measuring 281.330450 tok/s and 354.430746 ms (+2.79%).

A two-iteration software pipeline used independent accumulators to overlap
global loads and DP4A work. It stayed at 56 registers but measured 281.740252
tok/s and 350.803262 ms (+1.74%). Both variants were reverted.

## Validation

The focused Q8_0 single-column backend suite passes 17/17 cases, including the
odd-row and multidimensional generic fallbacks. Compute Sanitizer reports zero
errors over the same suite.

Eight-chunk WikiText-2 estimates remain exactly 18.3641 for Q8_0 and 21.8604
for Q4_0 at the displayed precision.
