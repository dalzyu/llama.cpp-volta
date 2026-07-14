# Volta dense Q8_0 fused gate rows

The Qwen 3.5 Q8_0 generation profile after experiment 052 spends
146.430842 ms over 9,216 launches in the dense fused gate MMVQ kernel. The
generic kernel gives two warps one output row, reduces through shared memory,
and evaluates both the value and gate projections inside each warp group.

The retained SM70 kernel instead assigns one complete output row to each warp.
Two independent rows share a 64-thread block, but each warp accumulates its
value and gate dot products, reduces both with shuffles, and lets lane zero
apply SiLU and write the result. The kernel uses typed row-base pointers and no
shared memory.

Dispatch is limited to SM70, Q8_0, normal K, one output column, no IDs, one
channel, one sample, an even dense row count, no bias or scale, and a SWIGLU
gate. Bias, scale, GEGLU, MoE, batched, multidimensional, odd-row, small-K,
non-Q8_0, and non-Volta cases continue to use the generic kernel.

## Same-binary generation result

All measurements used only the V100 UUID, a 150 W power limit, fixed 1200 MHz
SM and 877 MHz memory clocks, tg128, and 20 repetitions. One temporary binary
contained a switch that disabled only the dedicated gate dispatch. The switch
was removed from the retained source.

| Run | Generic fused MMVQ | Dedicated warp-row gate |
| --- | ---: | ---: |
| A | 284.236788 | 284.823653 |
| B | 283.334704 | 284.503289 |
| Mean | 283.785746 | 284.663471 |

The retained path improves Qwen 3.5 0.8B Q8_0 generation by 0.309%. The final
source without the temporary switch measures 285.319915 tok/s over 30 runs.
Its pp512 result is 14410.195735 tok/s; the single-column dispatch cannot run
on the prompt path.

Q4_0 measures 295.240177 tok/s, Qwen 3.6 27B Q2_K measures 31.779982 tok/s,
and Gemma 4 measures 68.850927 tok/s. These controls are unchanged within
normal run variation.

## Profile and resources

Across three tg128 runs, the dedicated gate kernel takes 137.106348 ms over
9,216 launches versus 146.430842 ms for the generic kernel, a 6.37% reduction.
The unchanged fused bias kernel takes 91.826014 ms in the final profile. Total
time across those two fused Q8_0 kernels falls by 3.18%.

| Kernel | Registers | Shared memory | Constant memory | Total time |
| --- | ---: | ---: | ---: | ---: |
| Generic fused Q8_0 MMVQ | 56 | 256 B | 512 B | 146.430842 ms |
| Dedicated warp-row gate | 48 | 0 B | 392 B | 137.106348 ms |

Neither kernel uses local or stack memory. Device allocations and model VRAM
are unchanged.

## Rejected variants

A matching dedicated warp-row bias kernel regresses its kernel total from
90.027923 ms to 91.986443 ms, or 2.18%, and was removed. The generic kernel
remains better for the smaller single-projection bias workload.

The full block-packing sweep retains two rows per block. Each result below is
from a matched profile comparison at the same launch count.

| Rows per block | Average gate kernel | Change | Result |
| --- | ---: | ---: | --- |
| 1 | 14.9072 us | +0.21% | rejected |
| 2 | 14.8753 us | control | retained |
| 4 | 14.9005 us | +0.26% | rejected |
| 8 | 15.2519 us | +2.63% | rejected |

The one-row and two-row shapes were also alternated from one binary. Their
end-to-end means are 284.860213 and 284.923623 tok/s respectively, a
noise-sized 0.02% advantage for two rows that agrees with the longer kernel
profile.

Experiment 042 already rejected sharing activation work across pairs of rows,
splitting value and gate projections across warps, and adding two-result ILP.
Those designs reduce registers but give up useful warp-level parallelism.

## Validation

All 30 Q8_0 matrix-vector fusion cases pass, including gate, bias, GEGLU,
SWIGLU, ID, broadcast, batched, and one-sample generic fallbacks. Compute
Sanitizer reports zero errors over the same suite.

Eight-chunk WikiText-2 estimates remain exactly 18.3641 for Q8_0 and 21.8604
for Q4_0 at the displayed precision.
