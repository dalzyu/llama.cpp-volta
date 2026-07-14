# Volta dense Q8_0 bias MMVQ

The post-experiment-063 Qwen 3.5 Q8_0 decode trace spends 9.607182 ms
over 990 launches in the generic fused Q8_0 matrix-vector kernel. These are
30 launches per evaluated token: the FFN-down and attention-output projections
whose following F32 additions are fused into MMVQ.

The retained SM70 kernel keeps the generic kernel's two-warps-per-row work
partition, per-lane partial-sum addition, warp reduction, and final bias
addition. It removes multidimensional indexing, unused gate and scale state,
runtime fusion branches, and the second shared-memory reduction array. This
preserves the exact floating-point order while reducing static shared memory
from 256 B to 128 B.

Dispatch requires Q8_0, one output column, normal K, exact SM70, one channel,
one sample, an even dense row count, a contiguous output, and only an F32 bias
fusion operand. IDs, gates, scales, small-K, batched, strided, prompt, and all
non-Volta cases continue to use the generic path.

## Fixed-clock generation result

Candidate and control came from one binary. A temporary environment switch
disabled only the new dispatch and was removed from the retained source. All
runs exposed only the V100 UUID and used a 150 W power limit, fixed 1200 MHz SM
clock, and 877 MHz memory clock.

| Run | Dedicated bias MMVQ | Generic control |
| --- | ---: | ---: |
| Candidate A | 342.752136 | - |
| Control | - | 341.096295 |
| Candidate B | 343.478451 | - |
| Candidate mean | 343.115294 | 341.096295 |

The bracket improvement is 0.5919%. The final source without the temporary
switch measures 342.429714 tok/s over 30 repetitions.

## Kernel profile

Node-level Nsight Systems traces cover 33 token evaluations. The dedicated
kernel reduces the affected GPU time by 1.096719 ms, or 33.23 us per evaluated
token.

| Variant | Launches | Average | Total | Registers | Static shared |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic fused Q8_0 | 990 | 9.7042 us | 9.607182 ms | 56 | 256 B |
| Dedicated bias Q8_0 | 990 | 8.5964 us | 8.510463 ms | 56 | 128 B |

This is an 11.42% reduction in the target kernels. Neither kernel uses local
or stack memory.

## Rejected variants

A one-warp-per-row version was faster end to end, measuring 342.751275 tok/s
against 340.625115 tok/s, but it changed the K accumulation order. Recurrent
decode amplified the numerical difference: context-128 PPL moved from 11.9470
to 12.0368. It was rejected.

A one-warp version with two independent accumulators reproduced the generic
kernel's K partitions and byte-exact output, but measured 339.671491 tok/s
against 341.260689 tok/s, a 0.4657% regression.

Packing two exact rows into a four-warp block was neutral end to end and made
the target kernel 0.14% slower, 8.6043 us versus 8.5922 us. Loading the bias
after the dot product left register use unchanged and made the kernel 4.42%
slower, so the retained kernel prefetches it.

## Validation and scope

Active CUDA Graph and eager execution both produce byte-identical context-128
logit files against the generic control. PPL remains 11.9470 and SHA-256 is
`acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274`.

The focused fusion suite passes 330/330 cases, including the exact Q8_0 dense
bias shape and multidimensional fallbacks. Compute Sanitizer memcheck and
initcheck report zero errors. The architecture graph test passes. Qwen 3.5
Q4_0, Qwen 3.6 Q2_K, and Gemma 4 smoke tests remain outside the dispatch and
measure 328.713465, 32.177194, and 70.399513 tok/s respectively. Q8_0 pp512
measures 15104.848063 tok/s and cannot enter the one-column specialization.
