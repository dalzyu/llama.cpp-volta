# Volta recurrent tail-copy specialization

Qwen DeltaNet recurrent convolution state uses the fused graph pattern from
experiment 047 with a narrower invariant: the copy source is the complete
dim0 tail of the concat output. Its view starts one float after each row and
contains every remaining dim0 value across all higher dimensions.

The retained SM70 specialization computes each concat value once. It writes
the full concat destination, skips dim0 index zero for the recurrent copy, and
writes every other value directly to the contiguous copy destination. The
generic fused kernel instead reconstructs four-dimensional source and
destination coordinates and computes the concat value a second time for copied
elements.

Dispatch requires a one-float view offset, a view dim0 exactly one smaller than
the concat dim0, equal dimensions 1 through 3, identical higher-dimensional
strides, and a contiguous copy destination. All other fused concat-copy cases
continue to use the generic kernel.

## Generation results

All measurements used only the V100 UUID, a 150 W power limit, fixed 1200 MHz
SM and 877 MHz memory clocks, tg128, and 20 repetitions. The control and
candidate came from the same binary through a temporary switch that disabled
only the tail specialization. The switch was removed from the retained source.

| Model | Generic fusion | Tail specialization | Change |
| --- | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q8_0 | 277.347692 | 280.164533 | +1.02% |
| Qwen 3.5 0.8B Q4_0 | 292.816709 | 295.692042 | +0.98% |

After tightening the final shape guards, Q8_0 reaches 279.621035 tok/s in a
10-run confirmation. Qwen 3.6 27B Q2_K reaches 31.822183 tok/s, +0.29% over
the preceding generic fusion measurement. Gemma 4 reaches 68.851794 tok/s and
is unchanged within run noise.

## Prompt results

The specialization is not a material prompt-path optimization. Q4_0 measures
14101.118718 tok/s at pp512 versus 14100.537250 for the generic fusion. The
Q8_0 10-run mean is 14382.396506 versus 14434.103017, but both runs include a
cold first sample and have standard deviations above 2700 tok/s. The nine
steady candidate samples average about 15254.86 tok/s, so this does not provide
evidence of a prompt regression.

## Profile

Across three Qwen 3.5 Q8_0 tg128 runs, the generic fused kernel takes
41.659548 ms over 6,912 launches. The specialized kernel takes 27.596555 ms
over the same launch count, a 33.76% reduction. Average launch duration falls
from 6.0271 us to 3.9926 us.

The SM70 specialized kernel uses 36 registers per thread versus 40 for the
generic fused kernel. Both use 256 threads and no shared, local, or stack
memory. Device allocations and model VRAM use are unchanged.

## Validation

The permanent backend test covers rows and sequence planes `(1,1)`, `(7,2)`,
and `(128,3)`; all three CUDA cases pass their CPU references. The focused
82-case fusion sweep and the full 370-case CONCAT/CPY sweep pass. Compute
Sanitizer reports zero errors for both the multidimensional backend test and a
Q8_0 pp128/tg16 model run.

Eight-chunk WikiText-2 estimates remain exactly 18.3641 for Q8_0 and 21.8604
for Q4_0 at the displayed precision.
