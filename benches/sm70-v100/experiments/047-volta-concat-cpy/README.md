# Volta concat and recurrent-state copy fusion

Qwen 3.5 DeltaNet layers build a convolution state by concatenating the new
state with the previous state, taking a strided view of the result, and copying
that view into a recurrent-state buffer:

`CONCAT(dim0) -> VIEW -> CPY`

The retained SM70 path computes the contiguous dim0 concat and the strided
F32 copy in one kernel. It is restricted to Volta, F32 tensors, contiguous
concat inputs and output, a direct view of that output, element-count matching,
float-aligned strides, and safe graph memory ranges. Other layouts, types, and
architectures keep the existing operations.

## Generation results

All measurements used only the V100 UUID, a 150 W power limit, fixed 1200 MHz
SM and 877 MHz memory clocks, and tg128. The control was a same-binary run
with a temporary environment switch disabling only this fusion; the switch was
removed from the retained source before the final build.

| Model | Candidate | Control | Change |
| --- | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q8_0 | 276.771896 | 274.248012 | +0.92% |
| Qwen 3.5 0.8B Q4_0 | 292.282380 | 289.129528 | +1.09% |

The Qwen 3.6 27B Q2_K smoke reaches 31.730442 tok/s. Gemma 4 reaches
68.732620 tok/s; neither model showed a regression outside normal run noise.

## Prompt results

The fusion is not a material prompt-path optimization for this graph. At pp512,
Q8_0 measures 14431.210032 tok/s versus 14434.103017 control (-0.02%), and
Q4_0 measures 14102.552059 versus 14100.537250 (+0.01%).

## Profile

Across three Qwen 3.5 Q8_0 tg128 runs, the control launches 6,912
`concat_cont` kernels totaling 21.629026 ms and 6,912 `cpy_scalar` kernels
totaling 24.893079 ms. The fused path launches 6,912
`concat_cpy_dim0_f32` kernels totaling 41.652533 ms. The chain falls from
46.522105 ms to 41.652533 ms, a 10.47% reduction, while its launch count falls
from 13,824 to 6,912.

The SM70 fused kernel uses 256 threads, 40 registers per thread, and no shared,
local, or stack memory. Device allocations and model VRAM use are unchanged.

## Validation

The permanent `CONCAT_CPY` backend test covers rows 1, 7, and 128; all 3 CUDA
and CPU cases pass. The focused CUDA sweep passes 82/82 cases covering
`CONT_SIGMOID_MUL`, `SIGMOID`, `CONT`, `GATED_DELTA_NET`, and `CONCAT_CPY`.
The full CUDA `CONCAT,CPY` sweep passes 370/370 cases. Compute Sanitizer
memcheck reports zero errors for a Q8_0 pp128/tg16 run.

Eight-chunk WikiText-2 estimates remain exactly 18.3641 for Q8_0 and 21.8604
for Q4_0 at the displayed precision.
