# Volta Q8_0 multi-row MMVQ

This experiment assigns two adjacent output rows to each normal-K, non-fused,
single-column Q8_0 MMVQ block on SM70. The existing two warps cooperate on
both rows, so the launch uses half as many x-grid blocks and can reuse the
same quantized activation data across two weight rows. Small-K, fused, other
quantization types, and other architectures retain their existing geometry.

## Row sweep

All measurements used only the V100 UUID, a 150 W power limit, fixed 1200 MHz
SM and 877 MHz memory clocks, and tg128. An initial sweep applied the row count
to both fused and non-fused Q8_0 kernels.

| Rows per block | Qwen 3.5 Q8_0 | Qwen 3.5 Q4_0 | Result |
| ---: | ---: | ---: | --- |
| 1 | 248.167 | 267.079 | control |
| 2 | 261.225 | 276.740 | refine |
| 3 | 261.396 | 276.630 | refine |
| 4 | 258.617 | 276.477 | reject |

The fused Q8_0 kernel regressed from 11.419 to 11.946 us with two rows. The
specialization was therefore limited to the non-fused kernel. This raised the
two-row Q8_0 result to 262.631 tok/s in the first refined run. Three non-fused
rows reached 262.787 tok/s, but grid-shape profiles showed 1% to 3% losses on
several smaller matrices and no gain on the dominant output projection. Two
rows were retained as the more robust configuration.

## Controlled results

The final values below compare the one-row control with the mean of two
two-row candidate runs bracketing the control.

| Model | Control | Candidate mean | Change |
| --- | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q8_0 tg128 | 248.166668 | 262.326062 | +5.71% |
| Qwen 3.5 0.8B Q4_0 tg128 | 267.079265 | 276.683254 | +3.60% |

Q4_0 benefits because its large output projection is Q8_0. Prompt processing
does not use this single-column specialization.

Nsight Systems shows the non-fused Q8_0 kernel average falling from 10.671 to
8.600 us (-19.40%). The 248320-row output-projection grid falls from 442.014
to 313.417 us (-29.09%). The fused kernel remains on one row and measures
11.398 us, effectively unchanged from its 11.419 us control.

On SM70, the retained non-fused kernel changes from 64 registers and 128 bytes
of shared memory per block to 48 registers and 256 bytes. It uses no stack or
local memory. The fused kernel remains at 56 registers and 256 bytes of shared
memory. No device allocation changes, so model VRAM use is unchanged.

## Cross-model checks

The Qwen 3.6 27B Q2_K model executes the affected Q8_0 output path. Its two
candidate tg128 runs were 31.262973 and 31.008424 tok/s around a 31.065052
control, for a candidate mean 0.23% above control. Gemma 4 contains no
affected Q8_0 MMVQ launch; its candidate mean was 0.09% below control and is
treated as neutral run noise.

## Validation

The focused Q8_0 single-column backend suite passes 16/16 CPU-reference cases,
including batched and permuted layouts. A temporary 17-row case also passed to
exercise the final partial two-row block. Compute Sanitizer memcheck reports
zero errors on a Q8_0 generation run.

Eight-chunk WikiText-2 estimates remain 18.3635 for Q8_0 and 21.8608 for Q4_0.
The arithmetic and reduction order within each output row are unchanged.
