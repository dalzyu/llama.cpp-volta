# Volta GDN gate-chain fusion

Qwen 3.5 and Qwen 3.6 form each gated-delta-net decay gate as three graph
operations:

`mul(softplus(add(alpha, dt_bias)), ssm_a)`

During token generation all three F32 tensors have the same contiguous shape.
The existing CUDA graph fusion already combined softplus and multiply, but the
preceding add still launched separately and materialized an intermediate.

The retained SM70 path recognizes the exact `ADD -> SOFTPLUS -> MUL` chain and
computes the expression in one kernel. Dispatch requires F32, equal shapes,
contiguous inputs and output, a single-use chain, and safe non-overlapping graph
memory ranges. Other shapes, types, operations, and architectures keep the
existing paths. In particular, prompt processing uses broadcast operands and
does not select this fusion.

## Generation results

All measurements used only the V100 UUID, a 150 W power limit, fixed 1200 MHz
SM and 877 MHz memory clocks, and tg128. Candidate A and B bracket a temporary
same-binary control in which only this fusion was disabled. That temporary
switch was removed from the retained source.

| Model | Candidate A | Control | Candidate B | Candidate mean change |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q8_0 | 273.306879 | 270.477029 | 273.367274 | +1.06% |
| Qwen 3.5 0.8B Q4_0 | 288.063833 | 284.440593 | 288.214076 | +1.30% |

Qwen 3.6 27B Q2_K reaches 31.501124 tok/s, +0.42% against the preceding
31.368633 tok/s bracket mean. Gemma 4 does not construct this graph pattern;
its tg128 smoke is 69.071792 tok/s and does not launch the fused kernel.

## Prompt results

The prompt graph retains its broadcast add and standalone softplus-multiply
path. Q8_0 reaches 15198.037782 tok/s and Q4_0 reaches 14856.672468 tok/s at
pp512, respectively -0.012% and -0.023% against the preceding measurements.
These differences are noise-sized and confirm that the restricted dispatch
does not perturb prompt processing.

## Profile

Across three Qwen 3.5 Q8_0 tg128 runs, the control gate chain launches 13,824
add kernels totaling 35.929 ms and 6,912 softplus-multiply kernels totaling
13.072 ms. The retained path launches 6,912 remaining add kernels totaling
17.247 ms and 6,912 fused gate kernels totaling 12.857 ms. The chain therefore
drops 6,912 launches and its measured GPU time falls from 49.001 ms to
30.105 ms, a 38.56% reduction.

The SM70 fused kernel uses 14 registers per thread and no shared, local, or
stack memory. It introduces no device allocation and does not change model
VRAM use.

## Validation

All 36 gated-delta-net reference cases and all four supported standalone
softplus cases pass their CPU references. Compute Sanitizer memcheck reports
zero errors for a Q8_0 pp128/tg16 run through the model graph.

Eight-chunk WikiText-2 estimates remain exactly 18.3641 for Q8_0 and 21.8604
for Q4_0 at the displayed precision, matching the preceding commit.
