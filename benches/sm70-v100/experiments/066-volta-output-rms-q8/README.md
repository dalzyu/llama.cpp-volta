# Volta output RMS, GET_ROWS, and Q8 fusion

Experiment 065 prequantizes 48 RMS-normalized Qwen 3.5 activations per decode
token. The final output normalization remained a three-launch chain: fused RMS
norm and weight multiplication, one-row GET_ROWS, then Q8_1 quantization for
the quantized vocabulary projection.

The retained SM70 path recognizes an exact 1024-element RMS_NORM, MUL, and
GET_ROWS subgraph whose selected result feeds a quantized one-column MUL_MAT.
The source and selected tensors both contain one row, so GET_ROWS can only
select row zero. One kernel now writes the normal `h_nextn` output, writes the
selected output, and creates the Q8_1 cache entry used by the vocabulary
projection.

Qwen keeps both F32 outputs live, and its allocator aliases `h_nextn` with the
RMS input. The exact kernel is safe in place because all 256 threads load four
source floats before the block reduction barrier and no output store precedes
that barrier. The general fusion overlap check is therefore not used for this
one-block pattern. The graph still verifies that every elided node is confined
to the subgraph.

Dispatch requires exact SM70, matching contiguous F32 tensors of shape
`[1024, 1, 1, 1]`, one I32 row index, and a later direct quantized MUL_MAT
consumer. The existing runtime cache checks still require stream zero,
alignment, and an active Q8_1 cache before prequantizing. If those checks fail,
the normal fused RMS kernel writes `h_nextn`, a device copy produces the
one-row selected result, and the later matvec performs its normal quantization.

The established 48-launch kernel retains its original five-argument launch
signature. A second wrapper with an inlined common body handles the additional
F32 output. This preserves the original SM70 resource layout for experiment
065 instead of adding an optional pointer and branch to every layer launch.

## Fixed-clock generation results

Candidate and control came from one binary. A temporary switch disabled only
the output-chain fusion and was removed from the retained source. All runs
exposed only the V100 UUID and used a 150 W limit, fixed 1200 MHz SM clock, and
877 MHz memory clock.

| Model | Candidate A | Control | Candidate B | Candidate mean | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 Q8_0 | 360.492777 | 359.693468 | 360.216319 | 360.354548 | +0.1838% |
| Qwen 3.5 Q4_0 | 344.977529 | 344.510790 | 344.816862 | 344.897196 | +0.1122% |

The final switch-free source measures 360.534984 tok/s for Q8_0 and
344.977036 tok/s for Q4_0.

## Launch profile

Node-level traces cover 33 evaluated tokens. The control runs three kernels
per final output while the retained path runs one.

| Path | Launches | GPU time | Average |
| --- | ---: | ---: | ---: |
| Control final RMS and multiply | 33 | 0.135332 ms | 4.1010 us |
| Control one-row GET_ROWS | 33 | 0.148099 ms | 4.4878 us |
| Control Q8_1 quantizer | 33 | 0.240969 ms | 7.3021 us |
| Fused RMS, copy, and Q8_1 | 33 | 0.168997 ms | 5.1211 us |

The final chain falls from 0.524400 ms to 0.168997 ms across the trace. It
saves 10.77 us and two launches per token, a 67.77% reduction for this chain.
The other 48 RMS-Q8 launches remain in their original no-copy kernel.

## Kernel variants

The first candidate kept the generic memory overlap check and never
dispatched because `h_nextn` aliases the RMS input. Its 359.708355 tok/s mean
matched the 359.704692 control. Instrumentation confirmed that the subgraph
and shapes were valid and only the conservative overlap test rejected it.

An optional output pointer inside the existing kernel proved the fusion and
measured +0.2484% for Q8_0 and +0.3226% for Q4_0. It also changed the launch
ABI for all 48 established calls. A compile-time branch removed the runtime
test but retained the enlarged parameter list. The final two-wrapper design
restores the original no-copy signature and isolates the extra pointer to the
single output launch.

Both final SM70 kernels use 38 registers, 4224 B dynamic shared memory, no
static shared memory, and no local or stack memory. The original kernel keeps
388 B of constant parameters; the copy wrapper uses 396 B.

## Correctness and fallback validation

Graph, eager, and cache-disabled Q8_0 execution produce byte-identical
context-128 logits against the control. PPL remains 11.9470 and SHA-256 is
`acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274`.
Graph and eager Q4_0 are also byte-identical, with PPL 12.6381 and SHA-256
`385e35ba44725c79a84b3ed86297d71de16280492d40086cb04fa0ed819e14a0`.

The focused RMS and matrix-vector fusion suite passes 351/351 cases.
Compute Sanitizer memcheck, initcheck, and synccheck report zero errors. The
broader non-backend CTest suite passes 53/53 cases, including the architecture
graph test.

Q8_0 pp512 measures 15134.773745 tok/s versus 15153.718556 for the disabled
control, a noise-sized -0.13% on a shape that cannot dispatch. Qwen 3.6 Q2_K
and Gemma 4 remain outside the width guard and measure 32.283918 and
70.572804 tok/s.
