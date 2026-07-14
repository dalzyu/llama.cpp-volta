# Volta GDN parameter projection fusion

Qwen 3.5 linear-attention layers independently project one decode activation
into small alpha and beta vectors. The original CUDA graph then applies alpha
bias, softplus, alpha scale, and beta sigmoid in separate kernels. Launch
latency dominates these small operations on SM70.

The retained path evaluates both quantized projections in one two-warp grid
and writes the final gate and sigmoid outputs directly. Q8_0 assigns one row
to each warp. Q4_0 lets both warps cooperate on one row through 128 bytes of
shared memory. Both variants reuse the graph-local Q8_1 activation cache.

Dispatch requires exact SM70, stream zero, one-column F32 decode, matching
Q8_0 or Q4_0 model weights, K of at least 1024 elements, contiguous F32
bias/scale/output tensors, and the exact nine-node GDN parameter subgraph.
The graph walker validates use counts, dependencies, and output memory ranges
before eliding any nodes. `GGML_CUDA_DISABLE_GDN_PROJ_FUSION=1` restores the
original launches.

## Fixed-clock generation results

Candidate and control use one binary. All runs expose only the V100 UUID and
pin it to 1200 MHz SM, 877 MHz memory, and the 150 W power limit.

| Model | Repetitions | Candidate A | Disabled control | Candidate B | Candidate mean | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q8_0 | 30 | 322.587142 | 310.056667 | 320.953334 | 321.770238 | +3.7779% |
| Qwen 3.5 0.8B Q4_0 | 50 | 331.541609 | 318.426579 | 331.271588 | 331.406598 | +4.0763% |

The host function rejects multi-column input, so prompt matrix paths and
their kernels are unchanged.

## Launch profiles

Each Nsight Systems trace covers a Qwen 3.5 tg32 run. In every GDN layer, one
new kernel replaces two projection kernels, the fused alpha
add/softplus/multiply kernel, and the beta sigmoid kernel. Across the trace,
36 launches replace 144 launches, saving 108 submissions.

| Quant | New kernel | New time | Removed projection launches | Removed elementwise launches | Estimated affected time saved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | 36 | 0.099040 ms | 72 | 72 | 0.202527 ms |
| Q4_0 | 36 | 0.104096 ms | 72 | 72 | 0.291298 ms |

The affected-time estimate subtracts the candidate and control totals for the
shared ordinary projection kernel, then includes the eliminated alpha and
beta elementwise kernels. Unrelated kernels remain outside the calculation.

On SM70, the Q8_0 fusion uses 48 registers and no shared, local, or stack
memory. The Q4_0 fusion uses 48 registers and 128 bytes of shared memory, with
no local or stack memory.

## Rejected precursor designs

The first design grouped the alpha projection into the existing large QKV
launch. It proved that early alpha evaluation was memory-safe and produced
byte-identical logits, but tied a small GDN optimization to the attention
kernel.

| Variant | Model | Candidate | Control | Change | Result |
| --- | --- | ---: | ---: | ---: | --- |
| Flattened QKV plus alpha | Q8_0 | 315.685692 | 310.578996 | +1.64% | superseded |
| Flattened QKV plus alpha | Q4_0 | 313.322352 | 318.498326 | -1.63% | rejected |
| Alpha embedded in QKV blocks | Q4_0 | 314.741094 | 319.223873 | -1.40% | rejected |

For Q4_0, extending the large QKV grid cost more GPU time than the saved
launch. Moving both small projections into their own fused kernel removes the
launches without perturbing QKV.

## Correctness and safety

Candidate and disabled-control runs save byte-identical all-logit files.

| Model | Context | SHA-256 | PPL |
| --- | ---: | --- | ---: |
| Qwen 3.5 Q8_0 | 128 | `acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274` | 11.9470 |
| Qwen 3.5 Q4_0 | 128 | `385e35ba44725c79a84b3ed86297d71de16280492d40086cb04fa0ed819e14a0` | 12.6381 |
| Qwen 3.6 Q2_K | 64 | `5790cafdade18db8b653eb9d0e0fae27e0e62f0614f37f101cc809e9a859e0e3` | 9.7193 |

Qwen 3.6 uses unsupported projection types and remains on the original path.
Focused one-column quantized matrix coverage passes 51/51. Compute Sanitizer
reports zero errors for both new kernel specializations. The all-target build
passes, and the full CTest suite passes 54/54, including the 206.55-second
exhaustive CUDA backend test.
