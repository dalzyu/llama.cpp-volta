# Volta RMS norm and Q8_1 cache fusion

Qwen 3.5 decode normalizes each layer input and FFN input with fused RMS norm
plus elementwise weight multiplication. Quantized projections then convert the
F32 result to Q8_1. The activation cache from experiment 055 removes duplicate
conversions across sibling projections, but one quantizer launch remains for
each distinct normalized tensor.

The retained path finds exact 1024-wide, one-token SM70 RMS outputs that feed a
quantized matrix-vector consumer. It allocates the existing graph-lifetime
Q8_1 cache entry before the consumer and uses one kernel to produce both the
normal F32 tensor and its cached Q8_1 representation.

The 256-thread kernel preserves the established F32 calculation and reduction
order. It stages all 1024 rounded F32 outputs in shared memory, then reproduces
the standalone quantizer's four 256-element iterations. Q8_1 max and sum
reductions therefore use the same 32-lane trees, and all quantized bytes remain
identical. The graph-visible F32 output is still written normally.

The graph walker enables prequantization only when it finds a direct quantized
one-column MUL_MAT consumer. Runtime dispatch additionally requires exact
SM70, stream zero, one contiguous 1024-element row, aligned F32 source, weight,
and destination tensors, and the active Q8_1 cache. Prompt, batched, strided,
other-width, non-quantized, non-Volta, and concurrent-stream cases retain the
existing RMS and quantizer paths.

## Fixed-clock generation results

Candidate and control came from one binary. A temporary switch disabled only
RMS prequantization and was removed from the retained source. All runs exposed
only the V100 UUID and used a 150 W limit, fixed 1200 MHz SM clock, and 877 MHz
memory clock.

| Model | Candidate A | Control | Candidate B | Candidate mean | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 Q8_0 | 359.730473 | 342.591737 | 359.661793 | 359.696133 | +4.9926% |
| Qwen 3.5 Q4_0 | 344.290729 | 331.768507 | 344.286809 | 344.288769 | +3.7738% |

The final source without temporary switches measures 360.036903 tok/s for
Q8_0 and 344.965985 tok/s for Q4_0.

## Launch profile

Node-level traces cover 33 evaluated tokens. The fusion applies to 48 of the
49 RMS-normalized activations per token. The final output normalization still
passes through GET_ROWS and retains its standalone quantizer.

| Variant | Launches | GPU time | Average | Registers | Dynamic shared |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control RMS, affected share | 1584 | 5.991418 ms | 3.7825 us | 32 | 128 B |
| Control Q8_1, affected share | 1584 | 4.349847 ms | 2.7461 us | 18 | 0 B |
| Fused RMS plus Q8_1 | 1584 | 7.286876 ms | 4.6003 us | 38 | 4224 B |

The fused path removes 1584 launches and saves 3.054389 ms across the trace,
or 92.56 us per evaluated token. It uses no static shared, local, or stack
memory.

## Shared staging variant

The first correct kernel reloaded normalized values from the F32 destination.
Its fused-kernel average was 4.7572 us. Staging the same rounded float values
in shared memory lowers this to 4.5925 us. A same-binary tg128 bracket measures
359.854513 tok/s for shared staging versus 357.405052 for the global reload,
an additional 0.6853%. Shared staging raises SM70 register use from 24 to 38
and dynamic shared memory from 128 B to 4224 B, while remaining faster.

## Correctness and fallback validation

Graph and eager Q8_0 execution produce byte-identical context-128 logit files
against the disabled control. PPL remains 11.9470 and SHA-256 remains
`acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274`.
Graph and eager Q4_0 are also byte-identical, with PPL 12.6381 and SHA-256
`385e35ba44725c79a84b3ed86297d71de16280492d40086cb04fa0ed819e14a0`.

The focused RMS and MMVQ fusion suite passes 351/351 cases. Compute Sanitizer
memcheck, initcheck, and synccheck report zero errors. The broader non-backend
CTest suite passes 53/53 cases, including the architecture graph test. Q8_0
pp512 measures 15136.740115 tok/s versus 15187.468643
for the disabled control, a noise-sized -0.33% on a path that cannot dispatch
the fusion. Qwen 3.6 Q2_K and Gemma 4 smoke tests remain outside the exact
width guard and measure 32.209873 and 70.435648 tok/s.
