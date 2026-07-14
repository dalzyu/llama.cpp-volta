# Volta grouped Q8 GDN projections

Qwen 3.5 recurrent layers project one decode activation into a combined QKV
tensor and, later in the graph, into the GDN alpha and beta parameters. The
standalone GDN projection fusion from experiment 058 already reduces alpha and
beta to one launch. For Q8_0, extending the earlier QKV launch to cover those
two projections removes one more launch per recurrent layer.

The retained kernel uses one warp per output row and a two-warp block. QKV
rows write their unmodified dot products. Alpha rows apply bias, softplus, and
scale before writing the final gate. Beta rows apply sigmoid before writing the
final beta tensor. All three matrices reuse the graph-local Q8_1 activation.

Dispatch requires exact SM70, stream zero, one-column F32 decode, contiguous
Q8_0 model weights, even row counts, and the exact Qwen GDN graph structure.
The graph walker validates dependencies and allocation-sized memory ranges
before computing the later gate and beta outputs early. Concurrent graph
streams and prompt matrix paths are excluded. `GGML_CUDA_DISABLE_GROUPED_GDN_Q8=1`
restores the standalone projection fusion from experiment 058.

## Fixed-clock generation result

Candidate and control use one binary. Every run exposes only the V100 UUID.
During measurement, the V100 reports 1200 MHz SM, 877 MHz memory, and a 150 W
power limit.

| Model | Repetitions | Candidate A | Disabled control | Candidate B | Candidate mean | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q8_0 | 50 | 326.549627 | 321.812637 | 325.831081 | 326.190354 | +1.3603% |

## Launch profile

Each Nsight Systems trace covers a tg32 run. Across 36 recurrent-layer
evaluations, the candidate replaces 36 ordinary QKV launches and 36 standalone
GDN launches with 36 grouped launches. It therefore removes 36 submissions.

| Variant | Ordinary Q8 launches | Standalone GDN launches | Grouped launches | Affected GPU time |
| --- | ---: | ---: | ---: | ---: |
| Disabled control | 110 | 36 | 0 | 0.474001 ms |
| Candidate | 74 | 0 | 36 | 0.408944 ms |

The control's affected time uses the 0.374669 ms difference in ordinary Q8
kernel totals plus the 0.099332 ms standalone GDN total. The candidate saves an
estimated 0.065057 ms of GPU kernel time. The larger end-to-end improvement is
consistent with removing CPU-side launch work from every recurrent layer.

On SM70, the grouped kernel uses 48 registers and no shared, local, or stack
memory.

## Correctness and safety

Candidate and disabled-control Q8_0 runs save byte-identical logits at context
128. Both files have SHA-256
`acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274`
and PPL 11.9470.

Compute Sanitizer reports zero errors. All 54 CTest cases pass, including the
architecture graph test and the 205.37-second exhaustive CUDA backend operation
test. Q4_0 does not enter the new host path and continues to use the standalone
GDN fusion.
