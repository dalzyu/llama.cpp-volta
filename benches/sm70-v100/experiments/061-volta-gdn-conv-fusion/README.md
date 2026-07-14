# Volta Q8 GDN projection and convolution fusion

Qwen 3.5 decode originally runs the grouped Q8_0 QKV/alpha/beta projection,
copies QKV into a four-tap convolution window while updating recurrent state,
runs depthwise convolution plus SiLU, and normalizes Q and K. These small
kernels are separated by graph storage that is useful for general shapes but
expensive for one-token SM70 decode.

The retained path uses two ordered launches:

1. The grouped projection kernel computes QKV, alpha, and beta. For each QKV
   row, it also evaluates the four-tap convolution and writes the shifted raw
   projection values directly to persistent convolution state. Convolved SiLU
   values go into the concat allocation as temporary storage.
2. A 32-thread finalize kernel applies the original L2 reduction to Q and K and
   copies V into their final graph allocations.

The intermediate is required for correctness. The allocator reuses the old
convolution-state allocation for final convolution output, so writing final
Q/K/V during the projection launch would race blocks that have not read their
old state. Stream ordering between the two launches provides the global phase
boundary without a cooperative kernel.

Dispatch requires exact SM70, stream zero, one-column F32 decode, contiguous
Q8_0 QKV/alpha/beta weights, a contiguous three-value old-state row, a
contiguous four-tap F32 convolution kernel, equal contiguous Q/K/V layouts,
matching L2 epsilon, and the exact 22-node Qwen graph subset. The graph walker
validates use counts, external outputs, pairwise output ranges, all first-phase
inputs, and allocations live between projection and convolution. Prompt,
multi-sequence, rollback, non-Q8_0, non-four-tap, and concurrent-stream graphs
fall back. `GGML_CUDA_DISABLE_GDN_CONV_FUSION=1` restores experiment 059.

## Fixed-clock generation result

Candidate and control use one binary. Every run exposes only the V100 UUID and
reports 1200 MHz SM, 877 MHz memory, and a 150 W power limit.

| Model | Repetitions | Candidate A | Disabled control | Candidate B | Candidate mean | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q8_0 | 50 | 341.417834 | 325.950756 | 340.835835 | 341.126834 | +4.6559% |

## Launch profile

Each Nsight Systems trace covers tg32. Across 36 recurrent evaluations, four
control kernel families become two candidate families. This removes 72
submissions and reduces affected GPU time by 0.172421 ms.

| Variant | Kernel family | Launches | GPU time |
| --- | --- | ---: | ---: |
| Control | Grouped QKV/GDN projection | 36 | 0.408421 ms |
| Control | Concat plus state copy | 36 | 0.147552 ms |
| Control | SSM convolution plus SiLU | 36 | 0.083009 ms |
| Control | Paired Q/K L2 norm | 36 | 0.091713 ms |
| Candidate | Projection, convolution, state, alpha/beta | 36 | 0.442434 ms |
| Candidate | Q/K normalize plus V finalize | 36 | 0.115840 ms |

The candidate's first kernel uses the same 48 registers as the prior grouped
projection. The finalizer uses 32 registers. Neither uses shared, local, or
stack memory on SM70.

## Correctness and scope

Candidate and disabled-control all-logit files are byte-identical at context
128 with PPL 11.9470 and SHA-256
`acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274`.
The same identity holds with CUDA Graphs disabled, exercising eager launches
and 128 recurrent-state updates in both modes.

Compute Sanitizer memcheck and initcheck report zero errors. A prompt trace
contains neither new kernel, Q4_0 remains on its standalone GDN projection
fusion, and Qwen 3.6 Q2_K remains on the generic path. The architecture graph
test passes across all model builders. All 54 CTest cases pass, including the
205.92-second exhaustive CUDA backend operation test.
