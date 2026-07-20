# Volta `sm_70` optimization branch

This branch is an experimental llama.cpp fork for NVIDIA Tesla V100 GPUs. It
contains retained CUDA kernel optimizations, rejected experiment records,
saved benchmark evidence, and the Git history used to develop them. It is not
an upstream-supported release.

The retained fast paths are selected by compute capability and, where needed,
exact shape and layout guards. Unsupported cases continue through the existing
generic paths. The extensive validation in this branch was performed on
V100-SXM2-16GB hardware; behavior on other devices has not received equivalent
coverage.

## Build

A minimal release build for Volta is:

```sh
cmake -S . -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

Use the normal llama.cpp command-line tools from `build/bin`. Benchmark
results are meaningful only when the clocks, power, cooling, model hashes, and
arguments match the protocol recorded with the result.

## Implementation map

| Area | Source |
| --- | --- |
| Quantized matrix-vector geometry, row packing, and fused exact-shape paths | [`mmvq.cu`](ggml/src/ggml-cuda/mmvq.cu), [`mmvq.cuh`](ggml/src/ggml-cuda/mmvq.cuh), [`vecdotq.cuh`](ggml/src/ggml-cuda/vecdotq.cuh) |
| Gated DeltaNet kernels and CUDA graph dispatch/fusions | [`gated_delta_net.cu`](ggml/src/ggml-cuda/gated_delta_net.cu), [`ggml-cuda.cu`](ggml/src/ggml-cuda/ggml-cuda.cu) |
| Volta FlashAttention tile selection and attention integration | [`fattn-tile.cuh`](ggml/src/ggml-cuda/fattn-tile.cuh), [`fattn-common.cuh`](ggml/src/ggml-cuda/fattn-common.cuh), [`fattn-mma-f16.cuh`](ggml/src/ggml-cuda/fattn-mma-f16.cuh), [`fattn.cu`](ggml/src/ggml-cuda/fattn.cu) |
| Vectorized and fused support kernels | [`getrows.cu`](ggml/src/ggml-cuda/getrows.cu), [`norm.cu`](ggml/src/ggml-cuda/norm.cu), [`rope.cu`](ggml/src/ggml-cuda/rope.cu), [`concat.cu`](ggml/src/ggml-cuda/concat.cu), [`unary.cu`](ggml/src/ggml-cuda/unary.cu) |
| CPU-reference and architecture coverage | [`test-backend-ops.cpp`](tests/test-backend-ops.cpp) |

The implementation should be read together with its experiment record. Those
records document the dispatch condition, benchmark control, correctness gates,
and prototypes that were removed after regressions or sanitizer failures.

## Results and evidence

- [Main V100 results and retained optimization index](benches/sm70-v100/README.md)
- [Active dual-V100 benchmark protocol](benches/sm70-v100/remote-dual-v100/README.md)
- [Experiment archive](benches/sm70-v100/experiments/)
- [Incomplete historical branch comparison](benches/sm70-v100/comparisons/001-origin-master-e8f19cc0a/README.md)
- [Publication normalization notes](benches/sm70-v100/PUBLICATION.md)

The retained single-V100 checkpoint recorded a complete 12,995/12,995 CUDA
backend pass and zero Compute Sanitizer errors over the full Q8_0 target graph.
Later experiment records repeat focused and complete validation as appropriate.
Each result remains tied to the exact revision and protocol named in its record.

The current dual-V100 protocol uses fixed 892 MHz graphics clocks at 150 W per
GPU. Historical single-V100 results used different power and clock rules and
must not be compared as if they came from the same benchmark environment.

## Repository status

The branch preserves its development sequence rather than squashing the
experiments into one patch. Its branch-only history was rewritten once to
remove environment identifiers from every snapshot; the publication notes map
legacy benchmark revision strings to their rewritten equivalents. It is
maintained as a research fork, not as one combined upstream contribution.
Upstream llama.cpp documentation remains the authority for general build and
usage instructions.
