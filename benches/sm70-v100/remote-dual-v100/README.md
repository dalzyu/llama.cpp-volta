# Remote dual-V100 benchmark protocol

This is the active protocol for the 2x Tesla V100-SXM2-16GB optimization
campaign. It supersedes the 300 W single-card rules in the parent README.
Experiment 083 established the safety and clock rules on 2026-07-20.
UUIDs and machine paths use the placeholders defined in
[`PUBLICATION.md`](../PUBLICATION.md).

## Fixed identities

| Item | Value |
| --- | --- |
| Candidate revision | `c55f89a84aad4e3003f40bef232cc9764732d083` |
| Upstream control | `e8f19cc0ad70a243c8012bf17b4be601abfc8ea2` |
| GPU 0 | `${V100_GPU_0_UUID}` |
| GPU 1 | `${V100_GPU_1_UUID}` |
| CUDA device code | `sm_70` only |
| Driver | 580.159.03 |
| CUDA compiler | 12.9.86 |
| Default power | 150 W per GPU |
| Hard power ceiling | 175 W per GPU |
| Canonical graphics clock | 892 MHz |
| Memory clock | 877 MHz |

The target input hashes are recorded in
[experiment 083](../experiments/083-dual-v100-allreduce-recovery/raw/input-sha256.txt).
The two main files are:

| Model | SHA-256 |
| --- | --- |
| Gemma 4 31B QAT Q4_0 | `9188a71055550f1e60b875d02b7abb63625ac11b4a6f148d6b22b3b28ba3d335` |
| Qwen 3.6 27B Q4_K_M | `6ed4d66ab779518031a617d64798085aee524ce460a621578060a130ef76a855` |

Every dual-GPU workload must expose exactly the two UUIDs above. A single-GPU
diagnostic must select one of them explicitly. Do not use numeric device
indices as the identity boundary.

## Why the command line follows the code

The benchmark front end does not parse `-ts` like the shared common argument
parser:

- [`llama-bench.cpp`](../../../tools/llama-bench/llama-bench.cpp) first splits
  a parameter sweep on commas, then splits one tensor allocation on `/` or
  `;`. Therefore an equal two-GPU benchmark uses `-ts 1/1`.
- [`common/arg.cpp`](../../../common/arg.cpp), used by tools such as
  `llama-perplexity`, accepts comma or slash inside one tensor split. The
  established perplexity commands use `-ts 1,1`.

Using `-ts 1,1` with `llama-bench` creates separate one-device sweep entries.
The failed and corrected commands are preserved under experiment 083.

Tensor mode is also a distinct execution path, not a faster spelling of layer
mode:

- [`common/arg.cpp`](../../../common/arg.cpp) labels tensor split experimental.
- [`src/llama.cpp`](../../../src/llama.cpp) builds a meta device from the
  selected physical devices.
- [`src/llama-context.cpp`](../../../src/llama-context.cpp) requires
  FlashAttention for tensor split.
- [`ggml-backend-meta.cpp`](../../../ggml/src/ggml-backend-meta.cpp) tries the
  CUDA backend all-reduce and uses its generic fallback when that call returns
  false.
- [`ggml-cuda.cu`](../../../ggml/src/ggml-cuda/ggml-cuda.cu) selects NCCL,
  internal all-reduce, or the generic fallback from
  `GGML_CUDA_ALLREDUCE`. On Linux, a build that actually links NCCL tries NCCL
  by default.
- [`allreduce.cu`](../../../ggml/src/ggml-cuda/allreduce.cu) implements the
  CUDA-only, two-device internal path with mapped pinned host staging. Its
  default F32 wire conversion is BF16. Setting
  `GGML_CUDA_AR_BF16_THRESHOLD=0` disables that conversion.
- [`ggml-cuda/CMakeLists.txt`](../../../ggml/src/ggml-cuda/CMakeLists.txt)
  controls whether `libggml-cuda` is linked to NCCL. A cache option set to ON
  is not proof of linkage; verify with `ldd`.

The NCCL implementation in `ggml-cuda.cu` reduces tensors below 32768 elements
as FP32 on two GPUs and converts larger F32 tensors to BF16 for the collective.
That reduction-order and precision behavior is why layer and tensor quality
results are kept as separate baselines.

Saved-logit comparisons also follow the implementation. In
[`tools/perplexity/perplexity.cpp`](../../../tools/perplexity/perplexity.cpp),
each token's log probabilities are clipped to a 16-log-unit range, scaled into
65535 steps, and stored as `uint16_t`. Every new reference must therefore be
compared against itself before candidate KLD, probability delta, or top-token
agreement is interpreted. Reconstructed `PPL(base)` is not substituted for a
normal perplexity result.

## Canonical timing rules

1. Set both boards to 150 W. The 175 W ceiling is reserved for an explicit
   power-scaling experiment and must never be exceeded.
2. Enable persistence mode and lock both graphics clocks to 892 MHz.
3. Require three consecutive idle samples in which both GPU and HBM
   temperatures are at or below 55 C and utilization is 0%.
4. Start one workload per process. Do not combine `pp32768` and `tg1024` in a
   heat-soaked invocation.
5. Capture both UUIDs, clocks, power, GPU and HBM temperature, utilization,
   VRAM, P-state, and all power/thermal cap reasons throughout the process.
6. A timed result is invalid if any busy sample is not 892/877 MHz or any cap
   reason is active.
7. Run each side enough times to distinguish an improvement from variance.
   Alternate revision order, or use a candidate/control/candidate bracket when
   a same-binary control exists.
8. After the final workload, reset the graphics-clock lock and restore 150 W.

The lock commands are:

```sh
: "${V100_GPU_0_UUID:?set V100_GPU_0_UUID to the first V100 UUID}"
: "${V100_GPU_1_UUID:?set V100_GPU_1_UUID to the second V100 UUID}"
GPU0=$V100_GPU_0_UUID
GPU1=$V100_GPU_1_UUID

for gpu in "$GPU0" "$GPU1"; do
    sudo nvidia-smi -i "$gpu" -pm 1
    sudo nvidia-smi -i "$gpu" -pl 150
    sudo nvidia-smi -i "$gpu" -lgc 892,892
done
```

Reset after the run:

```sh
for gpu in "$GPU0" "$GPU1"; do
    sudo nvidia-smi -i "$gpu" -rgc
    sudo nvidia-smi -i "$gpu" -pl 150
done
```

The canonical `llama-bench` arguments are:

```sh
CUDA_VISIBLE_DEVICES="$GPU0,$GPU1" \
LD_LIBRARY_PATH=/path/to/artifact/bin \
/path/to/artifact/bin/llama-bench \
    -m /path/to/model.gguf \
    -p 32768 -n 0 -r 1 \
    -b 2048 -ub 512 -t 4 \
    -ngl 999 -sm layer -ts 1/1 -mg 0 \
    -fa auto -ctk f16 -ctv f16 -o json
```

Use `-p 0 -n 1024` for the generation workload. A final comparison needs
multiple independent repetitions; the single repetition below only qualifies
the clock and memory configuration.

## Clock qualification

Gemma 4 31B `pp32768` provided the sustained qualification:

| Result | Value |
| --- | ---: |
| Throughput, one repetition | 729.237 tok/s |
| GPU 0 busy samples at 892/877 | 1139/1139 |
| GPU 1 busy samples at 892/877 | 1076/1076 |
| Cap samples | 0 |
| Peak power | 141.44 W / 147.17 W |
| Peak GPU temperature | 74 C / 69 C |
| Peak HBM temperature | 78 C / 73 C |
| Peak VRAM | 10966 MiB / 11572 MiB |

At 900 MHz, a layer-split correctness run produced one busy GPU 1 sample at
892 MHz. Lowering one supported bin to 892 MHz removed all departures during
the longer `pp32768` run. Raw JSON and telemetry are under
[experiment 083](../experiments/083-dual-v100-allreduce-recovery/raw/pp32768/).

## Correctness gates

Use layer split as the primary quality reference because it does not invoke
the tensor meta-backend all-reduce. For a numerical change:

1. Run the focused CUDA backend tests on each GPU.
2. Create an upstream layer-split saved-logit reference.
3. Compare upstream against that reference to establish the encoding floor.
4. Compare the candidate using identical model, tokens, context, batches, KV
   types, FlashAttention setting, and split.
5. Run plain perplexity and deterministic output checks in addition to KLD.
6. Test tensor/internal and tensor/NCCL separately. Do not compare a split-mode
   result to a layer baseline as if only the transport changed.
7. Run the complete CUDA backend suite and Compute Sanitizer before retaining a
   changed kernel.

The short four-chunk layer gate at revision `c55f89a84` passed with a 0.0132%
PPL increase, mean KLD 0.000028, 0.092% RMS probability delta, and 99.804%
top-token agreement. It is a screening gate, not a substitute for the longer
final quality run.

## Source selection checks for future optimizations

Do not infer kernel coverage from a model or quantization label:

- [`mmvq.cu`](../../../ggml/src/ggml-cuda/mmvq.cu) restricts the retained
  dedicated Q4_K SWIGLU warp-row path to SM70 and exactly
  `ncols_x == 4096 && nrows_x == 12288`. A different Q4_K gate shape uses the
  generic path.
- The same file restricts packed Q6_K output rows to dense, one-column Volta
  cases with at least 65536 output rows, and its fused Q6_K path to exactly
  4096 rows with bias-only fusion.
- [`fattn-tile.cuh`](../../../ggml/src/ggml-cuda/fattn-tile.cuh) has a Volta
  override for the exact 256/256 head configuration and delegates other
  configurations to the generic NVIDIA table.
- [`src/models/gemma4.cpp`](../../../src/models/gemma4.cpp) is the authority
  for Gemma 4 attention and GELU FFN graph construction.
- [`src/models/qwen35.cpp`](../../../src/models/qwen35.cpp) is the authority
  for the hybrid attention/Gated DeltaNet graph, SiLU FFN, and embedded MTP
  graph used by the Qwen 3.5/3.6 family.

Profile the target graph and inspect its tensor shapes before extending any
exact dispatch guard.

## Current limitations

- The 892 MHz qualification is complete only for Gemma 4 31B `pp32768`.
- Final repeated `pp32768` and `tg1024` branch-vs-upstream matrices for both
  target models are not part of experiment 083.
- The NCCL tensor path is stable in the short gate but differs numerically from
  the upstream internal-allreduce tensor reference. It is not yet a
  quality-equivalent default.
- MTP acceptance and exact-token validation remain separate server tests.
- The forced internal FP32 wire mode is prohibited on this host after the Xid
  79 incident documented in experiment 083.
