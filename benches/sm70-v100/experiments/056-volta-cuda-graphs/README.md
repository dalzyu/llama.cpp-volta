# CUDA Graph replay on Volta

The original CUDA Graph integration disabled every architecture below SM80
after early Pascal and Volta regressions. The current backend, CUDA 12.9
runtime, graph update path, and optimized SM70 kernels are materially different
from that 2024 configuration.

CUDA Graphs record a stable workflow and relaunch it with one host operation.
This is a good match for token generation, where llama.cpp otherwise submits
hundreds of short kernels per token. The current CUDA documentation describes
the launch-overhead benefit without an SM80 requirement:
<https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>.
The historical architecture decision is recorded in the initial integration:
<https://github.com/ggml-org/llama.cpp/pull/6766>.

The retained change enables the existing graph path by default on exactly
SM70. Pascal and Turing retain their previous policy, and
`GGML_CUDA_DISABLE_GRAPHS=1` remains the runtime opt-out.

## Fixed-clock generation results

Candidate and control use one binary. All runs pin only the V100 UUID at
1200 MHz SM and 877 MHz memory clocks. Qwen 3.5 and Gemma 4 use the 150 W
limit. Qwen 3.6 uses 225 W to remove power-throttling drift and returns to
150 W immediately afterward.

| Model | Graph A | Graph-disabled control | Graph B | Graph mean | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q8_0 | 306.305315 | 300.720015 | 306.279660 | 306.292487 | +1.8530% |
| Qwen 3.5 0.8B Q4_0 | 317.743908 | 312.047027 | 317.777429 | 317.760668 | +1.8310% |
| Qwen 3.6 27B Q2_K | 36.668172 | 36.327001 | 36.680629 | 36.674401 | +0.9563% |
| Gemma 4 12B Q4_K_XL | 70.513572 | 70.328656 | 70.436105 | 70.474839 | +0.2079% |

The smaller gain on larger models is expected: weight bandwidth occupies a
larger fraction of each token, leaving less host launch overhead to remove.

## Prompt capture tradeoff

CUDA Graph warmup and capture make the first measured pp512 call slower, then
replay improves the stable shape. The table includes the capture call in each
20-run average.

| Model | Graph mean | Graph-disabled | End-to-end change | Replay mean after capture |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3.5 Q8_0 | 15272.229 | 15283.129 | -0.071% | 15411.163 |
| Qwen 3.5 Q4_0 | 14971.836 | 14896.258 | +0.507% | 15102.584 |

The first graph samples are 12,568 and 12,697 tok/s for Q8_0, and 12,468 and
12,507 tok/s for Q4_0. This one-time capture cost matters to first-request
latency, but it is amortized during generation and long-running service.

## Launch profile and memory

Nsight Systems traces three Qwen 3.5 Q8_0 tg128 runs. Event tracing changes
absolute timings, so only API call counts are used.

| API | Graph replay | Direct execution | Change |
| --- | ---: | ---: | ---: |
| `cudaLaunchKernel` | 5,166 | 215,916 | -97.61% |
| `cudaGraphLaunch` | 378 | 0 | +378 |
| Stream captures | 3 | 0 | +3 |

The 378 graph launches cover 384 generated tokens after six warmup/capture
evaluations. Peak `nvidia-smi` sampling is 1,274 MiB with graphs and 1,268 MiB
without. The retained graph instances therefore add about 6 MiB on this model.

## Rejected concurrency variant

The existing `GGML_CUDA_GRAPH_OPT=1` path interleaves three attention branches
over multiple CUDA streams. On Volta it measures 304.469258 and 304.242188
tok/s versus 306.198481 for ordinary graph replay, a 0.6018% regression. The
extra event and stream synchronization outweighs projection overlap, so graph
concurrency remains disabled.

## Correctness and safety

Graph and direct executions save byte-identical all-logit files:

| Model | Context | SHA-256 | PPL |
| --- | ---: | --- | ---: |
| Qwen 3.5 Q8_0 | 128 | `acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274` | 11.9470 |
| Qwen 3.5 Q4_0 | 128 | `385e35ba44725c79a84b3ed86297d71de16280492d40086cb04fa0ed819e14a0` | 12.6381 |
| Qwen 3.6 Q2_K | 64 | `5790cafdade18db8b653eb9d0e0fae27e0e62f0614f37f101cc809e9a859e0e3` | 9.7193 |
| Gemma 4 12B Q4_K_XL | 32 | `079436a894fcb90281b8e10c2f19e0b7d60ead345877246e1faf99347f8f46f0` | 1203.2716 |

Focused quantized single-column coverage passes 51/51 with graphs enabled.
Compute Sanitizer reports zero errors for a graph-plus-cache Qwen 3.5 Q8_0
pp128/tg16 run.
