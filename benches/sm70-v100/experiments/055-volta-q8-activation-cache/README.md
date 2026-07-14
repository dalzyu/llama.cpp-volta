# Volta decode Q8_1 activation cache

Quantized MMVQ converts its F32 input to Q8_1 before multiplying it by a
quantized weight matrix. Decode graphs frequently send one activation tensor
to several sibling projections, but the original path quantizes that tensor
again for every projection.

The retained SM70 path keeps the Q8_1 conversion for the duration of one
backend graph execution. Entries are keyed by the source tensor and released
in reverse allocation order after the graph finishes. Reuse is limited to one
column, one sample, quantized weights, no expert IDs, and CUDA stream zero.
All other architectures and shapes keep the original operation-local
allocation.

## Same-binary generation results

The candidate and control use one binary. Setting
`GGML_CUDA_DISABLE_Q8_1_CACHE=1` disables only the new reuse path. The V100 was
pinned to 1200 MHz SM and 877 MHz memory clocks. Qwen 3.5 measurements used a
150 W limit, tg128, and 20 repetitions.

| Model | Candidate A | Control | Candidate B | Candidate mean | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q8_0 | 301.304505 | 285.581170 | 301.020694 | 301.162599 | +5.4560% |
| Qwen 3.5 0.8B Q4_0 | 315.694777 | 298.530135 | 315.672112 | 315.683445 | +5.7459% |

The cache is inactive during pp512. A Q8_0 prompt check measured 15221.880
tok/s with the candidate and 15253.477 tok/s with the control, a noise-sized
-0.21% difference.

Qwen 3.6 27B Q2_K was power-throttled at 150 W, so its decisive comparison
used the same fixed clocks at 225 W. Each entry contains seven repetitions.
The V100 was returned to 150 W immediately afterward.

| Run | tg128 tok/s | Standard deviation |
| --- | ---: | ---: |
| Candidate A | 36.343051 | 0.012596 |
| Control | 35.989482 | 0.026678 |
| Candidate B | 36.332431 | 0.016035 |

The candidate mean is 36.337741 tok/s, a 0.9677% improvement. The smaller
gain is consistent with a 27B decode workload dominated by weight bandwidth.

## Launch profile

Nsight Systems covered three Qwen 3.5 Q8_0 tg128 runs from each side of the
same binary.

| Measurement | Candidate | Control | Change |
| --- | ---: | ---: | ---: |
| Q8_1 quantizer launches | 37,248 | 62,592 | -40.49% |
| Q8_1 quantizer time | 78.556430 ms | 129.494674 ms | -39.34% |
| Dense Q8_0 MMVQ time | 341.521396 ms | 343.406825 ms | -0.55% |
| Q8_0 gate MMVQ time | 137.183722 ms | 136.959139 ms | +0.16% |
| Generic fused Q8_0 time | 91.194528 ms | 91.300673 ms | -0.12% |

The matrix-vector kernels remain flat while 25,344 redundant quantizer
launches disappear. The retained buffers are short single-token Q8_1 vectors
and live for only one graph execution; model weights and persistent VRAM are
unchanged.

## Correctness and safety

Candidate and disabled-cache runs saved every logit from batch-one
WikiText-2 evaluations. Each pair is byte-identical.

| Model | Context | Logit bytes | SHA-256 | PPL |
| --- | ---: | ---: | --- | ---: |
| Qwen 3.5 Q8_0 | 128 | 31,289,356 | `acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274` | 11.9470 |
| Qwen 3.5 Q4_0 | 128 | 31,289,356 | `385e35ba44725c79a84b3ed86297d71de16280492d40086cb04fa0ed819e14a0` | 12.6381 |
| Qwen 3.6 Q2_K | 64 | 15,396,364 | `5790cafdade18db8b653eb9d0e0fae27e0e62f0614f37f101cc809e9a859e0e3` | 9.7193 |

The standard eight-chunk Q8_0 estimate remains 18.3641. Focused Q8_0, Q4_0,
Q2_K, Q3_K, Q4_K, and Q6_K single-column coverage passes 51/51. Compute
Sanitizer reports zero errors for a whole Qwen 3.5 Q8_0 pp128/tg16 run.

