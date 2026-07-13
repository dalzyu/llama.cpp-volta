# sm_70 V100 optimization results

This directory records local optimization experiments for a Tesla V100-SXM2-16GB.
All inference commands set `CUDA_VISIBLE_DEVICES` to the V100 UUID so the RTX 4080
is not visible to llama.cpp.

## Baseline

- Revision: `99f3dc32296f825fec94f202da1e9fede1e78cf9`
- Build: Release, CUDA 12.9, native CUDA targets sm_70 and sm_89
- V100 UUID: `${V100_GPU_1_UUID}`
- Power limit: 150 W
- Model: `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`
- Model SHA-256: `cc9ff072e0a8203429ed854e6662c17a6c2bc1e5dca5b475dd4736caaacbc165`
- WikiText-2 test SHA-256: `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`

| Test | Result | Stddev | Peak VRAM |
| --- | ---: | ---: | ---: |
| pp512 | 1509.341 tok/s | 14.086 tok/s | 7582 MiB |
| tg128 | 69.685 tok/s | 0.061 tok/s | 6966 MiB |
| WikiText-2, 8 chunks, ctx 512 | PPL 308.3287 | 38.71784 | 9376 MiB |

Raw outputs, commands, and 200 ms GPU telemetry are under `baseline/`.
The short perplexity run is the per-experiment correctness gate. Accepted changes
will also be checked with broader backend tests and a longer perplexity run.

### Qwen baselines

The Qwen runs use the same revision, V100 UUID, 150 W limit, batch settings, and
WikiText-2 gate as the Gemma 4 baseline.

| Model | pp512 | tg128 | Bench peak VRAM | PPL | PPL peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q4_0 | 14371.067 tok/s | 304.638 tok/s | 1422 MiB | 21.8601 | 1894 MiB |
| Qwen 3.5 0.8B Q8_0 | 15253.253 tok/s | 277.522 tok/s | 1658 MiB | 18.3634 | 2130 MiB |
| Qwen 3.6 27B Q2_K | 691.610 tok/s | 27.721 tok/s | 12326 MiB | 7.1732 | 14566 MiB |

Model sources and hashes are recorded under `models/`.

## Profiling

Nsight Systems kernel summaries are under `profiles/`. For pp512, Q4_0
dequantization and tensor-core GEMM account for 79.3% of GPU kernel time. For
tg128, Q4_0 matrix-vector kernels account for 72.8% of GPU kernel time.
