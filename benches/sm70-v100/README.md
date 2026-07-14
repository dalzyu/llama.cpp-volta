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
Experiment [032](experiments/032-volta-q2-k-dequant-blocks/) packs two Q2_K
dequantization work units per 128-thread block; on Qwen 3.6 p512 it improves
the paired control from 690.861 to 719.799 tok/s (+4.19%).
Experiment [033](experiments/033-volta-q4_0-half2-dequant/) vectorizes the
Volta Q4_0 half destination and packs two dequantization work units per
64-thread block; the final Qwen 3.5 p512 comparison is 14011.421 to 14793.329
tok/s (+5.58%).
The remaining SSM, concat, and fused K-quant launch-bound candidates are
recorded as rejected in experiment [035](experiments/035-volta-remaining-kernels/).
Graph chunking, separate gate precomputation, multi-column warps, shared q/k,
and recurrent-loop unrolling are recorded as rejected in experiment
[037](experiments/037-volta-gdn-redesigns/).
Legacy cuBLAS algorithm forcing, Q8_0 dequantization packing, and exact-shape
cuBLASLt split-K integration are recorded as rejected in experiment
[039](experiments/039-volta-prompt-gemm/). The isolated split-K GEMM wins did
not translate consistently once dequantization overlap was included.

## Retained optimizations

Volta now uses two-warp blocks for normal-K single-column Q4_0 and Q8_0 MMVQ,
while their small-K specializations remain at four warps. Q2_K, Q3_K, Q4_K,
and Q6_K single-column MMVQ also use two warps. Fused normal-K Q2_K and Q3_K
use spill-free minimum occupancy bounds of 20 and 14 blocks per SM. Their inner
products avoid Q2_K byte broadcasting and unnecessary Q3_K saturation. Q2_K
dequantization uses guarded two-superblock blocks on Volta.

The profiled float-to-float `get_rows` path now copies two adjacent elements per
thread and halves its y-grid. Other source and destination type combinations
keep the existing scalar kernel.

Experiment [036](experiments/036-volta-gdn-warp-scalar/) evaluates the scalar
Gated DeltaNet gate once per warp and broadcasts it. At a fixed 1200 MHz SM
clock this improves Qwen 3.5 pp512 by 0.24% to 0.32% and reduces the GDN kernel
average by 3.67%.

Experiment [038](experiments/038-volta-gdn-vector-rows/) assigns four adjacent
rows to each lane for aligned 128-wide scalar GDN heads on Volta. It emits
128-bit state, q, and k accesses while leaving KDA and other architectures on
the original kernel. The GDN kernel average falls another 6.01%.

Experiment [040](experiments/040-volta-flash-attention/) adds exact SM70 tile
selections for 256-wide flash attention. The 64-column MMA kernel average falls
30.05%, while the single-token tile falls 35.73%. The external
`flash-attention-v100` implementation informed the 64-row KV tile and a shared
memory swizzle experiment. The swizzle was effective in its original kernel but
regressed ggml and was not retained.

| Controlled comparison | Before | After | Change |
| --- | ---: | ---: | ---: |
| Gemma 4 12B Q4_0 tg128 | 69.442 | 70.252 | +1.17% |
| Qwen 3.5 0.8B Q8_0 tg128 | 279.227 | 289.121 | +3.54% |
| Qwen 3.6 27B Q2_K tg128 | 27.835 | 28.826 | +3.56% |
| Qwen 3.6 fused Q3_K occupancy | 28.780 | 29.453 | +2.34% |
| Qwen 3.6 fused Q2_K occupancy | 29.501 | 29.659 | +0.53% |
| Qwen 3.6 K-quant device code | 29.758 | 30.716 | +3.22% |
| Qwen 3.6 Q2_K p512 dequantization | 690.861 | 719.799 | +4.19% |
| Qwen 3.5 Q4_0 p512 dequantization | 14011.421 | 14793.329 | +5.58% |
| Qwen 3.5 Q8_0 float get_rows tg128 | 288.953 | 291.321 | +0.82% |
| Qwen 3.6 float get_rows tg128 | 30.535 | 30.898 | +1.19% |
| Qwen 3.5 Q4_0 GDN vector rows pp512 | 14226.68 | 14428.77 | +1.42% |
| Qwen 3.5 Q8_0 GDN vector rows pp512 | 14557.47 | 14761.59 | +1.40% |
| Qwen 3.5 Q4_0 flash attention pp512 | 14425.28 | 14557.41 | +0.92% |
| Qwen 3.5 Q8_0 flash attention pp512 | 14769.94 | 14894.49 | +0.84% |
| Qwen 3.5 Q4_0 flash attention tg128 | 268.537 | 270.475 | +0.72% |
| Qwen 3.5 Q8_0 flash attention tg128 | 248.564 | 250.081 | +0.61% |

Peak VRAM is unchanged. The vector-row and attention reduction-order changes
produce final eight-chunk Qwen 3.5 PPL estimates of 21.8608 for Q4_0 and
18.3635 for Q8_0. Focused backend coverage passes 112/112 modified flash
attention cases in addition to the earlier K-quant, Q4_0, and Q8_0 suites.
The preceding retained-build measurements and commands are under `final/`.
