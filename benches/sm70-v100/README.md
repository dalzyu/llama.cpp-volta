# sm_70 V100 optimization results

This directory records local optimization experiments for a Tesla V100-SXM2-16GB.
All inference commands set `CUDA_VISIBLE_DEVICES` to the V100 UUID so the RTX 4080
is not visible to llama.cpp.

## Current canonical benchmark protocol

The V100 was replaced with another V100-SXM2-16GB on 2026-07-17. Results from
the original card remain historical data. New optimization comparisons use the
following fixed protocol unless an experiment explicitly studies power or clock
scaling:

- V100 UUID: `${V100_GPU_0_UUID}`
- Excluded RTX 4080 UUID: `${EXCLUDED_GPU_UUID}`
- Every CUDA process sets
  `CUDA_VISIBLE_DEVICES=${V100_GPU_0_UUID}`.
- Persistence mode is enabled and the board power limit is 300 W.
- The graphics clock is locked to 1192 MHz with
  `nvidia-smi -lgc 1192,1192` before benchmarking.
- The memory clock is 877 MHz, the only supported memory clock on this card.
  The driver reports that an explicit memory-clock lock is unsupported.
- Telemetry samples at 50 ms or faster record graphics and memory clocks,
  power, temperature, utilization, throttle reasons, and VRAM.
- A performance run is valid only when every busy sample is exactly 1192 MHz
  and no software power, software thermal, or hardware thermal cap is active.
  Unlocked or throttled runs are diagnostic data, not canonical results.
- Start a comparison group at or below 70 C. Candidate and control runs remain
  consecutive after that so both see the same thermal history.

The 1200 MHz clock survived two consecutive Gemma 4 12B `pp32768` passes with
all 1,467 busy samples at 1200 MHz, no throttle samples, and a 79 C peak. The
canonical clock is one supported bin lower to retain additional thermal margin.
At 1230 MHz, two passes succeeded but the third pass reached the 83 C software
thermal cap. Higher tested clocks also failed sustained load: 1275 and 1380 MHz
thermally capped, while 1530 MHz could not hold its requested bin.

The co-primary models and workloads are:

- Gemma 4 12B Q4_K_XL: `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`
- Qwen 3.5 9B Q4_K_M: `Qwen3.5-9B-Q4_K_M.gguf`
- Prompt processing: `pp32768`
- Token generation: `tg1024`

The initial 1192 MHz reference values are:

| Model | pp32768 | tg1024 | PP peak VRAM | TG peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 12B Q4_K_XL | 1318.107017 | 69.339006 | 8404 MiB | 7624 MiB |
| Qwen 3.5 9B Q4_K_M control | 1997.408054 | 100.096161 | 6912 MiB | 5812 MiB |

`pp512` and `tg128` remain useful screens but cannot justify retaining a
change. Long comparisons use the same binary and a candidate/control/candidate
bracket when a runtime control is possible. The two candidate values are
averaged against the center control. If a runtime control is not possible,
equivalent alternating builds are used. Default batch settings remain
`-b 2048 -ub 512`, FlashAttention remains `auto`, and all layers are offloaded.

Correctness gates include matched perplexity and saved-logit hashes on the
affected model, focused CUDA backend tests, the complete backend suite before a
checkpoint, and Compute Sanitizer for new or substantially changed kernels.
Performance improvements are not retained if these gates detect an accuracy or
memory-safety change.

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

## Final aggregate

Revision `d3b5d60f1` passes the complete 12,995-case CUDA backend suite. The
headline final measurements repeat the original baseline command at the 150 W
power limit with unlocked clocks and only the V100 visible.

| Model | pp512 baseline | pp512 final | Change | tg128 baseline | tg128 final | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 12B Q4_K_XL | 1509.341 | 1685.066 | +11.64% | 69.685 | 70.259 | +0.82% |
| Qwen 3.5 0.8B Q4_0 | 14371.067 | 15725.102 | +9.42% | 304.638 | 341.850 | +12.22% |
| Qwen 3.5 0.8B Q8_0 | 15253.253 | 15897.319 | +4.22% | 277.522 | 320.724 | +15.57% |
| Qwen 3.6 27B Q2_K | 691.610 | 726.682 | +5.07% | 27.721 | 31.963 | +15.30% |

The directly comparable eight-chunk perplexity changes are much smaller than
their estimate uncertainties. A separate 32-chunk final run and the full
results are under `final/`. No retained change adds a persistent GPU
allocation, so peak model VRAM is unchanged.

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

Experiment [041](experiments/041-volta-q8-multi-row/) computes two adjacent
rows per normal-K, non-fused Q8_0 MMVQ block on Volta. The non-fused kernel
average falls 19.40%, while the dominant output-projection grid falls 29.09%.
The fused kernel keeps one row, while small-K keeps its existing geometry.

Experiment [042](experiments/042-volta-generation-followups/) widens aligned
Volta float-to-float GET_ROWS copies from two scalar values to one 16-byte
transfer per thread. Total gather time falls 14.94%. Wider copies, fused Q8_0
warp partitioning, explicit activation reuse, and vector Q8_1 quantization are
recorded as rejected.

Experiment [043](experiments/043-volta-rms-norm-vec4/) uses 256 threads and
float4 transfers for aligned large fused RMS normalization. The affected
kernel falls 22.30%; Q8_0 and Q4_0 generation improve 0.88% and 0.96%.

Experiments [044](experiments/044-volta-l2-norm-pair/) through
[047](experiments/047-volta-concat-cpy/) remove small recurrent-graph launches:
paired Q/K L2 normalization, the GDN add-softplus-multiply chain, a strided
attention sigmoid gate, and concat-to-state copy. Their target chains fall by
46.49%, 38.56%, 59.32%, and 10.47% respectively.

Experiment [050](experiments/050-volta-concat-tail/) specializes the exact
Qwen recurrent concat tail. Its kernel falls 33.76% and Q8_0/Q4_0 generation
improves 1.02%/0.98% in same-binary controls.

Experiments [051](experiments/051-volta-q8-warp-rows/) through
[053](experiments/053-volta-q8-fused-gate/) replace dense non-fused and SWIGLU
Q8_0 MMVQ with one complete row per warp, eliminate repeated row-base
addressing, and retain two rows per block. The non-fused kernel falls 4.14%,
the gate kernel falls 6.37%, and paired generation gains are 1.18%, 0.168%,
and 0.309%.

Experiment [054](experiments/054-volta-fattn-320-fallback/) resolves a
pre-existing 320-wide Volta FlashAttention launch failure discovered by the
full validation sweep. That head size now stays on the existing tile kernel;
the optimized 256-wide path remains unchanged.

Experiment [079](experiments/079-volta-q6-large-rows/) computes two adjacent
rows per block for large, dense, non-fused Q6_K matrix-vector products on
Volta. It halves the Qwen 3.5 9B vocabulary projection grid, reduces that
kernel by 6.20%, and improves the fixed-1192-MHz `tg1024` bracket by 0.5251%.
Small, fused, odd-row, indirect, and non-Volta cases retain their original
geometry. A faster three-row prototype was rejected after initcheck exposed
an uninitialized tail read.

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
| Qwen 3.5 Q8_0 multi-row MMVQ tg128 | 248.167 | 262.326 | +5.71% |
| Qwen 3.5 Q4_0 multi-row output tg128 | 267.079 | 276.683 | +3.60% |
| Qwen 3.5 Q8_0 float4 get_rows tg128 | 262.520 | 263.281 | +0.29% |
| Qwen 3.6 Q2_K float4 get_rows tg128 | 31.075 | 31.389 | +1.01% |
| Qwen 3.5 Q8_0 float4 RMS norm tg128 | 263.288 | 265.605 | +0.88% |
| Qwen 3.5 Q4_0 float4 RMS norm tg128 | 276.652 | 279.316 | +0.96% |
| Qwen 3.5 Q8_0 paired L2 norm tg128 | 265.539 | 270.265 | +1.78% |
| Qwen 3.5 Q4_0 paired L2 norm tg128 | 279.091 | 284.526 | +1.95% |
| Qwen 3.5 Q8_0 GDN gate fusion tg128 | 270.477 | 273.337 | +1.06% |
| Qwen 3.5 Q4_0 GDN gate fusion tg128 | 284.441 | 288.139 | +1.30% |
| Qwen 3.5 Q8_0 strided sigmoid fusion tg128 | 273.303 | 275.967 | +0.97% |
| Qwen 3.5 Q4_0 strided sigmoid fusion tg128 | 288.163 | 291.107 | +1.02% |
| Qwen 3.5 Q8_0 concat-copy fusion tg128 | 274.248 | 276.772 | +0.92% |
| Qwen 3.5 Q4_0 concat-copy fusion tg128 | 289.130 | 292.282 | +1.09% |
| Qwen 3.5 Q8_0 recurrent tail copy tg128 | 277.348 | 280.165 | +1.02% |
| Qwen 3.5 Q4_0 recurrent tail copy tg128 | 292.817 | 295.692 | +0.98% |
| Qwen 3.5 Q8_0 warp-row MMVQ tg128 | 280.139 | 283.442 | +1.18% |
| Qwen 3.5 Q8_0 row-base addressing tg128 | 283.644 | 284.120 | +0.168% |
| Qwen 3.5 Q8_0 fused gate rows tg128 | 283.786 | 284.663 | +0.309% |

Peak VRAM is unchanged. The vector-row and attention reduction-order changes
produce final eight-chunk Qwen 3.5 PPL estimates of 21.8604 for Q4_0 and
18.3641 for Q8_0. The complete CUDA backend suite passes 12,995/12,995 cases,
and Compute Sanitizer reports zero errors for the whole Q8_0 target graph.
The final retained-build measurements and commands are under `final/`.
