# Final retained-build validation

Revision `d3b5d60f1` was rebuilt in Release mode with real sm_70 and sm_89
code. Every inference and CUDA test exposed only the Tesla V100 UUID. The
baseline-compatible headline runs used the original 150 W power limit,
unlocked SM clocks, `-p 512 -n 128 -r 5`, flash attention in auto mode, and
the original batch settings.

| Model | Baseline pp512 | Final pp512 | Change | Baseline tg128 | Final tg128 | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 12B Q4_K_XL | 1509.341 | 1685.066 | +11.64% | 69.685 | 70.259 | +0.82% |
| Qwen 3.5 0.8B Q4_0 | 14371.067 | 15725.102 | +9.42% | 304.638 | 341.850 | +12.22% |
| Qwen 3.5 0.8B Q8_0 | 15253.253 | 15897.319 | +4.22% | 277.522 | 320.724 | +15.57% |
| Qwen 3.6 27B Q2_K | 691.610 | 726.682 | +5.07% | 27.721 | 31.963 | +15.30% |

These aggregate comparisons include normal run-to-run clock and temperature
variation. Each individual optimization used fixed 1200 MHz SM and 877 MHz
memory clocks, adjacent or alternating controls, and larger repetition counts.
Those controlled results are under `../experiments/`.
The final fixed-clock Q8_0 generation kernel summary is in
`qwen35-q8-tg128-kernels.tsv`.

## Accuracy

The directly comparable eight-chunk WikiText-2 gate remains within a tiny
fraction of its estimate uncertainty for every model.

| Model | Baseline PPL | Final PPL | Final uncertainty |
| --- | ---: | ---: | ---: |
| Gemma 4 12B Q4_K_XL | 308.3287 | 308.9815 | 38.81265 |
| Qwen 3.5 0.8B Q4_0 | 21.8601 | 21.8604 | 1.50866 |
| Qwen 3.5 0.8B Q8_0 | 18.3634 | 18.3641 | 1.23699 |
| Qwen 3.6 27B Q2_K | 7.1732 | 7.1734 | 0.40718 |

The longer 32-chunk final run produces 185.5480, 18.6990, 15.4868, and
6.1046 respectively. Its uncertainties fall to 11.18965, 0.63220, 0.51042,
and 0.16680. Values from different chunk counts are not directly comparable;
the longer run is an additional final-output stability check.

## Retained work

The retained changes are restricted by architecture, shape, layout, type, or
graph invariants so unsupported cases keep the original path.

- Quantized generation: Volta-specific K-quant, Q4_0, and Q8_0 MMVQ launch
  geometry; spill-free fused Q2_K/Q3_K bounds; simpler K-quant dot products;
  and dedicated dense Q8_0 warp-row kernels for normal and SWIGLU gate paths.
- Prompt processing: packed Q2_K dequantization blocks, vector Q4_0 stores and
  packed blocks, and exact 256-wide Volta flash-attention tiles.
- Recurrent Qwen paths: warp-scalar and four-row GDN processing, paired Q/K L2
  normalization, fused gate chains, direct strided sigmoid gates, fused concat
  copies, and a specialized recurrent tail copy.
- Common support kernels: aligned float4 GET_ROWS and aligned float4 fused RMS
  normalization.
- Correctness: 320-wide Volta attention now stays on the launchable tile path
  instead of selecting a pre-existing unlaunchable MMA specialization.

No retained change adds a persistent device allocation. Model and runtime VRAM
usage are unchanged at the resolution of the collected telemetry.

## Exhausted high-impact alternatives

The remaining final Q8_0 generation hotspots have all had their meaningful
launch, vector-width, or work-partition axes measured.

| Area | Retained result | Rejected alternatives |
| --- | --- | --- |
| Dense Q8_0 MMVQ | one warp per row, two rows per block | 1/4/8 rows, VDR=4, three-row generic blocks, activation reuse, software pipeline, forced occupancy |
| Fused Q8_0 gate | one warp per row, two rows per block | 1/4/8 rows, split value/gate warps, activation reuse, two-accumulator ILP |
| Fused Q8_0 bias | generic kernel | dedicated warp-row bias was 2.18% slower |
| Q8_1 quantization | existing scalar 256-thread kernel | 64/128/512 threads and 2-wide/4-wide packing |
| Flash attention | 64-row KV prompt tile and smaller generation KV batch | direct external port, XOR swizzle, Q-in-register, vector kernels, other tile/occupancy/K/combine sizes |
| Prompt GEMM | existing cuBLAS plus retained dequant changes | legacy algorithms, cuBLASLt exact-shape split-K, forced workspace and compute modes |
| GDN | scalar broadcast plus four-row vector path | graph chunking, gate precompute, shared Q/K, multi-column warps, loop unrolling |
| Copy/gather/norm | float4 or exact graph fusion | float2/float8 endpoints and wider gather assignments |

Every material hotspot in the final Q8_0 generation profile now maps to a
retained change or a measured launch, vector-width, work-partition, or fusion
sweep. Experiments 021, 037, 039, 040, 042, and 048 through 053 contain the
detailed rejection measurements.

## Validation

The complete CUDA backend suite passes 12,995/12,995 cases. Focused coverage
also records the exact modified families: K-quant, Q4_0, Q8_0, GET_ROWS,
256-wide and 320-wide flash attention, RMS normalization, fused RMS/rope, L2
normalization, GDN, graph fusions, concat/copy, and Q8_0 matrix-vector fusion.

Compute Sanitizer reports zero errors for the repaired 320-wide attention
family, the focused Q8_0 fusion family, and a whole Qwen 3.5 Q8_0 pp128/tg16
model run through the retained graph fusions and kernels.
