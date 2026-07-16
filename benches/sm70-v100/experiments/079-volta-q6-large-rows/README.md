# Volta large dense Q6_K multi-row MMVQ

Qwen 3.5 9B Q4_K_M uses a Q6_K output projection with 248320 rows and
4096 columns. In the `tg1024` profile, its non-fused single-column MMVQ was
the third-largest CUDA kernel group: 1025 launches at 1.143763 ms each, or
11.2% of traced kernel time. Each launch used a 248320-block grid even though
the two-warp block had enough independent work to accumulate adjacent rows.

The retained SM70 path assigns two adjacent output rows to each block. It is
selected only for non-fused, normal-K Q6_K with one destination column, no
indirection, one dense channel and sample, a contiguous destination, at least
65536 rows, and an even row count. Other architectures and all fused, small-K,
multi-column, indirect, small, or odd-row cases use the existing kernel.
The regular kernel keeps its calculated row count through a defaulted template
parameter, so only this exact launch instantiates the two-row body.

## Prototype sweep

The first sweep changed all non-fused Q6_K single-column launches. Large grids
improved, but packing the five 1024-row projections reduced their occupancy and
made them slower. A split three-row version preserved the small kernels and had
the best benchmark result, but Compute Sanitizer initcheck found 28
uninitialized global reads in the partial final block of an odd 65537-row
boundary case. The writes were guarded, but all three input rows were loaded.
That prototype was rejected.

| Variant | tg1024 | Change vs. one row | 248320-row average | 1024-row average | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| One row | 100.268826 | control | 1.143763 ms | 11.0439 us | control |
| Two rows globally | 100.649889 | +0.3800% | 1.073024 ms | 11.8022 us | rejected |
| Three rows globally | 100.633293 | +0.3635% | 1.063313 ms | 12.6382 us | rejected |
| Four rows globally | 100.556038 | +0.2864% | not profiled | not profiled | rejected |
| Three rows, large only | 100.832555 | +0.6266% | 1.063465 ms | 11.0421 us | unsafe |
| Two rows, large even only | 100.693532 | +0.5251% | 1.072852 ms | 11.0504 us | retained |

The three-row kernel used 66 registers per thread and 384 bytes of shared
memory. Four rows increased that to 80 registers and 512 bytes. The retained
two-row kernel uses 64 registers and 256 bytes, with no local or stack spill.

## Fixed-clock performance

Canonical comparisons used the replacement V100 UUID
`${V100_GPU_0_UUID}`, a 300 W limit, 1192 MHz locked
graphics clock, 877 MHz memory, 50 ms telemetry, and only the V100 visible.
The final Qwen generation comparison used one temporary binary in
candidate/control/candidate order; the control disabled only the new dispatch.

| Model and test | Candidate A | Control | Candidate B | Candidate mean | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 9B Q4_K_M tg1024 | 100.750581 | 100.167528 | 100.636482 | 100.693532 | +0.5251% |

All busy samples in the bracket were exactly 1192 MHz, no power or thermal cap
was active, and peak VRAM remained 5812 MiB. The candidate A, control, and
candidate B runs had 231, 233, and 232 busy samples and peaked at 60, 58, and
58 C respectively.

The path is inactive for prompt GEMM and for Gemma's Q4_0 output matrix. Their
absolute production checks remained neutral:

| Model and test | Candidate | Reference | Change | Busy clock | Peak temperature | Peak VRAM |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Qwen 3.5 9B Q4_K_M pp32768 | 1995.130211 | 1997.408054 | -0.1140% | 1192-1192 MHz | 65 C | 6912 MiB |
| Gemma 4 12B Q4_K_XL pp32768 | 1318.813671 | 1318.107017 | +0.0536% | 1192-1192 MHz | 71 C | 8404 MiB |
| Gemma 4 12B Q4_K_XL tg1024 | 69.301057 | 69.339006 | -0.0547% | 1192-1192 MHz | 62 C | 7624 MiB |

## Kernel profile

The Qwen `tg1024` trace includes one warmup and 1024 measured tokens.

| Q6_K path | Grid x | Launches | Control average | Candidate average | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Large output projection | 248320 -> 124160 | 1025 | 1.143763 ms | 1.072852 ms | -6.20% |
| Five small projections | 1024 | 5125 | 11.0439 us | 11.0504 us | +0.06% |
| Fused projections | 4096 | 16400 | 71.5757 us | 71.5702 us | -0.01% |

The large projection saves 72.685 ms across the trace, or 70.912 us per model
evaluation. The small and fused Q6_K launches remain on their original
instantiations and are unchanged.

## Correctness and safety validation

The production candidate reports WikiText-2 PPL 5.9046 with an uncertainty of
1.87941. Its saved context-128 logits are byte-identical to experiment 078:
both have SHA-256
`be3f0c3b86215d5be67b9bba474d19156c3a9b6bfd87a55196772fd434fb5bd0`.

Temporary backend cases exercised Q6_K F32 matrix-vector multiplication at
`m=65536,n=1,k=1024`, which selects the two-row kernel, and at
`m=65537,n=1,k=1024`, which selects the one-row fallback. Both matched the CPU
reference. Memcheck, initcheck, and synccheck reported zero errors on both
cases; racecheck reported zero hazards on the two-row case. The temporary test
cases were then removed.

The standard focused Q6_K MUL_MAT sweep passes 11/11 cases. The complete CUDA
backend suite passes 12995/12995 cases, the architecture graph test exits
successfully, and the non-backend CTest suite passes 53/53 tests.
