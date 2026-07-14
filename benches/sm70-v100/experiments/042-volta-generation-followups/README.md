# Volta generation follow-up kernels

This experiment continues from the retained two-row Q8_0 MMVQ launch. It
tests fused Q8_0 work partitioning, Q8_1 vector quantization, and wider
float-to-float GET_ROWS transfers on SM70.

All throughput comparisons used only the V100 UUID, a 150 W power limit,
fixed 1200 MHz SM and 877 MHz memory clocks, and tg128.

## Retained float4 GET_ROWS

The existing float-to-float path assigns two scalar values to each thread.
The retained SM70 path assigns four adjacent values and uses one aligned
16-byte transfer for the common fully aligned layout. It halves the y-grid
again. Non-float types, other architectures, and layouts with a dimension or
stride that is not 16-byte compatible keep the existing path.

| Model | Candidate A | Control | Candidate B | Candidate mean change |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q8_0 | 263.260385 | 262.519619 | 263.300932 | +0.29% |
| Qwen 3.5 0.8B Q4_0 | 277.139439 | 276.536770 | 276.575080 | +0.12% |
| Qwen 3.6 27B Q2_K | 31.337033 | 31.075226 | 31.441491 | +1.01% |

Nsight Systems reduces total `k_get_rows_float<float,float>` time from 59.679
to 50.762 ms (-14.94%) over three Qwen 3.5 Q8_0 tg128 runs. The dominant
grid falls from 38.511 to 29.463 ms (-23.49%). The smaller 36-block control
grid is effectively flat after becoming an 18-block grid, so eight values per
thread was also tested. It reduced parallelism too far and was rejected.

| Values per thread | Qwen 3.5 Q8_0 | Qwen 3.5 Q4_0 | Result |
| ---: | ---: | ---: | --- |
| 2 | 262.519619 | 276.536770 | control |
| 4 | 263.280658 | 276.857259 | retained mean |
| 8 | 262.921239 | 276.295382 | rejected |

The SM70 float4 kernel uses 40 registers per thread, down from 56 for the
float2 control, with no shared, local, or stack memory. Device allocation and
model VRAM use are unchanged.

## Rejected fused Q8_0 designs

The fused Q8_0 MMVQ kernel remains one row per block. Three deeper designs
were tested:

| Design | Qwen 3.5 Q8_0 | Fused kernel average | Result |
| --- | ---: | ---: | --- |
| One-row retained | 262.326 mean | 11.398 us | control |
| Two rows, explicit activation reuse | 260.298 | 12.169 us | rejected |
| Value/gate split across warps | 261.457 | 11.777 us | rejected |
| Split warps, two K accumulators | 260.361 | not profiled | rejected |

Explicit activation reuse lowered registers from 56 to 54, and splitting the
value and gate projections lowered them to 36. Neither lower register count
offset the lost instruction-level parallelism and less favorable scheduling.

## Rejected Q8_1 vector quantization

The scalar Q8_1 quantizer already performs coalesced loads, warp reductions,
and coalesced byte stores. Two wider SM70 kernels were tested after the earlier
64/128/256/512-thread block-size sweep.

| Values per thread | Quantizer total | Qwen 3.5 Q8_0 | Qwen 3.5 Q4_0 | Result |
| ---: | ---: | ---: | ---: | --- |
| 1 | 130.093 ms | 262.326 mean | 276.683 mean | control |
| 2 | 131.978 ms | 262.425 | 276.216 | rejected |
| 4 | 133.994 ms | 262.352 | 276.026 | rejected |

The reduced grids did not compensate for per-thread max, sum, conversion, and
packing work. Both variants were reverted.

## Validation

All 47 supported GET_ROWS CPU-reference cases pass. Compute Sanitizer memcheck
reports zero errors for Qwen 3.5 Q8_0 generation. Eight-chunk WikiText-2
estimates remain 18.3635 for Q8_0 and 21.8608 for Q4_0. The retained path is a
bitwise copy and does not change arithmetic.
