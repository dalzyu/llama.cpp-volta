# Volta Q6_K fused bias multi-row MMVQ

Qwen 3.5 9B Q4_K_M contains 16 Q6_K feed-forward down projections with
12288 input columns and 4096 output rows. Decode fuses each matrix-vector
product with its following F32 residual addition. In a 128-token profile plus
one warmup, the generic fused Q6_K kernel launches 2064 times and averages
71.5241 us, accounting for 11.3% of kernel time.

The retained SM70 path computes two adjacent rows per block and halves the x
grid from 4096 to 2048. Dispatch requires Q6_K, one destination column,
normal K, no IDs, one channel and sample, a contiguous dense destination,
exactly 4096 even rows, and bias/add-only fusion. Fused gates, scales, multiple
channels or samples, small-K, indirect, non-dense, and non-Volta cases keep the
existing kernel. Experiment 079's large non-fused Q6_K output path is
unchanged.

## Geometry sweep

The generic two-warp kernel first received a row-count template override in
experiment 079. Reusing it for fused Q6_K keeps each row's two-warp reduction
order while accumulating two independent rows in one block.

| Q6_K fused geometry | Registers | Shared memory | Kernel average | Change | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| One row | 48 | 256 B | 71.5241 us | control | control |
| Two rows | 64 | 512 B | 69.8050 us | -2.40% | retained |
| Four rows | 110 | 1024 B | 78.6540 us | +9.97% | rejected |

None of these kernels uses local or stack memory. Four rows reduced the grid
again but its register pressure lowered occupancy enough to erase the gain.

The same experiment tested Q4_K before Q6_K. Applying two rows to all fused
Q4_K launches measured 100.743254 tok/s versus 100.906386 control. Profiling
showed that the 12288-row family slowed from 77.1823 to 77.9611 us (+1.01%),
while the 4096-row family improved from 35.3341 to 34.6910 us (-1.82%). A
4096-only Q4_K bracket still regressed: candidate mean 100.773631 versus
100.934080 control (-0.159%). Both Q4_K variants were removed.

## Fixed-clock performance

Canonical runs used only V100 UUID
`${V100_GPU_0_UUID}`, a 300 W limit, a 1192 MHz locked
graphics clock, 877 MHz memory, and 50 ms telemetry. A temporary runtime switch
disabled only the new Q6_K fused dispatch. It is absent from retained source.

The first five-pass `tg1024` bracket measured candidate A 101.037475, control
100.966395, and candidate B 101.134138 tok/s, a candidate mean of 101.085807
or +0.1183%. Three additional independent one-pass candidate/control/candidate
groups started at 60 C or below:

| Group | Candidate A | Control | Candidate B | Candidate mean | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 100.959768 | 100.875623 | 101.143727 | 101.051748 | +0.1746% |
| 2 | 100.962463 | 100.862397 | 101.132060 | 101.047261 | +0.1833% |
| 3 | 100.955926 | 100.856253 | 100.903354 | 100.929640 | +0.0728% |
| Combined | | 100.864758 | | 101.009550 | +0.1436% |

All 2100 busy samples in those groups were exactly 1192 MHz. Each group peaked
at 71 C with no software power, software thermal, or hardware thermal cap.
Peak VRAM remained 5812 MiB.

A ten-pass repeat and a later heat-soaked five-pass control were discarded.
They accumulated 397 and 150 software-thermal samples and dropped to 1125 and
1132 MHz respectively. They are diagnostic evidence only and are not included
in the retained comparison.

## Kernel profile

The final narrowed candidate profile includes one warmup and 128 measured
tokens.

| Variant | Grid x | Launches | Total time | Average |
| --- | ---: | ---: | ---: | ---: |
| Control one row | 4096 | 2064 | 147.625743 ms | 71.5241 us |
| Candidate two rows | 2048 | 2064 | 144.077483 ms | 69.8050 us |

The retained kernel is 2.40% faster and saves 3.548260 ms across the trace, or
27.506 us per model evaluation. The final restricted bias-only predicate still
selects all 16 target projections.

## Canonical inactive controls

Prompt processing does not use this single-column path, and Gemma has no
matching Q6_K 4096-row fused projection.

| Model and test | Production | Initial 1192 MHz reference | Change | Busy samples | Peak temperature | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 9B Q4_K_M pp32768 | 1994.705666 | 1997.408054 | -0.1353% | 674 | 64 C | 6912 MiB |
| Gemma 4 12B Q4_K_XL pp32768 | 1318.863866 | 1318.107017 | +0.0574% | 1001 | 71 C | 8404 MiB |
| Gemma 4 12B Q4_K_XL tg1024 | 69.295612 | 69.339006 | -0.0626% | 330 | 64 C | 7624 MiB |

Every busy sample was exactly 1192 MHz and no cap reason was active.

## Correctness and safety

The final candidate reports WikiText-2 PPL 5.9046 with uncertainty 1.87941.
Its context-128 logits are byte-identical to experiment 079, with SHA-256
`be3f0c3b86215d5be67b9bba474d19156c3a9b6bfd87a55196772fd434fb5bd0`.

A temporary `MUL_MAT_VEC_FUSION` case exercised Q6_K bias-only fusion at
`m=1,n=4096,k=1024`, selected the retained path, and matched the CPU reference.
Memcheck, initcheck, synccheck, and racecheck reported zero errors or hazards
on that exact case. The temporary test was then removed. The four tools also
reported zero findings on a real one-token Qwen 3.5 9B graph. The focused Q6_K
MUL_MAT sweep passes 11/11 cases. The first complete CUDA sweep passed
12994/12995 cases after an unrelated TOPK_MOE comparison exceeded its exact
tolerance. That case then passed 10/10 isolated runs, and a complete rerun
passed 12995/12995. The architecture graph tests completed successfully and
the remaining CTest set passed 53/53.
