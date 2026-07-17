# Rejected Volta Q5_K warp rows

Qwen 3.5 9B Q4_K_M has 24 Q5_K 4096x8192 attention QKV projections and 24
Q5_K 4096x4096 SSM output projections. Together their generic non-fused MMVQ
kernels account for 13.6% of the post-experiment-081 generation profile. The
generic Volta kernel gives four warps one row and reduces their partial sums
through 384 bytes of shared memory.

The prototype instead assigned one complete row to each warp. Independent
rows shared a block, each warp reduced only through shuffles, and no shared
memory was used. Dispatch was restricted to SM70, Q5_K, normal K, one output
column, no fusion or IDs, one channel and sample, dense output, and the exact
4096-column model shapes.

## Shape and geometry sweep

The first four-row prototype covered both Q5_K shapes. The 8192-row QKV kernel
improved, but the smaller SSM output kernel regressed enough to make their
combined device time worse.

| Shape | Generic total | Four-row total | Change |
| --- | ---: | ---: | ---: |
| 4096x8192 | 109.897722 ms | 107.809769 ms | -1.90% |
| 4096x4096 | 64.505413 ms | 67.522416 ms | +4.68% |
| Combined | 174.403135 ms | 175.332185 ms | +0.53% |

The 4096-row dispatch was removed before the row-packing sweep. One warmup
plus 128 measured tokens produced:

| Rows per block | Grid x | Kernel average | Change | Result |
| --- | ---: | ---: | ---: | --- |
| Generic | 8192 | 35.4967 us | control | control |
| 1 | 8192 | 35.9586 us | +1.3014% | rejected |
| 2 | 4096 | 35.4342 us | -0.1760% | rejected |
| 4 | 2048 | 34.8215 us | -1.9021% | best prototype |
| 8 | 1024 | 34.9441 us | -1.5567% | rejected |

Four rows use 38 SM70 registers, no shared, local, or stack memory, and 384
bytes of constant memory. The generic kernel uses 40 registers, 384 bytes of
shared memory, and 512 bytes of constant memory.

## Fixed-clock rejection

Two candidate/control/candidate `tg1024` groups used only V100 UUID
`${V100_GPU_0_UUID}`, a 300 W limit, locked 1192 MHz
graphics and 877 MHz memory clocks, and 50 ms telemetry.

| Group | Candidate A | Control | Candidate B | Candidate mean | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 101.950300 | 102.260293 | 101.990030 | 101.970165 | -0.2837% |
| 2 | 101.936557 | 102.220263 | 102.034479 | 101.985518 | -0.2296% |
| Combined | | 102.240278 | | 101.977842 | -0.2567% |

All 1386 busy samples were exactly 1192 MHz. Runs started between 45 C and
60 C, peaked at 65 C, and recorded no power or thermal cap. Peak VRAM was
5812 MiB.

The isolated 8192-row kernel gain did not translate to model throughput, and
the repeat reproduced the regression. The runtime selector, dedicated kernel,
and dispatch were removed. Experiment 081 remains the final retained source.
