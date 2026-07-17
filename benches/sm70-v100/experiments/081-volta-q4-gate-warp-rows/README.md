# Volta Q4_K fused gate warp rows

Qwen 3.5 9B Q4_K_M evaluates 32 fused Q4_K feed-forward up and gate
projections per decode token. Each projection has 4096 input columns and
12288 output rows. In a 128-token profile plus one warmup, the generic fused
kernel launches 4128 times, averages 77.2156 us, and accounts for 24.5% of
CUDA kernel time.

The generic Volta kernel gives two warps one output row and combines their
partial sums through shared memory. The retained kernel instead gives one
complete row to each warp. Two independent rows share a 64-thread block, but
each warp accumulates both the value and gate dot products, reduces them with
warp shuffles, applies SiLU in lane zero, and writes its result without shared
memory.

Dispatch requires SM70, Q4_K, normal K, one output column, no IDs, one channel
and sample, a contiguous dense destination, exactly 4096 input columns and
12288 output rows, no bias or scale, and a SWIGLU gate. Other shapes, types,
fusion modes, layouts, architectures, and batched or indirect cases keep the
generic kernel.

## Fixed-clock performance

All measurements exposed only V100 UUID
`${V100_GPU_0_UUID}`, used a 300 W limit, locked the
graphics clock at 1192 MHz and memory at 877 MHz, and sampled telemetry every
50 ms. A temporary same-binary switch disabled only the new dispatch. It is
absent from retained source.

Three independent candidate/control/candidate `tg1024` groups produced:

| Group | Candidate A | Control | Candidate B | Candidate mean | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 102.258508 | 100.944141 | 102.229445 | 102.243977 | +1.2877% |
| 2 | 102.255444 | 100.953726 | 102.247229 | 102.251337 | +1.2854% |
| 3 | 102.266348 | 100.956682 | 102.258569 | 102.262458 | +1.2934% |
| Combined | | 100.951516 | | 102.252590 | +1.2888% |

All 2074 busy samples were exactly 1192 MHz. Runs started between 43 C and
62 C, peaked at 67 C, and recorded no software power, software thermal, or
hardware thermal cap. Peak VRAM was 5812 MiB. The stripped production binary
measures 102.211735 tok/s in a separate `tg1024` run.

## Kernel profile and geometry

The final profile includes one warmup and 128 measured tokens.

| Kernel | Grid x | Launches | Total time | Average | Registers | Shared memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Generic two-warp row | 12288 | 4128 | 318.746110 ms | 77.2156 us | 48 | 256 B |
| Dedicated two warp-rows | 6144 | 4128 | 295.931673 ms | 71.6889 us | 48 | 0 B |

The retained kernel is 7.16% faster and saves 22.814437 ms across the trace,
or 176.856 us per model evaluation. Neither kernel uses local or stack memory.
The dedicated kernel uses 392 bytes of constant memory instead of 512 bytes.

A same-binary sweep varied only the number of independent warp-rows packed
into each block:

| Rows per block | Grid x | Kernel average | Change from two rows | Result |
| --- | ---: | ---: | ---: | --- |
| 1 | 12288 | 71.8164 us | +0.1779% | rejected |
| 2 | 6144 | 71.6889 us | control | retained |
| 4 | 3072 | 71.7695 us | +0.1124% | rejected |
| 8 | 1536 | 72.5324 us | +1.1766% | rejected |

All four variants use 48 SM70 registers and no shared, local, or stack memory.

## Canonical controls

The prompt path and Gemma do not select this exact single-column Q4_K gate
dispatch.

| Model and test | Production | Previous production | Change | Busy samples | Peak temperature | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 9B Q4_K_M pp32768 | 1993.939696 | 1994.705666 | -0.0384% | 670 | 65 C | 6912 MiB |
| Gemma 4 12B Q4_K_XL pp32768 | 1318.427188 | 1318.863866 | -0.0331% | 998 | 70 C | 8404 MiB |
| Gemma 4 12B Q4_K_XL tg1024 | 69.231227 | 69.295612 | -0.0929% | 336 | 64 C | 7624 MiB |

Every busy sample was exactly 1192 MHz and no cap reason was active.

## Correctness and safety

A temporary `MUL_MAT_VEC_FUSION` case exercised the exact Q4_K SWIGLU shape
at `m=1,n=12288,k=4096` and matched the CPU reference. Memcheck, initcheck,
synccheck, and racecheck reported zero errors or hazards on that case. The
temporary test was then removed, and the existing Q4_K fusion sweep passed
30/30 cases. The complete CUDA backend sweep passed 12995/12995 cases, the
architecture graph tests completed successfully, and the remaining CTest set
passed 53/53.

The candidate reports WikiText-2 PPL 5.9002 +/- 1.87612 versus control
5.9046 +/- 1.87941. Relative to control logits, it reports KL divergence
0.00047 +/- 0.00010, probability RMS difference 0.540 +/- 0.091%, and the
same top probability in 98.413 +/- 1.587% of comparisons. The reduction-order
change means its saved logits are not byte-identical to control.
