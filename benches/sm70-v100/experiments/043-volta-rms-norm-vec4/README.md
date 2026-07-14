# Volta vector RMS norm multiply

This experiment specializes the large fused RMS norm and multiply kernel for
aligned SM70 tensors. The retained path uses 256 threads and processes four
adjacent values per thread with `float4` loads and stores. The previous kernel
uses 1024 threads and scalar accesses.

Dispatch is limited to SM70, no fused add, at least 1024 columns, a multiplier
with the full column count, and 16-byte-compatible pointers and row strides.
All other architectures, shapes, broadcasts, and alignments retain the scalar
path.

## Vector-width sweep

All measurements used only the V100 UUID, a 150 W power limit, fixed 1200 MHz
SM and 877 MHz memory clocks, and tg128. The float2 and float8 variants reduce
or increase the thread count so that one block covers the same 1024 values.

| Values per thread | Threads | Qwen 3.5 Q8_0 | Qwen 3.5 Q4_0 | Result |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1024 | 263.288139 | 276.651721 | control |
| 2 | 512 | 265.282423 | 278.555000 | rejected |
| 4 | 256 | 265.605280 | 279.315562 | retained |
| 8 | 128 | 265.364586 | 278.633487 | rejected |

Float4 wins on both quantizations. Float2 leaves too many threads and float8
reduces block-level parallelism too far.

## Controlled results

The final values compare the scalar control with the mean of two float4 runs
bracketing the control.

| Model | Candidate A | Control | Candidate B | Candidate mean change |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q8_0 | 265.613845 | 263.288139 | 265.596715 | +0.88% |
| Qwen 3.5 0.8B Q4_0 | 279.442832 | 276.651721 | 279.188292 | +0.96% |
| Qwen 3.6 27B Q2_K | 31.607257 | 31.320814 | 31.419635 | +0.62% |
| Gemma 4 12B Q4_K_XL | 68.928171 | 68.575387 | 68.892311 | +0.49% |

Nsight Systems shows the affected Qwen 3.5 Q8_0 kernel falling from 65.954
to 51.244 ms over 18,816 launches (-22.30%). Average launch duration falls
from 3.505 to 2.723 us. The SM70 scalar and vector kernels both use 32
registers per thread, no local or stack memory, and 128 bytes of dynamic
shared memory. Reducing the block from 1024 to 256 threads improves scheduling
granularity without changing device allocations or model VRAM use.

## Validation

All 21 focused RMS norm CPU-reference cases and all 72 fused RMS norm, multiply,
and rope graph cases pass. Compute Sanitizer memcheck reports zero errors on a
Qwen 3.5 Q8_0 generation run.

The vector path changes floating-point reduction order. Eight-chunk WikiText-2
estimates are 18.3641 for Q8_0 and 21.8604 for Q4_0, versus 18.3635 and 21.8608
before this experiment. The differences are much smaller than the reported
1.23699 and 1.50866 estimate uncertainties and do not indicate an accuracy
regression.
