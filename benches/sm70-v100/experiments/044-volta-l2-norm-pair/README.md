# Volta paired L2 normalization

Qwen 3.5 and Qwen 3.6 build sibling Q and K L2 normalizations around a view
node. On SM70 generation, each normalization launches only 16 one-warp
blocks, so launch overhead is larger than the row arithmetic.

The retained fusion matches `L2_NORM -> VIEW -> L2_NORM` and writes both
normalization outputs with one kernel launch. The x-grid contains the rows of
both tensors; each block selects one source and destination while retaining
the original 32-thread row mapping, accumulation order, scale calculation,
and output mapping.

Dispatch requires SM70, F32 tensors, fewer than 1024 columns, equal input
layouts and epsilon, distinct contiguous outputs, and contiguous input rows.
All other patterns and architectures keep the existing kernels.

## Generation results

All measurements used only the V100 UUID, a 150 W power limit, fixed 1200 MHz
SM and 877 MHz memory clocks, and tg128. Candidate A and B bracket the control.

| Model | Candidate A | Control | Candidate B | Candidate mean change |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q8_0 | 270.288132 | 265.539164 | 270.241858 | +1.78% |
| Qwen 3.5 0.8B Q4_0 | 284.697730 | 279.091027 | 284.353841 | +1.95% |
| Qwen 3.6 27B Q2_K | 31.426142 | 31.201691 | 31.311124 | +0.53% |

The final restricted kernel, measured after sharing the equal stride and
epsilon arguments, reaches 270.407565 tok/s on Q8_0 and 284.709464 tok/s on
Q4_0. Gemma 4 does not launch this L2 normalization pattern and is unchanged.

## Prompt results

Pairing remains beneficial at pp512 even though the larger grids reduce the
relative importance of one launch.

| Model | Candidate A | Control | Candidate B | Candidate mean change |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q8_0 | 15188.492599 | 15167.441965 | 15225.363788 | +0.26% |
| Qwen 3.5 0.8B Q4_0 | 14846.893149 | 14841.820386 | 14896.798868 | +0.20% |

The final restricted kernel confirms 15199.822860 tok/s on Q8_0 and
14860.099165 tok/s on Q4_0.

## Profile

Over three Qwen 3.5 Q8_0 tg128 runs, the control launches 13,824 single-tensor
L2 kernels totaling 30.900 ms. The final path launches 6,912 paired kernels
totaling 16.534 ms. Launch count falls 50% and total L2 time falls 46.49%.
The paired SM70 kernel uses 40 registers per thread and no shared, local, or
stack memory. It does not change device allocations or model VRAM use.

## Validation

All 20 standalone L2 normalization cases and all 36 GDN cases pass their CPU
references. Compute Sanitizer memcheck reports zero errors for both pp128 and
tg16 model runs through the fused graph.

The pair preserves the original per-row reduction order. Eight-chunk
WikiText-2 estimates remain exactly 18.3641 for Q8_0 and 21.8604 for Q4_0 at
the displayed precision, matching the preceding RMS normalization commit.
