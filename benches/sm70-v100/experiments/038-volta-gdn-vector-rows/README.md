# Volta GDN vector rows

For aligned scalar-gated heads of width 128, the SM70 Gated DeltaNet kernel now
assigns four adjacent state rows to each lane. State, q, and k are loaded and
stored as `float4`. The previous mapping assigned rows separated by 32 elements
to a lane and emitted four scalar loads for each array.

The optimized specialization is selected only on Volta when all vectorized
pointers and strides are 16-byte aligned. KDA, other head widths, unaligned
layouts, and non-Volta devices retain the original kernel and reduction order.
Both specializations use 48 registers with no stack, shared, or local memory.

Measurements used only the V100 UUID, a 150 W power limit, and a fixed 1200 MHz
SM clock. Prompt values use 100 repetitions. The candidate was measured on
both sides of the control, and the table reports their mean.

| Model and test | Control | Candidate | Change |
| --- | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q4_0 pp512 | 14226.68 | 14428.77 | +1.42% |
| Qwen 3.5 0.8B Q8_0 pp512 | 14557.47 | 14761.59 | +1.40% |
| Qwen 3.5 0.8B Q4_0 tg128 | 263.56 | 264.88 | +0.50% |
| Qwen 3.5 0.8B Q8_0 tg128 | 246.85 | 246.89 | +0.02% |

Nsight Systems reduces the 54-call GDN kernel average from 436.535 us to
410.294 us (-6.01%). SM70 SASS emits one 128-bit state load and two 128-bit
q/k loads in place of twelve scalar loads, while preserving spill-free resource
usage.

All 36 GATED_DELTA_NET backend cases pass. Compute Sanitizer memcheck on a real
Q4_0 pp512 run reports zero errors. Eight-chunk WikiText-2 perplexity changes
from 21.8601 to 21.8595 for Q4_0 and from 18.3634 to 18.3631 for Q8_0. The small
decrease is not an accuracy loss; it comes from grouping the same FP32 terms by
adjacent rows before the warp reduction.
