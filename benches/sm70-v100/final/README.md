# Final retained-build validation

Revision `d2426432c` was rebuilt in Release mode and tested at the 150 W V100
power limit. Only the V100 UUID was visible to llama.cpp. The final benchmark
includes one cold sample in each reported mean.

| Model | pp512 tok/s | tg128 tok/s | PPL |
| --- | ---: | ---: | ---: |
| Gemma 4 12B Q4_0 | 1495.276 | 70.438 | 308.3287 |
| Qwen 3.5 0.8B Q4_0 | 14048.091 | 309.843 | 21.8601 |
| Qwen 3.5 0.8B Q8_0 | 14935.007 | 288.600 | 18.3634 |
| Qwen 3.6 27B Q2_K | 684.167 | 29.610 | 7.1732 |

Prompt throughput is not affected by the retained MMVQ changes and remains
within ordinary run variation. The fused Q2_K and Q3_K occupancy changes raise
Qwen 3.6 generation by 3.03% over the previous final build. Controlled
generation comparisons are recorded in experiments 016, 018, 019, 029, and
030. The changes do not alter memory allocation.

Focused final backend tests pass 20/20 K-quant, 14/14 Q4_0, and 16/16 Q8_0
cases. The PPL values are the final eight-chunk WikiText-2 checks; no source
changes followed those checks.
