# Final retained-build validation

Revision `31c84a820` was rebuilt in Release mode and tested at the 150 W V100
power limit. Only the V100 UUID was visible to llama.cpp. The final benchmark
includes one cold sample in each reported mean.

| Model | pp512 tok/s | tg128 tok/s | PPL |
| --- | ---: | ---: | ---: |
| Gemma 4 12B Q4_0 | 1494.353 | 70.338 | 308.3287 |
| Qwen 3.5 0.8B Q4_0 | 14121.782 | 310.030 | 21.8601 |
| Qwen 3.5 0.8B Q8_0 | 15084.376 | 288.197 | 18.3634 |
| Qwen 3.6 27B Q2_K | 685.239 | 28.740 | 7.1732 |

Prompt throughput is not affected by the retained MMVQ changes and remains
within ordinary run variation. Controlled paired generation comparisons are
recorded in experiments 016, 018, and 019. Peak VRAM is unchanged from the
baseline.

Focused final backend tests pass 20/20 K-quant, 14/14 Q4_0, and 16/16 Q8_0
cases. The PPL values are the final eight-chunk WikiText-2 checks; no source
changes followed those checks.
