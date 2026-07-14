# Final retained-build validation

Revision `08818b4e2` was rebuilt in Release mode and tested at the 150 W V100
power limit. Only the V100 UUID was visible to llama.cpp. The reported means
omit the first cold sample; repetition counts and standard deviations are in
`results.tsv`.

| Model | pp512 tok/s | tg128 tok/s | PPL |
| --- | ---: | ---: | ---: |
| Gemma 4 12B Q4_0 | 1684.107 | 70.014 | 308.3287 |
| Qwen 3.5 0.8B Q4_0 | 14793.329 | 310.945 | 21.8601 |
| Qwen 3.5 0.8B Q8_0 | 14908.723 | 289.009 | 18.3634 |
| Qwen 3.6 27B Q2_K | 708.070 | 30.535 | 7.1732 |

The Q2_K dequantizer block-packing follow-up is recorded in experiment 032. Its
paired p512 control and packed measurements are 690.861 and 719.799 tok/s
(+4.19%). The Q2_K and Q3_K device-code changes raise
bracketed Qwen 3.6 generation by 3.22%, from 29.758 to 30.716 tok/s. The final
revision measured 30.535 tok/s in the final five-run sample. The Q4_0
half2/two-work-unit change is recorded in experiment 033. Controlled generation
comparisons are recorded in experiments 016, 018, 019, 029, 030, 031, and 032.
The changes do not alter memory allocation.

Focused final backend tests pass 20/20 K-quant, 14/14 Q4_0, and 16/16 Q8_0
cases. The PPL values are the final eight-chunk WikiText-2 checks; no source
changes followed those checks.
