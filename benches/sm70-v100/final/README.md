# Final retained-build validation

Revision `6cb21d3ad` was rebuilt in Release mode and tested at the 150 W V100
power limit. Only the V100 UUID was visible to llama.cpp. The reported means
omit the first cold sample; repetition counts and standard deviations are in
`results.tsv`.

| Model | pp512 tok/s | tg128 tok/s | PPL |
| --- | ---: | ---: | ---: |
| Gemma 4 12B Q4_0 | 1672.267 | 70.199 | 308.3287 |
| Qwen 3.5 0.8B Q4_0 | 14575.425 | 313.484 | 21.8601 |
| Qwen 3.5 0.8B Q8_0 | 14759.825 | 291.220 | 18.3634 |
| Qwen 3.6 27B Q2_K | 709.457 | 30.834 | 7.1732 |

The Q2_K dequantizer block-packing follow-up is recorded in experiment 032. Its
paired p512 control and packed measurements are 690.861 and 719.799 tok/s
(+4.19%). The Q2_K and Q3_K device-code changes raise
bracketed Qwen 3.6 generation by 3.22%, from 29.758 to 30.716 tok/s. The earlier
retained-build revision measured 30.535 tok/s in its final sample. The Q4_0
half2/two-work-unit change is recorded in experiment 033. Controlled generation
comparisons are recorded in experiments 016, 018, 019, 029, 030, 031, and 032.
The float get_rows change and its paired checks are recorded in experiment 034.
The rejected remaining-kernel sweep is recorded in experiment 035.
The changes do not alter memory allocation.

Focused final backend tests pass 47/47 GET_ROWS, 20/20 K-quant, 14/14 Q4_0,
and 16/16 Q8_0 cases. CUDA memcheck reports zero errors for GET_ROWS. The PPL
values are the post-commit eight-chunk WikiText-2 checks.
