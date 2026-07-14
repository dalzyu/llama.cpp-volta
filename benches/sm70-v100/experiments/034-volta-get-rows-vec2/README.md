# Volta float get_rows vectorization

The profiled `get_rows_float<float,float>` path copied one element per thread.
The candidate keeps all other source and destination types on the existing
kernel, but gives each float-to-float thread two adjacent elements and halves
the y grid for that specialization.

The V100 generation checks below used the existing 128-thread flash-vector
path. GPU clocks vary at the 150 W limit, so the values are reported as a
paired sanity check rather than a fixed-clock claim.

| Model | Scalar control | Float2 path | Change |
| --- | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q8_0 tg128 | 288.953 | 291.321 | +0.82% |
| Qwen 3.6 27B Q2_K tg128 | 30.535 | 30.898 | +1.19% |

The Qwen 3.5 Q4_0 check was 313.449 tok/s with the candidate and was neutral
within the clock variation. The float path passed all 47 supported GET_ROWS
backend cases, and the eight-chunk WikiText-2 gates remained unchanged:

```
Qwen 3.5 Q4_0  21.8601 +/- 1.50861
Qwen 3.5 Q8_0  18.3634 +/- 1.23695
Qwen 3.6 Q2_K   7.1732 +/- 0.40717
Gemma 4 Q4     308.3287 +/- 38.71784
```

The 256-thread flash-vector and Volta 128-thread float matrix-vector launch
alternatives were measured in the same sweep and reverted because they did not
produce a reproducible end-to-end gain.
