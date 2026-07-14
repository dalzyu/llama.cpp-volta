# Rejected embedded Q8 GDN rows

Experiment 059 appends the Q8_0 alpha and beta rows to the combined QKV grid.
This follow-up instead let the first QKV blocks compute those GDN rows after
writing their normal QKV results. The intent was to hide the extra work under
the many remaining QKV waves and avoid extending the grid.

The temporary kernel was byte-identical to the flattened-grid control. It
reduced the grouped kernel total from 0.409863 ms to 0.403556 ms across 36 tg32
launches, a 0.006307 ms reduction. SM70 register use increased from 48 to 50,
with no shared, local, or stack memory.

The larger fixed-clock measurement was neutral to slightly negative:

| Model | Repetitions | Embedded A | Flattened control | Embedded B | Embedded mean | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q8_0 | 100 | 325.917954 | 325.991607 | 325.837733 | 325.877844 | -0.0349% |

The launch count is identical in both variants, so the tiny GPU-time reduction
does not remove any host submission work. The extra control flow and registers
offer no measurable end-to-end benefit. The embedded kernel was removed and
the flattened grid from experiment 059 remains active.

Candidate and control logits at context 128 have SHA-256
`acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274`.
