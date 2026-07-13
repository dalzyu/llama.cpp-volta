# Qwen uBatch sweep at 150 W

Control: `ub=512`. This experiment was rejected because the control is fastest
for all three models.

| uBatch | Qwen 3.5 Q4_0 pp512 | Qwen 3.5 Q8_0 pp512 | Qwen 3.6 Q2_K pp512 |
| ---: | ---: | ---: | ---: |
| 32 | 4187.407 | 3965.987 | 274.691 |
| 64 | 5209.054 | 6101.531 | 174.201 |
| 96 | 6319.188 | 7139.277 | 253.324 |
| 128 | 7920.875 | 8933.799 | 323.715 |
| 192 | 9093.507 | 10153.010 | 389.758 |
| 256 | 10829.645 | 11820.243 | 526.065 |
| 384 | 10889.176 | 11979.473 | 497.792 |
| 512 | 14054.652 | 14840.073 | 677.115 |

The non-monotonic Qwen 3.6 results at 64 and 384 do not change the conclusion.
The full 512-token micro-batch is 28.7% faster than the next-best result at 256.
Peak VRAM was 12328 MiB and peak temperature was 63 C.
