# Volta RMS normalization block size

Large-row RMS normalization was tested with 256, 512, and 1024 threads per
block on Volta. The 512-thread result preserves the existing 256-thread path
for rows smaller than 1024 columns.

| Large-row threads | Qwen 3.5 Q4_0 tg128 tok/s | Change |
| ---: | ---: | ---: |
| 256 | 309.230 | -0.54% |
| 512 | 310.659 | -0.08% |
| 1024 | 310.894 | baseline |

The existing 1024-thread launch remains unchanged. The reduction order differs
between launch sizes, so neither slower alternative proceeded to correctness
validation.
