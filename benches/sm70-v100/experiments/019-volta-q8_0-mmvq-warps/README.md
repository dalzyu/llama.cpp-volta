# Volta Q8_0 MMVQ warp tuning

The normal-K Q8_0 matrix-vector kernel now uses two warps on Volta. The
small-K specialization keeps four warps, and other types and architectures are
unchanged.

| Normal-K warps | Qwen 3.5 Q8_0 tg128 tok/s |
| ---: | ---: |
| 1 | 288.549 |
| 2 | 289.121 |
| 4 | 279.227 |
| 8 | 224.689 |

Two warps improves the 20-repeat comparison by 3.54%. The standard eight-chunk
perplexity remains exactly 18.3634. With batch size one, candidate perplexity
is 16.4784, compared with 16.4650 for four warps and 16.5004 on CPU; the tuned
path is closer to the CPU reference. Focused backend tests pass 16/16.

The retained build measured 15116.325 pp512 and 288.319 tg128 including one
cold generation sample; the remaining tg samples are 289.187-289.538 tok/s.
Peak VRAM is unchanged at 1658 MiB. Shared reduction storage falls from
384/768 bytes to 128/256 bytes for normal-K unfused/fused kernels.
