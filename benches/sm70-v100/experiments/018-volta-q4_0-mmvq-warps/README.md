# Volta Q4_0 MMVQ warp tuning

The normal-K Q4_0 matrix-vector kernel now uses two warps on Volta while the
small-K specialization keeps four. Other quant types, column counts, and GPU
architectures keep their previous launch parameters.

| Configuration | Gemma 4 tg128 tok/s |
| --- | ---: |
| One warp | 68.535 |
| Two warps | 70.453 |
| Four warps | 69.609 |
| Eight warps | 63.748 |

In the longer rebuilt pair, two warps improved Gemma 4 from 69.442 to 70.252
tok/s (+1.17%). Qwen 3.5 keeps four warps for its small-K path and measured
305.801 versus 305.448 tok/s (+0.12%, effectively neutral).

The standard eight-chunk perplexity results are exactly unchanged. A stricter
batch-1 comparison exercises the tuned path token by token. Candidate and
control have identical CPU-relative NMSE to nine decimal places on two
deterministic Gemma-sized tensor shapes, and both are well inside the project's
5e-4 tolerance. The temporary test cases and deterministic seed were removed.

The retained build measured 1509.482 pp512 and 70.718 tg128 on Gemma 4, with
7582 MiB peak VRAM. Normal-K shared reduction storage falls by 66.7%, from
384/768 bytes to 128/256 bytes for unfused/fused kernels. No model-level VRAM
change was measurable.
