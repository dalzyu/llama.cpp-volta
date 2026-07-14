# Remaining Volta kernel candidates

Three low-risk candidates were measured after the retained float get_rows
change. None was kept.

## Long-token SSM convolution

The long-token kernel was changed from a 32-token shared-memory tile to a
64-token tile. The larger tile reduced the number of blocks, but it lowered
Qwen 3.5 prompt throughput.

| Model | 32-token control | 64-token tile | Change |
| --- | ---: | ---: | ---: |
| Qwen 3.5 Q4_0 pp512 | 14575.4 | 14241.5 | -2.29% |
| Qwen 3.5 Q8_0 pp512 | 14759.8 | 14647.0 | -0.76% |

The 32-token split was restored.

## Non-contiguous concat block size

The non-contiguous concat kernel was tested with 128 threads instead of 256.
The paired results were mixed and stayed within V100 clock variation.

| Model | 256-thread control | 128-thread candidate | Change |
| --- | ---: | ---: | ---: |
| Qwen 3.5 Q4_0 pp512 | 14653.6 | 14721.6 | +0.46% |
| Qwen 3.5 Q8_0 pp512 | 14985.1 | 14912.2 | -0.49% |
| Gemma 4 Q4 pp512 | 1679.9 | 1679.5 | -0.02% |

The 256-thread launch was restored.

## Fused Q4_K and Q6_K launch bounds

Fused normal-K Q4_K and Q6_K MMVQ were given a Volta minimum occupancy bound
of 20 blocks per SM. Qwen 3.6 generation measured 31.034 tok/s with the bound
and 30.995 tok/s without it (+0.13%), which is not a reproducible gain. The
default bound was restored.
