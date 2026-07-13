# Volta Q2_K and Q3_K two-warp MMVQ

This candidate changes single-column Q2_K and Q3_K MMVQ launches on Volta from
four warps to two. Other quant types and GPU architectures are unchanged.

| Revision | pp512 tok/s | tg128 tok/s | Peak VRAM |
| --- | ---: | ---: | ---: |
| Baseline | 691.610 | 27.721 | 12326 MiB |
| Q2_K and Q3_K two warps | 692.854 | 28.640 | 12326 MiB |

The candidate improves tg128 by 3.3%. The 0.2% pp512 difference is within run
variation, as expected because pp512 does not use the modified launch shape.
