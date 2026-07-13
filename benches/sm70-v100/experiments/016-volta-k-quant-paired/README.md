# Paired Volta K-quant MMVQ comparison

The control and candidate were each rebuilt immediately before measurement.
The compile interval allowed the V100 to cool between runs. Both runs reached a
peak temperature of 57 C.

| Configuration | tg128 tok/s | Avg. active power | Avg. active SM clock |
| --- | ---: | ---: | ---: |
| Four warps | 27.835 | 145.9 W | 1100.7 MHz |
| Two warps | 28.826 | 144.4 W | 1083.5 MHz |

Two warps improves tg128 by 3.56% despite the candidate running at a 1.6% lower
average SM clock. This confirms the speedup is not a boost-clock artifact.
