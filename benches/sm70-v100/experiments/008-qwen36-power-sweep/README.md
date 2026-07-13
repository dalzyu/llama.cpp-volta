# Qwen 3.6 27B power sweep

This diagnostic sweep uses three repetitions at each power limit. It is not a
software optimization. The V100 was restored to 150 W when the sweep finished.

| Power limit | pp512 tok/s | tg128 tok/s | Avg. active power | Avg. active SM clock |
| ---: | ---: | ---: | ---: | ---: |
| 150 W | 689.633 | 27.839 | 147.7 W | 1083.1 MHz |
| 225 W | 838.503 | 32.660 | 218.4 W | 1329.9 MHz |
| 300 W | 915.386 | 35.365 | 285.8 W | 1497.0 MHz |

Raising the limit from 150 W to 300 W improves pp512 by 32.7% and tg128 by
27.0%. Qwen 3.6 decode is therefore substantially compute/power limited on this
V100, unlike the Gemma 4 Q4_0 decode result that plateaued near 225 W.

Peak VRAM was unchanged at 12326 MiB. The highest observed temperature was
77 C during the final 300 W run.
