# Gemma 4 12B power sweep

The V100 was restored to 150 W after the sweep.

| Power limit | pp512 tok/s | vs 150 W | tg128 tok/s | vs 150 W |
| ---: | ---: | ---: | ---: | ---: |
| 100 W | 1018.817 | -31.5% | 58.121 | -16.1% |
| 150 W | 1486.908 | control | 69.275 | control |
| 180 W | 1626.656 | +9.4% | 72.315 | +4.4% |
| 200 W | 1708.068 | +14.9% | 73.599 | +6.2% |
| 225 W | 1784.745 | +20.0% | 74.808 | +8.0% |
| 250 W | 1840.111 | +23.8% | 74.997 | +8.3% |
| 275 W | 1898.474 | +27.7% | 75.045 | +8.3% |
| 300 W | 1943.326 | +30.7% | 75.042 | +8.3% |

TG plateaus at about 225 W, where the SM reaches its 1530 MHz maximum during
generation and board draw remains below the higher limits. PP continues to
scale because it draws more power and remains compute/power constrained.
Peak VRAM did not change with the power limit.
