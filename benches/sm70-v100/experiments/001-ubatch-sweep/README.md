# Gemma 4 12B uBatch sweep at 150 W

Control: `ub=512`. This experiment was rejected.

| uBatch | pp512 tok/s |
| ---: | ---: |
| 8 | 297.781 |
| 16 | 473.471 |
| 32 | 693.544 |
| 48 | 803.267 |
| 56 | 818.265 |
| 64 | 352.965 |
| 80 | 380.579 |
| 96 | 467.909 |
| 112 | 501.907 |
| 128 | 606.186 |
| 192 | 763.391 |
| 256 | 1076.174 |
| 384 | 985.354 |
| 512 | 1466.862 |
| 768 | 1465.839 |
| 1024 | 1465.262 |

The V100 reached about 62 C by the end of the ordered sweep. Despite that
disadvantage, the control remained fastest. Values above 512 cannot increase
the physical micro-batch for a 512-token prompt.
