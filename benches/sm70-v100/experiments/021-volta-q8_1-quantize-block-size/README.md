# Volta Q8_1 quantization block size

The Q8_1 row quantization kernel was tested with 64, 128, 256, and 512 threads
per block on Volta. The arithmetic and memory layout were unchanged.

| Threads | Qwen 3.5 Q4_0 tg128 tok/s | Qwen 3.5 Q8_0 tg128 tok/s |
| ---: | ---: | ---: |
| 64 | 310.922 | not run |
| 128 | 310.936 | 289.005 |
| 256 | 310.894 | 289.048 |
| 512 | 310.616 | not run |

The 64- and 128-thread results differ from the 256-thread baseline by less
than 0.02%, and 512 threads is slower. The existing 256-thread launch remains
unchanged.
