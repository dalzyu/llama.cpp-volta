# CUDA graph optimization

Control: `GGML_CUDA_GRAPH_OPT` unset. This experiment was rejected.

| Model | Mode | pp512 tok/s | tg128 tok/s |
| --- | --- | ---: | ---: |
| Gemma 4 12B Q4_0 | control | 1500.789 | 69.655 |
| Gemma 4 12B Q4_0 | graph opt | 1500.345 | 69.315 |
| Qwen 3.5 0.8B Q4_0 | control | 14461.548 | 301.428 |
| Qwen 3.5 0.8B Q4_0 | graph opt | 14447.237 | 302.689 |
| Qwen 3.6 27B Q2_K | control | 686.274 | 27.579 |
| Qwen 3.6 27B Q2_K | graph opt | 680.695 | 27.402 |

The Qwen 3.5 tg128 increase is only 0.4% and is smaller than the run variance.
Gemma 4 and Qwen 3.6 regress in both tests. The default remains unchanged.
