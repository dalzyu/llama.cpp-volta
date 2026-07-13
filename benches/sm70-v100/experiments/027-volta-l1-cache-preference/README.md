# Volta L1 cache preference

NVIDIA documents that Volta combines L1 and shared memory in a 128 KB cache
and exposes a runtime preference for the split. A Volta-only
`cudaFuncCachePreferL1` device policy was tested against the retained build.

| Model and test | Default | Prefer L1 | Change |
| --- | ---: | ---: | ---: |
| Qwen 3.6 27B pp512 | 685.239 | 651.895 | -4.87% |
| Qwen 3.6 27B tg128 | 28.740 | 29.028 | +1.00% |
| Qwen 3.5 Q8_0 tg128 | 289.048 | 288.959 | -0.03% |
| Gemma 4 12B tg128 | 70.364 | 70.236 | -0.18% |

The isolated Qwen 27B generation increase does not reproduce on the other
models and comes with a large prompt-processing loss. The default cache policy
was restored.

Reference: https://docs.nvidia.com/cuda/volta-tuning-guide/
