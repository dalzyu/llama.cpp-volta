# Nsight Systems kernel profiles

These profiles use the same model, V100 UUID, 150 W power limit, and llama.cpp
revision as the baseline. Each profile is a single no-warmup measurement under
Nsight Systems 2025.1.3. Profiler overhead makes the reported llama-bench speed
unsuitable for performance comparison; only the kernel-time distribution is
used to choose optimization targets.

## Prompt processing

The pp512 profile measured 1257.89 tok/s under the profiler. Its leading CUDA
kernels were:

| Kernel | Time | Calls |
| --- | ---: | ---: |
| CUTLASS sm_70 tensor-op GEMM, f16 128x128 TN | 45.2% | 240 |
| Q4_0 to f16 dequantization | 34.1% | 328 |
| Flash attention f16 256x256 | 4.5% | 40 |
| Volta tensor-op GEMM, fp16 64x128 TN | 4.3% | 80 |
| f32 to f16 conversion | 3.1% | 328 |

Q4_0 dequantization plus tensor-core GEMM account for 79.3% of GPU kernel time.

## Token generation

The tg128 profile measured 66.61 tok/s under the profiler. Its leading CUDA
kernels were:

| Kernel | Time | Calls |
| --- | ---: | ---: |
| Q4_0 matvec, unfused | 42.0% | 29824 |
| Q4_0 matvec, fused | 30.8% | 6144 |
| Fused RMS normalization | 5.1% | 12288 |
| Flash attention vector kernel | 4.7% | 5120 |
| Q8_1 quantization | 4.5% | 35968 |

The two Q4_0 matrix-vector kernels account for 72.8% of GPU kernel time.

The full Nsight reports are intentionally kept in `/tmp` because each report is
about 100 MiB. The CSV summaries retain the kernel totals needed for this work.

## Static resource usage

Nsight Compute hardware counters are restricted by the NVIDIA driver on this
host. `cuobjdump --dump-resource-usage` still provides the compiler resource
counts for the exact sm_70 cubins:

| Kernel | Registers/thread | Shared memory | Local memory |
| --- | ---: | ---: | ---: |
| Q4_0 to f16 dequantization | 20 | 0 B | 0 B |
| Q4_0 matvec, unfused | 54 | 384 B | 0 B |
| Q4_0 matvec, fused | 40 | 768 B | 0 B |

The dequantization kernel launches one 32-thread warp per block. GV100 supports
up to 32 resident blocks and 64 resident warps per SM, so the block limit caps
this kernel at 50% theoretical warp occupancy even though registers and shared
memory are not limiting it.

## Qwen 3.5 Q4_0

Qwen 3.5 uses three Gated DeltaNet layers for every full-attention layer. In the
0.8B Q4_0 pp512 profile, Gated DeltaNet is the largest individual kernel at
21.7%, followed by Q4_0 dequantization at 13.9%. Dequantization remains a useful
cross-model target, but the upper bound is lower than for Gemma 4.

For tg128, quantized matrix-vector kernels account for 54.1% of GPU kernel time:
17.7% Q8_0, 36.4% Q4_0 across fused, unfused, and small-K variants. Gated
DeltaNet itself accounts for 2.3% during generation.

## Qwen 3.6 27B Q2_K

The pp512 profile spends 50.9% of GPU kernel time in the sm70 tensor-core GEMM
and 29.5% in K-quant and Q8_0 dequantization. Q2_K and Q3_K dequantization alone
account for 24.5%. Gated DeltaNet accounts for 8.1%.

For tg128, quantized matrix-vector kernels account for 82.5% of GPU kernel time.
Q2_K and Q3_K variants account for 66.1%, making K-quant matrix-vector work the
primary generation target for this model. Gated DeltaNet accounts for 1.2%.
