# Volta prompt GEMM exploration

This experiment tested three ways to reduce the prompt-processing time left in
dequantization and cuBLAS GEMMs after the Gated DeltaNet optimization. None was
retained in the runtime.

All end-to-end measurements used only the V100 UUID, a 150 W power limit, a
fixed 1200 MHz SM clock, and 100 llama-bench repetitions unless noted.

## Legacy cuBLAS algorithms

The Qwen 3.5 Q4_0 prompt graph uses eight FP16-input, FP32-output GEMM shapes.
Forcing legacy tensor-op algorithm IDs 100 through 115 did not improve the
default algorithm. The default measured 14436.36 tok/s. Algorithm 101 was the
closest explicit selection at 14191.20 tok/s (-1.70%).

## Q8_0 dequantization packing

A 64-thread block processed two independent 2048-value dequantization units,
with separate shared-memory slices and warp synchronization. The Q8_0 prompt
result changed from 14768.56 to 14777.09 tok/s (+0.06%), and Qwen 3.6 Q2_K
changed from 719.65 to 720.56 tok/s (+0.13%). Nsight Systems measured only a
0.34% reduction in the Q8_0 dequantization kernel average. A one-warp
`__syncwarp()` variant measured 14763.29 tok/s. Both changes were noise or
negative and were reverted.

## cuBLASLt shape probe

`volta_gemm_probe.cu` compares the legacy default with every cuBLASLt heuristic
returned for one FP16-input, FP32-output shape. A 64 MiB workspace was allowed.
The strongest isolated results used split-K for the two Qwen projection shapes:

| M | N | K | Legacy ms | Best cuBLASLt ms | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 512 | 1024 | 0.029962 | 0.027157 | -9.36% |
| 1024 | 512 | 1024 | 0.036884 | 0.037110 | +0.61% |
| 2048 | 512 | 1024 | 0.049418 | 0.049603 | +0.37% |
| 3584 | 512 | 1024 | 0.081265 | 0.079831 | -1.76% |
| 4096 | 512 | 1024 | 0.080814 | 0.080732 | -0.10% |
| 6144 | 512 | 1024 | 0.124006 | 0.122040 | -1.59% |
| 1024 | 512 | 2048 | 0.073380 | 0.054743 | -25.40% |
| 1024 | 512 | 3584 | 0.098959 | 0.086140 | -12.95% |

The 16x512x1024 shape was also a tie: 0.009974 ms for legacy and 0.009892 ms
for cuBLASLt.

## Graph integration

The two large projection shapes were integrated with cached descriptors and
the probe-selected algorithms. They reduced their profiled GEMM kernel total,
but the end-to-end result was quantization-dependent:

| Selection | Q4_0 pp512 | Change | Q8_0 pp512 | Change |
| --- | ---: | ---: | ---: | ---: |
| Adjacent legacy control | 14473.26 | - | 14763.83 | - |
| K=2048 and K=3584 | 14479.90 | +0.05% | 14806.72 | +0.29% |
| K=2048 only | 14405.12 | -0.47% | 14765.22 | +0.01% |
| K=3584 only | 14463.06 | -0.07% | 14780.31 | +0.11% |
| Five probe-positive shapes | 14382.38 | -0.63% | 14704.34 | -0.40% |

Split-K adds reduction kernels and changes overlap with dequantization in the
real graph. The isolated GEMM wins therefore did not translate consistently.
The runtime integration was reverted instead of adding a Volta-only descriptor
cache, workspace allocation, and exact-shape policy for a 0.05% to 0.29% result.
