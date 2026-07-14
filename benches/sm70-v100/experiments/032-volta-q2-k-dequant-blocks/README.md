# Volta Q2_K dequantizer block packing

The Q2_K conversion kernel originally used one 64-thread CUDA block for one
256-value superblock. On SM70, two independent work units fit naturally in a
128-thread block. The retained mapping assigns each half-block to one
superblock and keeps the original per-thread value mapping:

```text
i   = 2 * blockIdx.x + threadIdx.x / 64
tid = threadIdx.x % 64
```

The launch count is reduced from `nb` to `(nb + 1) / 2`. An explicit
superblock-count guard handles odd `nb` values, including the 256-element
backend test case, without reading or writing a second nonexistent block.

| Variant | Threads | p512 tok/s | Stddev | Change |
| --- | ---: | ---: | ---: | ---: |
| Original one-superblock launch | 64 | 690.861 | 1.885 | - |
| Packed two-superblock launch | 128 | 719.799 | 2.326 | +4.19% |

The paired runs used the Qwen 3.6 27B Q2_K model at 150 W on the V100. The
clean retained build also measured 716.047 tok/s in a separate five-run sample;
the variation is consistent with the V100 thermal ramp seen throughout the
benchmark set.

The final Nsight Systems p512 profile reports 254.151 ms over 896 Q2_K
dequantizer calls, 9.0% of GPU kernel time (283.651 us per call). The same
profile reports 251.747 ms for Q3_K dequantization, 250.908 ms for GDN, and
1563.499 ms for tensor-core GEMM. No memory allocation or perplexity change was
observed.

Focused backend coverage passes 20/20 K-quant cases, 14/14 Q4_0 cases, and
16/16 Q8_0 cases. Compute Sanitizer memcheck reports zero errors for the odd
superblock Q2_K case. Eight-chunk WikiText-2 perplexity remains exactly
`7.1732 +/- 0.40717`.

Nearby candidates were rejected: Q3_K and Q4_K two-superblock packing did not
beat noise, Q4_0 packing was neutral, packed Q2_K scale loads were neutral, the
fast exponential GDN variant was neutral, and reusing the stored Q8 sum failed
the backend accuracy tolerance.
