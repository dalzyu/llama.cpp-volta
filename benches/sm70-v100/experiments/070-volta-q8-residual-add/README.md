# Volta Q8 output projection residual epilogue

Qwen 3.5 evaluates 18 GDN layers per decode token. Each layer ends with a
contiguous Q8_0 2048x1024 output projection, reshapes its 1024-element result,
and launches a separate F32 residual add.

The retained SM70 path recognizes the exact MUL_MAT, RESHAPE, and ADD
subgraph. It requires one contiguous 2048-element F32 input column, matching
contiguous 1024-element residual and output tensors, stream zero, and the exact
Q8_0 weight shape. The subgraph checker proves that the projection and reshape
intermediates have no external users. All other types, shapes, streams, prompt
batches, cache-disabled runs, and non-Volta devices retain the original path.

The final output aliases the residual exactly. This is safe because lane zero
reads residual row `r` before overwriting the same row, and no thread reads
another row. Partial aliases and overlap with the projection weights or input
remain rejected.

The fused kernel preserves the retained two-rows-per-CTA dot products and warp
reduction order. Lane zero adds the residual to the already-rounded F32 dot
product and writes the final output. If the projection input is not already in
the graph-local Q8_1 cache, the launcher performs the same on-demand
quantization and cache allocation as the original MMVQ path.

## Fixed-clock generation result

Candidate and control came from one binary. A temporary environment switch
disabled only this graph fusion and was removed from the retained source. All
runs exposed only the V100 UUID and used a 150 W power limit, fixed 1200 MHz SM
clock, and 877 MHz memory clock.

| Run | Path | tg128 tok/s | Standard deviation |
| --- | --- | ---: | ---: |
| A | Fused projection and residual | 373.572616 | 0.673676 |
| Control | Separate projection and residual | 366.063720 | 0.494524 |
| B | Fused projection and residual | 373.429265 | 0.569379 |
| Candidate mean | Fused projection and residual | 373.500941 | |

The bracket improvement is 2.0317%. The final switch-free binary measures
373.511733 tok/s, 2.0346% above the matched control.

## Launch profile

Node-level traces cover 33 evaluated tokens. Grid dimensions isolate the
1024-row GDN output projections from the vocabulary projection.

| Path | Launches | GPU time | Average |
| --- | ---: | ---: | ---: |
| Control output projection | 594 | 4.556747 ms | 7.6713 us |
| Control residual add | 594 | 2.137221 ms | 3.5980 us |
| Fused projection and residual | 594 | 4.689451 ms | 7.8947 us |

The affected chain falls from 6.693968 ms to 4.689451 ms, a 29.95% reduction.
It saves 60.74 us and 18 launches per token. The fused and bare projection
kernels both use 48 registers per thread with no shared, local, or stack
memory on SM70. The fused constant parameter area grows from 384 B to 392 B.

## Cache dispatch precursor

The first launcher required the Q8_1 cache entry to exist before graph fusion
and therefore did not dispatch. The ordinary projection creates this cache
entry on demand. Mirroring that established behavior inside the exact fused
launcher activates the path without changing cache-disabled execution.

## Correctness and fallback validation

Candidate, disabled-control, eager, cache-disabled, and final switch-free Q8_0
runs produce byte-identical context-128 logits. PPL remains 11.9470 and SHA-256
is `acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274`.
Q4_0 graph and eager output are also byte-identical, with PPL 12.6381 and
SHA-256 `385e35ba44725c79a84b3ed86297d71de16280492d40086cb04fa0ed819e14a0`.

The focused RMS and matrix-vector fusion suite passes 351/351 cases. Compute
Sanitizer memcheck, initcheck, and synccheck report zero errors. The architecture
graph test passes, and the non-backend CTest suite passes 53/53 tests. Q4_0
tg128 and Q8_0 pp512 remain outside the guard and measure 344.490628 and
15105.436850 tok/s. Qwen 3.6 27B Q2_K and Gemma 4 12B also remain outside the
guard and measure 32.402055 and 70.439694 tok/s respectively.
