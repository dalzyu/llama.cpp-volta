# Volta Q8 gate projection epilogue

Qwen 3.5 evaluates 18 GDN layers per decode token. After the recurrent core,
each layer normalizes a contiguous 128x16 output and independently projects a
1024-element layer input to a 2048-element gate. The graph then reshapes the
gate and launches a separate kernel for `silu(gate) * normalized`.

The retained SM70 path recognizes the exact MUL_MAT, RESHAPE, SILU, and MUL
subgraph. It requires a contiguous Q8_0 1024x2048 projection, one contiguous
F32 input column, a matching contiguous 2048-element multiplier and output,
stream zero, and an existing Q8_1 activation cache entry. The subgraph checker
also proves that the projection, reshape, and SILU intermediates have no
external users.

Qwen allocates the final result exactly over the normalized multiplier. This
alias is safe because lane zero reads multiplier row `r` before overwriting the
same row, and no thread reads another row. Partial aliases and overlap with the
projection weights or input remain rejected. All other types, shapes, streams,
prompt batches, cache-disabled runs, and non-Volta devices use the original
path.

The fused kernel preserves the retained Q8_0 two-rows-per-CTA dot products and
warp reduction order. Lane zero applies SiLU to the already-rounded F32 dot
product, multiplies it by the normalized value, and stores the final output.

## Fixed-clock generation result

Candidate and control came from one binary. A temporary environment switch
disabled only this graph fusion and was removed from the retained source. All
runs exposed only the V100 UUID and used a 150 W power limit, fixed 1200 MHz SM
clock, and 877 MHz memory clock.

| Run | Path | tg128 tok/s | Standard deviation |
| --- | --- | ---: | ---: |
| A | Fused projection epilogue | 366.214899 | 0.470453 |
| Control | Separate projection and gate | 360.948186 | 0.663424 |
| B | Fused projection epilogue | 366.006958 | 0.499173 |
| Candidate mean | Fused projection epilogue | 366.110929 | |

The bracket improvement is 1.4303%. The final switch-free binary measures
366.188657 tok/s, 1.4519% above the matched control.

## Launch profile

Node-level traces cover 33 evaluated tokens. Grid dimensions separate the
2048-row gate projection from the 1024-row output projections and the
vocabulary projection.

| Path | Launches | GPU time | Average |
| --- | ---: | ---: | ---: |
| Control 1024-block gate projection | 594 | 4.218231 ms | 7.1014 us |
| Control gated SiLU | 594 | 1.770269 ms | 2.9803 us |
| Fused projection epilogue | 594 | 4.478700 ms | 7.5399 us |

The affected chain falls from 5.988500 ms to 4.478700 ms, a 25.21% reduction.
It saves 45.75 us and 18 launches per token. The fused and bare projection
kernels both use 48 registers per thread with no shared, local, or stack
memory on SM70. The fused constant parameter area grows from 384 B to 392 B.

## Rejected precursor

The first prototype placed the gated epilogue in the weighted RMS kernel. It
could not dispatch because the independent gate projection is scheduled
between weighted RMS and the final multiplication. Moving the epilogue to the
projection preserves graph order and requires no extra synchronization.

## Correctness and fallback validation

Candidate, disabled-control, eager, cache-disabled, and final switch-free Q8_0
runs produce byte-identical context-128 logits. PPL remains 11.9470 and SHA-256
is `acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274`.
Q4_0 graph and eager output are also byte-identical, with PPL 12.6381 and
SHA-256 `385e35ba44725c79a84b3ed86297d71de16280492d40086cb04fa0ed819e14a0`.

The focused RMS and matrix-vector fusion suite passes 351/351 cases. Compute
Sanitizer memcheck, initcheck, and synccheck report zero errors. The architecture
graph test passes, and the non-backend CTest suite passes 53/53 tests. Q4_0 and
Q8_0 pp512 remain outside the guard and measure 344.688135 and 15138.490238
tok/s. Qwen 3.6 27B Q2_K and Gemma 4 12B also remain outside the guard and
measure 32.433594 and 70.465291 tok/s respectively.
