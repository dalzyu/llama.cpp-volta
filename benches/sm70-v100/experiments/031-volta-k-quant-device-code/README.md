# Volta K-quant MMVQ device code

SM70 disassembly showed that the fused Q3_K kernel synthesized the saturating
packed-byte subtraction in `vec_dot_q3_K_q8_1_impl_mmvq` as a long sequence of
integer shifts and logic. The low operands are in the range 0 through 3 and the
high operands are either 0 or 4, so wrapping byte subtraction produces the same
signed byte bit patterns without saturation.

The Q2_K minimum term previously broadcast its 4-bit scale to four bytes before
`dp4a`. Computing the sum with packed byte values of one and multiplying the
integer result by the scale is algebraically identical and avoids the broadcast
sequence.

| Variant | Q3_K instructions | Q2_K instructions | Q3_K registers | Q2_K registers |
| --- | ---: | ---: | ---: | ---: |
| Control | 552 | 392 | 72 | 48 |
| Q3_K wrapping subtract | 512 | 392 | 72 | 48 |
| Q3_K plus Q2_K packed sum | 512 | 376 | 72 | 48 |

All kernels remained spill-free. Bracketed Qwen 3.6 27B generation runs used
10 repetitions at 128 tokens.

| Variant | Runs (tok/s) | Paired average | Change from control |
| --- | --- | ---: | ---: |
| Control | 29.858, 29.658 | 29.758 | - |
| Q3_K wrapping subtract | 30.600, 30.421 | 30.510 | +2.53% |
| Q3_K plus Q2_K packed sum | 30.817, 30.616 | 30.716 | +3.22% |

The Q2_K rewrite contributed another 0.68% over the Q3_K-only result. The final
revision measured 30.951 tok/s. Focused K-quant MUL_MAT coverage passed 20/20
cases, and eight-chunk Qwen 3.6 perplexity remained exactly 7.1732.
