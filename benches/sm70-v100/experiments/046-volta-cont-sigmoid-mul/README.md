# Volta strided gate-view fusion

Qwen 3.5 and Qwen 3.6 full-attention layers build the attention gate from an
interleaved projection view. The graph materializes that view with
`CONT(view)`, then applies sigmoid and multiplies the attention output:

`mul(attention_output, sigmoid(cont(interleaved_gate_view)))`

The retained SM70 path matches the exact `CONT -> SIGMOID -> MUL` chain and
reads the row-contiguous source view directly through the existing strided
unary-gated kernel. It therefore skips the contiguous materialization while
preserving the original sigmoid and multiply operations. Dispatch is limited
to Volta F32 tensors with row-contiguous source views, flat contiguous other
input and output, equal element counts, and safe graph memory ranges. Other
layouts, types, and architectures retain the ordinary copy and fusion paths.

## Generation results

All measurements used only the V100 UUID, a 150 W power limit, fixed 1200 MHz
SM and 877 MHz memory clocks, and tg128. Candidate A and B bracket a temporary
same-binary control in which only this fusion was disabled. The temporary switch
was removed from the retained source.

| Model | Candidate A | Control | Candidate B | Candidate mean change |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q8_0 | 275.864527 | 273.303356 | 276.069017 | +0.97% |
| Qwen 3.5 0.8B Q4_0 | 291.153159 | 288.163160 | 291.061535 | +1.02% |

Qwen 3.6 27B Q2_K reaches 31.585661 tok/s, +0.27% against the preceding
31.501124 tok/s run. Gemma 4 does not materially use this chain; its tg128
smoke is 69.029370 tok/s versus 69.071792 previously (-0.06%).

## Prompt results

The same graph pattern occurs in prompt processing. Q8_0 reaches 15321.098255
tok/s at pp512, +0.81% against the preceding retained build. Q4_0 reaches
14940.239599 tok/s, +0.56%.

## Profile

Across three Qwen 3.5 Q8_0 tg128 runs, the control launches 2,304 strided-view
copy kernels totaling 6.230 ms and 2,304 sigmoid-multiply kernels totaling
5.048 ms. The retained path removes all 2,304 copy launches; the direct-view
sigmoid-multiply kernels total 4.588 ms. This chain falls from 11.278 ms to
4.588 ms, a 59.32% reduction, and its launch count falls from 4,608 to 2,304.
The other 6,912 copy launches in the profile are unrelated recurrent-state
copies and remain unchanged.

The retained path uses the existing SM70 F32 sigmoid-gated kernel: 16
registers per thread and no shared, local, or stack memory. Device allocations
and model VRAM use are unchanged.

## Validation

The focused graph test covers three strided source layouts, including
multi-token rows; all 3 CUDA and CPU cases pass. The combined V100 backend
sweep passes 79/79 cases covering CONT_SIGMOID_MUL, SIGMOID, CONT, and
GATED_DELTA_NET. Compute Sanitizer memcheck reports zero errors for a Q8_0
pp128/tg16 run through the fused graph.

Eight-chunk WikiText-2 estimates remain exactly 18.3641 for Q8_0 and 21.8604
for Q4_0 at the displayed precision.
