# Rejected Volta Q8_0 three-row MMVQ blocks

The retained normal-K Q8_0 MMVQ path computes two rows per two-warp block on
SM70. This probe changed only the non-fused Volta path to compute three rows per
block, reducing the grid while increasing per-block partial sums and shared
state.

A same-build Qwen 3.5 0.8B Q8_0 tg128 comparison measured 276.727796 tok/s for
the retained two-row control and 277.043170 tok/s for three rows (+0.1140%).
The difference is within run noise, so the candidate was rejected and the
retained two-row geometry was restored.
