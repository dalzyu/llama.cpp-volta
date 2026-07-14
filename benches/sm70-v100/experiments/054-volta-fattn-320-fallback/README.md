# Volta 320-wide flash-attention fallback

The final full CUDA backend sweep exposed a pre-existing Volta launch failure
for 320-wide K/Q and 256-wide V flash attention. With a GQA ratio of 32 and a
permuted Q layout, the selector chose the MMA F16 specialization and aborted
when `cudaFuncSetAttribute` rejected its dynamic shared-memory request.

The optimized branch changes only the exact 256-wide Volta MMA and tile
configurations. A diff against the original baseline confirms that the
320-wide selector, configuration, and kernel were unchanged, so this was not
a regression from the retained attention optimization.

The fix keeps 320-wide attention on the existing tile kernel on Volta. Smaller
effective widths already selected that kernel, and the full 320-wide test
family proves it supports the required layouts. Turing and newer GPUs retain
their existing MMA selection. The Qwen and Gemma target models do not use a
320-wide attention head, so their dispatch is unchanged.

## Validation

Before the fix, the full suite aborted in the
`hsk=320,hsv=256,nr23=[32,1]` F16 case with permutation `[0,2,1,3]`. After the
fix:

- all 16 supported 320-wide F16 flash-attention cases pass;
- Compute Sanitizer reports zero errors over the same cases;
- all 112 optimized 256-wide F16 flash-attention cases still pass; and
- the complete CUDA backend suite passes 12,995/12,995 cases.

A Qwen 3.5 Q8_0 smoke after the fix measures 15990.619463 tok/s at pp512 and
319.050782 tok/s at tg128. Its steady generation samples are 321.35 to 321.55
tok/s, matching the preceding final run.
