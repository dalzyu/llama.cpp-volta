# Grouped Volta decode attention projections

Single-token attention evaluates Q, V, and K projections from the same F32
activation. The normal CUDA path submits one MMVQ kernel for each projection.
On SM70, the launch cost remains visible after the Q8_1 activation conversion
is cached and the surrounding graph is replayed.

The retained path flattens the rows of all three projections into one grid. It
supports same-type Q8_0 and Q4_0 projections, plus the Q2_K/Q4_K/Q2_K and
Q6_K/Q4_K/Q2_K combinations used by the tested Qwen 3.6 model. The latter
uses compile-time type dispatch per grid range while preserving the existing
two-warp reduction order.

Dispatch is limited to exact SM70, one-column F32 decode, stream zero, K of at
least 1024 elements, model-weight buffers, and a recognized graph structure.
Before V or K is evaluated early, the graph walker rejects any destination
range that overlaps an intervening output or input. CUDA graph concurrency is
also excluded. `GGML_CUDA_DISABLE_GROUPED_QKV=1` restores separate launches.

## Fixed-clock generation results

Candidate and control use one binary. All runs pin only the V100 UUID at
1200 MHz SM and 877 MHz memory clocks. Qwen 3.5 uses the 150 W limit. Qwen
3.6 uses 225 W to avoid power-throttling drift and returns to 150 W
immediately afterward.

| Model | Repetitions | Grouped A | Disabled control | Grouped B | Grouped mean | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q8_0 | 20 | 310.217303 | 306.429074 | 310.264861 | 310.241082 | +1.2440% |
| Qwen 3.5 0.8B Q4_0 | 50 | 318.739559 | 317.259595 | 318.547667 | 318.643613 | +0.4362% |
| Qwen 3.6 27B Q2_K | 7 | 36.847870 | 36.663854 | 36.850848 | 36.849359 | +0.5060% |

A Q8_0 pp512 check measures 15,264.720 tok/s grouped and 15,279.987
tok/s disabled, a noise-sized -0.10% difference. The grouped function rejects
multi-column input, so prompt kernels are unchanged.

## Launch profiles

Nsight Systems traces include graph warmup and capture, so kernel counts are
used to isolate the submission change.

For Qwen 3.5 Q8_0, 36 grouped kernels replace 108 Q/V/K kernels. The ordinary
Q8_0 row kernel falls from 654 to 546 launches, and the candidate adds 36
grouped launches. This saves 72 submissions. Time in the affected kernels
falls from 5.351182 ms to 5.163979 ms, including 0.341460 ms in the grouped
kernel.

For the Qwen 3.6 n32/r3 trace, 96 mixed grouped kernels replace 288 separate
Q2_K, Q4_K, and Q6_K projection kernels. This saves 192 submissions. The two
mixed specializations each launch 48 times.

## Scheduling exclusions

The optimization is intentionally narrower than a general projection
scheduler:

- Reordering the three existing kernels without grouping was byte-exact but
  neutral at about -0.07%.
- Qwen GDN destinations overlap live convolution scratch in the allocator, so
  early writes are unsafe.
- Gemma Q, K, and V reuse the same output address and are rejected by the
  overlap checks.
- Small-K shapes may use a different four-warp MMVQ reduction and are rejected.

## Correctness and safety

Grouped and disabled runs save byte-identical all-logit files:

| Model | Context | SHA-256 | PPL |
| --- | ---: | --- | ---: |
| Qwen 3.5 Q8_0 | 128 | `acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274` | 11.9470 |
| Qwen 3.5 Q4_0 | 128 | `385e35ba44725c79a84b3ed86297d71de16280492d40086cb04fa0ed819e14a0` | 12.6381 |
| Qwen 3.6 Q2_K | 64 | `5790cafdade18db8b653eb9d0e0fae27e0e62f0614f37f101cc809e9a859e0e3` | 9.7193 |
| Gemma 4 12B Q4_K_XL | 32 | `079436a894fcb90281b8e10c2f19e0b7d60ead345877246e1faf99347f8f46f0` | 1203.2716 |

Focused Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, and Q6_K single-column coverage passes
51/51. Compute Sanitizer reports zero errors for a Qwen 3.5 Q8_0 pp128/tg16
run and a one-token Qwen 3.6 mixed-kernel run. The all-target build and full
54-test CTest suite pass, including the exhaustive CUDA backend test.
