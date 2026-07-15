# Volta MROPE K/V cache-store fusion

Qwen 3.5 uses MROPE for its full-attention K tensors. The CUDA backend already
fused NORMAL and NEOX RoPE with the following VIEW and SET_ROWS, but MROPE
always materialized an F32 tensor and then launched an F32-to-F16 cache store.
The V tensor used another cache-store launch.

The retained SM70 path recognizes the exact decode graph sequence:

```text
ROPE(K) -> VIEW -> SET_ROWS(K) -> VIEW(V) -> SET_ROWS(V)
```

A separate MROPE kernel rotates K directly into its F16 cache. The same threads
also load adjacent F32 V elements, convert them to `half2`, and write the V
cache using its own row indices. The standard MROPE entry point and parameter
list remain unchanged for Q rotation and unrelated graphs.

The combined path is restricted to Volta and requires MROPE or IMROPE, F32 K
and V inputs, F16 caches, I64 row indices, contiguous flattened views, matching
K/V element counts, one outer sequence dimension, and distinct cache
destinations. If the V pattern does not match, Volta can still use the narrower
MROPE-to-K-cache fusion. Other devices and layouts retain the original path.

## Prototypes

The first prototype added only MROPE-to-K-cache support. Its 60-run Q8_0
candidate mean was 424.268840 tok/s versus 423.181070 control, a 0.2570%
improvement. It removed six launches per token but left the six V stores.

The first K+V prototype routed both ordinary and fused MROPE through one kernel
signature. It improved the initial bracket by 0.9380%, but made ordinary Q
rotation carry five unused cache arguments. The retained split entry point
restores the original standard kernel ABI and contains the extra parameter and
register cost in the fused K kernel.

## Fixed-clock generation results

Candidate and control measurements came from one binary. A temporary
environment switch disabled MROPE cache fusion and was removed from retained
source. Every run exposed only the V100 UUID and used a 150 W power limit,
1200 MHz SM clock, and 877 MHz memory clock.

| Run | Path | Q8_0 tg128 tok/s | Standard deviation |
| --- | --- | ---: | ---: |
| A | Fused MROPE K/V store | 426.505995 | 0.405336 |
| Control | Separate MROPE and K/V stores | 423.552067 | 0.788527 |
| B | Fused MROPE K/V store | 426.129465 | 0.484197 |
| Candidate mean | Fused MROPE K/V store | 426.317730 | |
| Final switch-free | Fused MROPE K/V store | 426.820411 | 0.606465 |

The conservative A/B candidate mean improves Q8_0 generation by 0.6530%. The
later switch-free result is 0.7717% above the matched control.

Q4_0 candidate A, control, and candidate B results are 362.600291,
360.755729, and 362.460426 tok/s. The 20-run candidate mean is 362.530359
tok/s, a 0.4919% improvement.

Q8_0 pp512 is neutral at 15170.951287 tok/s versus 15169.226575 control, a
0.0114% difference.

The Qwen-family 27B Q2_K model also matches the fused graph. Candidate A,
control, and candidate B are 32.574371, 32.179245, and 32.456726 tok/s. The
candidate mean improves by 1.0451%. Gemma 4 remains outside the MROPE path and
measures 70.506557 tok/s, consistent with the previous experiment.

## Launch profile

Node-level traces include one warmup evaluation and 32 measured Q8_0 decode
tokens. Each path therefore evaluates six attention layers 33 times, for 198
affected chains. Grid dimensions separate the control Q and K instances even
though both use the same kernel symbol.

| Path | Launches | GPU time | Average |
| --- | ---: | ---: | ---: |
| Control K MROPE | 198 | 0.634426 ms | 3.2042 us |
| Control K/V SET_ROWS | 396 | 1.189457 ms | 3.0037 us |
| Control affected chain | 594 | 1.823883 ms | |
| Final fused K MROPE and K/V store | 198 | 0.945942 ms | 4.7775 us |

The affected chain is 48.1358% faster. It saves 26.6043 us and twelve launches
per token evaluation. Across the complete trace, launches fall from 8256 to
7860 and summed GPU kernel time falls from 73.604150 to 72.721062 ms, a
1.1998% reduction.

The SM70 fused kernel uses 23 registers per thread, no local, stack, or static
shared memory, and 520 bytes of constant parameter space. The unchanged
standard SM70 MROPE kernel uses 21 registers and 465 bytes of constant parameter
space.

## Correctness and fallback validation

Final, temporary-control, graph-disabled, and experiment-076 Q8_0 runs produce
byte-identical context-128 logits. PPL remains 11.9470 and SHA-256 is
`acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274`.

The two-sequence result remains byte-identical to its experiment-076 reference,
with SHA-256
`58f244f417288747e3b2951b610cf2e04719559fbd91b251756ebd68e07d18b9`.
Q4_0 PPL remains 12.6381 and SHA-256 is
`385e35ba44725c79a84b3ed86297d71de16280492d40086cb04fa0ed819e14a0`.

The expanded RMS, matrix-vector fusion, GDN, flash-attention, and RoPE backend
suite passes 3695/3695 cases. Compute Sanitizer memcheck, initcheck, synccheck,
and racecheck report zero errors or hazards. The architecture graph test exits
successfully, and the non-backend CTest suite passes 53/53 tests.
