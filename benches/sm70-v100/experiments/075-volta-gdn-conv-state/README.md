# Volta direct GDN convolution-state reads

Qwen 3.5 0.8B evaluates 18 gated delta net blocks for each decode token.
Before this experiment, every block gathered one 18432-element F32
convolution-state row from the persistent cache into graph scratch. That is a
72 KiB copy and one launch per block, or 1296 KiB and 18 launches per token.
The fused Q8_0 projection and convolution kernel then read the copied row once
and immediately wrote the shifted state back to the persistent cache.

The retained SM70 path reads the selected persistent-cache row directly. A
graph prepass recognizes the GET_ROWS -> RESHAPE -> fused projection and
convolution chain, records the persistent pointer, device row-id pointer, and
row stride, and skips GET_ROWS. The fused kernel resolves the dynamic row on
device before reading each three-value convolution history.

The rewrite is limited to the existing exact Volta Q8_0, single-token,
single-sequence fused path. The gathered row and reshape must have no other
consumers, the persistent source cannot be written between the original
gather and the fused launch, and an overlapping state update must begin on a
whole cache-row boundary and fit within that row. Concurrent graph regions,
prompt batches, multiple tokens or sequences, other quantizations, other GPU
architectures, and mismatched layouts retain the original path. If the final
kernel preflight unexpectedly rejects a prepared graph, the skipped gather is
executed before normal fallback.

## Fixed-clock generation results

Candidate and control measurements came from one binary. A temporary
environment switch disabled only direct convolution-state reads and was
removed from the retained source. Every run exposed only the V100 UUID and
used a 150 W power limit, fixed 1200 MHz SM clock, and 877 MHz memory clock.

| Run | Path | Q8_0 tg128 tok/s | Standard deviation |
| --- | --- | ---: | ---: |
| A | Direct convolution state | 418.767067 | 1.104328 |
| Control | Gathered convolution state | 409.789562 | 0.482319 |
| B | Direct convolution state | 417.780236 | 1.026890 |
| Candidate mean | Direct convolution state | 418.273652 | |
| Final switch-free | Direct convolution state | 419.425010 | 0.948535 |

The Q8_0 bracket improvement is 2.0704%. The final switch-free result is
2.3513% above the matched control.

Q4_0 does not use the specialized Q8_0 projection and convolution fusion, so
it remains a fallback. Ten-run candidate and temporary-control measurements
were 357.653532 and 360.360133 tok/s, respectively. The negative difference is
run noise rather than an executed code-path change. Q8_0 pp512 also remains a
fallback and measures 15119.162607 tok/s versus 15158.014144 tok/s control.

## Launch profile

Node-level traces include one warmup evaluation and 32 measured Q8_0 decode
tokens. Each path therefore executes 594 fused kernels, or 18 kernels for each
of 33 token evaluations.

| Path | Launches | GPU time | Average |
| --- | ---: | ---: | ---: |
| Control fused projection and convolution | 594 | 8.591086 ms | 14.4631 us |
| Control 72 KiB convolution-state GET_ROWS | 594 | 2.834036 ms | 4.7711 us |
| Control combined chain | 1188 | 11.425122 ms | |
| Direct-state fused kernel | 594 | 8.820649 ms | 14.8496 us |
| Remaining convolution-state GET_ROWS | 0 | 0 ms | 0 us |

Dynamic row addressing adds 0.229563 ms to the fused kernels, but removing the
copies reduces the affected chain by 2.604473 ms, or 22.7960%. This saves
78.9234 us and 18 launches per token evaluation. The final switch-free trace
measures 8.840850 ms for the direct-state kernels and contains no grid
(1,18,1) GET_ROWS launches.

The gathered-state and direct-state kernel variants both use 48 registers per
thread, no static shared, local, or stack memory, and 496 bytes of constant
parameter space. Direct addressing therefore does not reduce SM70 occupancy.

## Correctness and fallback validation

Final, temporary-control, experiment-074 reference, and eager Q8_0 runs
produce byte-identical context-128 logits. PPL remains 11.9470 and SHA-256 is
`acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274`.
A two-sequence fallback is byte-identical to its control and experiment-074
reference, with SHA-256
`58f244f417288747e3b2951b610cf2e04719559fbd91b251756ebd68e07d18b9`.

The final Q4_0 fallback remains byte-identical to the experiment-074 reference.
PPL is 12.6381 and SHA-256 is
`385e35ba44725c79a84b3ed86297d71de16280492d40086cb04fa0ed819e14a0`.

The focused RMS, matrix-vector fusion, and GDN suite passes 387/387 cases.
Compute Sanitizer memcheck, initcheck, and synccheck report zero errors. The
architecture graph test passes, and the non-backend CTest suite passes 53/53
tests, including recurrent-state rollback.

Qwen 3.6 27B Q2_K and Gemma 4 12B remain outside the exact fusion and measure
32.269087 and 70.494935 tok/s, respectively.
