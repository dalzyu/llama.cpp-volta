# Volta direct GDN recurrent-state reads

Qwen 3.5 0.8B evaluates 18 gated delta net blocks for each decode token.
Before this experiment, every block used GET_ROWS to copy one 128x128x16 F32
recurrent-state row from the persistent cache into graph scratch. That is one
1 MiB copy and one launch per block, or 18 MiB and 18 launches per token. The
GDN kernel then read the copied state exactly once.

The retained SM70 path reads the selected persistent-cache row directly. The
graph walker recognizes the GET_ROWS -> RESHAPE -> GATED_DELTA_NET chain,
records the cache pointer, dynamic row-id pointer, and row stride in graph-local
CUDA context metadata, and skips the redundant GET_ROWS node. A specialized GDN
kernel resolves the row id on device and applies the existing head and column
offsets to that cache row.

The graph rewrite requires exact SM70, K=1, one token, one sequence, scalar
gating, F32 state storage, 128 values per head, 16 heads, contiguous gathered
state, and the expected 262144-element cache rows. GET_ROWS and its reshape
must have no other consumers, and no intervening computed node may write an
allocation overlapping the persistent source. Concurrent graph regions,
prompt batches, multiple tokens or sequences, rollback snapshots, other
layouts, KDA gating, non-Volta devices, and mismatched graphs keep the original
materialized-state path.

## Fixed-clock generation results

Candidate and control measurements came from one binary. A temporary
environment switch disabled only direct-state reads and was removed from the
retained source. Every run exposed only the V100 UUID and used a 150 W power
limit, fixed 1200 MHz SM clock, and 877 MHz memory clock.

| Run | Path | Q8_0 tg128 tok/s | Standard deviation |
| --- | --- | ---: | ---: |
| A | Direct recurrent state | 409.783484 | 0.545949 |
| Control | Gathered recurrent state | 395.207878 | 0.648851 |
| B | Direct recurrent state | 409.651115 | 0.585513 |
| Candidate mean | Direct recurrent state | 409.717299 | |
| Final switch-free | Direct recurrent state | 409.490393 | 0.434252 |

The Q8_0 bracket improvement is 3.6713%. The final switch-free result is
3.6139% above the matched control.

The state path is independent of weight quantization. Q4_0 does not use the
raw Q/K/V core from experiment 073, but it removes the same state gather:

| Run | Path | Q4_0 tg128 tok/s | Standard deviation |
| --- | --- | ---: | ---: |
| A | Direct recurrent state | 359.075348 | 0.677356 |
| Control | Gathered recurrent state | 346.958690 | 0.514706 |
| B | Direct recurrent state | 360.021065 | 1.618971 |
| Candidate mean | Direct recurrent state | 359.548206 | |
| Final switch-free | Direct recurrent state | 359.162167 | 0.692400 |

The Q4_0 bracket improvement is 3.6285%. The final switch-free result is
3.5173% above the matched control.

## Launch profile

Node-level traces cover 32 Q8_0 decode tokens. The control already uses the
raw Q/K/V core retained in experiment 073.

| Path | Launches | GPU time | Average |
| --- | ---: | ---: | ---: |
| Control raw-Q/K/V GDN core | 576 | 3.291936 ms | 5.7152 us |
| Control 1 MiB state GET_ROWS | 576 | 2.709277 ms | 4.7036 us |
| Control combined chain | 1152 | 6.001213 ms | |
| Direct-state GDN core | 576 | 3.832926 ms | 6.6544 us |
| Remaining 1 MiB state GET_ROWS | 0 | 0 ms | 0 us |

Direct cache reads make the GDN core 0.540990 ms slower over 32 tokens because
the gathered scratch no longer prewarms the state. Removing the copy still
reduces the affected chain by 2.168287 ms, or 36.1308%. This saves 67.7590 us
and 18 launches per token. The final switch-free trace measures the direct
core at 3.832936 ms and contains no grid (1,256,1) GET_ROWS launches. The
smaller recurrent convolution-state gather remains visible at grid (1,18,1).

The Q8_0 materialized-state and direct-state kernels both use 55 registers per
thread, 8 bytes of static shared memory, and no local or stack memory. The
corresponding Q4_0 kernels both use 48 registers and no shared, local, or stack
memory. The extra cache address therefore does not reduce SM70 occupancy.

## Correctness and fallback validation

Final, disabled-control, and eager Q8_0 runs produce byte-identical context-128
logits. PPL remains 11.9470 and SHA-256 is
`acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274`.
Final, disabled-control, and eager Q4_0 runs are also byte-identical, with PPL
12.6381 and SHA-256
`385e35ba44725c79a84b3ed86297d71de16280492d40086cb04fa0ed819e14a0`.

A two-token batched fallback is byte-identical to its disabled control. Q8_0
pp512 remains outside the decode guard and measures 15121.861423 tok/s versus
15149.704895 tok/s control, within run noise. The focused RMS, matrix-vector
fusion, and GDN suite passes 387/387 cases, including K=2, K=3, and K=4 state
snapshots. Compute Sanitizer memcheck, initcheck, and synccheck report zero
errors. The architecture graph test passes, and the non-backend CTest suite
passes 53/53 tests, including recurrent-state rollback.

Qwen 3.6 27B Q2_K and Gemma 4 12B remain outside the exact state shape and
measure 32.435478 and 70.542025 tok/s respectively.
