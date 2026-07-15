# Volta flash-attention gate Q8_1 epilogue

Qwen 3.5 0.8B has six full-attention layers in each decode evaluation. Before
this experiment, every layer launched a split-result flash-attention combine
kernel and then a second kernel for the strided gate copy, sigmoid, multiply,
F32 output, and Q8_1 prequantization. The following quantized output projection
consumed that Q8_1 cache.

The retained SM70 path folds the gate, sigmoid, multiply, and Q8_1
prequantization into the mandatory flash-attention combine. It does not
materialize the attention or gated F32 intermediate because the quantized
projection is the gated result's sole consumer.

A graph prepass recognizes the exact
FLASH_ATTN_EXT -> RESHAPE -> VIEW -> CONT -> SIGMOID -> MUL -> quantized
MUL_MAT chain. It verifies use counts, output flags, types, layouts, the
256-element head width, eight query heads, two KV heads, and the 256-token KV
extent. It also preallocates the persistent Q8_1 cache before flash attention
allocates temporary buffers. The launch is limited to Volta stream 0 and the
existing `DV=256`, `ncols1=1`, `ncols2=4` split-result tile.

Missing metadata, a missing cache entry, concurrent graph regions, prompt or
multi-sequence shapes, non-Volta devices, non-quantized projections, additional
consumers, or any layout mismatch retain the original combine and gate path.

## Rejected prototypes

The first prototype wrote both the gated F32 tensor and Q8_1 cache from the
combine. The graph memory-range check correctly rejected it because the graph
allocator aliases the gated F32 destination with earlier live scratch.

The first Q8_1-only prototype allocated its persistent cache inside
`launch_fattn`. That placed the allocation above flash-attention temporaries in
the VMM pool and triggered the pool's LIFO assertion on function exit. The
retained prepass allocation preserves pool order and the launch falls back if
that allocation is absent.

## Fixed-clock generation results

Candidate and control measurements came from one binary. A temporary
environment switch disabled only this epilogue and was removed from retained
source. Every run exposed only the V100 UUID and used a 150 W power limit,
1200 MHz SM clock, and 877 MHz memory clock.

| Run | Path | Q8_0 tg128 tok/s | Standard deviation |
| --- | --- | ---: | ---: |
| A | Fused combine epilogue | 423.964055 | 0.948599 |
| Control | Separate combine and gate | 418.079878 | 1.221648 |
| B | Fused combine epilogue | 423.629740 | 0.899365 |
| Candidate mean | Fused combine epilogue | 423.796898 | |
| Final switch-free | Fused combine epilogue | 424.281570 | 0.761659 |

The bracket improvement is 1.3674%. The later switch-free result is 1.4834%
above the matched control.

The same exact attention path is useful with Q4_0 weights because its output
projection also consumes a Q8_1 activation. Ten-run Q4_0 candidate A, control,
and candidate B results are 360.785034, 358.896124, and 360.711301 tok/s. The
candidate mean improves by 0.5160%.

Q8_0 pp512 remains a fallback and measures 15100.024500 tok/s versus
15123.047548 tok/s control, a -0.1522% noise-level difference.

## Launch profile

Node-level traces include one warmup evaluation and 32 measured Q8_0 decode
tokens. Each path therefore evaluates six attention layers 33 times, for 198
affected chains.

| Path | Launches | GPU time | Average |
| --- | ---: | ---: | ---: |
| Control flash combine | 198 | 1.065346 ms | 5.3805 us |
| Control gate and Q8_1 kernel | 198 | 0.653600 ms | 3.3010 us |
| Control combined chain | 396 | 1.718946 ms | |
| Final fused combine epilogue | 198 | 1.246413 ms | 6.2950 us |

The affected chain is 27.4897% faster. It saves 14.3192 us and six launches
per token evaluation. Across the complete trace, launches fall from 8454 to
8256 and summed GPU kernel time falls from 74.459963 to 73.568418 ms, a
1.1973% reduction.

The SM70 fused kernel uses 66 registers per thread, no local or stack memory,
408 bytes of constant parameter space, and the combine's existing dynamic
shared metadata. The standard SM70 combine uses 65 registers, no local or
stack memory, and 380 bytes of constant parameter space.

## Correctness and fallback validation

Final, temporary-control, eager, and experiment-075 Q8_0 runs produce
byte-identical context-128 logits. PPL remains 11.9470 and SHA-256 is
`acb248fa3c62d5f085a16556485a9d0d5e3b6bdf5c58a392cf734724f885c274`.
Disabling the shared Q8_1 cache cleanly falls back and produces the same hash.

The two-sequence result is byte-identical to its control and experiment-075
reference, with SHA-256
`58f244f417288747e3b2951b610cf2e04719559fbd91b251756ebd68e07d18b9`.
Q4_0 candidate, control, and experiment-075 logits are also byte-identical.
PPL is 12.6381 and SHA-256 is
`385e35ba44725c79a84b3ed86297d71de16280492d40086cb04fa0ed819e14a0`.

The focused RMS, matrix-vector fusion, GDN, and flash-attention suite passes
3267/3267 cases. Compute Sanitizer memcheck, initcheck, and synccheck report
zero errors. The architecture graph test exits successfully, and the
non-backend CTest suite passes 53/53 tests.

Qwen 3.6 27B Q2_K and Gemma 4 12B remain outside the exact fusion and measure
32.559085 and 70.479877 tok/s, respectively.
