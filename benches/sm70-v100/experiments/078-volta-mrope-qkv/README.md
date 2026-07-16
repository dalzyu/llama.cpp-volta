# Volta combined Q/K MROPE K/V cache update

Qwen 3.5 full-attention layers emit Q MROPE before the V and K projection
chain, but Q is not consumed until after K MROPE and the K/V cache updates.
Experiment 077 already combined K MROPE with both cache stores. This experiment
defers the independent in-place Q rotation and adds its rows to that launch.

The retained SM70 prepass recognizes this exact 13-node sequence:

```text
ROPE(Q)
MUL_MAT(V) -> RESHAPE(V)
MUL_MAT(K) -> RESHAPE(K) -> RMS_NORM(K) -> MUL(K) -> ROPE(K)
VIEW(K) -> SET_ROWS(K) -> VIEW(V) -> SET_ROWS(V) -> VIEW(Q)
```

Q is marked precomputed and associated with the later K node. When the existing
K/V MROPE fusion is selected, one combined kernel rotates K into its F16 cache,
stores V as `half2`, and rotates the in-place F32 Q rows. This removes the
standalone Q MROPE launch without changing its arithmetic. Dimensions beyond
`n_dims` are left untouched because Q input and output alias. The original
K-only and K/V fused kernel entry points remain separate, so unrelated graphs
do not carry the new arguments or register cost.

The prepass is restricted to Volta and requires the exact graph edges, the
existing K/V cache-fusion conditions, identical Q/K MROPE parameters and
position tensors, contiguous in-place F32 Q, matching head and token dimensions,
one Q consumer, a non-output Q tensor, legal deferred reads, and disjoint Q,
K, V, cache, index, and position ranges. The global fusion-disable setting
gates both precomputation and kernel selection.

## Prototype results

The first combined kernel shared more of the K branch with Q. It activated on
the live partial-rotary shape (`n_dims=64`, head width 256) but changed Qwen
3.5 0.8B Q8_0 PPL from 11.9470 to 12.0110. That version was rejected. A test
that deferred the unchanged Q kernel to the K location produced exact logits,
proving that graph ordering was safe. Splitting Q into its own in-kernel branch
then restored byte-identical logits.

Qwen 3.5 0.8B Q8_0 generation trials for block sizes 32, 64, 128, and 256 were
426.546374, 426.831181, 427.089068, and 427.116239 tok/s. The original 256-thread
geometry was retained. The temporary block-size and fusion-control switches are
not present in retained source.

## Canonical fixed-clock results

The replacement V100 used UUID
`${V100_GPU_0_UUID}`. Canonical comparisons used a 300 W
limit, 1192 MHz locked graphics clock, 877 MHz memory, 50 ms telemetry, and only
the V100 visible. Every busy sample in both brackets was exactly 1192 MHz, and
no power or thermal throttle reason was active. Candidate and control came from
one temporary binary in candidate/control/candidate order.
The Qwen model SHA-256 is
`03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`.

| Model and test | Candidate A | Control | Candidate B | Candidate mean | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 9B Q4_K_M tg1024 | 100.148795 | 100.096161 | 100.354407 | 100.251601 | +0.1553% |
| Qwen 3.5 9B Q4_K_M pp32768 | 1997.847176 | 1997.408054 | 1998.254287 | 1998.050732 | +0.0322% |

Generation is measurably faster and prompt processing is neutral. Peak VRAM is
unchanged at 5812 MiB for tg1024 and 6912 MiB for pp32768.
The retained switch-free build measures 100.165961 tok/s on tg1024, 0.0697%
above the matched temporary control.

Gemma 4 has no matching MROPE graph. A temporary-control tg1024 bracket at the
earlier 1230 MHz trial measured candidate mean 70.314727 versus 70.241058 tok/s
control, a noise-level +0.105%. Its pp32768 bracket was discarded because the
third run reached the 83 C thermal cap. The canonical lock was subsequently
lowered to 1192 MHz as recorded in the top-level benchmark protocol. The
switch-free production build measures 1318.107017 tok/s on pp32768 and
69.339006 tok/s on tg1024 at that lock, with no throttle samples.

## Launch profile

The node-level Qwen 3.5 0.8B trace includes one warmup and 32 measured tokens.
Six full-attention layers therefore produce 198 affected chains.

| Path | Launches | GPU time | Average |
| --- | ---: | ---: | ---: |
| Control K/V fused MROPE | 198 | 0.947740 ms | 4.7866 us |
| Control Q MROPE | 198 | 0.785082 ms | 3.9651 us |
| Candidate Q/K/V fused MROPE | 198 | 0.981836 ms | 4.9588 us |

The affected chain is 43.34% faster. Across 33 evaluations it removes 198
launches and saves 0.729671 ms, or six launches and 22.11 us per evaluation.
Total traced kernel time falls from 72.877057 to 72.147386 ms, while launches
fall from 7860 to 7662.

The SM70 combined kernel uses 24 registers per thread, no local, stack, or
static shared memory, and 532 bytes of constant parameter space. The unchanged
standard MROPE kernel uses 21 registers and 465 constant bytes. The separate
K/V fused kernel uses 24 registers and 520 constant bytes.

## Correctness and safety validation

The real Qwen 3.5 9B Q4_K_M candidate and temporary control produce identical
context-128 logits. Both report PPL 5.9046 and SHA-256
`be3f0c3b86215d5be67b9bba474d19156c3a9b6bfd87a55196772fd434fb5bd0`.
The retained switch-free build produces the same hash.

The two-sequence shape is also byte-identical at PPL 8.9254 with SHA-256
`4dff4a6846610731d43b303125c3ed5b1726c87eacffe271ef46ce94608c9a62`.
The global fusion-disabled fallback completes normally at PPL 5.9200; its
different hash is expected because it also disables all previously retained
fusion paths and their reduction orders.
The focused SET_ROWS, RMS_NORM, MUL, and ROPE backend sweep passes 679/679
supported cases. Compute Sanitizer memcheck, initcheck, and synccheck report
zero errors; racecheck reports zero hazards. The complete CUDA backend suite
passes 12995/12995 cases. The architecture graph test exits successfully, and
the non-backend CTest suite passes 53/53 tests.
