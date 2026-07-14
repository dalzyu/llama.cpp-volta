# Volta flash attention tile tuning

This experiment tunes the existing ggml CUDA flash attention kernels for the
256-wide K/Q and V heads used by the Qwen 3.5 models. The retained changes are
configuration-only and apply only to SM70.

For 64 query columns, the Volta MMA kernel changes from the Ampere fallback to
the following configuration:

| Parameter | Control | Retained |
| --- | ---: | ---: |
| Threads | 128 | 256 |
| Target occupancy | 2 | 1 |
| KV rows per iteration | 32 | 64 |
| K half2 batch | 128 | 128 |
| V half2 batch | 128 | 128 |
| Combine half2 batch | 128 | 64 |
| Pipeline target | 2 | 1 |
| Keep Q in registers | yes | no |

For four query columns, the generation tile keeps 128 threads and target
occupancy two, reduces the KV batch from 64 to 32 rows, and increases the K
half2 batch from 64 to 128. Other head sizes, column counts, and architectures
retain their existing selection.

## End-to-end results

All comparisons used only the V100 UUID, a 150 W power limit, fixed 1200 MHz
SM and 877 MHz memory clocks, and adjacent candidate/control/candidate builds.
Prompt values use 100 repetitions and generation values use 30 repetitions.
Candidate values below are the mean of the two bracket runs.

| Model and test | Control | Candidate | Change |
| --- | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q4_0 pp512 | 14425.28 | 14557.41 | +0.92% |
| Qwen 3.5 0.8B Q8_0 pp512 | 14769.94 | 14894.49 | +0.84% |
| Qwen 3.5 0.8B Q4_0 tg128 | 268.537 | 270.475 | +0.72% |
| Qwen 3.5 0.8B Q8_0 tg128 | 248.564 | 250.081 | +0.61% |

Nsight Systems reduces the prompt flash attention kernel from 179.139 us to
125.300 us (-30.05%). The generation tile changes from 11.403 us to 7.328 us
(-35.73%) in the isolated adjacent profiles. The retained SM70 prompt kernel
uses 255 registers and 48 bytes of stack per thread. The generation tile uses
98 registers, no stack, and 11008 bytes of static shared memory.

## flash-attention-v100 study

The external
[flash-attention-v100](https://github.com/ai-bond/flash-attention-v100)
repository was inspected at revision
`c91cad40c0539805754819e6ea96c75184d816a6` under its BSD-3-Clause license.
Its forward path uses explicit SM70 WMMA fragments, 512-thread blocks,
128-bit global/shared transfers, online softmax, and an XOR shared-memory
swizzle. Its 256-wide specialization uses a 32-row Q tile and a 64-row KV
tile. The 64-row KV tile independently supports the retained ggml prompt
batch size.

The repository's standalone forward kernels were extended locally with warmup
and repeated event timing. A safely aligned `+8` half padding variant was used
as the non-swizzled control because a raw 256-half stride is bank-conflicted.

| Standalone case | Layout | Average |
| --- | --- | ---: |
| D128, M=N=128, H=1 | unpadded | 27.332 us |
| D128, M=N=128, H=1 | padded | 16.350 us |
| D128, M=N=128, H=1 | XOR swizzled | 14.674 us |
| D256, M=N=512, H=1 | padded | 105.970 us |
| D256, M=N=512, H=1 | XOR swizzled | 86.790 us |
| D256, M=N=512, H=8 | padded | 166.310 us |
| D256, M=N=512, H=8 | XOR swizzled | 138.010 us |

All standalone cases passed their FP32 CPU reference with maximum error below
0.00051. The swizzle is effective for that kernel and layout.

ggml already uses explicit SM70 `mma.sync.m8n8k4`, 16-byte vector transfers,
GQA-aware splitting, stream-K fixup, and padded shared-memory rows. Porting the
external kernel would discard these integrations. The standalone D256 H8 case
is not directly comparable to ggml's GQA-aware model kernel, but its 138.010 us
result and 88 KiB shared-memory block provide no evidence that a wholesale
replacement would beat the integrated 125.300 us ggml path.

The XOR mapping was nevertheless adapted experimentally to ggml's exact
SM70/D256/64-column path. It removed ggml's row padding and changed Q, K, and V
fragment loads to use the swizzled addresses. It compiled and passed a model
run, but the attention kernel regressed to 162.871 us. Q4_0 pp512 fell from
14557.19 to 14454.92 tok/s (-0.70%), and Q8_0 fell from 14893.52 to 14794.33
tok/s (-0.67%). The adaptation was reverted.

Other rejected variants include 64- and 256-thread vector attention, smaller
prompt column tiles, keeping Q in registers, larger generation KV batches,
different K batches and occupancy targets, and a wider result-combine batch.
The most useful external ideas were therefore tile sizing and a bank-conflict
diagnostic, not a wholesale kernel replacement.

## Validation

The Release build contains both sm_70 and sm_89 real code. The focused
`FLASH_ATTN_EXT` backend suite passes 112/112 FP16 cases for 256-wide K/Q and V
heads, including masks, sinks, ALiBi bias, KV lengths 113 through 1024, GQA
permutations, and query batches 1 through 75. Compute Sanitizer memcheck reports
zero errors for a combined Q4_0 pp512 and tg16 model run.

An unfiltered flash-attention sweep reaches a pre-existing V100 failure in the
unmodified 320x256 shape when it requests unsupported dynamic shared memory.
The retained selectors cannot affect that shape; the targeted modified shapes
complete successfully.

Eight-chunk WikiText-2 checks and larger-model smoke tests are:

| Model | pp512 | tg128 | PPL |
| --- | ---: | ---: | ---: |
| Qwen 3.5 0.8B Q4_0 | 14556.08 | 270.47 | 21.8608 |
| Qwen 3.5 0.8B Q8_0 | 14910.36 | 250.08 | 18.3635 |
| Qwen 3.6 27B Q2_K | 728.51 | 31.30 | 7.1735 |
| Gemma 4 12B Q4_0 | 1703.40 | 68.99 | 308.2710 |

The small PPL differences relative to the preceding retained build are
consistent with the changed FP32 accumulation order and are covered by the
CPU-comparison operator tests.
