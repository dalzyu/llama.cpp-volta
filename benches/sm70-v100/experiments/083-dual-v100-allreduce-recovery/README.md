# Dual-V100 all-reduce recovery and clock qualification

Experiment 083 closes the interrupted Gemma 4 31B tensor-parallel numerical
experiment, validates the NCCL alternative after recovery, and establishes the
remote server's locked-clock rule.

## Outcome

- Reject `GGML_CUDA_AR_BF16_THRESHOLD=0` on this host. The only forced-FP32
  internal-allreduce run coincided with GPU 0 Xid 79 and Xid 154.
- Retain layer split as the primary quality reference.
- NCCL 2.30.7 is stable in the four-chunk tensor gate at 892 MHz, but its
  numerical delta against the upstream internal-allreduce tensor reference is
  too large to call quality-equivalent from this short test.
- Use 892 MHz graphics, 877 MHz memory, and 150 W per GPU for timed work.
- Use `-ts 1/1` with `llama-bench`; comma is the benchmark sweep delimiter.

No source optimization was added or removed by this experiment.

## Source audit

The internal all-reduce source is identical in the candidate and upstream
control. The two file hashes and the empty Git diff are recorded in
[raw/source-audit.txt](raw/source-audit.txt). This matters because the Xid
cannot be attributed to a branch-only all-reduce edit.

The behavior under test comes from:

- [`ggml_backend_cuda_comm_init`](../../../../ggml/src/ggml-cuda/ggml-cuda.cu),
  which selects `nccl`, `internal`, or the generic fallback from
  `GGML_CUDA_ALLREDUCE`.
- [`ggml_backend_cuda_comm_allreduce_nccl`](../../../../ggml/src/ggml-cuda/ggml-cuda.cu),
  which uses FP32 below 32768 elements for two GPUs and BF16 for larger F32
  reductions.
- [`ggml_cuda_ar_pipeline_init`](../../../../ggml/src/ggml-cuda/allreduce.cu),
  which requires exactly two sm70-or-newer CUDA devices and reads
  `GGML_CUDA_AR_BF16_THRESHOLD`.
- [`ggml_cuda_ar_allreduce`](../../../../ggml/src/ggml-cuda/allreduce.cu),
  which selects the internal BF16 wire conversion and copy-engine or chunked
  kernel path.
- [`ggml_backend_meta_graph_compute`](../../../../ggml/src/ggml-backend-meta.cpp),
  which invokes the backend collective between tensor subgraphs and falls back
  when it is unavailable.

The internal default threshold is 1 byte, so every nonempty F32 reduction uses
the BF16 wire conversion. Setting the threshold to zero disables it and sends
F32 instead. That doubles F32 wire bytes, but the experiment does not prove
that the extra PCIe traffic caused the Xid.

The saved-logit implementation in
[`tools/perplexity/perplexity.cpp`](../../../../tools/perplexity/perplexity.cpp)
clips each token distribution to 16 log units and stores scaled `uint16_t`
values. The upstream self-comparison is therefore a required control.

## Fixed inputs

| Item | Value |
| --- | --- |
| Candidate | `c55f89a84aad4e3003f40bef232cc9764732d083` |
| Upstream | `e8f19cc0ad70a243c8012bf17b4be601abfc8ea2` |
| Model | Gemma 4 31B QAT Q4_0 |
| Model SHA-256 | `9188a71055550f1e60b875d02b7abb63625ac11b4a6f148d6b22b3b28ba3d335` |
| WikiText-2 SHA-256 | `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08` |
| GPU 0 | `${V100_GPU_0_UUID}` |
| GPU 1 | `${V100_GPU_1_UUID}` |

See [raw/environment.txt](raw/environment.txt),
[raw/build.txt](raw/build.txt), [raw/artifact-sha256.txt](raw/artifact-sha256.txt),
and [raw/input-sha256.txt](raw/input-sha256.txt) for the captured records.

## Incident and recovery

Two default internal-BF16 tensor runs completed before the incident. The next
run changed only the candidate's internal wire setting by adding:

```sh
GGML_CUDA_ALLREDUCE=internal
GGML_CUDA_AR_BF16_THRESHOLD=0
```

At 2026-07-17 10:16:54 UTC, PCI device `0000:01:00.0` reported Xid 79 twice,
followed by Xid 154 with `GPU Reset Required`. The process then failed at
`cudaStreamSynchronize` on CUDA device 1. The exact command, application log,
and kernel lines are under [raw/incident](raw/incident/).

The captured command status is 143. GPU 0 remained unavailable and GPU 1
retained the failed process allocation until reboot. No result from that
invocation is accepted.

After reboot:

- both exact UUIDs enumerated;
- both volatile corrected and uncorrected ECC counts were zero;
- no compute processes remained;
- both PCIe links trained at Gen3 x8;
- each UUID passed 99/99 focused CUDA ADD cases independently;
- the current boot remained free of Xid and PCIe error messages through all
  subsequent gates.

The focused logs are under [raw/post-reboot-smoke](raw/post-reboot-smoke/).
The post-reset idle state and final current-boot error check are in
[raw/final-health.txt](raw/final-health.txt).

## Layer-split numerical control

All three layer cases used context 512, four chunks, batch and ubatch 512, F16
KV, FlashAttention on, full GPU offload, and an equal layer split. They ran at
900/877 MHz and 150 W.

| Case | Plain/current PPL | Mean KLD | RMS probability delta | Same top token |
| --- | ---: | ---: | ---: | ---: |
| Upstream reference generation | 1642.749188 +/- 310.717690 | N/A | N/A | N/A |
| Upstream against its own reference | 1642.749188 +/- 310.717690 | -0.000000 +/- 0.000000 | 0.000% | 100.000% |
| Candidate against upstream reference | 1642.966281 +/- 310.743955 | 0.000028 +/- 0.000004 | 0.092% | 99.804% |

The candidate PPL change is +0.217093, or +0.013215%. The self-comparison's
maximum probability delta is 0.003%, which establishes the encoding floor for
this reference.

`PPL(base)` reconstructed from the saved reference is about 1284.69 even in
the upstream self-test. It must not be compared to the plain 1642.75 PPL as if
it were an independently evaluated model.

One candidate GPU 1 busy sample reported 892 MHz instead of the requested 900
MHz. These are correctness runs, not accepted timing measurements. That sample
motivated the one-bin reduction used by the NCCL and sustained clock gates.

Raw records are under [raw/layer-kl](raw/layer-kl/).

## Tensor all-reduce results

The pre-incident upstream internal run created the tensor-split saved-logit
reference. The default candidate internal run completed. Both were unlocked
correctness runs and are not timing measurements.

| Case | PPL | Mean KLD | RMS probability delta | Same top token | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Upstream internal default | 1667.1081 +/- 315.86140 | N/A | N/A | N/A | reference |
| Candidate internal default | 1637.190026 +/- 309.896604 | 0.003081 +/- 0.001649 | 1.386% | 98.922% | completed |
| Candidate internal forced FP32 | N/A | N/A | N/A | N/A | rejected after Xid 79 |
| Candidate NCCL 2.30.7 | 1609.402167 +/- 303.308101 | 0.002752 +/- 0.000308 | 1.248% | 97.941% | stable, not equivalent |

The NCCL run held 892/877 MHz with no cap samples. NCCL's own log reports two
channels, `isAllCudaP2p 1`, and `SHM/direct` for both directions. That recorded
transport is used here instead of inferring a path from `nvidia-smi topo`.

Raw internal records are under
[raw/tensor-internal-preincident](raw/tensor-internal-preincident/). NCCL
records are under [raw/tensor-nccl](raw/tensor-nccl/).

## Sustained clock qualification

The first `llama-bench` command used `-ts 1,1` and failed to load because
[`llama-bench.cpp`](../../../../tools/llama-bench/llama-bench.cpp) treats comma
as its outer sweep delimiter. Its peak allocation was only 316 MiB per GPU, so
this was not evidence that `pp32768` could not fit.

The corrected command used `-ts 1/1`:

```sh
llama-bench \
    -m ${MODEL_STAGE}/target.gguf \
    -p 32768 -n 0 -r 1 \
    -b 2048 -ub 512 -t 4 \
    -ngl 999 -sm layer -ts 1/1 -mg 0 \
    -fa auto -ctk f16 -ctv f16 -o json
```

It completed at 729.237 tok/s.

| GPU | Busy samples | Wrong-clock samples | Cap samples | Peak power | Peak GPU/HBM | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1139 | 0 | 0 | 141.44 W | 74/78 C | 10966 MiB |
| 1 | 1076 | 0 | 0 | 147.17 W | 69/73 C | 11572 MiB |

This single repetition qualifies the clock, thermals, and memory placement. It
is not a performance baseline and is not used to claim a branch speedup.
Raw records, including the failed parser control, are under
[raw/pp32768](raw/pp32768/).

## Decision

1. Never set `GGML_CUDA_AR_BF16_THRESHOLD=0` on this server again.
2. Use 892/877 MHz and 150 W per GPU for canonical timing.
3. Reset the graphics lock after timed work.
4. Keep layer and tensor quality baselines separate.
5. Require a longer tensor reference and deterministic output gate before
   promoting NCCL tensor split for quality-sensitive inference.
6. Preserve the NCCL build as a validation artifact, not as an accepted source
   optimization.

Machine-readable summaries are in [results.tsv](results.tsv) and
[telemetry-summary.tsv](telemetry-summary.tsv).
