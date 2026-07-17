# origin/master comparison at e8f19cc0a

This comparison measures the merged `sm70-opt` branch against its exact
`origin/master` parent after the 2026-07-17 upstream merge.

## Revisions

- Control: `e8f19cc0ad70a243c8012bf17b4be601abfc8ea2`
- Candidate: `e33e5bf79b8aaab5017882fed46a3ab7a7552169`

Both builds use Release mode, CUDA 12.9, native CPU code, and real sm_70 and
sm_89 device code. The control is built in a detached Git worktree so neither
source tree nor binary changes during measurement.

## Protocol

The canonical V100 rules in `../../README.md` apply. In particular, only V100
UUID `${V100_GPU_0_UUID}` is visible, power is fixed at
300 W, graphics is locked to 1192 MHz, and memory runs at 877 MHz. Every timed
invocation starts after both GPU and HBM temperatures remain at or below 50 C
for three checks. Every run records 50 ms telemetry. A result is accepted only
when every sample with nonzero GPU utilization remains at 1192/877 MHz and
reports no software power, software thermal, or hardware thermal cap.

The full matrix covers five fully offloaded standalone text models in
`build/bin/models/`:

- Qwen 3.5 0.8B Q4_0
- Qwen 3.5 0.8B Q8_0
- Qwen 3.5 9B Q4_K_M
- Gemma 4 12B Q4_K_XL
- Qwen 3.6 27B Q2_K

Tokenizer fixtures, synthetic architecture-test files, `mmproj`, and MTP
draft models are not standalone benchmark targets and are excluded. Gemma 4
31B is also excluded from the full comparison at the user's direction: it
requires partial CPU offload to leave enough VRAM for `pp32768` and cuBLAS
workspace. Its earlier smoke and offload probes remain under `smoke-combined/`
as diagnostic records. All five full-matrix models use `-ngl 99`.

Each revision receives six independent repetitions of `pp512`, `pp4096`,
`pp32768`, `tg128`, and `tg1024`. Each workload is a separate cooled
`llama-bench` invocation so a long model cannot heat-soak later workloads in
the same process. Revision order alternates each round. Runs use the established
`-b 2048 -ub 512 -sm none -mg 0 -fa auto` settings and the built-in warmup.
This yields 300 timed measurements.

## Thermal qualification

The first full-matrix attempt placed all five workloads in one invocation and
cooled only before each revision pair. It was discarded after telemetry showed
software thermal capping and clock reductions in every completed Qwen 3.6 27B
run and in one heat-soaked Qwen 3.5 9B run.

A standalone candidate Qwen 3.6 27B `pp32768` probe beginning at 49 C validated
the replacement protocol. All 1,755 busy samples remained at 1192/877 MHz with
no cap, peak GPU and HBM temperatures were both 79 C, peak power was 217.65 W,
and throughput was 734.665724 tok/s. The raw probe is under `thermal-probe/`.

Run a compatibility smoke test first, then the full resumable matrix:

```sh
benches/sm70-v100/comparisons/001-origin-master-e8f19cc0a/run.sh smoke
benches/sm70-v100/comparisons/001-origin-master-e8f19cc0a/run.sh full
```

If a workload pair is interrupted or its telemetry is invalid, both sides of
that pair are repeated. Fully completed and telemetry-valid pairs are retained
when the script resumes.
