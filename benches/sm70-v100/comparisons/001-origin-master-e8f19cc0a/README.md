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
300 W, graphics is locked to 1192 MHz, memory runs at 877 MHz, and each pair
starts at or below 70 C. Every run records 50 ms telemetry.

The full matrix covers every standalone text model in `build/bin/models/`:

- Qwen 3.5 0.8B Q4_0
- Qwen 3.5 0.8B Q8_0
- Qwen 3.5 9B Q4_K_M
- Gemma 4 12B Q4_K_XL
- Qwen 3.6 27B Q2_K
- Gemma 4 31B Q4_K_XL

Tokenizer fixtures, synthetic architecture-test files, `mmproj`, and MTP
draft models are not standalone benchmark targets and are excluded. The 31B
model uses a fixed 42 GPU layers because larger offloads do not leave enough
VRAM for `pp32768` and cuBLAS workspace. All other models use `-ngl 99`.

Each revision receives six independent repetitions of `pp512`, `pp4096`,
`pp32768`, `tg128`, and `tg1024`. Revision order alternates each round. Runs
use the established `-b 2048 -ub 512 -sm none -mg 0 -fa auto` settings and
the built-in warmup. This yields 360 timed measurements.

Run a compatibility smoke test first, then the full resumable matrix:

```sh
benches/sm70-v100/comparisons/001-origin-master-e8f19cc0a/run.sh smoke
benches/sm70-v100/comparisons/001-origin-master-e8f19cc0a/run.sh full
```

If a pair is interrupted, both sides of that pair are repeated. Fully
completed pairs are retained when the script resumes.
