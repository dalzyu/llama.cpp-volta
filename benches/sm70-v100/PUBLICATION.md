# Publication notes

This public experiment archive was mechanically normalized to remove
machine-specific identifiers while retaining the technical record. The
normalization applies to documentation, saved commands, telemetry, and logs
under `benches/sm70-v100/`.

## Normalized values

| Public value | Meaning |
| --- | --- |
| `${LLAMA_CPP_ROOT}` | Candidate source or build root on the machine that ran the command |
| `${UPSTREAM_WORKTREE}` | Separate upstream-control worktree |
| `${ARTIFACT_ROOT}` | Built artifact staging root |
| `${MODEL_STAGE}` | Model staging root |
| `${VALIDATION_ROOT}` | Validation workspace |
| `${MODEL_ROOT}` | Model storage root |
| `${USER_LOCAL}` | User-local installation root |
| `${V100_GPU_0_UUID}` | First V100 UUID |
| `${V100_GPU_1_UUID}` | Second V100 UUID |
| `${EXCLUDED_GPU_UUID}` | Non-target GPU UUID excluded from the historical local runs |
| `v100-server` | Benchmark host name |
| `192.0.2.10` | Documentation-only replacement for the host's private IPv4 address |

The post-reboot smoke filenames use `smoke-gpu0` and `smoke-gpu1` rather than
hardware UUIDs. One NCCL interface inventory was reduced to the selected
documentation address plus a redaction marker because it contained ephemeral
container and virtual-interface identifiers. Statistical output uses ASCII
`+/-` and `dp` in place of display-only Unicode glyphs.

## Preserved values

Benchmark results, telemetry values, timestamps, revision IDs, model and input
hashes, test counts, and pass/fail outcomes were not recomputed or rounded
during publication cleanup. The command structure and relevant software and
hardware versions remain present.

The normalized text logs are not byte-for-byte copies of the private
environment logs. Any hashes recorded in the archive identify models, inputs,
saved logits, or build artifacts as labeled; they must not be interpreted as
checksums of the normalized log files.

The branch intentionally preserves its complete development history. Earlier
commits therefore retain the original local paths, host name, private network
address, and GPU UUIDs that existed before this cleanup. A history-wide scan
found no private keys or recognizable API, GitHub, Hugging Face, AWS, Google,
Slack, bearer, or basic-auth credentials. Removing non-secret machine
identifiers from every historical Git object would require rewriting the
commit IDs and is outside this publication cleanup.

For commands that use the UUID placeholders, export values for the target
machine before running the protocol:

```sh
export V100_GPU_0_UUID="GPU-..."
export V100_GPU_1_UUID="GPU-..."
```
