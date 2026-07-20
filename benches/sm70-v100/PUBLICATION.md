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

Benchmark results, telemetry values, timestamps, model and input hashes, test
counts, and pass/fail outcomes were not recomputed or rounded during
publication cleanup. The command structure and relevant software and hardware
versions remain present.

The normalized text logs are not byte-for-byte copies of the private
environment logs. Any hashes recorded in the archive identify models, inputs,
saved logits, or build artifacts as labeled; they must not be interpreted as
checksums of the normalized log files.

## Rewritten history

Branch-only history was mechanically rewritten for publication. The initial
pass removed environment identifiers. After upstream `master` through
`571d0d540df04f25298d0e159e520d9fc62ed121` was merged, the branch-side author
and committer metadata was set to the fork owner's GitHub-linked identity,
`dalzyu <85581627+dalzyu@users.noreply.github.com>`. Shared upstream commits
were not rewritten and retain their original identities.

The identity pass preserved commit trees, timestamps, subjects, message bodies
apart from the rewrite-version trailer, order, and merge topology. Every
branch-side commit in this published state also carries
`Assisted-by: OpenAI Codex` and
`History-Rewritten: public-sanitization-v3` trailers.

The rewrite removes the original local paths, host name, private network
address, and GPU UUIDs from every branch-only historical snapshot. A
history-wide scan found no private keys or recognizable API, GitHub, Hugging
Face, AWS, Google, Slack, bearer, or basic-auth credentials.

Archived benchmark output retains the `build_commit` value emitted by the
binary that produced it. Those legacy values are measurements rather than live
Git object names. The legacy revisions present in the archive map to the
rewritten equivalents below:

| Recorded revision | Published revision |
| --- | --- |
| `275a8d6c507ffca0d32f98b05cf26c13c652727f` | `5854d11f18bb5b778d7168a7b0cf191f5bfd80b8` |
| `d3b5d60f1534b71f5dd1a3ac0fe0ee1fafeb9a64` | `b194b518eb01d9c56c8a9209f997b280ed9ef2ba` |
| `e33e5bf79b8aaab5017882fed46a3ab7a7552169` | `25fc8122e9bed9f83df11f6ad63631239a06e2bf` |
| `a01b5bcd8ecdaf26f15475a77143e1a9336264be` | `11023dd82144986d7fd861cd9bcdc9f132a89d5b` |

For commands that use the UUID placeholders, export values for the target
machine before running the protocol:

```sh
export V100_GPU_0_UUID="GPU-..."
export V100_GPU_1_UUID="GPU-..."
```
