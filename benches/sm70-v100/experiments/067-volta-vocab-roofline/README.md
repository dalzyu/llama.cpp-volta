# Volta Q8_0 vocabulary projection roofline

The final Qwen 3.5 Q8_0 vocabulary projection is the largest remaining
single decode kernel. It multiplies 248320 Q8_0 rows of width 1024 by one
Q8_1 activation column. The retained experiment 051/052 kernel assigns one
output row to each warp and packs two warps into each 64-thread block.

A same-binary trace after experiment 066 measures this kernel at 313.922 us
for the retained path and 313.768 us with only the preceding output fusion
disabled. The 0.05% difference is noise and confirms that the projection
itself is unchanged.

## Traffic lower bound

Each row contains 32 `block_q8_0` values of 34 bytes, or 1088 bytes. Reading
all weights therefore requires 270172160 bytes. Writing 248320 F32 outputs
requires another 993280 bytes. Before counting activation reads, cache-line
overfetch, or any other traffic, the kernel must move 271165440 bytes.

| Quantity | Value |
| --- | ---: |
| Minimum traffic | 271.165 MB |
| Candidate time | 313.922 us |
| Control time | 313.768 us |
| Minimum effective bandwidth at control time | 864.222 GB/s |
| Fixed-clock V100 HBM2 theoretical bandwidth | about 898 GB/s |
| Lower-bound utilization | 96.24% |

The real traffic includes the Q8_1 activation, so actual HBM utilization is
slightly higher than this conservative lower bound. Hardware performance
counters are unavailable on this host because the NVIDIA module restricts
profiling to administrators, but the byte-count lower bound is already close
enough to the physical limit to rule out a useful arithmetic-only redesign.

## Rejected design space

Experiment 041 cut the output kernel from about 442 us to 313 us by packing
two independent row warps per block. Experiments 048, 051, and 052 then tested
wider dot products, one to eight warps per block, two rows per warp with
activation reuse, and software-pipelined accumulators. The best later change
was only about 0.3%; all alternate work assignments regressed.

The projection is therefore treated as an HBM roofline, not as a remaining
compute optimization target. Further risky work should eliminate launch-bound
activation passes around smaller kernels instead of adding complexity here.
