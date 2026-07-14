# Rejected CUDA Graph host fast path

Experiment 062 removed 78.86 us of traced GPU work per generated token but did
not produce a stable end-to-end gain. This follow-up measured the steady-state
host path to test whether CUDA Graph validation or cache bookkeeping had become
the bottleneck.

Temporary microsecond counters were placed around graph compatibility and
property checks, graph lookup, the Q8_1 cache reset, and `cudaGraphLaunch`.
They accumulated 256 steady-state replays of the 1374-node Qwen 3.5 0.8B Q8_0
decode graph. The counters and environment switch were removed afterward.

| Segment | Average |
| --- | ---: |
| Set current CUDA device | 0.031 us |
| Compatibility and property checks | 2.203 us |
| Q8_1 cache/reset setup | 0.023 us |
| CUDA graph map lookup | 0.035 us |
| Pre-launch work | 0.016 us |
| `cudaGraphLaunch` host API | 29.406 us |
| Post-launch work | 0.012 us |

The graph UID fast path is already active. Removing every remaining validation
and lookup would recover only about 2.3 us from a roughly 2.6 ms token. The
29.4 us CUDA API call dominates host submission time, but it is asynchronous
and overlaps the much longer GPU graph.

CUDA device graph launch was also rejected as an architectural direction. The
CUDA Programming Guide requires a device graph to be launched from another
graph and gives it fixed structure between host re-instantiations. Llama token
generation still needs a host boundary to inspect or sample logits and update
the next graph inputs. A device-side wrapper would therefore add a kernel
without removing the required per-token host interaction.

No source change was retained. Host graph bookkeeping is not a useful target
until GPU execution falls by much more than the current gap.
