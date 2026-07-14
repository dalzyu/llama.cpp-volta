# Rejected Volta GDN redesigns

Five deeper Gated DeltaNet prefill designs were tested after the retained
warp-scalar gate optimization. Measurements used only the V100 UUID, a 150 W
power limit, and a fixed 1200 MHz SM clock. The retained experiment 036 result
is the control.

| Variant | Q4_0 pp512 | Change | Q8_0 pp512 | Change | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Graph chunking | 9474.16 | -33.43% | 9700.43 | -33.54% | Rejected |
| Separate gate precompute | 14230.07 | -0.01% | 14577.93 | -0.12% | Rejected |
| Two columns per warp | 14165.59 | -0.46% | 14506.71 | -0.60% | Rejected |
| Shared q/k double buffer | 13879.05 | -2.47% | - | - | Rejected |
| Two-token loop unroll | 14195.07 | -0.25% | 14460.39 | -0.92% | Rejected |

The graph chunking path replaces the fused recurrent kernel with the existing
chunked graph construction. Its extra operations and launches dominate on
Volta. Separately precomputing the scalar gate removes duplicate exponentials,
but adds a launch and a full gate tensor round trip; end-to-end performance is
neutral to negative.

Assigning two columns to each warp reuses q, k, and the scalar gate, but halves
the number of independent column warps. The SM70 kernel rises from 48 to 62
registers and its average time is 446.958 us. A shared-memory double buffer
preserves the column warps and reduces global load instructions, but its block
barrier on every recurrent token is more expensive than the saved loads.

Finally, unrolling the recurrent loop by two raises the kernel to 51 registers
without hiding enough latency to offset the lower residency. None of these
source variants is retained.
