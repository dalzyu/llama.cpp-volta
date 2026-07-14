# Rejected Volta Q8_0 MMVQ VDR=4

The retained Q8_0 MMVQ dot product processes two packed integer values per
thread call (`VDR_Q8_0_Q8_1_MMVQ=2`). This probe changed the Q8_0 MMVQ path to
process four values per call, reducing the number of dot-product calls while
leaving the rest of the Volta launch geometry unchanged.

The candidate was measured on the V100 at the fixed 150 W limit and 1200 MHz
SM clock. Qwen 3.5 0.8B Q8_0 tg128 fell from the retained 276.771896 tok/s to
264.934306 tok/s (-4.2770%). The candidate was rejected and the source was
restored to VDR=2 without a correctness gate because the performance loss was
decisive. A three-run Nsight Systems profile also showed the normal and fused
Q8 kernels becoming materially slower; the candidate profile's dominant normal
Q8 MMVQ kernel totaled 321.344661 ms over 34,944 instances.
