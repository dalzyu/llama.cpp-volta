# Benchmark model inventory

All model files are stored under `build/bin/models/` and are ignored by Git.
Only provenance, hashes, and benchmark metadata are tracked here.

## Qwen 3.5 0.8B

Source: https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| Qwen3.5-0.8B-Q4_0.gguf | 563036064 | 57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf |
| Qwen3.5-0.8B-Q8_0.gguf | 811843488 | 75526add2fec8543a78d412a5546dd1c13dcf1ade237b245915d0f12dd43bb3d |

Both files report architecture `qwen35`, 752.39M parameters, and load with all
layers on the V100. Short smoke tests completed successfully before baseline
measurement.

## Qwen 3.6 27B

Source: https://huggingface.co/bartowski/Qwen_Qwen3.6-27B-GGUF

The Q2_K file was selected because its 12.05 GB size leaves enough of the 16 GB
V100 for compute and short-context buffers. Higher-quality Qwen 3.6 27B quants
start near the V100's full capacity and would force CPU offload, obscuring CUDA
kernel performance.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| Qwen_Qwen3.6-27B-Q2_K.gguf | 12051776000 | 5c2bb1598727e0307f968db0304201fecce135c6659aff94d63d72cb80f978a1 |
