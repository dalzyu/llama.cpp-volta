# Gemma 4 12B Flash Attention sweep at 150 W

The default `auto` setting already enables Flash Attention. No runtime change is
needed.

| Setting | pp512 tok/s | tg128 tok/s | Peak VRAM |
| --- | ---: | ---: | ---: |
| off | 1424.334 | 66.029 | 7592 MiB |
| on | 1479.351 | 68.828 | 7582 MiB |
| auto | 1479.887 | 68.763 | 7582 MiB |

Forced `on` and `auto` are equivalent within run variance. Disabling Flash
Attention costs about 3.9% PP and 4.1% TG in this run.
