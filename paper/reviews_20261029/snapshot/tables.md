# Generated tables

## quality

| Seed | Stock QR10 CE | Full BF16 CE | CE increase | Perplexity increase |
| --- | ---: | ---: | ---: | ---: |
| 204 | 3.411152 | 3.422389 | +0.011237 | +1.130% |
| 205 | 3.409272 | 3.419354 | +0.010082 | +1.013% |

## timing

| Seed | Matched calls | Stock step ms | BF16 step ms | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| 204 | 101–200 | 380.740 | 314.638 | 17.36% |
| 205 | 101–10000 | 380.548 | 309.533 | 18.66% |

## controls

| Policy | CE at 2000 | Step ms | Optimizer ms |
| --- | ---: | ---: | ---: |
| Warm10/QR50 (cached) | 3.993939 | 314.142 | 26.415 |
| QR20 | 3.990383 | 329.641 | 40.460 |
| QR50 | 3.998679 | 313.551 | 22.864 |

## profile

| Dimension (batch 12) | Original CUDA ms | Reuse CUDA ms | Time reduction | Reordered columns |
| --- | ---: | ---: | ---: | ---: |
| 768 | 31.210 | 30.914 | 0.95% | 0/9216 |
| 1536 | 82.053 | 80.535 | 1.85% | 12/18432 |
| 2304 | 145.854 | 141.566 | 2.94% | 40/27648 |
| 3072 | 225.660 | 215.218 | 4.63% | 160/36864 |

## pilot

| Development arm | CE at 2000 | Step ms | Matrix-state GiB |
| --- | ---: | ---: | ---: |
| Stock QR10 | 3.985905 | 382.773 | 2.320 |
| Original warm, FP32 state | 3.995753 | 337.172 | 2.320 |
| Compiled FP32 storage / BF16 arithmetic | 3.991560 | 316.281 | 2.320 |
| Full BF16 | 3.993939 | 313.162 | 1.160 |
| BF16 M/V/C, FP32 bases | 3.995611 | 315.376 | 1.608 |
