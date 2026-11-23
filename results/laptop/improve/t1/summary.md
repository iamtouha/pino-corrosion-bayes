| n | seed offset | cases | cov90 xc | yc | r | delta | delta r^2 | capacity | capacity (200 sub) | cap. 90% width kN | unique frac | evals | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 0 | 10 | 0.90 | 0.70 | 0.90 | 1.00 | 0.80 | 0.80 | 0.70 | 9.7 | 0.86 | 8788 | 76 |
| 1000 | 1000 | 10 | 0.90 | 0.80 | 0.90 | 1.00 | 0.80 | 0.80 | 0.70 | 9.9 | 0.85 | 8495 | 93 |
| 4000 | 0 | 10 | 0.90 | 0.70 | 0.80 | 0.90 | 0.70 | 0.70 | 0.80 | 9.8 | 0.86 | 38234 | 792 |

Repeatability (seed vs seed):

- n=1000 (10 cases): W1/width mean xc 0.031, yc 0.028, r 0.038, delta 0.045; capacity q05/q95 shift 0.08/0.03 of the 90% width; coverage flips 2
- n=1000 vs n=4000 (seed offset 0, 10 cases): W1/width mean xc 0.022, yc 0.021, r 0.021, delta 0.018

Step 7 baseline capacity misses (n = 1000, all damaged grid56 cases):

- pino_N100_SG: cov90 0.78 (27 cases), truth below q05: 2, above q95: 4
- pino_N2000_SG: cov90 0.81 (27 cases), truth below q05: 1, above q95: 4
- pino_N100_noSG: cov90 0.78 (27 cases), truth below q05: 2, above q95: 4
- pino_N2000_noSG: cov90 0.81 (27 cases), truth below q05: 1, above q95: 4
