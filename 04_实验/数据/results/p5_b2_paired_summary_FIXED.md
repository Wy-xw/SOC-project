# B2 paired stats -- FIXED grouping (from p5_b2_perfile.csv)

Regenerated from `04_实验/数据/results/p5_b2_perfile.csv` using
explicit filename-based temperature classification (the `eval_group`
column is contaminated: its `ood(0C)` holds 36 files = all four OOD
temperature groups, due to the `"0degC_"` substring matching
`"10degC_"` / `"n10degC_"` / `"n20degC_"`).

Per-file RMSE first averaged over the 3 seeds, then paired at file level.

## Group means (RMSE, pp)

| group | n | A2-0 | A2-1 | A2-2 | A2-3 | A2-4 |
|---|---|---|---|---|---|---|
| test_id(25C) | 4 | 0.620 | 0.596 | 0.623 | 0.621 | 0.681 |
| ood(10C) | 9 | 3.384 | 3.715 | 3.489 | 3.329 | 3.339 |
| ood(0C) | 9 | 8.926 | 9.470 | 9.074 | 8.824 | 8.887 |
| ood(-10C) | 9 | 11.525 | 12.081 | 11.782 | 11.329 | 11.283 |
| ood(-20C) | 9 | 14.872 | 15.429 | 15.168 | 14.635 | 14.591 |

## A2-3 - A2-0 (full diag vs identity)

| group | n | delta (pp) | 95% CI | improved |
|---|---|---|---|---|
| test_id(25C) | 4 | +0.001 | [-0.066, +0.067] | 2/4 |
| ood(10C) | 9 | -0.056 | [-0.141, +0.029] | 7/9 |
| ood(0C) | 9 | -0.102 | [-0.201, -0.003] **excl-zero** | 8/9 |
| ood(-10C) | 9 | -0.196 | [-0.315, -0.078] **excl-zero** | 8/9 |
| ood(-20C) | 9 | -0.238 | [-0.359, -0.117] **excl-zero** | 8/9 |

## A2-3 - A2-2 (K/P increment)

| group | n | delta (pp) | 95% CI | improved |
|---|---|---|---|---|
| test_id(25C) | 4 | -0.002 | [-0.053, +0.049] | 2/4 |
| ood(10C) | 9 | -0.161 | [-0.219, -0.102] **excl-zero** | 8/9 |
| ood(0C) | 9 | -0.251 | [-0.379, -0.122] **excl-zero** | 9/9 |
| ood(-10C) | 9 | -0.453 | [-0.529, -0.378] **excl-zero** | 9/9 |
| ood(-20C) | 9 | -0.534 | [-0.642, -0.426] **excl-zero** | 9/9 |

## A2-1 - A2-0 (nu alone)

| group | n | delta (pp) | 95% CI | improved |
|---|---|---|---|---|
| test_id(25C) | 4 | -0.024 | [-0.101, +0.054] | 2/4 |
| ood(10C) | 9 | +0.331 | [+0.308, +0.354] **excl-zero** | 0/9 |
| ood(0C) | 9 | +0.545 | [+0.446, +0.643] **excl-zero** | 0/9 |
| ood(-10C) | 9 | +0.556 | [+0.458, +0.655] **excl-zero** | 0/9 |
| ood(-20C) | 9 | +0.557 | [+0.382, +0.731] **excl-zero** | 0/9 |

## A2-4 - A2-0 (K/P only vs control)

| group | n | delta (pp) | 95% CI | improved |
|---|---|---|---|---|
| test_id(25C) | 4 | +0.061 | [+0.031, +0.091] **excl-zero** | 0/4 |
| ood(10C) | 9 | -0.045 | [-0.128, +0.038] | 8/9 |
| ood(0C) | 9 | -0.039 | [-0.197, +0.119] | 6/9 |
| ood(-10C) | 9 | -0.242 | [-0.350, -0.134] **excl-zero** | 9/9 |
| ood(-20C) | 9 | -0.282 | [-0.380, -0.183] **excl-zero** | 9/9 |

## A2-4 - A2-3

| group | n | delta (pp) | 95% CI | improved |
|---|---|---|---|---|
| test_id(25C) | 4 | +0.060 | [+0.005, +0.115] **excl-zero** | 0/4 |
| ood(10C) | 9 | +0.011 | [-0.048, +0.069] | 3/9 |
| ood(0C) | 9 | +0.063 | [-0.069, +0.195] | 4/9 |
| ood(-10C) | 9 | -0.045 | [-0.090, -0.001] **excl-zero** | 7/9 |
| ood(-20C) | 9 | -0.044 | [-0.164, +0.076] | 4/9 |

## A2-4 - A2-2

| group | n | delta (pp) | 95% CI | improved |
|---|---|---|---|---|
| test_id(25C) | 4 | +0.058 | [+0.043, +0.073] **excl-zero** | 0/4 |
| ood(10C) | 9 | -0.150 | [-0.173, -0.128] **excl-zero** | 9/9 |
| ood(0C) | 9 | -0.188 | [-0.396, +0.020] | 7/9 |
| ood(-10C) | 9 | -0.499 | [-0.550, -0.448] **excl-zero** | 9/9 |
| ood(-20C) | 9 | -0.577 | [-0.596, -0.559] **excl-zero** | 9/9 |
