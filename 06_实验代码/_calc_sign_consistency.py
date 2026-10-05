# -*- coding: utf-8 -*-
"""按附录 A.5 定义，算 10 种子下的符号一致率。

定义：对某温度组，先求每个种子对该组所有文件的组均值；
比较「各种子 (A2-x − A2-0) 组均值」与「跨种子合并 (A2-x − A2-0) 组均值」是否同号。
合计仅统计 4 个 OOD 温度（10/0/−10/−20 °C），分母 = 4 × 种子数。
"""
import sys

import pandas as pd
import os
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RES = os.path.join(__ROOT__, "04_实验", "数据", "results")
OOD = ["ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]

d = pd.read_csv(f"{RES}/p5_b2_perfile_seed10.csv")
seeds = sorted(d.seed.unique())
print(f"种子数 {len(seeds)}: {seeds}\n")

# 每个 (exp, seed, group) 的文件均值
g = d.groupby(["exp", "seed", "eval_group"]).RMSE.mean().unstack("eval_group")

for arm in ["A2-1", "A2-2", "A2-3", "A2-4"]:
    hit = tot = 0
    per_temp = []
    for T in OOD:
        delta_s = g.loc[arm][T] - g.loc["A2-0"][T]
        merged = delta_s.mean()
        agree = (delta_s > 0).sum() if merged > 0 else (delta_s < 0).sum()
        hit += agree
        tot += len(delta_s)
        per_temp.append(f"{agree}/{len(delta_s)}")
    print(f"{arm} − A2-0:  {hit}/{tot}  ({hit/tot*100:.1f}%)   逐温度 {' '.join(per_temp)}")
