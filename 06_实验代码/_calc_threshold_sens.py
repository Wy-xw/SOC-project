# -*- coding: utf-8 -*-
"""效应/噪声比的显式定义 + 阈值敏感性（0.5 / 1.0 / 2.0）。"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RES = r"E:\SOC论文项目\04_实验\数据\results"
OOD = ["ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]
d = pd.read_csv(f"{RES}/p5_b2_perfile_seed10.csv")
g = d.groupby(["exp", "seed", "eval_group"]).RMSE.mean().unstack("eval_group")

print("定义：对每个温度组，先取每个种子相对 A2-0 的组均值 Δ 与跨种子标准差 SD，")
print("      效应/噪声比 = |mean_s(Δ)| / SD_s(Δ)。\n")
print(f"{'臂':<6}{'温度':<12}{'组均值Δ':>10}{'跨种子SD':>10}{'比值':>8}")
rows = {}
for arm in ["A2-1", "A2-2", "A2-3", "A2-4"]:
    rs = []
    for T in OOD:
        dl = g.loc[arm][T] - g.loc["A2-0"][T]
        ratio = abs(dl.mean()) / dl.std(ddof=1)
        rs.append(ratio)
        print(f"{arm:<6}{T:<12}{dl.mean():>10.3f}{dl.std(ddof=1):>10.3f}{ratio:>8.2f}")
    rows[arm] = rs
    print(f"{'':6}{'→ 区间':<12}{'':>10}{'':>10}{min(rs):>7.2f}–{max(rs):.2f}\n")

print("=" * 62)
print("阈值敏感性：各臂在各阈值下是否被判为「确立」（比值 > 阈值）")
print(f"{'阈值':<8}" + "".join(f"{a:>18}" for a in rows))
for th in [0.5, 1.0, 2.0]:
    line = f"{th:<8}"
    for a in rows:
        n = sum(1 for r in rows[a] if r > th)
        line += f"{f'{n}/4 组':>18}"
    print(line)
