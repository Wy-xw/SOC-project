# -*- coding: utf-8 -*-
"""CB-2 报告：读 cb2_rescore_perfile.csv，输出名义 vs 逐文件容量（S1）对照。"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

R = os.path.join(__ROOT__, "04_实验", "数据", "results")
df = pd.read_csv(os.path.join(R, "cb2_rescore_perfile.csv"))
ORDER = ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]
df["eval_group"] = pd.Categorical(df.eval_group, ORDER, ordered=True)


def paired(a, c, g, col):
    s = df[df.eval_group == g]
    w = s.pivot_table(index="cycle_id", columns=["exp", "seed"], values=col)
    if a not in w.columns.get_level_values(0) or c not in w.columns.get_level_values(0):
        return None
    d = w[a] - w[c]
    f = d.mean(axis=1).dropna()                      # 跨种子平均（手稿 A.5 口径）
    lo, hi = stats.t.interval(0.95, len(f) - 1, loc=f.mean(), scale=stats.sem(f))
    return f.mean(), lo, hi, len(f), d


print("=" * 100)
print("A. 每个对比的位移（S1 - 名义），按温度组")
print("=" * 100)
print(f"{'对比':<12}{'组':<13}{'名义均值':>10}{'S1均值':>10}{'位移':>9}"
      f"{'名义CI':>22}{'S1 CI':>22}")
summary = {}
for a, c in [("A2-1", "A2-0"), ("A2-3", "A2-0"), ("A2-4", "A2-0"),
             ("A2-2", "A2-0")]:
    for g in ORDER:
        rn = paired(a, c, g, "RMSE_nominal")
        rs = paired(a, c, g, "RMSE_S1")
        if not rn or not rs:
            continue
        summary[(a, g)] = (rn[0], rs[0])
        print(f"{a+'-'+c:<12}{g:<13}{rn[0]:>10.4f}{rs[0]:>10.4f}{rs[0]-rn[0]:>+9.4f}"
              f"   [{rn[1]:+.4f},{rn[2]:+.4f}]   [{rs[1]:+.4f},{rs[2]:+.4f}]")

print()
print("=" * 100)
print("B. 符号一致性（三种子方向），名义 -> S1")
print("=" * 100)
print(f"{'对比':<12}{'组':<13}{'名义符号':>10}{'S1符号':>10}{'变化':>8}")
for a, c in [("A2-1", "A2-0"), ("A2-3", "A2-0"), ("A2-4", "A2-0")]:
    for g in ORDER:
        s = df[df.eval_group == g]
        w = s.pivot_table(index="cycle_id", columns=["exp", "seed"], values=None,
                          aggfunc="first")
        wn = s.pivot_table(index="cycle_id", columns=["exp", "seed"], values="RMSE_nominal")
        ws = s.pivot_table(index="cycle_id", columns=["exp", "seed"], values="RMSE_S1")
        try:
            dn = (wn[a] - wn[c]).mean(axis=0)
            ds = (ws[a] - ws[c]).mean(axis=0)
        except KeyError:
            continue
        sn = "".join("+" if v > 0 else "-" for v in dn.values)
        ss = "".join("+" if v > 0 else "-" for v in ds.values)
        mark = " <== 翻转" if (sn.count("+") == 3) == (ss.count("+") == 3) and sn != ss else ""
        print(f"{a+'-'+c:<12}{g:<13}{sn:>10}{ss:>10}{'':>8}{mark}")

print()
print("=" * 100)
print("C. 参考偏移 Delta 的幅值分级")
print("=" * 100)
g = df.groupby("eval_group", observed=True).agg(
    n_files=("cycle_id", "nunique"),
    dabs=("dabs_mean_pp", "mean"),
    dabs_max=("dabs_mean_pp", "max"),
    d_end_mean=("delta_end_pp", "mean"),
    d_end_max=("delta_end_pp", "max"))
print(g.round(4).to_string())

print()
print("D. RMSE 位移（S1 - 名义）的跨文件分布")
sh = df.assign(shift=df.RMSE_S1 - df.RMSE_nominal)
print(sh.groupby("eval_group", observed=True)["shift"]
      .agg(["mean", "min", "max"]).round(4).to_string())

print()
print("=" * 100)
print("D2. 各档绝对 RMSE 排名（名义 vs S1）——看中心论断是否存活")
print("=" * 100)
for col in ["RMSE_nominal", "RMSE_S1"]:
    print(f"\n[{col}] 均值 pp，行=温度组 列=实验档")
    p = df.pivot_table(index="eval_group", columns="exp", values=col,
                       aggfunc="mean", observed=True).round(4)
    p["最优档"] = p.idxmin(axis=1)
    print(p.to_string())

print()
print("=" * 100)
print("E. 中心论断是否存活（判据：A2-4 在 OOD 各温度是否仍优于 A2-0，且符号一致）")
print("=" * 100)
for a, c in [("A2-4", "A2-0"), ("A2-3", "A2-0"), ("A2-1", "A2-0")]:
    print(f"\n[{a} vs {c}]")
    for g in ORDER:
        s = df[df.eval_group == g]
        wn = s.pivot_table(index="cycle_id", columns=["exp", "seed"], values="RMSE_nominal")
        ws = s.pivot_table(index="cycle_id", columns=["exp", "seed"], values="RMSE_S1")
        try:
            dn = (wn[a] - wn[c]).mean(axis=1).dropna()
            ds = (ws[a] - ws[c]).mean(axis=1).dropna()
        except KeyError:
            continue
        print(f"  {g:<13} 名义 {dn.mean():+.4f} (改善 {int((dn<0).sum())}/{len(dn)} 文件)"
              f"   S1 {ds.mean():+.4f} (改善 {int((ds<0).sum())}/{len(ds)} 文件)"
              f"   位移 {ds.mean()-dn.mean():+.4f}")
