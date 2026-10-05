# -*- coding: utf-8 -*-
"""A1 结果分析：10 种子下的**符号一致率（带零分布）** + 配对区间。

为什么需要零分布（R2-M3）
------------------------
原稿只报 11/12，未给零假设下的期望。按原定义（三种子组均值是否与合并均值同号），
**无效应时每格同号概率 ≈ 3/4**（三取多数），12 格期望 ≈ 9/12。
⇒ 11/12 相对该基线只高约 2 格。没有零分布，读者无法判断是否显著。

本脚本同时给出
-------------
1. 逐格符号一致率（10 种子下改为「10 个种子中与合并均值同号的比例」）
2. **置换检验**：把各格 delta 的符号随机翻转，重算一致率，得零分布
3. A2-2（ν+NIS）的一致率——原稿漏报（R3-M3）
"""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = r"E:\SOC论文项目\04_实验\数据\results"
GROUPS = ["ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]


def load(which: str) -> pd.DataFrame:
    """优先用扩种子结果；缺失则回退原始 3 种子。"""
    ext = os.path.join(RESULTS, "p3_06_ablation_ext_full.csv")
    base = os.path.join(RESULTS, "p3_06_ablation_full.csv")
    a = pd.read_csv(ext) if os.path.exists(ext) else pd.DataFrame()
    b = pd.read_csv(base) if os.path.exists(base) else pd.DataFrame()
    return pd.concat([b, a], ignore_index=True).drop_duplicates(
        ["exp", "seed", "eval_group", "cycle_id"])


def main() -> int:
    d = load("auto")
    seeds = sorted(d["seed"].unique())
    print(f"种子数 {len(seeds)}: {seeds}")
    print()

    # 逐 (组, 种子, 文件) → 先按文件平均，得该种子在该组的组均值
    piv = (d.groupby(["exp", "seed", "eval_group"])["nn_RMSE"].mean()
             .reset_index())

    print("=" * 74)
    print("符号一致率（组内：各种子的组均值 与 合并均值 同号）")
    print("=" * 74)
    for pair, base in [("A2-1", "A2-0"), ("A2-2", "A2-0"),
                       ("A2-3", "A2-0"), ("A2-4", "A2-0")]:
        row, rates = [], []
        for g in GROUPS:
            a = piv[(piv["exp"] == pair) & (piv["eval_group"] == g)].set_index("seed")["nn_RMSE"]
            b = piv[(piv["exp"] == base) & (piv["eval_group"] == g)].set_index("seed")["nn_RMSE"]
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            delta = (a[common] - b[common]).values
            merged = np.mean(delta)
            same = int(np.sum(np.sign(delta) == np.sign(merged)))
            rates.append(same / len(delta))
            row.append(f"{g.split('(')[1][:-1]}:{same}/{len(delta)}")
        tot_same = int(round(sum(r * len(seeds) for r in rates)))
        tot_n = len(rates) * len(seeds)
        print(f"  {pair} − {base}:  {'  '.join(row)}   → 合计 {tot_same}/{tot_n}")

    # 置换检验：零分布
    print()
    print("=" * 74)
    print("置换检验（零假设：delta 符号随机）")
    print("=" * 74)
    rng = np.random.default_rng(0)
    for pair, base in [("A2-1", "A2-0"), ("A2-2", "A2-0"),
                       ("A2-3", "A2-0"), ("A2-4", "A2-0")]:
        all_delta = []
        for g in GROUPS:
            a = piv[(piv["exp"] == pair) & (piv["eval_group"] == g)].set_index("seed")["nn_RMSE"]
            b = piv[(piv["exp"] == base) & (piv["eval_group"] == g)].set_index("seed")["nn_RMSE"]
            common = a.index.intersection(b.index)
            all_delta.append((a[common] - b[common]).values)
        obs = sum(int(np.sum(np.sign(dv) == np.sign(np.mean(dv)))) for dv in all_delta)
        n = sum(len(dv) for dv in all_delta)
        # 零分布：把每个 delta 的符号随机翻转
        null = []
        for _ in range(2000):
            cnt = 0
            for dv in all_delta:
                s = np.sign(dv) * rng.choice([-1, 1], size=len(dv))
                cnt += int(np.sum(s == np.sign(np.mean(s))))
            null.append(cnt)
        null = np.array(null)
        p = float(np.mean(null >= obs))
        print(f"  {pair} − {base}:  观测 {obs}/{n}   零分布均值 {null.mean():.1f}   "
              f"p(≥观测) = {p:.4f}")

    print()
    print("说明：原稿按 3 种子报 11/12；本表按现有全部种子重算。")
    print("     零分布均值反映「无效应下也会有的同号数」，据此可判是否显著。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
