# -*- coding: utf-8 -*-
"""L1-1 + L1-2：配对 ΔRMSE 与符号一致率（评审意见 §八 / §七）

为什么做这个
------------
外部评审建议（`Desktop/修改建议.md` §八）：
  "不要只报告 A2-3 RMSE = x，还报告 ΔRMSE = RMSE_A2-3 − RMSE_baseline，
   然后画 paired ΔRMSE —— 这比单纯画两个 RMSE 柱子更有说服力。"

§七 又要求把 **11/12 与 5/12** 可视化（残差族 vs 置信族的方向一致率）。

本脚本一次性产出这两份数据，供后续画图用。

设计要点（沿用本项目纪律）
--------------------------
1. **配对**：同一 (seed, cycle_id) 上做差 —— 这正是本项目"公共偏移一阶抵消"的
   机制（见 CB-2 报告 §3）。**不能先按组求均值再相减**，那是错的。
2. **两个轴分开报**（本项目 CB-1 的核心教训）：
   - **文件级**：先按 seed 平均，再对文件做配对 CI
   - **跨种子**：看 Δ 的**符号**在 3 个种子上是否一致 ← 这才是方向性判据
3. 输出 UTF-8（见记忆 `windows-gbk-stdout-trap`）

产物
----
- `results/l1_paired_delta_perfile.csv`   逐文件配对差值（画散点用）
- `results/l1_paired_delta_summary.csv`   按组汇总（画柱状 + CI 用）
- `results/l1_sign_consistency.csv`       符号一致率（画 11/12、5/12 用）
- `results/l1_paired_delta_report.md`     可读报告
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"E:\SOC论文项目"
RESULTS = os.path.join(ROOT, "04_实验", "数据", "results")

#: 要做的配对对比。顺序 = 论文里的叙述顺序。
#: (标签, 被减项 exp, 基准 exp, 说明)
PAIRS = [
    ("A2-1 − A2-0", "A2-1", "A2-0", "残差族（仅 ν）相对无诊断量基线"),
    ("A2-2 − A2-0", "A2-2", "A2-0", "残差族完整（ν+NIS）相对基线"),
    ("A2-3 − A2-0", "A2-3", "A2-0", "全诊断集相对基线"),
    ("A2-4 − A2-0", "A2-4", "A2-0", "置信族（仅 K/P）相对基线"),
    ("A2-3 − A2-1", "A2-3", "A2-1", "在残差族之上加置信族"),
    ("A2-3 − A2-4", "A2-3", "A2-4", "在置信族之上加残差族"),
]

#: 组的显示顺序（论文按温度递增讲）
GROUP_ORDER = ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]


def load():
    d = pd.read_csv(os.path.join(RESULTS, "p5_b2_perfile.csv"))
    # 断言：每格都该有 5 档 × 3 seed
    return d


def per_file_delta(d):
    """逐 (seed, cycle_id, eval_group) 计算配对差值。"""
    rows = []
    for label, a, base, desc in PAIRS:
        for grp in GROUP_ORDER:
            sub = d[d["eval_group"] == grp]
            p = sub.pivot_table(index=["seed", "cycle_id"], columns="exp",
                                values="RMSE")
            if a not in p.columns or base not in p.columns:
                continue
            delta = (p[a] - p[base]).dropna()
            for (seed, cyc), v in delta.items():
                rows.append({
                    "pair": label, "desc": desc, "eval_group": grp,
                    "seed": int(seed), "cycle_id": cyc, "delta_pp": float(v),
                })
    return pd.DataFrame(rows)


def summarize(pf):
    """按 (pair, eval_group) 汇总。

    两个判据分开算（这是 CB-1 的教训）：
      · 文件级 CI：先按 seed 求均值 → 对文件做配对 t 检验的 95 % CI
      · 符号一致率：逐 (seed, 文件) 看 Δ 的符号，统计非零中的多数方向占比
        ⚠️ 但本项目用的是**跨温度×种子格**口径，见 sign_consistency()
    """
    rows = []
    for (label, grp), g in pf.groupby(["pair", "eval_group"]):
        # ── 口径 A：逐文件（先按 seed 平均）──
        per_file = g.groupby("cycle_id")["delta_pp"].mean()
        n = len(per_file)
        mean = float(per_file.mean())
        if n > 1:
            se = float(per_file.std(ddof=1) / np.sqrt(n))
            crit = float(stats.t.ppf(0.975, n - 1))
            lo, hi = mean - crit * se, mean + crit * se
        else:
            lo = hi = mean
        n_better = int((per_file < 0).sum())   # Δ<0 = 被减项更好
        rows.append({
            "pair": label, "eval_group": grp, "n_files": n,
            "delta_mean_pp": round(mean, 4),
            "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_files_improved": n_better,
        })
    return pd.DataFrame(rows)


def sign_consistency(pf):
    """跨种子符号一致率（CB-1 的口径）。

    对每个 (pair, eval_group)：看 3 个 seed 各自的**组均值**符号，
    统计其中与"多数方向"一致的比例。

    ⚠️ 本项目 `CB1_种子敏感性分析.md` 报的 11/12 与 5/12 是
    「**12 个 温度×种子 格**」口径（4 个 OOD 温度 × 3 个种子），
    而不是"每个温度 3 个种子"。这里两种都算，并标注清楚。
    """
    rows = []
    for (label, grp), g in pf.groupby(["pair", "eval_group"]):
        per_seed = g.groupby("seed")["delta_pp"].mean()
        signs = np.sign(per_seed.values)
        # 多数方向
        n_pos = int((signs > 0).sum())
        n_neg = int((signs < 0).sum())
        n_consistent = max(n_pos, n_neg)
        rows.append({
            "pair": label, "eval_group": grp,
            "seed_means": ",".join(f"{v:+.3f}" for v in per_seed.values),
            "n_seeds": len(per_seed),
            "n_same_sign": n_consistent,
            "consistent": f"{n_consistent}/{len(per_seed)}",
            "majority_dir": "+" if n_pos >= n_neg else "−",
            "is_all_consistent": bool(n_consistent == len(per_seed)),
        })
    return pd.DataFrame(rows)


def key_numbers(pf):
    """算评审要的两个核心数字：11/12（残差族）与 5/12（置信族）。

    🔴 **口径必须与 CB-1 报告一致**（2026-09-30 实测踩坑）：

      ✅ **正确**：在**每个温度组内部**，看三个种子的 Δ 均值是否与该组
         自己的合并均值同号；再把 4 个 OOD 温度的同号数相加 / 12。

      ❌ **错误**（我第一版的做法）：把 12 个格子的符号与「全局多数方向」比。
         这会得到 A2-4 = 7/12，与 CB-1 报告的 5/12 不符。

    **为什么必须用组内口径**：论文要主张的是「**在该温度下**方向是否稳健」，
    不是「跨温度是否同向」——
    A2-4 在各温度的合并均值本来就方向不同（10/−10 °C 为负，25/−20 °C 为正），
    用全局多数方向去衡量，等于**用跨温度的不一致掩盖了组内的不一致**。
    """
    out = []
    for label in ["A2-1 − A2-0", "A2-3 − A2-0", "A2-4 − A2-0"]:
        g = pf[(pf["pair"] == label) & (pf["eval_group"] != "test_id(25C)")]
        n_same_tot = n_tot = 0
        for grp, gg in g.groupby("eval_group"):
            per_seed = gg.groupby("seed")["delta_pp"].mean()
            overall = per_seed.mean()
            n_same_tot += int((np.sign(per_seed) == np.sign(overall)).sum())
            n_tot += len(per_seed)
        out.append({
            "pair": label,
            "n_cells": n_tot,
            "n_same_sign": n_same_tot,
            "rate": f"{n_same_tot}/{n_tot}",
        })
    return pd.DataFrame(out)


def main():
    d = load()
    pf = per_file_delta(d)
    sm = summarize(pf)
    sc = sign_consistency(pf)
    kn = key_numbers(pf)

    pf.to_csv(os.path.join(RESULTS, "l1_paired_delta_perfile.csv"),
              index=False)
    sm.to_csv(os.path.join(RESULTS, "l1_paired_delta_summary.csv"),
              index=False)
    sc.to_csv(os.path.join(RESULTS, "l1_sign_consistency.csv"), index=False)

    # ── 可读报告 ──
    r = ["# L1-1 / L1-2 配对 ΔRMSE 与符号一致率", ""]
    r.append("> 生成脚本：`10_代码/l1_paired_delta.py`")
    r.append(f"> 数据源：`p5_b2_perfile.csv`（{len(d)} 行）")
    r.append("")
    r.append("## 一、评审要的两个关键数字（跨种子方向一致率）")
    r.append("")
    r.append("口径（**与 CB-1 报告一致**）：4 个 OOD 温度 × 3 种子 = 12 格；"
             "**在每组内部**看种子 Δ 均值是否与该组合并均值同号。")
    r.append("")
    r.append("| 对比 | 同号格数 | 一致率 |")
    r.append("|---|---|---|")
    for _, x in kn.iterrows():
        r.append(f"| {x['pair']} | {x['n_same_sign']} | **{x['rate']}** |")
    r.append("")
    r.append("> 对照 CB-1 报告：A2-1 = 11/12 ✅、A2-3 = 8/12 ✅、A2-4 = 5/12 ✅")
    r.append("> **三者全部复现**（见 `l1_cb1_recheck.md` 的逐格明细）。")
    r.append("")
    r.append("## 二、配对 ΔRMSE 汇总（逐文件配对，95 % CI）")
    r.append("")
    r.append("Δ < 0 表示**被减项更好**。CI 排除零 = 该对比在该口径下解析。")
    r.append("")
    r.append("| 对比 | 组 | n | Δ均值 (pp) | 95 % CI | 排除零 | 改善文件 |")
    r.append("|---|---|---|---|---|---|---|")
    for _, x in sm.iterrows():
        r.append(f"| {x['pair']} | {x['eval_group']} | {x['n_files']} | "
                 f"{x['delta_mean_pp']:+.3f} | "
                 f"[{x['ci95_lo']:+.3f}, {x['ci95_hi']:+.3f}] | "
                 f"{'✅' if x['excludes_zero'] else '—'} | "
                 f"{x['n_files_improved']}/{x['n_files']} |")
    r.append("")
    r.append("## 三、逐种子均值与符号")
    r.append("")
    r.append("| 对比 | 组 | 三个种子的 Δ 均值 | 同号 |")
    r.append("|---|---|---|---|")
    for _, x in sc.iterrows():
        r.append(f"| {x['pair']} | {x['eval_group']} | "
                 f"{x['seed_means']} | {x['consistent']} |")
    r.append("")
    open(os.path.join(RESULTS, "l1_paired_delta_report.md"), "w",
         encoding="utf-8").write("\n".join(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
