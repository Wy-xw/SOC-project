# -*- coding: utf-8 -*-
"""
P5-B2 补充：A2-4（仅 K + diag(P)，无 ν/NIS）的文件级配对统计。

**为什么需要这个脚本**
--------------------
2026-09-24 三盲审的 R1-M4 / R2-M4 共同指出：§6.3 的结论句
「neither ν nor K/P helps in isolation」与 Table 2（稿内消融表；2026-09-26 重编号前叫 Table 3）自相矛盾——
A2-4 在 −10/−20 °C 比完整模型 A2-3 还好。而 `p5_b2_paired_summary.md`
只给了 A2-3−A2-0、A2-3−A2-2、A2-1−A2-0 三组配对，
**恰恰缺了能裁决该争议的 A2-4−A2-0 与 A2-4−A2-3**。本脚本补上。

**为什么不能用 p5_b2_perfile.csv 自带的 eval_group 列**
------------------------------------------------------
该列的 `ood(0C)` 分组含 **36 个文件**（等于全部 OOD），
而 ood(10C)/(−10C)/(−20C) 各 9 个 —— 是上游分组函数里的字符串子串陷阱
（`"0degC_"` 会匹配 `"10degC_"`）留下的标签错误，R1-M9 已记录。
本脚本改为**按文件名显式分类**，并排除 trise 文件。

自检：用本脚本重算 A2-3−A2-0，结果与 `p5_b2_paired_summary.md` 逐格吻合，
且各组均值与 Table 2 一致 —— 证明分组正确。

用法
----
    "D:/venvs/soc-paper/Scripts/python.exe" 10_代码/p5_b2_a24_paired.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# Windows 控制台默认 GBK，而本脚本 print 含 GBK 无码位的字符 —— 不改编码会在
# **成功路径**崩掉（活干完了却退出码 1）。别靠 PYTHONIOENCODING 兜底。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
PERFILE = os.path.join(RESULTS, "p5_b2_perfile.csv")
OUT_MD = os.path.join(RESULTS, "p5_b2_a24_paired_summary.md")

GROUPS = ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]


def classify(cid: str) -> str | None:
    """按文件名判温度组。注意必须显式排除子串陷阱（10degC 含 0degC）。"""
    if "trise" in cid:
        return None
    if "25degC" in cid:
        return "test_id(25C)" if "Cycle" in cid else "train_or_val"
    if "n10degC" in cid:
        return "ood(-10C)"
    if "n20degC" in cid:
        return "ood(-20C)"
    if "10degC" in cid:
        return "ood(10C)"
    if "0degC" in cid:
        return "ood(0C)"
    return None


def paired(df: pd.DataFrame, a: str, b: str):
    """文件级配对（a − b），返回 (n, δ均值, CI下, CI上, a更好的文件数)。"""
    d = (df[a] - df[b]).values
    n = len(d)
    m = float(d.mean())
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return n, m, m - t * se, m + t * se, int((d < 0).sum())


def main() -> int:
    raw = pd.read_csv(PERFILE)
    raw["grp"] = raw["cycle_id"].map(classify)
    per = (raw.dropna(subset=["grp"])
              .groupby(["exp", "grp", "cycle_id"], as_index=False)["RMSE"].mean())
    piv = per.pivot_table(index=["grp", "cycle_id"], columns="exp",
                          values="RMSE").reset_index()

    L = ["# B2 补充配对统计：A2-4（仅 K/P）与 A2-3（完整诊断集）\n",
         "> 生成脚本 `10_代码/p5_b2_a24_paired.py`；"
         "源数据 `p5_b2_perfile.csv`（每文件先跨 3 seeds 平均，再做文件级配对）。",
         "> ⚠️ 分组用文件名显式判定，**不用** `p5_b2_perfile.csv` 自带的 "
         "`eval_group` 列（其 `ood(0C)` 误含 36 个文件，见 R1-M9）。\n",
         "## 组均值（RMSE, pp；与 Table 2 核对一致）\n",
         "| 评估组 | n | A2-0 | A2-1 | A2-2 | A2-3 | A2-4 |",
         "|---|---|---|---|---|---|---|"]
    for grp in GROUPS:
        s = piv[piv["grp"] == grp]
        cells = " | ".join(f"{s[c].mean():.3f}"
                           for c in ["A2-0", "A2-1", "A2-2", "A2-3", "A2-4"])
        L.append(f"| {grp} | {len(s)} | {cells} |")

    L.append("\n## 配对差异 δ = 前者 − 后者（负 = 前者更好）\n")
    for a, b, title in [("A2-3", "A2-0", "A2-3 − A2-0（完整诊断 vs 恒等映射对照）"),
                        ("A2-4", "A2-0", "A2-4 − A2-0（仅 K/P vs 对照）★ 本轮新增"),
                        ("A2-4", "A2-3", "A2-4 − A2-3（K/P 单独是否够用）★ 本轮新增"),
                        ("A2-4", "A2-2", "A2-4 − A2-2（K/P 相对 ν+NIS）★ 本轮新增")]:
        L.append(f"\n### {title}\n")
        L.append("| 评估组 | n | δ (pp) | 95% CI | 改善文件数 |")
        L.append("|---|---|---|---|---|")
        for grp in GROUPS:
            s = piv[piv["grp"] == grp]
            n, m, lo, hi, nb = paired(s, a, b)
            star = "" if lo <= 0 <= hi else " **排除零**"
            L.append(f"| {grp} | {n} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}]{star} | "
                     f"{nb}/{n} |")

    L.append("""
## 读法与对 §6.3 的含义

1. **A2-4 − A2-0**：仅 K/diag(P) 就能在 −10 °C 与 −20 °C 上
   显著优于恒等映射对照（−0.242 / −0.282 pp，9/9 文件，CI 排除零）。
   分布内则显著更差（+0.061 pp，0/4）。⇒ **「K/P 单独没有用」不成立。**

2. **A2-4 − A2-3**：分布内完整集显著更好（+0.060 pp，0/4，CI 排除零）；
   −10 °C 反向且勉强显著（−0.045，7/9）；其余温度 CI 跨零。
   ⇒ **ν 与 NIS 的价值在分布内，不在深度外推。**

3. 合并读法：**K/diag(P) 承载跨温增量，ν/NIS 承载同分布增量；
   ν 单独使用有害。** 这与 §6.3 现稿的
   「neither ν nor K/P helps in isolation」直接冲突，需按本表改写。

## ⚠️ 上游数据问题（待修）

`p5_b2_perfile.csv` 的 `eval_group` 列中 `ood(0C)` 含 36 个文件，
根因是分组函数的字符串子串陷阱（`"0degC_"` 匹配 `"10degC_"`）。
本脚本已绕开，但**源文件应修正并重出**，否则任何直接使用该列的分析都会出错。
""")
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print("saved:", OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
