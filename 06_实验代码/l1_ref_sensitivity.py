# -*- coding: utf-8 -*-
"""L1-3：参考口径敏感性数据整理（评审 §七 Fig.8 / §十六）

评审要求单独成图：
    Figure 8  Reference sensitivity
    Nominal reference vs capacity-corrected reference

数据已存在（`cb2_scenarios_perfile.csv`，3600 行），本脚本只做**整理与汇总**，
不重跑任何模型 —— 符合"先改实验、不重做实验"的约束。

产物
----
- `results/l1_ref_sensitivity_summary.csv`  按 (口径, 档) 汇总，供画图
- `results/l1_ref_sensitivity_report.md`    可读报告
"""
import os
import sys

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"E:\SOC论文项目"
RESULTS = os.path.join(ROOT, "04_实验", "数据", "results")

#: 口径的显示顺序与中文名（与 CB-2 报告 §2 一致）
SCEN_LABEL = {
    "S0": "S0 名义（额定 2.9 Ah）",
    "S1": "S1 逐文件（日期插值）",
    "S2": "S2 两端 2.80/2.35",
    "S3": "S3 统一 2.80",
    "S4": "S4 统一 2.35",
    "S5": "S5 名义×0.8 = 2.32",
}

GROUP_ORDER = ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]


def main():
    d = pd.read_csv(os.path.join(RESULTS, "cb2_scenarios_perfile.csv"))
    scen_in_data = sorted(d["scenario"].unique())
    exps = sorted(d["exp"].unique())

    # ── 汇总：先按 (scenario, exp, eval_group) 求各文件均值 ──
    rows = []
    for (sc, ex, grp), g in d.groupby(["scenario", "exp", "eval_group"]):
        per_file = g.groupby("cycle_id")["RMSE"].mean()
        rows.append({
            "scenario": sc,
            "scenario_label": SCEN_LABEL.get(sc, sc),
            "exp": ex,
            "eval_group": grp,
            "n_files": len(per_file),
            "rmse_mean_pp": round(float(per_file.mean()), 4),
            "rmse_sd_pp": round(float(per_file.std(ddof=1)), 4)
            if len(per_file) > 1 else 0.0,
        })
    sm = pd.DataFrame(rows)
    sm.to_csv(os.path.join(RESULTS, "l1_ref_sensitivity_summary.csv"),
              index=False)

    # ── 关键对比：EKF(基线) vs A2-3(本文)，逐个口径 ──
    r = ["# L1-3 参考口径敏感性（评审 §七 Fig.8 / §十六）", ""]
    r.append(f"> 数据源：`cb2_scenarios_perfile.csv`（{len(d)} 行）")
    r.append(f"> 口径：{', '.join(scen_in_data)}")
    r.append(f"> 档位：{', '.join(exps)}")
    r.append("")
    # 🔴 **重要更正（2026-09-30）**：
    #    `cb2_scenarios_perfile.csv` **只含神经网络档（A2-0…A2-4），没有 EKF 档**。
    #    A2-0 是"无诊断量残差学习基线"，**它本身也是带 NN 的**，不能当 EKF 代理。
    #    真 EKF 的数字在 `cb2b_ekf_physical_check.csv`（第二张表）。
    #    第一版我误用 A2-0 当 EKF，得到"S0 下 EKF 0.620 vs A2-3 0.621"，
    #    而真值是 **EKF 2.734 vs A2-3 0.621** —— 差 4 倍，是**严重的误读**。
    r.append("## 一、神经网络各档在各口径下的 RMSE（同分布 test_id 25 °C）")
    r.append("")
    r.append("> ⚠️ 本表**只有 NN 档**（A2-0…A2-4），**不含 EKF 基线**。")
    r.append("> EKF 的数字见下方第二张表 —— 两者不可混用。")
    r.append("")
    r.append("| 口径 | " + " | ".join(exps) + " |")
    r.append("|---" * (len(exps) + 1) + "|")

    for sc in scen_in_data:
        sub = sm[(sm["scenario"] == sc) & (sm["eval_group"] == "test_id(25C)")]
        cells = []
        for exp in exps:
            m = sub[sub["exp"] == exp]["rmse_mean_pp"]
            cells.append(f"{float(m.iloc[0]):.3f}" if len(m) else "—")
        r.append(f"| {SCEN_LABEL.get(sc, sc)} | " + " | ".join(cells) + " |")
    r.append("")
    r.append("## 二、EKF 在物理口径下的真实精度（CB-2 §8 独立校验）")
    r.append("")
    try:
        pc = pd.read_csv(os.path.join(RESULTS, "cb2b_ekf_physical_check.csv"))
        t = pc[pc["eval_group"] == "test_id(25C)"]
        r.append("| 组 | Cn=2.9→名义 | Cn=2.9→**物理** | "
                 "Cn=实测→名义 | Cn=实测→**物理** |")
        r.append("|---|---|---|---|---|")
        for grp in GROUP_ORDER:
            s = pc[pc["eval_group"] == grp]
            if s.empty:
                continue
            r.append(f"| {grp} | {s['A_vs_nom'].mean():.3f} | "
                     f"**{s['A_vs_S1'].mean():.3f}** | "
                     f"{s['B_vs_nom'].mean():.3f} | "
                     f"{s['B_vs_S1'].mean():.3f} |")
        r.append("")
        r.append("> `A` = EKF 内部 Cn 固定 2.9；`B` = 内部 Cn 换成实测。")
        r.append("> 「物理」= 以逐文件实测容量构造的参考。")
    except Exception as e:
        r.append(f"（cb2b 读取失败：{e}）")
    r.append("")
    open(os.path.join(RESULTS, "l1_ref_sensitivity_report.md"), "w",
         encoding="utf-8").write("\n".join(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
