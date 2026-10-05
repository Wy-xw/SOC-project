# -*- coding: utf-8 -*-
"""复核 CB-1 报告的符号一致率表：逐格重算，核对合计。

背景：CB-1 报告的表加不起来
  A2-1 行：2+3+3+3+2 = 13，报告写 11/12
  A2-4 行：2+1+1+1+2 = 7，报告写 5/12
本脚本独立重算，判断"是报告抄错，还是口径不同"。
"""
import os
import sys

import numpy as np
import pandas as pd
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.join(__ROOT__, "SOC论文项目")
RESULTS = os.path.join(ROOT, "04_实验", "数据", "results")
OUT = os.path.join(RESULTS, "l1_cb1_recheck.md")

PAIRS = [("A2-1 − A2-0", "A2-1"), ("A2-3 − A2-0", "A2-3"),
         ("A2-4 − A2-0", "A2-4")]
GROUPS = ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]


def main():
    d = pd.read_csv(os.path.join(RESULTS, "p5_b2_perfile.csv"))
    r = ["# CB-1 符号一致率复核", ""]
    r.append("> 触发：CB-1 报告的表加不起来（A2-4 行 2+1+1+1+2=7，报告写 5/12）")
    r.append("")

    for label, a in PAIRS:
        r.append(f"## {label}")
        r.append("")
        r.append("| 温度组 | 逐种子 Δ 均值 | 合并均值 | 同号数 |")
        r.append("|---|---|---|---|")
        tot_same = tot_n = 0
        for grp in GROUPS:
            sub = d[d["eval_group"] == grp]
            p = sub.pivot_table(index=["seed", "cycle_id"], columns="exp",
                                values="RMSE")
            delta = (p[a] - p["A2-0"]).dropna()
            per_seed = delta.groupby(level=0).mean()
            overall = per_seed.mean()
            same = int((np.sign(per_seed) == np.sign(overall)).sum())
            tot_same += same
            tot_n += len(per_seed)
            sm = ", ".join(f"{v:+.3f}" for v in per_seed.values)
            r.append(f"| {grp} | {sm} | {overall:+.4f} | {same}/{len(per_seed)} |")
        r.append("")
        r.append(f"**五个温度合计（含 25 °C）**：{tot_same}/{tot_n}")
        # 只算 4 个 OOD（不含 25 °C）
        ood_same = ood_n = 0
        for grp in GROUPS[1:]:
            sub = d[d["eval_group"] == grp]
            p = sub.pivot_table(index=["seed", "cycle_id"], columns="exp",
                                values="RMSE")
            delta = (p[a] - p["A2-0"]).dropna()
            per_seed = delta.groupby(level=0).mean()
            ood_same += int((np.sign(per_seed)
                             == np.sign(per_seed.mean())).sum())
            ood_n += len(per_seed)
        r.append(f"**仅 4 个 OOD 温度**：{ood_same}/{ood_n}")
        r.append("")

    open(OUT, "w", encoding="utf-8").write("\n".join(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
