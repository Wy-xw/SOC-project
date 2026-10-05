# -*- coding: utf-8 -*-
"""R1-M5b：同一池子（test_ood 36 文件、正确初始化）下的三方对比。

- 默认 EKF       : R=1e-4, q=1e-9
- 公平调参 EKF   : 在 6 个 25 °C 训练/验证文件上从 9 点网格选出（不碰测试）
- 事后 oracle EKF: 每个温度组用**本组自己**的测试文件选最优点（乐观上界）
- 融合 A2-3      : 10 种子组均值

输出: r1m5b_compare.md / .csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

RES = r"E:\SOC论文项目\04_实验\数据\results"
GRP = ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]
LAB = {"test_id(25C)": "25 °C", "ood(10C)": "10 °C", "ood(0C)": "0 °C",
       "ood(-10C)": "−10 °C", "ood(-20C)": "−20 °C"}


def main() -> int:
    fb = pd.read_csv(os.path.join(RES, "r1m5_fair_baseline_test_ood.csv"))
    fb_id = pd.read_csv(os.path.join(RES, "r1m5_fair_baseline_test_id.csv"))
    fb_id["eval_group"] = "test_id(25C)"
    fb_all = pd.concat([fb, fb_id], ignore_index=True)
    a = pd.read_csv(os.path.join(RES, "p5_b2_perfile_seed10.csv"))
    fuse = (a[a.exp == "A2-3"].groupby(["eval_group", "cycle_id"]).RMSE.mean()
            .reset_index())
    # 公平基线 CSV 无 eval_group，从 A2-3 表按 cycle_id 映射
    gmap = (a[a.exp == "A2-3"].groupby("cycle_id").eval_group.first().to_dict())
    fb_all["eval_group"] = fb_all.apply(
        lambda r: "test_id(25C)" if r["eval_group"] == "test_id(25C)"
        else gmap.get(r["cycle_id"]), axis=1)
    # 公平调参点在**调参池**（25 °C train/val，两版共用）上选出
    tune = fb_all[fb_all.pool == "tune"]
    fair_pt = tune.groupby("variant").rmse.mean().idxmin()

    fb = fb_all[fb_all.eval_group.notna() & (fb_all.pool == "eval")]
    # 逐文件表：默认 / 网格各点（用于事后 oracle）
    piv = fb.pivot_table(index=["eval_group", "cycle_id"], columns="variant",
                         values="rmse")
    grid_cols = [c for c in piv.columns if c.startswith("grid_")]

    rows = []
    for g in GRP:
        if g not in piv.index.get_level_values(0):
            continue
        sub = piv.loc[g]
        f_ = fuse[fuse.eval_group == g].set_index("cycle_id").RMSE
        # 事后 oracle：每组内选使组均值最小的那个网格点
        gm = sub[grid_cols].mean().sort_values()
        oracle_pt = gm.index[0]
        rows.append({
            "grp": LAB[g], "n": int(sub.shape[0]),
            "默认EKF": sub["default"].mean(),
            "公平调参": sub[fair_pt].mean(),
            "事后oracle": sub[oracle_pt].mean(),
            "融合A2-3": f_.reindex(sub.index).mean(),
            "_oracle_pt": oracle_pt, "_fair_pt": fair_pt,
        })
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, "r1m5b_compare.csv"), index=False, encoding="utf-8")

    L = ["# R1-M5b 三方对比（test_ood 36 文件，正确初始化）\n",
         f"- 公平调参点在 25 °C 调参池选出：**{fair_pt}**",
         f"- 事后 oracle 点为各组用本组测试文件选出（乐观上界）",
         "",
         "| 组 | n | 默认 EKF | 公平调参 | 事后 oracle | 融合 A2-3 | 融合 vs 公平 |",
         "|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        d = r["融合A2-3"] - r["公平调参"]
        L.append(f"| {r['grp']} | {r['n']} | {r['默认EKF']:.2f} | "
                 f"{r['公平调参']:.2f} | {r['事后oracle']:.2f} | "
                 f"{r['融合A2-3']:.2f} | {d:+.2f} |")
    L += ["", "## 事后 oracle 各组的选点", ""]
    for _, r in t.iterrows():
        L.append(f"- {r['grp']}: {r['_oracle_pt']}")
    ood = t[t["grp"] != "25 °C"]
    L += ["", "## 关键差值（OOD 四组）", "",
          f"- 默认 → 公平调参："
          f"{ood['默认EKF'].mean():.3f} → {ood['公平调参'].mean():.3f} pp",
          f"- 默认 → 事后 oracle："
          f"{ood['默认EKF'].mean():.3f} → {ood['事后oracle'].mean():.3f} pp",
          f"- 公平调参的收益占事后 oracle 收益的比例："
          f"{(ood['默认EKF'].mean()-ood['公平调参'].mean())/(ood['默认EKF'].mean()-ood['事后oracle'].mean())*100:.0f}%",
          "",
          f"- 融合 vs 公平调参（OOD 四组）："
          f"{(ood['融合A2-3']-ood['公平调参']).mean():+.3f} pp "
          f"（{'融合输' if (ood['融合A2-3']-ood['公平调参']).mean()>0 else '融合赢'}）",
          f"- 融合 vs 公平调参（25 °C）："
          f"{(t[t['grp']=='25 °C']['融合A2-3']-t[t['grp']=='25 °C']['公平调参']).iloc[0]:+.3f} pp"]
    open(os.path.join(RES, "r1m5b_compare.md"), "w", encoding="utf-8").write(
        "\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
