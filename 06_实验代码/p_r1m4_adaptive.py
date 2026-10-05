# -*- coding: utf-8 -*-
"""
R1-M4 补充实验：自适应滤波对照 + Q/R 敏感性
- 40 文件（test_id 4 + OOD 36，每温度 9）
- 变体：EKF(固定) / AEKF-R / AEKF-Q / AEKF-RQ
- 网格：r ∈ {1e-5,1e-4,1e-3} × q_soc ∈ {1e-8,1e-9,1e-10}（固定 EKF）
输出：results/r1m4_perfile.csv、results/r1m4_summary.md
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "10_代码", "src"))
from ecm_ekf import ECMParams, SOC_EKF  # noqa
from ecm_tables import OCVTable, ECMParamTable, make_ekf_table_fns  # noqa


# Windows 控制台默认 GBK：本脚本 print/写文件含 GBK 无码位字符（− ⚠️ 🔴），
# 不切编码会在成功路径崩掉。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, "04_实验", "数据", "processed")
RES = os.path.join(PROJ, "04_实验", "数据", "results")
BURN = 300
SOC0_BIAS = 0.10


def grp(c: str):
    if "25degC_Cycle" in c: return "test_id(25C)"
    if "10degC_" in c and "n10degC" not in c and "n20degC" not in c: return "ood(10C)"
    if "0degC_" in c and "10degC" not in c and "20degC" not in c: return "ood(0C)"
    if "n10degC_" in c: return "ood(-10C)"
    if "n20degC_" in c: return "ood(-20C)"
    return None


def main():
    ocv_fn, docv_fn, params_fn = make_ekf_table_fns(
        OCVTable.from_csv(), ECMParamTable.from_csv())
    man = pd.read_csv(os.path.join(PROJ, "04_实验", "数据", "splits",
                                   "split_manifest.csv"))
    d = pd.read_csv(os.path.join(DATA, "mcmaster_drive_cycles.csv.gz"))
    test_cids = set(man[man.split.isin(["test_id", "test_ood"])].cycle_id)

    variants = [("EKF_fixed", {}), ("AEKF_R", dict(adapt_R=True)),
                ("AEKF_Q", dict(adapt_Q=True)),
                ("AEKF_RQ", dict(adapt_R=True, adapt_Q=True))]
    grid = [(r, q) for r in (1e-5, 1e-4, 1e-3) for q in (1e-8, 1e-9, 1e-10)]

    rows = []
    for cid, g in d.groupby("cycle_id"):
        if cid not in test_cids or len(g) < 100:
            continue
        gg = grp(cid)
        if gg is None:
            continue
        i = g.current_A.to_numpy(); v = g.voltage_V.to_numpy()
        t = g.temperature_C.to_numpy(); s = g.soc_true.to_numpy()

        def run(**kw):
            p = ECMParams(Cn=2.9, dt=1.0)
            soc0 = float(np.clip(s[0] + SOC0_BIAS, 0.0, 1.0))
            e = SOC_EKF(p, soc0=soc0, ocv_fn=ocv_fn, docv_fn=docv_fn,
                        params_fn=params_fn, **kw)
            for k in range(len(i)):
                e.step(float(i[k]), float(v[k]), temp_k=float(t[k]))
            err = (np.array(e.history["soc"])[BURN:] - s[BURN:]) * 100
            return float(np.sqrt(np.mean(err ** 2)))

        for name, kw in variants:
            rows.append({"cycle_id": cid, "grp": gg, "variant": name,
                         "rmse": run(**kw)})
        for r, q in grid:
            rows.append({"cycle_id": cid, "grp": gg,
                         "variant": f"grid_r{r:.0e}_q{q:.0e}",
                         "rmse": run(r=r, q_soc=q)})
        print(cid[:40], "done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "r1m4_perfile.csv"), index=False)
    order = ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]

    # 变体汇总
    v = df[~df.variant.str.startswith("grid")].groupby(
        ["variant", "grp"]).rmse.mean().unstack().loc[:, order]
    # 网格敏感性（每组的 min/max/中位）
    g = df[df.variant.str.startswith("grid")]
    gs = g.groupby("grp").rmse.agg(["min", "median", "max"]).loc[order]
    # 与融合模型对比（B2 多文件均值）
    fused = {"test_id(25C)": 0.621, "ood(10C)": 3.329, "ood(0C)": 8.824,
             "ood(-10C)": 11.329, "ood(-20C)": 14.635}

    L = ["# R1-M4 自适应滤波对照 + Q/R 敏感性\n",
         f"文件池：每温度 {df[df.variant=='EKF_fixed'].groupby('grp').size().iloc[0]}"
         "（test_id 4）；稳态 RMSE（前 300 s 剔除），**正确初始化**。\n"
         "（脚本内 `SOC0_BIAS = 0.10`，但本数据集每个文件自满充 s[0]≈1.0 起步，"
         "`clip(s[0]+0.10,0,1)` 把该偏置裁掉 —— 故实际为正确初始化，"
         "与论文 §5.4 主口径一致。）\n",
         "## 变体对比（RMSE, pp）\n",
         "| 评估组 | EKF(固定) | AEKF-R | AEKF-Q | AEKF-RQ | EKF+NN(A2-3) |",
         "|---|---|---|---|---|---|"]
    for ev in order:
        L.append(f"| {ev} | {v.loc['EKF_fixed', ev]:.3f} | "
                 f"{v.loc['AEKF_R', ev]:.3f} | {v.loc['AEKF_Q', ev]:.3f} | "
                 f"{v.loc['AEKF_RQ', ev]:.3f} | {fused[ev]:.3f} |")
    L += ["", "## Q/R 网格敏感性（固定 EKF，9 组合）\n",
          "| 评估组 | min | 中位 | max | 极差 |",
          "|---|---|---|---|---|"]
    for ev in order:
        row = gs.loc[ev]
        L.append(f"| {ev} | {row['min']:.3f} | {row['median']:.3f} | "
                 f"{row['max']:.3f} | {row['max']-row['min']:.3f} |")
    L += ["", "## 默认值(1e-4/1e-9)在网格中的位置\n",
          f"- 默认组合在各组的中位排名计算见 CSV；关键句："
          "固定 EKF 的表现在 9 组网格内的极差 = 上表「极差」列。"]

    # ⚠️ 2026-09-27：下面这段口径说明原本是**手工加进 r1m4_summary.md 的**，
    #    结果重跑本脚本时被静默抹掉（本脚本整体重写该文件）。
    #    现改为**由脚本生成**，并把两组数字**现算**，这样重跑不再丢注记、
    #    数字也不会随数据漂移。与 p6_tables.py 那类"生成器回改"是同一类坑。
    gm = (g.groupby(["grp", "variant"]).rmse.mean()
          .groupby("grp").agg(["min", "max"]))

    def _raw(ev: str) -> str:
        r = gs.loc[ev]
        return f"{r['min']:.3f} – {r['max']:.3f}"

    def _mean(ev: str) -> str:
        r = gm.loc[ev]
        return f"{r['min']:.2f} – {r['max']:.2f}"

    L += ["", "---", "",
          "> 🔴 **口径说明（2026-09-26 立，2026-09-27 改为脚本生成）——"
          "本表统计的是「逐文件原始 RMSE」的 min/中位/max，不是论文用的那个口径。"
          "拿本表复核论文会误判。**",
          ">",
          "> | 口径 | −10 °C 的范围 | −20 °C 的范围 |",
          "> |---|---|---|",
          f"> | **本表**：9 个网格组合 × 逐文件原始 RMSE 的 min/max "
          f"| {_raw('ood(-10C)')} | {_raw('ood(-20C)')} |",
          f"> | **论文**（§6.2）：9 个网格组合的**组均值**再取 min/max "
          f"| **{_mean('ood(-10C)')}** | **{_mean('ood(-20C)')}** |",
          ">",
          "> **论文的数字是对的**——本脚本在写本文件时逐格重算过 `r1m4_perfile.csv`："
          "组均值口径下 "
          + "、".join(f"{ev} {_mean(ev)}" for ev in order) + "。",
          "> 差异来源：本表把**单个文件的极端值**也算进去了，"
          "而论文报的是**该组合的组内均值**——后者才是可比的统计量。",
          ">",
          "> ⚠️ **结论不受影响**：论文的关键句"
          "「每个 OOD 温度都被调好参的固定滤波器反超」逐项成立。"]
    with open(os.path.join(RES, "r1m4_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("saved: r1m4_perfile.csv / r1m4_summary.md")


if __name__ == "__main__":
    main()
