# -*- coding: utf-8 -*-
"""意见8 §二.3：融合规则敏感性分析。

论文式(12) 为**固定加性**修正：SOC_fused = SOC_EKF − ŷ/100。
本脚本检验该选择对结论的影响：换成缩放 / 门控 / 限幅融合后，
A2-3 的**相对排序**是否变化（尤其 OOD 上是否仍不占优）。

变体（均在 A2-3 的同一批预测 ŷ 上重算，不改网络）
------------------------------------------------
- base        : SOC − ŷ/100                        （论文式12，全权重）
- no_corr     : SOC                              （ŷ≡0，≈A2-0 的极限对照）
- scale_a     : SOC − a·ŷ/100,  a ∈ {0.5, 1.5}
- gate_nu     : 权重 w=1/(1+|ν|/θ)，θ=0.01 V       （新息大→少信修正）
- gate_P      : 权重 w=min(1, P00/θ)，θ=(0.01)²     （协方差大→多信修正）
- clip_c      : SOC − clip(ŷ,−c,c)/100, c ∈ {2,5} pp

产物：results/r1m7_fusion_rules.csv / .md
"""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(__ROOT__, "06_实验代码", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nn_features import GROUPS, build_split_arrays, load_history  # noqa: E402
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PROJ = os.path.join(__ROOT__, "SOC论文项目")
RES = os.path.join(PROJ, "04_实验", "数据", "results")
MODELS = os.path.join(PROJ, "04_实验", "models")
SEEDS = [7, 13, 42, 101, 202, 303, 404, 505, 606, 707]
EXP = "A2-3"
THETA_NU = 0.01      # V
THETA_P = 1e-4       # V²


def group_of(cid: str) -> str | None:
    """按文件名判温度。顺序关键：'10degC' ⊂ '0degC'、'0degC' ⊂ 'n10degC'/'n20degC'，
    故负温 → '10degC' → '0degC' 依次判。"""
    if "25degC" in cid:
        return "25 °C"
    for tag, lab in [("n20degC", "−20 °C"), ("n10degC", "−10 °C"),
                     ("10degC", "10 °C"), ("0degC", "0 °C")]:
        if tag in cid:
            return lab
    return None


def main() -> int:
    from tensorflow import keras
    hist = load_history(os.path.join(RES, "ekf_full_history.npz"))
    models = {s: keras.models.load_model(
        os.path.join(MODELS, f"gru16_{EXP}_seed{s}.keras"), compile=False)
        for s in SEEDS}

    rows = []
    for cid, h in hist.items():
        g = group_of(cid)
        if g is None or len(h["nu"]) < 400:
            continue
        X, y, ends_by_file = build_split_arrays(hist, [cid], GROUPS[EXP])
        if X is None:
            continue
        ends = ends_by_file[cid]
        soc = h["soc"][ends]
        nu = h["nu"][ends]
        P00 = h["P00"][ends]
        for s in SEEDS:
            yh = models[s].predict(X, verbose=0).ravel()          # pp
            variants = {"base": soc - yh / 100.0}
            variants["no_corr"] = soc.copy()
            for a in (0.5, 1.5):
                variants[f"scale_{a}"] = soc - a * yh / 100.0
            w_nu = 1.0 / (1.0 + np.abs(nu) / THETA_NU)
            variants["gate_nu"] = soc - w_nu * yh / 100.0
            w_p = np.clip(P00 / THETA_P, 0.0, 1.0)
            variants["gate_P"] = soc - w_p * yh / 100.0
            for c in (2.0, 5.0):
                variants[f"clip_{int(c)}"] = soc - np.clip(yh, -c, c) / 100.0
            for vname, soc_f in variants.items():
                e = (soc_f - h["soc_true"][ends]) * 100
                rows.append({"exp": EXP, "seed": s, "eval_group": g,
                             "cycle_id": cid, "rule": vname,
                             "RMSE": float(np.sqrt(np.mean(e ** 2)))})
        print(f"  {cid[:44]} {g}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "r1m7_fusion_rules.csv"), index=False)

    piv = (df.groupby(["eval_group", "rule"]).RMSE.mean()
           .unstack("rule"))
    order = ["no_corr", "scale_0.5", "base", "scale_1.5", "clip_5",
             "clip_2", "gate_nu", "gate_P"]
    piv = piv[[c for c in order if c in piv.columns]]
    print("\n" + piv.round(3).to_string())
    L = ["# R1-M7 融合规则敏感性（A2-3，10 种子）\n",
         "各格为**种子×文件均值** RMSE (pp)。`base` = 论文式(12) 固定加性融合。\n",
         "| 组 | " + " | ".join(piv.columns) + " |",
         "|---" * (len(piv.columns) + 1) + "|"]
    for g, r in piv.iterrows():
        L.append(f"| {g} | " + " | ".join(f"{r[c]:.2f}" for c in piv.columns) + " |")
    L += ["", "`no_corr` 为 ŷ≡0（不修正）的极限参照。"]
    open(os.path.join(RES, "r1m7_fusion_rules.md"), "w",
         encoding="utf-8").write("\n".join(L) + "\n")
    print("\n产物: r1m7_fusion_rules.csv / .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
