# -*- coding: utf-8 -*-
"""意见8 §三.5：自适应 EKF 超参敏感性扫描。

论文 §5.4.2(b) 用 W=60、α=0.2 的自适应 EKF（新息窗口匹配 R）。
本脚本扫 W×α 网格，检验"自适应实现设置影响相对性能"这一主张对超参的敏感性。

产物：results/r1m8_aekf_sweep.csv / .md
"""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(__ROOT__, "06_实验代码", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ecm_ekf import ECMParams, SOC_EKF  # noqa: E402
from ecm_tables import OCVTable, ECMParamTable, make_ekf_table_fns  # noqa: E402
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PROJ = __ROOT__
DATA = os.path.join(PROJ, "04_实验", "数据", "processed")
RES = os.path.join(PROJ, "04_实验", "数据", "results")
BURN = 300
SOC0_BIAS = 0.0
WINS = [20, 60, 120]
ALPHAS = [0.05, 0.2, 0.4]


def temp_of(cid: str) -> float | None:
    for tag, T in [("n20degC", -20.0), ("n10degC", -10.0),
                   ("10degC", 10.0), ("0degC", 0.0), ("25degC", 25.0)]:
        if tag in cid:
            return T
    return None


def main() -> int:
    ocv_fn, docv_fn, params_fn = make_ekf_table_fns(
        OCVTable.from_csv(), ECMParamTable.from_csv())
    man = pd.read_csv(os.path.join(PROJ, "04_实验", "数据", "splits",
                                   "split_manifest.csv"))
    d = pd.read_csv(os.path.join(DATA, "mcmaster_drive_cycles.csv.gz"))
    ids = sorted(set(man[man.split.isin(["test_ood", "test_id"])].cycle_id))

    variants = [("fixed", None, None)] + [
        (f"R_W{w}_a{a}", w, a) for w in WINS for a in ALPHAS]
    # RQ（R 与 Q 同时自适应）单点参照：论文用的即此变体
    variants.append(("RQ_W60_a0.2", 60, 0.2))

    shard_dir = os.path.join(RES, "_shards_r1m8")
    os.makedirs(shard_dir, exist_ok=True)

    def shard_of(cid, tag):
        safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in cid)
        return os.path.join(shard_dir, f"r1m8_{tag}_{safe}.csv")

    for n, cid in enumerate(ids, 1):
        gp = d[d.cycle_id == cid]
        if len(gp) < BURN + 100:
            continue
        i = gp.current_A.to_numpy(); v = gp.voltage_V.to_numpy()
        t = gp.temperature_C.to_numpy(); s = gp.soc_true.to_numpy()
        T = temp_of(cid)
        for tag, w, a in variants:
            sp = shard_of(cid, tag.replace(".", "p"))
            if os.path.exists(sp):
                continue
            if w is None:
                kw = {}
            else:
                kw = {"adapt_R": True, "adapt_Q": tag.startswith("RQ_"),
                      "adapt_win": w, "adapt_alpha": a}
            e = SOC_EKF(ECMParams(Cn=2.9, dt=1.0),
                        soc0=float(np.clip(s[0] + SOC0_BIAS, 0, 1)),
                        ocv_fn=ocv_fn, docv_fn=docv_fn, params_fn=params_fn, **kw)
            for k in range(len(i)):
                e.step(float(i[k]), float(v[k]), temp_k=float(t[k]))
            err = (np.array(e.history["soc"])[BURN:] - s[BURN:]) * 100
            pd.DataFrame([{"cycle_id": cid, "T": T, "variant": tag,
                           "rmse": float(np.sqrt(np.mean(err ** 2)))}]
                         ).to_csv(sp, index=False)
        print(f"  [{n}/{len(ids)}] {cid[:42]} T={T}", flush=True)

    parts = []
    for cid in ids:
        for tag, _, _ in variants:
            fp = shard_of(cid, tag.replace(".", "p"))
            if os.path.exists(fp):
                parts.append(pd.read_csv(fp))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(os.path.join(RES, "r1m8_aekf_sweep.csv"), index=False)

    piv = df.pivot_table(index="T", columns="variant", values="rmse",
                         aggfunc="mean")
    piv = piv[[v[0] for v in variants if v[0] in piv.columns]]
    print("\n" + "=" * 70)
    print(piv.round(3).to_string())
    L = ["# R1-M8 自适应 EKF 超参敏感性（W×α 网格，正确初始化）\n",
         "各格为区间内文件均值 RMSE (pp)。`fixed` = 非自适应默认 EKF。\n",
         "| 温度 | " + " | ".join(piv.columns) + " |",
         "|---" * (len(piv.columns) + 1) + "|"]
    for T, r in piv.iterrows():
        L.append(f"| {T:.0f} °C | " + " | ".join(f"{r[c]:.2f}" for c in piv.columns) + " |")
    L += ["", "## 各区间的自适应极差（相对 fixed 的最小改进 → 最大改进）", ""]
    for T, r in piv.iterrows():
        ood = r.drop("fixed")
        best = ood.min() - r["fixed"]
        L.append(f"- {T:.0f} °C: fixed {r['fixed']:.2f} pp；"
                 f"自适应区间 {ood.min():.2f}–{ood.max():.2f} pp；"
                 f"相对 fixed 最好 {best:+.2f} pp")
    open(os.path.join(RES, "r1m8_aekf_sweep.md"), "w",
         encoding="utf-8").write("\n".join(L) + "\n")
    print("产物: r1m8_aekf_sweep.csv / .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
