# -*- coding: utf-8 -*-
"""意见8 §二.2 (B)：温度相关 OCV 的 EKF 实跑对照。

对照
----
- base : 原装 `OCVTable.from_csv()`（单一 25 °C 曲线）—— 论文现状，**忠实复现**。
- ocvT : OCV = 25 °C 曲线 + 温度漂移 ΔOCV(soc,T)。

ΔOCV(soc,T) 由 `ecm_params_raw.csv` 各温度 HPPC 弛豫 v_eq 相对 25 °C 的
**同方法差值** 得到（抵消辨识方法偏差），Savitzky-Golay 平滑，
在该温度 SOC 覆盖外取端点值持平延拓（再叠加 OCVTable 的线性延拓）。

关键：base 与 ocvT 用**同一 soc 网格、同一 docv 计算路径**，差异只来自 ΔOCV。
"""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(__ROOT__, "06_实验代码", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

from ecm_ekf import ECMParams, SOC_EKF  # noqa: E402
from ecm_tables import OCVTable, ECMParamTable, make_ekf_table_fns  # noqa: E402
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PROJ = os.path.join(__ROOT__, "SOC论文项目")
DATA = os.path.join(PROJ, "04_实验", "数据", "processed")
RES = os.path.join(PROJ, "04_实验", "数据", "results")
BURN = 300
TEMPS = [-20.0, -10.0, 0.0, 10.0]


def temp_of(cid: str) -> float | None:
    """按文件名判温度（注意 '10degC' ⊂ '0degC' 顺序陷阱）。"""
    for tag, T in [("n20degC", -20.0), ("n10degC", -10.0),
                   ("10degC", 10.0), ("0degC", 0.0)]:
        if tag in cid:
            return T
    return None


def smooth(x: np.ndarray, win: int = 9) -> np.ndarray:
    w = max(5, min(win, len(x) if len(x) % 2 else len(x) - 1))
    return savgol_filter(x, w, 2)


def main() -> int:
    tab0 = OCVTable.from_csv()
    soc_grid = tab0._soc
    ocv25 = tab0._ocv
    docv25 = tab0._docv

    # 各温度 v_eq 相对 25 °C 的漂移（同方法差值 + 二次多项式拟合）。
    # 用多项式而非插值：v_eq 在覆盖边缘有 ~200 mV 伪尖峰，其梯度会使
    # dOCV/dSOC 变负 → EKF 修正符号翻转 → 发散（实测 0 °C 炸到 75 pp）。
    raw = pd.read_csv(os.path.join(RES, "ecm_params_raw.csv"))
    cov = raw.groupby("temperature_c").soc.agg(["min", "max"])
    ref = raw[raw.temperature_c == 25].sort_values("soc")
    v25 = np.interp(soc_grid, ref.soc.values, ref.v_eq_V.values)

    def shift_on_grid(T: float) -> np.ndarray:
        s = raw[raw.temperature_c == T].sort_values("soc")
        vT = np.interp(soc_grid, s.soc.values, s.v_eq_V.values)
        lo = max(cov.loc[T, "min"], 0.10)
        hi = cov.loc[T, "max"]
        m = (soc_grid >= lo) & (soc_grid <= hi)
        c = np.polyfit(soc_grid[m], (vT - v25)[m], 2)
        sh = np.polyval(c, soc_grid)
        sh = np.clip(sh, -0.06, 0.06)
        return sh

    def make_table(T: float | None) -> OCVTable:
        if T is None:
            return tab0
        sh = shift_on_grid(T)
        ocv_T = ocv25 + sh
        docv_T = docv25 + np.gradient(sh, soc_grid)   # d(ocv+sh)/dsoc
        return OCVTable(soc_grid, ocv_T, docv_T)

    _, _, params_fn = make_ekf_table_fns(tab0, ECMParamTable.from_csv())
    man = pd.read_csv(os.path.join(PROJ, "04_实验", "数据", "splits",
                                   "split_manifest.csv"))
    d = pd.read_csv(os.path.join(DATA, "mcmaster_drive_cycles.csv.gz"))
    eval_ids = sorted(man[man.split == "test_ood"].cycle_id)

    shard_dir = os.path.join(RES, "_shards_r1m6")
    os.makedirs(shard_dir, exist_ok=True)

    def shard_of(cid, tag):
        safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in cid)
        return os.path.join(shard_dir, f"r1m6_{tag}_{safe}.csv")

    for n, cid in enumerate(eval_ids, 1):
        gp = d[d.cycle_id == cid]
        if len(gp) < BURN + 100:
            continue
        i = gp.current_A.to_numpy(); v = gp.voltage_V.to_numpy()
        t = gp.temperature_C.to_numpy(); s = gp.soc_true.to_numpy()
        T = temp_of(cid)
        for tag, TT in [("base", None), ("ocvT", T)]:
            sp = shard_of(cid, tag)
            if os.path.exists(sp):
                continue
            tab = make_table(TT)
            e = SOC_EKF(ECMParams(Cn=2.9, dt=1.0), soc0=float(np.clip(s[0], 0, 1)),
                        ocv_fn=tab.ocv, docv_fn=tab.docv_dsoc,
                        params_fn=params_fn)
            for k in range(len(i)):
                e.step(float(i[k]), float(v[k]), temp_k=float(t[k]))
            err = (np.array(e.history["soc"])[BURN:] - s[BURN:]) * 100
            pd.DataFrame([{"cycle_id": cid, "T": T, "variant": tag,
                           "rmse": float(np.sqrt(np.mean(err ** 2)))}]
                         ).to_csv(sp, index=False)
        print(f"  [{n}/{len(eval_ids)}] {cid[:44]} T={T}", flush=True)

    parts = []
    for cid in eval_ids:
        for tag in ("base", "ocvT"):
            fp = shard_of(cid, tag)
            if os.path.exists(fp):
                parts.append(pd.read_csv(fp))
    t = pd.concat(parts, ignore_index=True)
    t.to_csv(os.path.join(RES, "r1m6b_ocv_ekf.csv"), index=False)

    piv = t.pivot_table(index="T", columns="variant", values="rmse", aggfunc="mean")
    piv["减少"] = piv["base"] - piv["ocvT"]
    print("\n" + "=" * 60)
    print(piv.round(3).to_string())
    L = ["# R1-M6b 温度相关 OCV 的 EKF 实跑对照\n",
         "| 温度 | 单一 25 °C OCV | T 相关 OCV | RMSE 变化 | 贡献占比 |",
         "|---|---|---|---|---|"]
    for T in piv.index:
        b, o = piv.loc[T, "base"], piv.loc[T, "ocvT"]
        L.append(f"| {T:.0f} °C | {b:.3f} | {o:.3f} | {o-b:+.3f} | "
                 f"{(b-o)/b*100:.1f}% |")
    L += ["", "**贡献占比** = (base−ocvT)/base，即 OCV 失配可解释的 EKF 误差份额；",
          "为负表示换成温度相关 OCV 后误差**不降反升**（见正文解读）。"]
    open(os.path.join(RES, "r1m6b_ocv_ekf.md"), "w", encoding="utf-8").write(
        "\n".join(L) + "\n")
    print("产物: r1m6b_ocv_ekf.csv / .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
