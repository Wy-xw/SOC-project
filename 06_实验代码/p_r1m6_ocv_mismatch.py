# -*- coding: utf-8 -*-
"""意见8 §二.2：OCV 失配对低温残差的贡献估计。

事实基础
--------
- EKF 的 OCV 表 `ocv_soc_curve.csv` 只在 **25 °C** 标定（单一曲线，不随温度变）。
- ECM 参数 {$R_0,R_1,C_1$} 则是 (SOC, 温度) 索引 —— 参数随温度更新，OCV 不更新。
- `ecm_params_raw.csv` 里有各温度 HPPC 弛豫外推的平衡电位 `v_eq_V`。

方法（两个层次）
----------------
**(A) 静态失配量**：ΔOCV(soc,T) = v_eq_T(soc) − v_eq_25(soc)（同方法相减，
抵消辨识方法偏差），再用 dOCV/dSOC 换算为等效 SOC 偏差 ΔSOC_ocv = ΔOCV/(dOCV/dSOC)。

**(B) EKF 实跑对照**：把 25 °C 曲线换成 T 相关曲线重跑 EKF，
比较各区间的 RMSE —— 给出 OCV 失配对低温误差的**贡献量级**。

产物：results/r1m6_ocv_static.csv / .md
"""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PROJ = os.path.join(__ROOT__, "SOC论文项目")
RES = os.path.join(PROJ, "04_实验", "数据", "results")
TEMPS = [-20.0, -10.0, 0.0, 10.0, 25.0]


def build_veq_curves(smooth_win: int = 9):
    """各温度 v_eq(soc) 曲线（共同 SOC 网格插值 + Savitzky-Golay 平滑）。

    v_eq 由 HPPC 弛豫外推得到，相邻点有 8–21 mV 量级散点；
    温度信号同量级，故先平滑再作差（只保留平滑后的温度趋势）。
    """
    from scipy.signal import savgol_filter
    d = pd.read_csv(os.path.join(RES, "ecm_params_raw.csv"))
    grid = np.linspace(0.0, 1.0, 1001)
    curves = {}
    for T in TEMPS:
        s = d[d.temperature_c == T].sort_values("soc")
        if len(s) < 5:
            continue
        v = np.interp(grid, s.soc.values, s.v_eq_V.values)
        w = min(smooth_win, len(s) if len(s) % 2 else len(s) - 1)
        w = max(w, 5)
        curves[T] = savgol_filter(v, w, 2)
    return grid, curves


def main() -> int:
    ocv = pd.read_csv(os.path.join(RES, "ocv_soc_curve.csv"))
    grid, curves = build_veq_curves()
    ocv25 = np.interp(grid, ocv.soc.values, ocv.ocv_V.values)
    docv = np.interp(grid, ocv.soc.values, ocv.docv_dsoc.values)

    # 各温度 SOC 有效覆盖
    d = pd.read_csv(os.path.join(RES, "ecm_params_raw.csv"))
    cov = d.groupby("temperature_c").soc.agg(["min", "max"])

    print("=" * 68)
    print("(A) 静态失配 ΔOCV(soc,T) = v_eq_T − v_eq_25  [mV]")
    print("=" * 68)
    rows = []
    for T in TEMPS:
        if T == 25:
            continue
        dv = (curves[T] - curves[25]) * 1000.0
        s_lo, s_hi = cov.loc[T, "min"], cov.loc[T, "max"]
        m = (grid >= s_lo) & (grid <= s_hi)
        # 等效 SOC 偏差 [pp] = ΔOCV[V] / (dOCV/dSOC)[V per unit SOC]
        dpp = (dv / 1000.0) / docv * 100.0
        rows.append({
            "T": T, "soc_lo": s_lo, "soc_hi": s_hi,
            "dOCV_mean_mV": dv[m].mean(), "dOCV_min_mV": dv[m].min(),
            "dOCV_max_mV": dv[m].max(),
            "dSOC_mean_pp": dpp[m].mean(),
            "dSOC_absmax_pp": np.abs(dpp[m]).max(),
        })
        print(f"  T={T:5.0f}°C  SOC∈[{s_lo:.2f},{s_hi:.2f}]  "
              f"ΔOCV {dv[m].mean():+.1f} mV (范围 {dv[m].min():+.1f}~{dv[m].max():+.1f})  "
              f"等效ΔSOC 均 {dpp[m].mean():+.2f} pp / 峰值 {np.abs(dpp[m]).max():.2f} pp")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "r1m6_ocv_static.csv"), index=False,
              encoding="utf-8")

    L = ["# R1-M6 OCV 失配的静态量化（意见8 §二.2）\n",
         "EKF 的 OCV 表只在 25 °C 标定；下表给出各温度**实测平衡电位**相对 25 °C 的漂移，",
         "及其换算的等效 SOC 偏差（等效偏差 = ΔOCV / (dOCV/dSOC)，单位 pp）。\n",
         "| 温度 | SOC 覆盖 | ΔOCV 均值 (mV) | ΔOCV 范围 (mV) | 等效 ΔSOC 均值 (pp) | 等效 ΔSOC 峰值 (pp) |",
         "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r['T']:.0f} °C | {r['soc_lo']:.2f}–{r['soc_hi']:.2f} | "
                 f"{r['dOCV_mean_mV']:+.1f} | {r['dOCV_min_mV']:+.1f} ~ {r['dOCV_max_mV']:+.1f} | "
                 f"{r['dSOC_mean_pp']:+.2f} | {r['dSOC_absmax_pp']:.2f} |")
    open(os.path.join(RES, "r1m6_ocv_static.md"), "w",
         encoding="utf-8").write("\n".join(L) + "\n")
    print("\n产物: r1m6_ocv_static.csv / .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
