# -*- coding: utf-8 -*-
"""P2-02 诊断：按支路解剖 C/20 OCV 数据（滞后带归因的基础事实）。

论证目标（skills 第 0 步）：
    论文要论证 OCV(SOC) 放电支路可用于 EKF；需要先判断
    「充电/放电支路之间是否存在不可忽略的滞后带」。
本脚本只负责拿事实，不画图：判断滞后带是真实物理量，
还是 soc_true 坐标被 clip 破坏造成的对齐错位。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATH = PROJECT_ROOT + r"\04_实验\数据\processed\mcmaster_ocv_c20.csv.gz"

REST_A = 0.005          # |I| 以下视为静置
CLIP_LOGICAL = True     # 用于对比：soc_true 是否被 clip 改变（构造反事实）

def main() -> int:
    d = pd.read_csv(PATH)
    i = np.asarray(d["current_A"])
    s_raw = np.asarray(d["soc_true"])           # 已 clip 到 [0,1] 的版本
    v = np.asarray(d["voltage_V"])

    # 反推未 clip 的真值：soc = 1 + Ah/2.9，直接用 ah_throughput
    ah = np.asarray(d["ah_throughput"])
    s_unclipped = 1.0 + ah / 2.9

    dis = i > REST_A
    chg = i < -REST_A
    sd, vd = s_unclipped[dis], v[dis]
    sc, vc = s_unclipped[chg], v[chg]

    print("== 未 clip 坐标 ==")
    print(f"放电支路 soc 未clip: n={len(sd)} min={sd.min():.4f} max={sd.max():.4f}")
    print(f"充电支路 soc 未clip: n={len(sc)} min={sc.min():.4f} max={sc.max():.4f}")
    c1, c99 = np.quantile(sc, 0.01), np.quantile(sc, 0.99)
    print(f"充电支路 q01={c1:.4f} q99={c99:.4f}  "
          f"[0,1]内占比={( (sc>=0)&(sc<=1) ).mean()*100:.1f}%")
    print(f"放电支路最低: soc={sd.min():.3f} 电压={vd[sd.argmin()]:.3f} V")
    print(f"充电支路最高: soc={sc.max():.3f} 电压={vc[sc.argmax()]:.3f} V")

    # 公共区间滞后带（对比同 SOC 的两支路电压）
    g = np.linspace(0.0, 1.0, 201)
    lo = max(float(sd.min()), float(sc.min()))
    hi = min(float(sd.max()), float(sc.max()))
    common = (g >= lo) & (g <= hi)
    vgi = np.interp(g, np.sort(sd), vd[np.argsort(sd)])
    vgic = np.interp(g, np.sort(sc), vc[np.argsort(sc)])
    lag = (vgi - vgic) * 1000.0                    # mV，放电 - 充电
    lc = lag[common]
    print(f"\n== 公共 SOC 区间 [{lo:.3f}, {hi:.3f}] 的滞后带 ==")
    print(f"放电-充电：median={np.median(lc):.1f} mV, "
          f"q25={np.quantile(lc, .25):.1f}, q75={np.quantile(lc, .75):.1f}, "
          f"max|.|.{np.abs(lc).max():.1f}")

    # 用 clip 后的版本重算一遍（复现之前的错位，验证截断影响）
    sd_c, vd_c = np.clip(s_raw[dis], 0, 1), v[dis]
    sc_c, vc_c = np.clip(s_raw[chg], 0, 1), v[chg]
    vgi_c = np.interp(g, np.sort(sd_c), vd_c[np.argsort(sd_c)])
    vgic_c = np.interp(g, np.sort(sc_c), vc_c[np.argsort(sc_c)])
    lag_c = (vgi_c - vgic_c) * 1000.0
    lcc = lag_c[common]
    print("\n== 同区间但用已clip坐标（复现错位源）==")
    print(f"放电-充电：median={np.median(lcc):.1f} mV (若与此前 -1075 接近则错位来自 clip)")

    # 分支内有效范围占比（判断充电支路本身完整覆盖多大 SOC 区间）
    print(f"\n放电支路有效 soc 占比: {np.ptp(sd):.3f}")
    print(f"充电支路有效 soc 占比: {np.ptp(sc):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
