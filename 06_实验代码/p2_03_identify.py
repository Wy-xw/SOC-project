# -*- coding: utf-8 -*-
"""P2-03：ECM 参数辨识（HPPC 脉冲分析，按 SOC × 温度）

方法
----
- R0：脉冲起始电压骤降 ÷ 脉冲电流（放电为正 → 电压降）
- R1/C1：弛豫段指数恢复拟合  v(t) = v_eq ∓ A·exp(-t/τ)（放电取 −，充电取 +）
  R1 = A/|I|，C1 = τ/R1
- SOC 锚点：脉冲前静置段尾部均值

过滤
----
- dur < 5 s 的脉冲（电压触底被截止）丢弃
- 拟合失败 / A≤0 / τ 超界丢弃
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(PROJECT_ROOT, "04_实验", "数据", "processed")
RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")

I_TH = 0.05
MIN_PULSE_S = 5.0
TAIL_S = 60.0          # V_eq 取弛豫段尾 60 s 中位数
FIT_MAX_S = 1200.0     # 指数拟合窗口（弛豫段前 N 秒）


def _segments(i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    st = np.where(i > I_TH, 1, np.where(i < -I_TH, -1, 0))
    b = np.concatenate(([0], np.where(np.diff(st) != 0)[0] + 1, [len(i)]))
    return st, b


def _fit_exp(x: np.ndarray, v: np.ndarray, sign: int):
    """弛豫指数拟合。sign=+1 放电后回升，sign=-1 充电后回落。

    v_eq 初值取尾段 0.9 分位（略高于已恢复值）；
    只对未平衡点（y > 1e-6）线性化并拟合，平衡点剔除。
    """
    if len(x) < 30:
        return None
    tail = v[-int(TAIL_S):] if len(v) > 100 else v[-10:]
    v_eq0 = float(np.quantile(tail, 0.9))
    y = sign * (v_eq0 - v)
    m = y > 1e-6
    if m.sum() < 30:
        return None
    ly = np.log(y[m])
    slope, _ = np.polyfit(x[m], ly, 1)
    if slope >= 0:
        return None
    tau0 = float(max(-1.0 / slope, 1.0))
    A0 = float(np.exp(np.interp(0.0, x[m], ly)))

    def f(xx, ve, A, tau):
        return ve - sign * A * np.exp(-xx / tau)

    try:
        p, _ = curve_fit(f, x[m], v[m], p0=[v_eq0, A0, tau0], maxfev=10000)
    except Exception:
        return None
    ve, A, tau = p
    if A <= 0 or tau < 1 or tau > 5000:
        return None
    return ve, A, tau


def main() -> int:
    d = pd.read_csv(os.path.join(DATA, "mcmaster_hppc.csv.gz"))
    rows = []
    for (tmp, cid), g in d.groupby(["temperature_c", "cycle_id"]):
        t = g["timestamp"].to_numpy()
        i = g["current_A"].to_numpy()
        v = g["voltage_V"].to_numpy()
        s = g["soc_true"].to_numpy()
        st, b = _segments(i)
        segs = list(zip(b[:-1], b[1:]))
        for k, (a, bb) in enumerate(segs):
            kind = int(st[a])
            dur = t[bb - 1] - t[a]
            if kind == 0 or dur < MIN_PULSE_S:
                continue
            if k == 0 or int(st[segs[k - 1][0]]) != 0:
                continue                    # 脉冲前必须是静置段
            pa, pb = segs[k - 1]
            win = max(pa, pb - 30)
            v_rest = float(np.mean(v[win:pb]))
            soc_rest = float(np.mean(s[win:pb]))
            i_pulse = float(np.mean(np.abs(i[a:bb])))
            if i_pulse < 0.1:
                continue
            v_pulse0 = float(v[a])
            r0 = ((v_rest - v_pulse0) / i_pulse if kind == 1
                  else (v_pulse0 - v_rest) / i_pulse)
            if r0 < 0:
                continue
            if k + 1 >= len(segs) or int(st[segs[k + 1][0]]) != 0:
                continue                    # 脉冲后必须是弛豫静置
            ra, rb = segs[k + 1]
            x = t[ra:rb + 1] - t[ra]
            vv = v[ra:rb + 1]
            m = x <= FIT_MAX_S
            p = _fit_exp(x[m], vv[m], kind)
            if p is None:
                continue
            ve, A, tau = p
            r1 = A / i_pulse
            c1 = tau / r1 if r1 > 0 else np.nan
            if not np.isfinite(c1) or c1 <= 0:
                continue
            rows.append({
                "temperature_c": tmp, "cycle_id": cid, "kind": kind,
                "soc": soc_rest, "i_pulse_A": i_pulse, "v_rest_V": v_rest,
                "r0_ohm": r0, "v_eq_V": ve, "v_amp_mV": A * 1000.0,
                "tau_s": tau, "r1_ohm": r1, "c1_F": c1,
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "ecm_params_raw.csv"), index=False)
    print(f"有效脉冲数: {len(df)}")
    bins = np.arange(0.0, 1.05, 0.1)
    gb = df.groupby([df["temperature_c"], pd.cut(df["soc"], bins)])
    print("\n== 每温度×SOC 档 参数均值 ==")
    print(gb[["r0_ohm", "r1_ohm", "tau_s", "c1_F"]]
          .mean().round(4).to_string())
    print("\n== 脉冲数 ==")
    print(gb.size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
