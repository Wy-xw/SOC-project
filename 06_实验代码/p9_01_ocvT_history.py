# -*- coding: utf-8 -*-
"""P9-01 生成**温度相关 OCV** 内核下的 EKF 历史（回应审稿人 R2-M2）。

为什么做这个
------------
R2-M2 质疑：主导 OOD 失配是本文**刻意保留**的 OCV 温度失配（§3.2），
而**新息 ν 正是对该静态失配最敏感的通道**。故 A2-1 − A2-4 的族间差，
可能只是"哪个通道更能反映这处**人为未建模**的误差"，而非两族的**功能**差异。

`p_r1m6b_ocv_ekf.py`（意见8 §二.2）只把温度相关 OCV 用在 **EKF 上**，
**未生成可训练的历史**，所以消融无法在它上面重跑。本脚本补上这一步。

OCV 代理（**逐字复用 `p_r1m6b_ocv_ekf.py` 的构造**，不另造）
-----------------------------------------------------------
    OCV(soc, T) = OCV_25(soc) + ΔOCV(soc, T)
    ΔOCV 由 `ecm_params_raw.csv` 各温度 HPPC 弛豫 v_eq 相对 25 °C 的
    **同方法差值**给出（抵消辨识方法偏差），对 SOC 覆盖段做**二次多项式**拟合
    （不用插值：v_eq 边缘有 ~200 mV 伪尖峰，其梯度会使 dOCV/dSOC 变负 →
    EKF 修正符号翻转 → 发散，实测 0 °C 炸到 75 pp），裁剪到 ±60 mV。

⚠️ **除 OCV 表外一切不变**：同 SOC0 bias、同 ECMParamTable、同划分、同 burn-in。
   这样 base 与 ocvT 的差异**只能来自 OCV 表**。

产出
----
- results/ekf_full_history_ocvT.npz   测试（40 文件）
- results/ekf_train_history_ocvT.npz  训练 5 + 验证 1

用法
----
    python p9_01_ocvT_history.py
"""
from __future__ import annotations

import os
import sys
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(__ROOT__, "06_实验代码", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ecm_ekf import ECMParams, SOC_EKF  # noqa: E402
from ecm_tables import OCVTable, ECMParamTable, make_ekf_table_fns  # noqa: E402

PROJ = __ROOT__
DATA = os.path.join(PROJ, "04_实验", "数据", "processed")
RES = os.path.join(PROJ, "04_实验", "数据", "results")
SOC0_BIAS = 0.10          # 与 p5_00_full_history.py 一致


def temp_of(cid: str) -> float | None:
    """按文件名判温度 —— 注意 '10degC' ⊂ '0degC' 的顺序陷阱。"""
    for tag, T in [("n20degC", -20.0), ("n10degC", -10.0),
                   ("10degC", 10.0), ("0degC", 0.0)]:
        if tag in cid:
            return T
    return None


def build_temp_tables(tab0: OCVTable) -> dict[float, OCVTable]:
    """构造各 OOD 温度的温度相关 OCV 表（复用 r1m6b 的代理）。"""
    soc_grid = tab0._soc
    ocv25 = tab0._ocv
    docv25 = tab0._docv

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
        return np.clip(sh, -0.06, 0.06)

    out = {}
    for T in (-20.0, -10.0, 0.0, 10.0):
        sh = shift_on_grid(T)
        out[T] = OCVTable(soc_grid, ocv25 + sh,
                          docv25 + np.gradient(sh, soc_grid))
    return out


def run_ekf(df, cid, tab, params_fn) -> dict:
    g = df[df.cycle_id == cid]
    i = g.current_A.to_numpy()
    v = g.voltage_V.to_numpy()
    t = g.temperature_C.to_numpy()
    s = g.soc_true.to_numpy()
    ts = g.timestamp.to_numpy()
    p = ECMParams(Cn=2.9, dt=float(np.median(np.diff(ts[:10])) or 1.0))
    ekf = SOC_EKF(p, soc0=float(np.clip(s[0] + SOC0_BIAS, 0.0, 1.0)),
                  ocv_fn=tab.ocv, docv_fn=tab.docv_dsoc, params_fn=params_fn)
    for k in range(len(i)):
        ekf.step(float(i[k]), float(v[k]), temp_k=float(t[k]))
    h = ekf.history
    return {"timestamp": ts, "soc_true": s, "current_A": i,
            "temperature_C": t, "voltage_V": v,
            **{k: np.asarray(vv) for k, vv in h.items()}}


def main() -> int:
    tab0 = OCVTable.from_csv()
    _, _, params_fn = make_ekf_table_fns(tab0, ECMParamTable.from_csv())
    tabs = build_temp_tables(tab0)

    man = pd.read_csv(os.path.join(PROJ, "04_实验", "数据", "splits",
                                   "split_manifest.csv"))
    d = pd.read_csv(os.path.join(DATA, "mcmaster_drive_cycles.csv.gz"))

    ood = sorted(man[man.split == "test_ood"].cycle_id)
    tid = sorted(man[man.split == "test_id"].cycle_id)
    tr = sorted(man[man.split == "train"].cycle_id)
    va = sorted(man[man.split == "val"].cycle_id)

    # 测试集（40 文件）：25 °C 用原表（无漂移），低温用温度相关表
    test_hist = {}
    for cid in tid + ood:
        T = temp_of(cid)
        tab = tabs.get(T, tab0)      # 25 °C → tab0（ΔOCV≡0）
        test_hist[cid] = run_ekf(d, cid, tab, params_fn)
        print(f"  [test] {cid[:46]}  T={T}", flush=True)

    # 训练 + 验证（25 °C → 原表；温度相关 OCV 对 25 °C 无影响）
    train_hist = {}
    for cid in tr + va:
        train_hist[cid] = run_ekf(d, cid, tab0, params_fn)
        print(f"  [train] {cid[:46]}", flush=True)

    def save(hist, name):
        flat = {}
        for cid, h in hist.items():
            for k, v in h.items():
                flat[f"{cid}::{k}"] = v
        np.savez_compressed(os.path.join(RES, name), **flat)
        print(f"  -> {name}（{len(hist)} 文件）", flush=True)

    save(test_hist, "ekf_full_history_ocvT.npz")
    save(train_hist, "ekf_train_history_ocvT.npz")
    print("\n完成。注意：25 °C 文件的 OCV 表与 base 相同（ΔOCV≡0），"
          "故 25 °C 结果应与 base 完全一致，可作自检。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
