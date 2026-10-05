# -*- coding: utf-8 -*-
"""P2-05/06：EKF 在 McMaster 真实数据上的基线指标

设置
----
- 模型：1RC-ECM + 表驱动 OCV（P2-02）+ 时变参数（P2-03 表，SOC×温度插值）
- 初始 SOC 偏差：脚本设 soc0 = soc_true[0] + 0.10 并 clamp 到 [0,1]，**但实际无效** ——
  本数据集全部 55 个文件 soc_true[0] = 1.0000，clamp(1.1,0,1) = 1.0，
  该 +10 pp 偏置在每个文件上都被裁掉、从未生效（实为正确初始化）。
  真正测初始偏差的脚本是 p5_03b_initial_bias_fixed.py
- 噪声参数：r=1e-4（10 mV 量测噪声标准差），q_soc=1e-9, q_v1=1e-6
  （沿用合成自检值；这些是滤波器超参，不代表真实噪声，P3 可调）
- 记录全部 history（nu/K/P/NIS）→ P2-07 残差分析与 P2-09 特征导出的输入

产出
----
- results/ekf_baseline_metrics.csv：每文件 RMSE/MAE/MAX（百分点）
  + 稳态段（前 300 s 收敛期剔除）指标
- results/ekf_baseline_history.npz：test_id 4 文件 + 每温度 1 个 OOD 文件的
  完整 history（P2-07 直接消费）
- stdout 汇总：split × 温度
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "06_实验代码", "src"))

from ecm_ekf import ECMParams, SOC_EKF, metrics   # noqa: E402
from ecm_tables import OCVTable, ECMParamTable, make_ekf_table_fns  # noqa: E402

DATA = os.path.join(PROJECT_ROOT, "04_实验", "数据", "processed")
RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")

BURN_IN_S = 300        # 收敛期（秒），稳态指标剔除
SOC0_BIAS = 0.10       # 初始 SOC 故意偏高 10 个百分点


def main() -> int:
    ocv_tab = OCVTable.from_csv()
    ptab = ECMParamTable.from_csv()
    ocv_fn, docv_fn, params_fn = make_ekf_table_fns(ocv_tab, ptab)

    man = pd.read_csv(os.path.join(
        PROJECT_ROOT, "04_实验", "数据", "splits", "split_manifest.csv"))
    d = pd.read_csv(os.path.join(DATA, "mcmaster_drive_cycles.csv.gz"))

    # history 存档选择：test_id 全部 + 每个名义温度各 1 个 OOD 文件
    hist_keep: dict[str, list] = {}
    ood_by_temp: set[str] = set()
    for tmp, g in man[man.split == "test_ood"].groupby("nominal_temp_c"):
        ood_by_temp.add(g.sort_values("cycle_id").cycle_id.iloc[0])

    rows = []
    for cid, g in d.groupby("cycle_id"):
        if len(g) < 100:
            continue
        i = g.current_A.to_numpy()
        v = g.voltage_V.to_numpy()
        t = g.temperature_C.to_numpy()
        s = g.soc_true.to_numpy()
        ts = g.timestamp.to_numpy()
        meta = man[man.cycle_id == cid]
        split = meta.split.iloc[0] if len(meta) else "?"

        p = ECMParams(Cn=2.9, dt=float(np.median(np.diff(ts[:10])) or 1.0))
        soc0 = float(np.clip(s[0] + SOC0_BIAS, 0.0, 1.0))
        ekf = SOC_EKF(
            p, soc0=soc0, ocv_fn=ocv_fn, docv_fn=docv_fn, params_fn=params_fn)
        for k in range(len(i)):
            ekf.step(float(i[k]), float(v[k]), temp_k=float(t[k]))

        h = ekf.history
        soc_est = np.array(h["soc"])
        full = metrics(soc_est, s)
        m_ss = metrics(soc_est[BURN_IN_S:], s[BURN_IN_S:]) \
            if len(soc_est) > BURN_IN_S * 2 else full
        rows.append({
            "cycle_id": cid, "split": split,
            "nominal_temp_c": meta.nominal_temp_c.iloc[0] if len(meta) else np.nan,
            "n": len(g),
            "full_RMSE": full["RMSE"], "full_MAE": full["MAE"],
            "full_MAX": full["MAX"],
            "ss_RMSE": m_ss["RMSE"], "ss_MAE": m_ss["MAE"],
            "ss_MAX": m_ss["MAX"],
        })

        if split == "test_id" or cid in ood_by_temp:
            hist_keep[cid] = {
                "timestamp": ts, "soc_true": s, "current_A": i,
                "temperature_C": t, "voltage_V": v,
                **{k2: np.array(v2) for k2, v2 in h.items()},
            }

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "ekf_baseline_metrics.csv"), index=False)
    np.savez_compressed(
        os.path.join(RESULTS, "ekf_baseline_history.npz"),
        **{f"{cid}::{k2}": v2 for cid, dd in hist_keep.items()
           for k2, v2 in dd.items()})

    print("== EKF 基线（SOC 误差，百分点；正确初始化）==")
    agg = (df.groupby(["split", "nominal_temp_c"])
           .agg(RMSE=("ss_RMSE", "mean"), MAE=("ss_MAE", "mean"),
                MAX=("ss_MAX", "max"), n_files=("cycle_id", "count"))
           .round(2))
    print(agg.to_string())
    print(f"\nhistory 存档文件数: {len(hist_keep)}（test_id 全部 + 每温度 1 OOD）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
