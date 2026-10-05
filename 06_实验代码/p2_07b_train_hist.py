# -*- coding: utf-8 -*-
"""P2-07 辅助：生成 train/val 文件的 EKF history 存档（探针 A/B 用）

与 p2_05_06_ekf_baseline.py 完全相同的 EKF 配置，
只跑 train(5) + val(1) 文件，追加存为 ekf_train_history.npz。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "06_实验代码", "src"))

from ecm_ekf import ECMParams, SOC_EKF   # noqa: E402
from ecm_tables import OCVTable, ECMParamTable, make_ekf_table_fns  # noqa: E402

DATA = os.path.join(PROJECT_ROOT, "04_实验", "数据", "processed")
RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
SOC0_BIAS = 0.10


def main() -> int:
    ocv_tab = OCVTable.from_csv()
    ptab = ECMParamTable.from_csv()
    ocv_fn, docv_fn, params_fn = make_ekf_table_fns(ocv_tab, ptab)
    d = pd.read_csv(os.path.join(DATA, "mcmaster_drive_cycles.csv.gz"))
    man = pd.read_csv(os.path.join(
        PROJECT_ROOT, "04_实验", "数据", "splits", "split_manifest.csv"))
    want = man[man.split.isin(["train", "val"])].cycle_id.tolist()

    out: dict[str, np.ndarray] = {}
    for cid in want:
        g = d[d.cycle_id == cid]
        if len(g) < 100:
            continue
        i = g.current_A.to_numpy()
        v = g.voltage_V.to_numpy()
        t = g.temperature_C.to_numpy()
        s = g.soc_true.to_numpy()
        p = ECMParams(Cn=2.9, dt=1.0)
        ekf = SOC_EKF(p, soc0=float(np.clip(s[0] + SOC0_BIAS, 0, 1)),
                      ocv_fn=ocv_fn, docv_fn=docv_fn, params_fn=params_fn)
        for k in range(len(i)):
            ekf.step(float(i[k]), float(v[k]), temp_k=float(t[k]))
        rec = {"timestamp": g.timestamp.to_numpy(), "soc_true": s,
               "current_A": i, "temperature_C": t, "voltage_V": v,
               **{k2: np.array(v2) for k2, v2 in ekf.history.items()}}
        for k2, v2 in rec.items():
            out[f"{cid}::{k2}"] = v2
        print(f"完成 {cid}（{len(g)} 步）")
    np.savez_compressed(
        os.path.join(RESULTS, "ekf_train_history.npz"), **out)
    print(f"\n存档 {len(want)} 文件 → ekf_train_history.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
