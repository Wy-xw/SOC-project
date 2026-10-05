# -*- coding: utf-8 -*-
"""P7-01 补证实验：生成 0 °C 训练集/同分布测试的 EKF 历史。

为什么需要（审稿人 R1-M1）
--------------------------
主设定是"25 °C 训练 → 低温外推"。R1 要求在一个**第二训练温度**上
原样重跑同一套消融，检验"两族不可互换"是否随训练温度改变。

本脚本用 `split_manifest_0C.csv` 的角色划分，对 **0 °C 的 9 个文件**
（训练 4 + 验证 1 + 同分布测试 4）跑与 `p2_07b_train_hist.py`
**完全相同的 EKF 配置**，生成：
    ekf_train_history_0C.npz   ← 训练 5（含验证）文件
    ekf_full_history_0C.npz    ← 全部 9 个 0 °C 文件

⚠️ 与主设定的唯一差别是**文件集合**，EKF 参数、初值偏置、步进方式全部一致。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "06_实验代码", "src"))

from ecm_ekf import ECMParams, SOC_EKF          # noqa: E402
from ecm_tables import OCVTable, ECMParamTable, make_ekf_table_fns  # noqa: E402

DATA = os.path.join(PROJECT_ROOT, "04_实验", "数据", "processed")
RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
SPLITS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "splits")
SOC0_BIAS = 0.10          # 与 p2_07b 一致


def run_files(cycle_ids, d, ocv_fn, docv_fn, params_fn):
    out: dict[str, np.ndarray] = {}
    for cid in cycle_ids:
        g = d[d.cycle_id == cid]
        if len(g) < 100:
            print(f"  跳过 {cid}（样本不足）")
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
        print(f"  完成 {cid}（{len(g)} 步）")
    return out


def main() -> int:
    ocv_tab = OCVTable.from_csv()
    ptab = ECMParamTable.from_csv()
    ocv_fn, docv_fn, params_fn = make_ekf_table_fns(ocv_tab, ptab)
    d = pd.read_csv(os.path.join(DATA, "mcmaster_drive_cycles.csv.gz"))
    man = pd.read_csv(os.path.join(SPLITS, "split_manifest_0C.csv"))

    # 训练历史：train + val
    tv = man[man.split.isin(["train", "val"])].cycle_id.tolist()
    print(f"[1/2] 训练历史：{len(tv)} 个文件")
    h_tr = run_files(tv, d, ocv_fn, docv_fn, params_fn)
    np.savez_compressed(os.path.join(RESULTS, "ekf_train_history_0C.npz"), **h_tr)
    print(f"  → ekf_train_history_0C.npz（{len(h_tr)//9} 个文件）")

    # 全历史：只生成 **0 °C 的 9 个文件**（训练 5 + 同分布测试 4）。
    #   ⚠️ **不要**在这里重跑 25 °C：它作为 OOD 测试集，
    #      其 EKF 历史**已在既有 `ekf_full_history.npz` 里**（25 °C × 4 文件），
    #      重跑会与主实验产生不一致的基线。p7_02 直接从既有 npz 取。
    ids_0c = man[man.split.isin(["train", "val", "test_id"])].cycle_id.tolist()
    print(f"\n[2/2] 0 °C 全历史：{len(ids_0c)} 个文件")
    h_all = run_files(ids_0c, d, ocv_fn, docv_fn, params_fn)
    np.savez_compressed(os.path.join(RESULTS, "ekf_full_history_0C.npz"), **h_all)
    print(f"  → ekf_full_history_0C.npz（仅 0 °C）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
