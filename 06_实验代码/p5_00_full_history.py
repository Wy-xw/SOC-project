# -*- coding: utf-8 -*-
"""
P5-00：为全部测试文件生成完整 EKF history（供 P5-01 评估用）

原来 p2_05_06 只存了 test_id 4 + 每温度 1 OOD = 8 文件的 history。
P5-01 需要全部 OOD 文件的 history 才能评估 EKF+NN 融合方法。

产出
----
- results/ekf_full_history.npz：全部文件的 EKF history（含 soc_true/current_A 等元数据）
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "06_实验代码", "src"))

from ecm_ekf import ECMParams, SOC_EKF  # noqa: E402
from ecm_tables import OCVTable, ECMParamTable, make_ekf_table_fns  # noqa: E402

DATA = os.path.join(PROJECT_ROOT, "04_实验", "数据", "processed")
RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")

SOC0_BIAS = 0.10


def main() -> int:
    ocv_tab = OCVTable.from_csv()
    ptab = ECMParamTable.from_csv()
    ocv_fn, docv_fn, params_fn = make_ekf_table_fns(ocv_tab, ptab)

    man = pd.read_csv(os.path.join(
        PROJECT_ROOT, "04_实验", "数据", "splits", "split_manifest.csv"))
    d = pd.read_csv(os.path.join(DATA, "mcmaster_drive_cycles.csv.gz"))

    # 🔴 2026-10-03：增加 --with-trise 选项，把 9 个温度斜坡文件也跑上。
    #    用途：为"公平调参基线"提供不泄露测试温度的资源（意见8 §三.4）。
    #    默认仍只跑 test_id + test_ood，**不改变原产物**。
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-trise", action="store_true",
                    help="额外处理 test_ood_trise（输出到 *_withtrise.npz）")
    args, _ = ap.parse_known_args()

    splits = ["test_id", "test_ood"]
    suffix = ""
    if args.with_trise:
        splits.append("test_ood_trise")
        suffix = "_withtrise"
    test_splits = man[man.split.isin(splits)]
    test_cids = set(test_splits.cycle_id.tolist())

    # 检查已有 history
    existing_path = os.path.join(RESULTS, f"ekf_full_history{suffix}.npz")
    if os.path.exists(existing_path):
        z = np.load(existing_path, allow_pickle=False)
        existing_cids = set(k.split("::")[0] for k in z.files)
        missing = test_cids - existing_cids
        if not missing:
            print(f"已有完整 history（{len(existing_cids)} 文件），无需重新生成")
            return 0
        print(f"已有 {len(existing_cids)} 文件，缺 {len(missing)} 文件，补跑...")
        hist_keep = {}
        for cid in existing_cids:
            hist_keep[cid] = {k.split("::")[1]: z[k]
                              for k in z.files if k.startswith(cid + "::")}
    else:
        hist_keep = {}
        missing = test_cids

    count = 0
    for cid, g in d.groupby("cycle_id"):
        if cid not in missing:
            continue
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
        hist_keep[cid] = {
            "timestamp": ts, "soc_true": s, "current_A": i,
            "temperature_C": t, "voltage_V": v,
            **{k2: np.array(v2) for k2, v2 in h.items()},
        }
        count += 1
        print(f"  [{count}/{len(missing)}] {cid} ({split}, {len(g)} 点)")

    # 保存
    flat = {}
    for cid, dd in hist_keep.items():
        for k2, v2 in dd.items():
            flat[f"{cid}::{k2}"] = v2
    np.savez_compressed(existing_path, **flat)
    print(f"\n保存 {len(hist_keep)} 文件 → {existing_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
