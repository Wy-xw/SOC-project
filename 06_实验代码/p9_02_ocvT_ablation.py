# -*- coding: utf-8 -*-
"""P9-02 在**温度相关 OCV 内核**上重跑消融评估（回应审稿人 R2-M2）。

关键设计事实（已在 `p9_01` 出口处验证）
-------------------------------------
训练集**只有 25 °C**，而温度相关 OCV 只改变**非 25 °C** 的表
（ΔOCV(25 °C) ≡ 0，实测 25 °C 的 EKF 轨迹与 base 逐位一致、训练特征矩阵完全相同）。
⇒ **训练数据一字未变**，故**模型可原样复用**，本脚本**不重训**，
  只在 `ekf_full_history_ocvT.npz` 上重新评估。

这使对照更纯粹：**模型完全相同，唯一变化是评分所用的 OCV 内核**。
若族间差在 OCV 修正后**保持** → R2-M2 的替代解释被排除；
若**衰减/消失** → 主张须改为"在特定 OCV 失配下两族表现不同"。

⚠️ 注意本实验的自洽性：训练目标 y = SOC^EKF − SOC* 是在 **25 °C（base）** 上算的，
   而评测用的 SOC^EKF 来自 **ocvT 内核**。这是**有意为之**——它正是"训练目标与
   评分口径不一致"这一本文 RQ4 主题的又一次实例，故在解读时不可忽略。

产出
----
- results/p9_02_ocvT_ablation.csv   逐 臂 × 种子 × 评测组
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "06_实验代码", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from nn_features import GROUPS, build_split_arrays, load_history  # noqa: E402

RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "04_实验", "models")
OUT_CSV = os.path.join(RESULTS, "p9_02_ocvT_ablation.csv")

SEEDS_ALL = [7, 13, 42, 101, 202, 303, 404, 505, 606, 707]
EXPECT_TEST = {"test_id(25C)": 4, "ood(10C)": 9, "ood(0C)": 9,
               "ood(-10C)": 9, "ood(-20C)": 9}


def test_groups(hist):
    """逐字照抄主文权威实现（含温度子串陷阱防护）。"""
    ids = sorted(hist.keys())
    return {
        "test_id(25C)": [c for c in ids if "25degC_Cycle" in c],
        "ood(10C)":   [c for c in ids if "10degC_" in c and "trise" not in c
                       and "n10degC" not in c and "n20degC" not in c],
        "ood(0C)":    [c for c in ids if "0degC_" in c and "trise" not in c
                       and "10degC" not in c and "20degC" not in c],
        "ood(-10C)":  [c for c in ids if "n10degC_" in c and "trise" not in c],
        "ood(-20C)":  [c for c in ids if "n20degC_" in c and "trise" not in c],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()
    seeds = ([int(s) for s in args.seeds.split(",")]
             if args.seeds else SEEDS_ALL)

    hist = load_history(os.path.join(RESULTS, "ekf_full_history_ocvT.npz"))
    groups = test_groups(hist)

    print("测试库存自检（应 4/9/9/9/9）:", flush=True)
    bad = []
    for k, want in EXPECT_TEST.items():
        got = len(groups.get(k, []))
        if got != want:
            bad.append(f"{k}: {got}≠{want}")
        print(f"  [{'OK' if got==want else '!!'}] {k}: {got}", flush=True)
    if bad:
        print("[!!] 库存不符，终止：", bad, flush=True)
        return 1

    from tensorflow import keras
    records: list[dict] = []
    total = len(GROUPS) * len(seeds)
    n = 0
    t0 = time.time()

    for g in sorted(GROUPS.keys()):
        feats = GROUPS[g]
        for seed in seeds:
            n += 1
            mp = os.path.join(MODELS_DIR, f"gru16_{g}_seed{seed}.keras")
            if not os.path.exists(mp):
                print(f"[{n}/{total}] 缺模型 {mp}，跳过", flush=True)
                continue
            model = keras.models.load_model(mp)
            for gname, cids in groups.items():
                for cid in cids:
                    if cid not in hist:
                        continue
                    X, y, _ = build_split_arrays(hist, [cid], feats)
                    if len(X) == 0:
                        continue
                    pred = model.predict(X, verbose=0).ravel()
                    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
                    records.append(dict(exp=g, seed=seed, eval_group=gname,
                                        cycle_id=cid, RMSE=rmse))
            print(f"[{n}/{total}] {g} seed{seed}  {time.time()-t0:.0f}s",
                  flush=True)

    pd.DataFrame(records).to_csv(OUT_CSV, index=False)
    print(f"\n已写出 {OUT_CSV}（{len(records)} 行）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
