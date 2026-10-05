# -*- coding: utf-8 -*-
"""P7-02 补证实验：0 °C 训练设定下的分层消融（对应审稿人 R1-M1）。

与 `p3_06_ablation.py` **逻辑完全同构**，只换三处：
  1. 数据源：`ekf_train_history_0C.npz` / `ekf_full_history_0C.npz`
  2. 测试分组：同分布 = 0 °C 的 4 个 Cycle；OOD = 25 °C 的 10 个文件
  3. 输出前缀：`p7_02_0C_*`

目的：检验主设定的核心结论——"两族直接对比 A2-1 − A2-4 稳健"——
是否随**训练温度**改变。若方向与量级保持，则结论不依赖 25 °C 这一特定训练温度。

用法：
    python p7_02_0C_ablation.py                # 全量（5 臂 × 10 种子）
    python p7_02_0C_ablation.py --seeds 7,13,42  # 先跑 3 种子探路
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "06_实验代码", "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "06_实验代码"))

from nn_features import GROUPS, build_split_arrays, load_history  # noqa: E402

RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "04_实验", "模型")
OUT_CSV = os.path.join(RESULTS, "p7_02_0C_ablation.csv")

EPOCHS, BATCH, PATIENCE = 200, 256, 20
SEEDS_ALL = [7, 13, 42, 101, 202, 303, 404, 505, 606, 707]


def load_all():
    """训练历史用 0 °C 新建的；**OOD 的 25 °C 历史取自既有全历史**。

    ⚠️ 25 °C 不能重跑：它是主设定的训练集，其 EKF 历史已在
       `ekf_full_history.npz` 中；重跑会与主实验基线不一致（实测踩坑）。
    """
    train_hist = load_history(os.path.join(RESULTS, "ekf_train_history_0C.npz"))
    test_hist = load_history(os.path.join(RESULTS, "ekf_full_history_0C.npz"))

    # 25 °C 的 **10 个**文件 —— 分两处取齐（实测踩坑）：
    #   `ekf_full_history.npz` 只有 25 °C 的同分布测试 4 个；
    #   原训练 5 + 验证 1 在 `ekf_train_history.npz` 里。
    n25 = 0
    for src in ("ekf_full_history.npz", "ekf_train_history.npz"):
        h = load_history(os.path.join(RESULTS, src))
        for c in h:
            if "25degC" in c and c not in test_hist:
                test_hist[c] = h[c]
                n25 += 1
    print(f"从既有 npz 补入 25 °C：{n25} 个文件（应为 10）")

    val_ids = [c for c in train_hist if "UDDS" in c]
    tr_ids = [c for c in train_hist if c not in val_ids]
    return train_hist, test_hist, tr_ids, val_ids


def test_groups(test_hist):
    """0 °C 设定下的评测分组。

    同分布测试 = 0 °C 的 4 个 Cycle（与主设定的 25 °C Cycle 结构对齐）
    OOD 测试   = 25 °C 的 10 个文件（原主设定的训练 5 + 验证 1 + 测试 4）
    """
    ids = sorted(test_hist.keys())
    return {
        "test_id(0C)": [c for c in ids if "0degC_Cycle" in c],
        "ood(25C)": [c for c in ids if "25degC" in c],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None,
                    help="逗号分隔的种子；默认全部 10 个")
    args = ap.parse_args()
    seeds = ([int(s) for s in args.seeds.split(",")]
             if args.seeds else SEEDS_ALL)

    train_hist, test_hist, tr_ids, val_ids = load_all()
    print(f"训练文件 {len(tr_ids)} 个，验证 {len(val_ids)} 个")
    print(f"  训练: {[c[:30] for c in tr_ids]}")

    groups = test_groups(test_hist)
    for k, v in groups.items():
        print(f"  {k}: {len(v)} 个文件")
    groups_eval = {"val(0C)": (train_hist, val_ids)}
    for k, v in groups.items():
        groups_eval[k] = (test_hist, v)

    # 特征缓存
    feats_cache = {}
    for g in GROUPS:
        feats = GROUPS[g]
        Xtr, ytr, _ = build_split_arrays(train_hist, tr_ids, feats)
        Xva, yva, _ = build_split_arrays(train_hist, val_ids, feats)
        feats_cache[g] = (Xtr, ytr, Xva, yva)
        print(f"  {g}: Xtr={Xtr.shape}")

    from tensorflow import keras
    records: list[dict] = []
    os.makedirs(MODELS_DIR, exist_ok=True)
    total = len(GROUPS) * len(seeds)
    n = 0

    for g in sorted(GROUPS.keys()):
        Xtr, ytr, Xva, yva = feats_cache[g]
        for seed in seeds:
            n += 1
            mp = os.path.join(MODELS_DIR, f"gru16_0C_{g}_seed{seed}.keras")
            if os.path.exists(mp):
                model = keras.models.load_model(mp)
                print(f"[{n}/{total}] {g} seed{seed} 载入缓存")
            else:
                from p3_06_ablation import build_model
                model = build_model(Xtr.shape[2], seed)
                cbs = [
                    keras.callbacks.EarlyStopping(
                        patience=PATIENCE, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(
                        patience=8, factor=0.5, min_lr=1e-5),
                ]
                model.fit(Xtr, ytr, validation_data=(Xva, yva),
                          epochs=EPOCHS, batch_size=BATCH,
                          callbacks=cbs, verbose=0)
                model.save(mp)
                print(f"[{n}/{total}] {g} seed{seed} 训练完成")
            for gname, (hist, cids) in groups_eval.items():
                for cid in cids:
                    if cid not in hist:
                        continue
                    X, y, _ = build_split_arrays(hist, [cid], GROUPS[g])
                    if len(X) == 0:
                        continue
                    pred = model.predict(X, verbose=0).ravel()
                    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
                    records.append(dict(exp=g, seed=seed,
                                        eval_group=gname, cycle_id=cid,
                                        RMSE=rmse))
    pd.DataFrame(records).to_csv(OUT_CSV, index=False)
    print(f"\n已写出 {OUT_CSV}（{len(records)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
