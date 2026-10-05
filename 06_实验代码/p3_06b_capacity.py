# -*- coding: utf-8 -*-
"""
P3-06 附加：容量检查 —— GRU16 结论是否受网络容量限制？

背景
----
多 seed 消融（3 seeds）显示：同分布下 ν 的增量 ≈ 0.02 pp（噪声量级）。
在判 S2 之前必须排除「容量不足导致学不出 ν 的贡献」这一替代解释。

设计
----
- A2-0 / A2-3 × GRU(units=32) × seeds {7, 13}（与 GRU16 的同 seed 配对比较）
- 判据：若 GRU32 下 (A2-3 − A2-0) 缺口显著大于 GRU16 下的缺口
  → 容量是瓶颈，消融需在 GRU32 上重跑；
  若缺口依旧 ≈ 0 → 容量不是原因，S2 材料可以定稿
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "06_实验代码", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from nn_features import GROUPS, WINDOW_L, build_split_arrays, load_history  # noqa: E402
from p3_06_ablation import load_all, test_groups, eval_one  # noqa: E402

# Windows 控制台默认 GBK，而本脚本 print 含 GBK 无码位的字符 —— 不改编码会在
# **成功路径**崩掉（活干完了却退出码 1）。别靠 PYTHONIOENCODING 兜底。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "04_实验", "models")
CAP_SEEDS = [7, 13]
UNITS = 32
EPOCHS = 200
BATCH = 256


def build_model(n_feats, seed):
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(seed)
    inp = keras.Input(shape=(WINDOW_L, n_feats))
    x = keras.layers.GRU(UNITS, unroll=True)(inp)
    x = keras.layers.Dense(8, activation="relu")(x)
    out = keras.layers.Dense(1)(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def main() -> int:
    from tensorflow import keras
    train_hist, test_hist, tr_ids, val_ids = load_all()
    groups = test_groups(test_hist)
    groups_eval = {"val(25C)": (train_hist, val_ids)}
    for k, v in groups.items():
        groups_eval[k] = (test_hist, v)

    records = []
    for g in ["A2-0", "A2-3"]:
        feats = GROUPS[g]
        Xtr, ytr, _ = build_split_arrays(train_hist, tr_ids, feats)
        Xva, yva, _ = build_split_arrays(train_hist, val_ids, feats)
        for seed in CAP_SEEDS:
            mp = os.path.join(MODELS_DIR, f"gru32_{g}_seed{seed}.keras")
            if os.path.exists(mp):
                model = keras.models.load_model(mp)
            else:
                model = build_model(Xtr.shape[2], seed)
                cbs = [
                    keras.callbacks.EarlyStopping(
                        patience=20, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(
                        patience=8, factor=0.5, min_lr=1e-5),
                ]
                model.fit(Xtr, ytr, validation_data=(Xva, yva),
                          epochs=EPOCHS, batch_size=BATCH,
                          callbacks=cbs, verbose=0)
                model.save(mp)
            for gname, (hist, cids) in groups_eval.items():
                for r in eval_one(model, hist, cids, feats):
                    records.append({"exp": g, "seed": seed,
                                    "eval_group": gname, **r})
            print(f"完成 gru32 {g} seed={seed}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(RESULTS, "p3_06b_capacity_full.csv"), index=False)

    print("\n==== GRU32 vs GRU16：组均值 nn_RMSE（pp）====")
    f16 = pd.read_csv(os.path.join(RESULTS, "p3_06_ablation_full.csv"))
    g16 = f16.groupby(["exp", "eval_group"]).nn_RMSE.mean().unstack(0)
    g32 = df.groupby(["exp", "eval_group"]).nn_RMSE.mean().unstack(0)
    both = pd.concat([g16.add_prefix("16_"), g32.add_prefix("32_")], axis=1).round(3)
    cols = [f"{u}_{g}" for g in ["A2-0", "A2-3"] for u in [16, 32]]
    print(both[cols].to_string())

    print("\n==== 关键缺口 (A2-0 − A2-3)：正 = 内部诊断量有增量 ====")
    gaps = pd.DataFrame({
        "gru16": (g16["A2-0"] - g16["A2-3"]).round(3),
        "gru32": (g32["A2-0"] - g32["A2-3"]).round(3),
    })
    print(gaps.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
