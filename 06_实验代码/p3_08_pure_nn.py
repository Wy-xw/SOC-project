# -*- coding: utf-8 -*-
"""
P3-08：纯 NN baseline —— 不借助 EKF，测量量序列直接回归 SOC

定位
----
论文 Table 1（Baseline 对比；2026-09-26 重编号前叫 Table 2）的关键对照：证明「EKF+NN 混合」优于
「纯数据驱动」。审稿人必看这条——如果纯 NN 就够，混合方法就没有存在意义。

设计
----
- 输入：测量量滑窗 (I, |I|, T, V_t, ΔV_t)，L=30 —— **不使用任何 EKF 输出**
  （不用 SOC_EKF/ν/NIS/K/P —— 这正是它与 A2-0 的区别：
   A2-0 有 SOC_EKF（EKF 的输出），纯 NN 连这个都没有）
- 输出：SOC 绝对值（不是残差！）——纯 NN 只能直接回归
- 网络与训练配置主线一致（GRU16 unroll），公平比较
- 评估：val 早停，test_id + OOD 各组；与融合版同口径（稳态 RMSE pp）

符号注意
--------
纯 NN 的一个天然缺陷要如实报告：初始 SOC 不可知时，纯 NN 只能靠
V_t 反推 SOC（隐含 OCV 映射），冷启动段误差大是预期行为——
这本身就是混合方法优势的一部分（EKF 提供物理先验的安时积分骨架）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "06_实验代码", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from nn_features import (  # noqa: E402
    BURN_IN, FEATURE_TRANSFORMS, WINDOW_L, load_history,
)
from p3_06_ablation import load_all, test_groups  # noqa: E402

RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "04_实验", "models")
SEEDS = [7, 13, 42, 101, 202, 303, 404, 505, 606, 707]  # 意见9①：扩到 10 种子与消融臂口径统一
EPOCHS = 200
BATCH = 256

#: 纯 NN 特征（只有测量量，无任何 EKF 信息）
PURE_FEATS = ["i", "iabs", "temp", "vt", "dvt"]


def file_to_samples(h, feats=PURE_FEATS, window_l=WINDOW_L, burn_in=BURN_IN):
    """测量量滑窗 → (X, y=SOC_true×100)。"""
    n = len(h["nu"])
    if n < burn_in + window_l + 1:
        return None
    F = np.stack(
        [np.clip(FEATURE_TRANSFORMS[f](h), -10.0, 10.0) for f in feats],
        axis=1).astype(np.float32)
    y_all = h["soc_true"] * 100.0                     # 绝对 SOC（pp）
    starts = np.arange(burn_in, n - window_l + 1)
    idx = starts[:, None] + np.arange(window_l)[None, :]
    X = np.take(F, idx, axis=0)
    ends = starts + window_l - 1
    return X, y_all[ends].astype(np.float32)


def build_model(n_feats, seed):
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(seed)
    inp = keras.Input(shape=(WINDOW_L, n_feats))
    x = keras.layers.GRU(16, unroll=True)(inp)
    x = keras.layers.Dense(8, activation="relu")(x)
    out = keras.layers.Dense(1)(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def main() -> int:
    from tensorflow import keras
    train_hist, test_hist, tr_ids, val_ids = load_all()

    def stack(hist, cids):
        xs, ys = [], []
        for c in cids:
            out = file_to_samples(hist[c])
            if out is not None:
                xs.append(out[0]); ys.append(out[1])
        return np.concatenate(xs), np.concatenate(ys)

    Xtr, ytr = stack(train_hist, tr_ids)
    Xva, yva = stack(train_hist, val_ids)
    print(f"纯 NN 样本: train {len(ytr)} / val {len(yva)}，特征 {PURE_FEATS}")

    groups = test_groups(test_hist)
    groups_eval = {"val(25C)": (train_hist, val_ids)}
    groups_eval.update({k: (test_hist, v) for k, v in groups.items()})

    records = []
    for seed in SEEDS:
        mp = os.path.join(MODELS_DIR, f"pureNN_seed{seed}.keras")
        if os.path.exists(mp):
            model = keras.models.load_model(mp)
        else:
            model = build_model(Xtr.shape[2], seed)
            cbs = [
                keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-5),
            ]
            model.fit(Xtr, ytr, validation_data=(Xva, yva),
                      epochs=EPOCHS, batch_size=BATCH, callbacks=cbs, verbose=0)
            model.save(mp)
        for gname, (hist, cids) in groups_eval.items():
            for cid in cids:
                out = file_to_samples(hist[cid])
                if out is None:
                    continue
                X, y = out
                pred = model.predict(X, verbose=0).ravel()
                e = pred - y                                    # pp
                ekf_e = (hist[cid]["soc"][BURN_IN + WINDOW_L - 1:]
                         - hist[cid]["soc_true"][BURN_IN + WINDOW_L - 1:]) * 100
                records.append({
                    "seed": seed, "eval_group": gname, "cycle_id": cid,
                    "nn_RMSE": float(np.sqrt(np.mean(e**2))),
                    "nn_MAE": float(np.mean(np.abs(e))),
                    "nn_MAX": float(np.max(np.abs(e))),
                    "ekf_RMSE": float(np.sqrt(np.mean(ekf_e**2))),
                })
        print(f"完成 seed={seed}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(RESULTS, "p3_08_pure_nn.csv"), index=False)
    agg = df.groupby("eval_group").agg(
        pureNN_RMSE=("nn_RMSE", "mean"), pureNN_MAE=("nn_MAE", "mean"),
        pureNN_MAX=("nn_MAX", "mean"), ekf_RMSE=("ekf_RMSE", "mean"),
        n_seeds=("seed", "nunique")).round(3)
    print("\n==== 纯 NN baseline（3 seeds 均值，pp）====")
    print(agg.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
