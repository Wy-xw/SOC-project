# -*- coding: utf-8 -*-
"""
P3-02/03/04：GRU 残差网络 —— 数据 pipeline + 基线训练 + 评估

网络
----
GRU(16, unroll=True) + Dense(8, relu) + Dense(1)
- unroll=True 是 TFLite 转换硬约束（`07_环境与工具.md`：默认 Keras GRU
  用动态 TensorListReserve，TFLite 转换失败）
- 参数量 **1 633**（§4.4 精确值）；导出体积：float16 **105 kB**、int8 **205 kB**
  （§4.5 / `p3_12_int8_results_full40.csv`）。⚠️ 原注释写"~1.8 k、int8 后 ~50 KB"
  与论文冲突，已于 2026-09-26 更正——**旧值不是本模型的实测值**。
- 输入 (L=30, F)，输出窗末时刻残差预测 y_hat（pp）

训练
----
- train = 5 文件（25 °C），val = 1 文件（UDDS，25 °C），早停看 val
- 训练集仅 12.7 h（P1 结论 R2 风险）：小网络 + 早停 + 小学习率收尾
- 损失 MSE（目标是回归 pp 级残差）

评估（与 EKF 基线同口径，可直接对照 ekf_baseline_metrics.csv）
----
- 逐文件：EKF RMSE → 融合后 RMSE（pp，稳态段 = 前 300 s 剔除）
- 分组汇总：test_id / OOD 各温度
- 消融组由命令行 --group 指定（A2-0 ~ A2-4）

用法
----
    python p3_04_train.py --group A2-3          # 完整特征
    python p3_04_train.py --group A2-0          # 对照基线
    python p3_04_train.py --group A2-3 --skip-train  # 只评估已存模型
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "06_实验代码", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from nn_features import (  # noqa: E402
    BURN_IN, GROUPS, WINDOW_L, build_split_arrays, load_history,
)

RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "04_实验", "models")
# ⚠️ 2026-09-26：本脚本固定 SEED = 7 且**无 --seed 参数**；论文的 3 seed
#    （7/13/42）由 p3_06_ablation.py 提供，不由本脚本产生。要复现多 seed 请用后者。
SEED = 7
EPOCHS = 200
BATCH = 256
PATIENCE = 20


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #

def load_all():
    train_hist = load_history(os.path.join(RESULTS, "ekf_train_history.npz"))
    test_hist = load_history(os.path.join(RESULTS, "ekf_baseline_history.npz"))
    # train npz 含 train 5 + val 1；val 是 UDDS
    val_ids = [c for c in train_hist if "UDDS" in c]
    tr_ids = [c for c in train_hist if c not in val_ids]
    return train_hist, test_hist, tr_ids, val_ids


def test_groups(test_hist: dict) -> dict[str, list[str]]:
    """评估分组，与 P2-07c 对齐。

    baseline history 每个 OOD 温度各存 1 文件：
    10 °C / −10 °C / −20 °C 是 HWFET，0 °C 是 Cycle_1（首字母排序所致）。
    """
    ids = sorted(test_hist.keys())
    # ⚠️ "10degC_" 是 "n10degC_" 的子串 → ood(10C) 必须显式排除负温度文件
    return {
        "test_id(25C)": [c for c in ids if "25degC_Cycle" in c],
        "ood(10C)":  [c for c in ids if "10degC_" in c and "trise" not in c
                      and "n10degC" not in c],
        "ood(0C)":   [c for c in ids if "0degC_Cycle" in c],
        "ood(-10C)": [c for c in ids if "n10degC_" in c],
        "ood(-20C)": [c for c in ids if "n20degC_" in c and "trise" not in c],
    }


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #

def build_model(n_feats: int, window_l: int):
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(SEED)
    inp = keras.Input(shape=(window_l, n_feats))
    x = keras.layers.GRU(16, unroll=True)(inp)
    x = keras.layers.Dense(8, activation="relu")(x)
    out = keras.layers.Dense(1)(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="mse", metrics=["mae"])
    return m


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #

def evaluate(model, hist, cids, feats):
    """逐文件评估：透传逐点 y_hat，与 EKF 基线同口径算稳态 RMSE。"""
    rows = []
    for cid in cids:
        out = build_split_arrays(hist, [cid], feats)
        X, y, _ = out
        if X is None:
            continue
        y_hat = model.predict(X, verbose=0).ravel()
        # 稳态口径：X 窗末索引 ≥ BURN_IN 已由采样窗口保证；
        # EKF 基线的 ss_ 剔除前 300 s，窗末起点恰为 burn_in → 完全对齐
        e_before = y                       # EKF 误差（pp）
        e_after = y - y_hat                # 融合后误差（pp）
        h = hist[cid]
        n = len(h["nu"])
        ss = slice(BURN_IN, n)
        ss_mask_ends = np.arange(BURN_IN + WINDOW_L - 1, n)
        # 与基线同长对齐：直接用全部样本（其窗末已 ≥ burn_in+L−1 > 300）
        rows.append({
            "cycle_id": cid,
            "n": len(y),
            "ekf_RMSE": float(np.sqrt(np.mean(e_before**2))),
            "nn_RMSE": float(np.sqrt(np.mean(e_after**2))),
            "ekf_MAE": float(np.mean(np.abs(e_before))),
            "nn_MAE": float(np.mean(np.abs(e_after))),
            "ekf_MAX": float(np.max(np.abs(e_before))),
            "nn_MAX": float(np.max(np.abs(e_after))),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="A2-3", choices=sorted(GROUPS.keys()))
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    feats = GROUPS[args.group]
    print(f"特征组 {args.group}: {len(feats)} 特征 → {feats}")

    train_hist, test_hist, tr_ids, val_ids = load_all()
    print(f"train {len(tr_ids)} / val {len(val_ids)} / test {len(test_hist)} 文件")

    Xtr, ytr, _ = build_split_arrays(train_hist, tr_ids, feats)
    Xva, yva, _ = build_split_arrays(train_hist, val_ids, feats)
    print(f"样本: train {len(ytr)} / val {len(yva)}（X: {Xtr.shape}）")

    os.makedirs(MODELS_DIR, exist_ok=True)
    tag = f"gru16_{args.group}_L{WINDOW_L}"
    model_path = os.path.join(MODELS_DIR, f"{tag}.keras")
    metrics_csv = os.path.join(MODELS_DIR, f"{tag}_metrics.csv")

    if not args.skip_train:
        model = build_model(Xtr.shape[2], WINDOW_L)
        from tensorflow import keras
        cbs = [
            keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-5),
        ]
        hist = model.fit(
            Xtr, ytr, validation_data=(Xva, yva),
            epochs=args.epochs, batch_size=BATCH, callbacks=cbs, verbose=2)
        model.save(model_path)
        best = min(hist.history["val_loss"])
        print(f"训练完成，best val_loss={best:.4f} → {model_path}")
    else:
        from tensorflow import keras
        model = keras.models.load_model(model_path)

    # 分组评估
    all_rows = []
    groups = test_groups(test_hist)
    groups["val(25C)"] = val_ids
    for gname, cids in groups.items():
        df = evaluate(model, train_hist if gname == "val(25C)" else test_hist,
                      cids, feats)
        if df.empty:
            continue
        mean_row = {
            "group": gname, "n_files": len(df),
            "ekf_RMSE": df.ekf_RMSE.mean(), "nn_RMSE": df.nn_RMSE.mean(),
            "ekf_MAE": df.ekf_MAE.mean(), "nn_MAE": df.nn_MAE.mean(),
            "ekf_MAX": df.ekf_MAX.max(), "nn_MAX": df.nn_MAX.max(),
        }
        all_rows.append(mean_row)
        print(f"\n[{gname}] {len(df)} 文件")
        print(df.to_string(index=False,
                           float_format=lambda v: f"{v:7.3f}"))

    summary = pd.DataFrame(all_rows)
    summary.to_csv(metrics_csv, index=False)
    print("\n==== 汇总（pp，均值口径）====")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print(f"\n保存: {metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
