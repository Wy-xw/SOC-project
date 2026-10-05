# -*- coding: utf-8 -*-
"""
P3-05：超参搜索（学习率 / 窗长 / GRU 单元数）

搜索空间
--------
- learning_rate: [5e-4, 1e-3, 2e-3]
- window_l: [15, 30, 60]
- gru_units: [8, 16, 32]

设计
----
- 只在 A2-3（完整特征）上搜索，seed=7
- 搜索 3×3×3 = 27 组合（每组合 < 1 min，总计 ~30 min）
- 评估指标：val RMSE（选最优）+ test_id RMSE（确认）+ OOD RMSE（鲁棒性）
- 结论写入 results/p3_05_hparam_results.csv

用法
----
    python p3_05_hparam_search.py              # 跑全部 27 组合
    python p3_05_hparam_search.py --quick       # 跑 9 组合（窗长×学习率，固定 units=16）
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "06_实验代码", "src"))

from nn_features import (  # noqa: E402
    BURN_IN, GROUPS, WINDOW_L, build_split_arrays, load_history,
)

RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "04_实验", "models")
SEED = 7
EPOCHS = 200
BATCH = 256
PATIENCE = 20

# 超参搜索空间
SEARCH_SPACE = {
    "learning_rate": [5e-4, 1e-3, 2e-3],
    "window_l": [15, 30, 60],
    "gru_units": [8, 16, 32],
}

QUICK_SPACE = {
    "learning_rate": [5e-4, 1e-3, 2e-3],
    "window_l": [15, 30, 60],
    "gru_units": [16],
}


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #

def load_all():
    train_hist = load_history(os.path.join(RESULTS, "ekf_train_history.npz"))
    test_hist = load_history(os.path.join(RESULTS, "ekf_baseline_history.npz"))
    val_ids = [c for c in train_hist if "UDDS" in c]
    tr_ids = [c for c in train_hist if c not in val_ids]
    return train_hist, test_hist, tr_ids, val_ids


def test_groups(test_hist: dict) -> dict[str, list[str]]:
    ids = sorted(test_hist.keys())
    return {
        "test_id(25C)": [c for c in ids if "25degC_Cycle" in c],
        "ood(10C)":  [c for c in ids if "10degC_" in c and "trise" not in c
                      and "n10degC" not in c],
        "ood(0C)":   [c for c in ids if "0degC_Cycle" in c],
        "ood(-10C)": [c for c in ids if "n10degC_" in c],
        "ood(-20C)": [c for c in ids if "n20degC_" in c and "trise" not in c],
    }


# --------------------------------------------------------------------------- #
# 模型（支持动态窗长和单元数）
# --------------------------------------------------------------------------- #

def build_model(n_feats: int, window_l: int, gru_units: int, lr: float):
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(SEED)
    inp = keras.Input(shape=(window_l, n_feats))
    x = keras.layers.GRU(gru_units, unroll=True)(inp)
    x = keras.layers.Dense(8, activation="relu")(x)
    out = keras.layers.Dense(1)(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss="mse", metrics=["mae"])
    return m


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #

def evaluate(model, hist, cids, feats, window_l):
    rows = []
    for cid in cids:
        X, y, _ = build_split_arrays(hist, [cid], feats, window_l=window_l)
        if X is None:
            continue
        y_hat = model.predict(X, verbose=0).ravel()
        e_after = y - y_hat
        rows.append({
            "cycle_id": cid,
            "nn_RMSE": float(np.sqrt(np.mean(e_after**2))),
            "nn_MAE": float(np.mean(np.abs(e_after))),
            "nn_MAX": float(np.max(np.abs(e_after))),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def run_one(config: dict, train_hist, test_hist, tr_ids, val_ids, feats):
    """跑一组超参，返回各评估组的指标。"""
    import tensorflow as tf

    lr = config["learning_rate"]
    wl = config["window_l"]
    units = config["gru_units"]

    # 重新采样数据（窗长变了）
    Xtr, ytr, _ = build_split_arrays(train_hist, tr_ids, feats, window_l=wl)
    Xva, yva, _ = build_split_arrays(train_hist, val_ids, feats, window_l=wl)

    if Xtr is None or Xva is None:
        return None

    # 建模 + 训练
    tf.random.set_seed(SEED)
    model = build_model(Xtr.shape[2], wl, units, lr)
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            patience=PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=8, factor=0.5, min_lr=1e-5),
    ]
    hist = model.fit(
        Xtr, ytr, validation_data=(Xva, yva),
        epochs=EPOCHS, batch_size=BATCH, callbacks=cbs, verbose=0)

    best_val_loss = min(hist.history["val_loss"])
    n_epochs = len(hist.history["loss"])

    # 评估
    groups = test_groups(test_hist)
    groups["val(25C)"] = val_ids

    results = {"val_loss": best_val_loss, "n_epochs": n_epochs,
               "lr": lr, "window_l": wl, "gru_units": units}
    for gname, cids in groups.items():
        h = train_hist if gname == "val(25C)" else test_hist
        df = evaluate(model, h, cids, feats, wl)
        if df.empty:
            continue
        results[f"{gname}_RMSE"] = df.nn_RMSE.mean()
        results[f"{gname}_MAE"] = df.nn_MAE.mean()

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: 9 combos (lr x wl, fixed units=16)")
    args = ap.parse_args()

    space = QUICK_SPACE if args.quick else SEARCH_SPACE
    combos = list(itertools.product(*space.values()))
    keys = list(space.keys())
    print(f"Search space: {len(combos)} combinations")
    for k, v in space.items():
        print(f"  {k}: {v}")

    train_hist, test_hist, tr_ids, val_ids = load_all()
    feats = GROUPS["A2-3"]

    all_results = []
    for i, vals in enumerate(combos):
        config = dict(zip(keys, vals))
        print(f"\n[{i+1}/{len(combos)}] {config}")

        result = run_one(config, train_hist, test_hist, tr_ids, val_ids, feats)
        if result is None:
            print("  SKIP (data issue)")
            continue

        all_results.append(result)
        val_rmse = result.get("val(25C)_RMSE", -1)
        test_rmse = result.get("test_id(25C)_RMSE", -1)
        print(f"  val_loss={result['val_loss']:.4f}  "
              f"val_RMSE={val_rmse:.3f}  test_RMSE={test_rmse:.3f}  "
              f"epochs={result['n_epochs']}")

    # 汇总
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS, "p3_05_hparam_results.csv")
    df.to_csv(csv_path, index=False)

    # 找最优（按 val RMSE）
    if "val(25C)_RMSE" in df.columns:
        best = df.loc[df["val(25C)_RMSE"].idxmin()]
        print(f"\n{'='*60}")
        print(f"BEST (by val RMSE):")
        print(f"  lr={best['lr']}, window_l={int(best['window_l'])}, "
              f"gru_units={int(best['gru_units'])}")
        print(f"  val_RMSE={best['val(25C)_RMSE']:.3f}  "
              f"test_RMSE={best.get('test_id(25C)_RMSE', -1):.3f}")
        print(f"  OOD 10C={best.get('ood(10C)_RMSE', -1):.3f}  "
              f"OOD 0C={best.get('ood(0C)_RMSE', -1):.3f}  "
              f"OOD -10C={best.get('ood(-10C)_RMSE', -1):.3f}  "
              f"OOD -20C={best.get('ood(-20C)_RMSE', -1):.3f}")

    print(f"\nResults saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
