# -*- coding: utf-8 -*-
"""
P3-06 消融正式版：A2-0 ~ A2-4 × 多 seed，S2 判定材料

设计
----
- 5 个特征组 × 3 seeds（7/13/42）+ 已训 seed 7 复用（p3_04_train.py 产物）
- 每组每 seed 记录：逐文件 EKF/融合 RMSE/MAE/MAX + 分组均值
- 汇总：组均值 ± std（跨 seed），配对比较 key 增量：
    * A2-1 vs A2-0   ← S2 核心问题①：ν 有增量吗？
    * A2-2 vs A2-1   ← NIS 有增量吗？
    * A2-3 vs A2-2   ← S2 核心问题②：K/P 还有增量吗？（定稿 §4 判据）
    * A2-4 vs A2-0   ← K/P 独立（无 ν）有用吗？
- 显著性：seed 少（3），用配对 Wilcoxon 或仅报均值±范围，明确注明
  「n=3 seeds，无统计检验力，以效应方向一致性为准」——诚实报告

产出
----
- results/p3_06_ablation_full.csv   逐 seed × 组 × 评估组
- results/p3_06_ablation_summary.md S2 判定材料

用法
----
    python p3_06_ablation.py            # 全量（缺啥补啥，已有则复用）
    python p3_06_ablation.py --summary-only
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

from nn_features import GROUPS, WINDOW_L, build_split_arrays, load_history  # noqa: E402

RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "04_实验", "models")
SEEDS = [7, 13, 42]
EPOCHS = 200
BATCH = 256
PATIENCE = 20


def load_all():
    train_hist = load_history(os.path.join(RESULTS, "ekf_train_history.npz"))
    test_hist = load_history(os.path.join(RESULTS, "ekf_baseline_history.npz"))
    val_ids = [c for c in train_hist if "UDDS" in c]
    tr_ids = [c for c in train_hist if c not in val_ids]
    return train_hist, test_hist, tr_ids, val_ids


def test_groups(test_hist):
    ids = sorted(test_hist.keys())
    # ⚠️ "10degC_" 是 "n10degC_" 的子串 → 必须显式排除负温度文件
    return {
        "test_id(25C)": [c for c in ids if "25degC_Cycle" in c],
        "ood(10C)":  [c for c in ids if "10degC_" in c and "trise" not in c
                      and "n10degC" not in c],
        "ood(0C)":   [c for c in ids if "0degC_Cycle" in c],
        "ood(-10C)": [c for c in ids if "n10degC_" in c],
        "ood(-20C)": [c for c in ids if "n20degC_" in c and "trise" not in c],
    }


def build_model(n_feats, seed):
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(seed)
    inp = keras.Input(shape=(WINDOW_L, n_feats))
    x = keras.layers.GRU(16, unroll=True)(inp)
    x = keras.layers.Dense(8, activation="relu")(x)
    out = keras.layers.Dense(1)(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


def eval_one(model, hist, cids, feats) -> list[dict]:
    rows = []
    for cid in cids:
        X, y, _ = build_split_arrays(hist, [cid], feats)
        if X is None:
            continue
        y_hat = model.predict(X, verbose=0).ravel()
        e_after = y - y_hat
        rows.append({
            "cycle_id": cid, "n": len(y),
            "ekf_RMSE": float(np.sqrt(np.mean(y**2))),
            "nn_RMSE": float(np.sqrt(np.mean(e_after**2))),
            "ekf_MAE": float(np.mean(np.abs(y))),
            "nn_MAE": float(np.mean(np.abs(e_after))),
            "ekf_MAX": float(np.max(np.abs(y))),
            "nn_MAX": float(np.max(np.abs(e_after))),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    train_hist, test_hist, tr_ids, val_ids = load_all()
    groups = test_groups(test_hist)
    groups_eval = {"val(25C)": (train_hist, val_ids)}
    for k, v in groups.items():
        groups_eval[k] = (test_hist, v)

    feats_cache: dict[str, tuple] = {}
    for g in GROUPS:
        feats = GROUPS[g]
        Xtr, ytr, _ = build_split_arrays(train_hist, tr_ids, feats)
        Xva, yva, _ = build_split_arrays(train_hist, val_ids, feats)
        feats_cache[g] = (Xtr, ytr, Xva, yva)

    full_path = os.path.join(RESULTS, "p3_06_ablation_full.csv")
    records: list[dict] = []

    if not args.summary_only:
        from tensorflow import keras
        for g in sorted(GROUPS.keys()):
            Xtr, ytr, Xva, yva = feats_cache[g]
            for seed in SEEDS:
                mp = os.path.join(MODELS_DIR, f"gru16_{g}_seed{seed}.keras")
                if os.path.exists(mp):
                    model = keras.models.load_model(mp)
                else:
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
                for gname, (hist, cids) in groups_eval.items():
                    for r in eval_one(model, hist, cids, GROUPS[g]):
                        records.append({"exp": g, "seed": seed, "eval_group": gname, **r})
                print(f"完成 {g} seed={seed}")
        pd.DataFrame(records).to_csv(full_path, index=False)
    else:
        records = pd.read_csv(full_path).to_dict("records")

    # ---------- 汇总 ----------
    df = pd.DataFrame(records)
    agg = (df.groupby(["exp", "eval_group"])
             .agg(nn_RMSE_mean=("nn_RMSE", "mean"),
                  nn_RMSE_std=("nn_RMSE", "std"),
                  nn_MAE_mean=("nn_MAE", "mean"),
                  nn_MAX_mean=("nn_MAX", "mean"),
                  ekf_RMSE=("ekf_RMSE", "mean"),
                  n_seeds=("seed", "nunique"))
             .round(3).reset_index())

    print("\n==== 消融汇总（nn_RMSE，mean ± std over 3 seeds，pp）====")
    piv = agg.pivot_table(index="eval_group", columns="exp",
                          values="nn_RMSE_mean").round(3)
    piv = piv[sorted(GROUPS.keys())]
    print(piv.to_string())

    # seed 间 std
    std_piv = agg.pivot_table(index="eval_group", columns="exp",
                              values="nn_RMSE_std").round(3)
    print("\n==== seed 间 std（pp）====")
    print(std_piv[sorted(GROUPS.keys())].to_string())

    # 配对增量（逐 seed 配对，报均值方向一致性）
    print("\n==== 配对增量（Δnn_RMSE，负 = 更好，pp）====")
    wide = df.groupby(["exp", "seed", "eval_group"]).nn_RMSE.mean().reset_index()
    for hi, lo, label in [
            ("A2-0", "A2-1", "A2-1 vs A2-0 (+ν)"),
            ("A2-1", "A2-2", "A2-2 vs A2-1 (+NIS)"),
            ("A2-2", "A2-3", "A2-3 vs A2-2 (+K,P)"),
            ("A2-0", "A2-4", "A2-4 vs A2-0 (+K,P 独立)")]:
        d = (wide[wide.exp == lo].set_index(["seed", "eval_group"]).nn_RMSE
             - wide[wide.exp == hi].set_index(["seed", "eval_group"]).nn_RMSE)
        d = d.rename("delta").reset_index()
        sm = d.groupby("eval_group").delta.agg(["mean", "std"]).round(3)
        # 方向一致性：3 个 seed 里 ≤0 的个数
        cons = d.groupby("eval_group").delta.apply(
            lambda s: f"{(s <= 0).sum()}/{len(s)}")
        print(f"\n[{label}]")
        print(pd.concat([sm, cons.rename("seeds_improved")], axis=1).to_string())

    agg.to_csv(os.path.join(RESULTS, "p3_06_ablation_agg.csv"), index=False)
    print(f"\n保存: {full_path} / p3_06_ablation_agg.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
