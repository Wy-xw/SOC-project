# -*- coding: utf-8 -*-
"""
P5-01 Layer 1：Baseline 对比 — 6 方法 × 3 工况 × 3 温度

方法
----
M1: Coulomb Counting（安时积分，无模型校正）
M2: EKF alone（1RC-ECM + 扩展卡尔曼）
M3: EKF + NN A2-0（仅测量量，对照基线）
M4: EKF + NN A2-3（完整特征，主模型）
M5: Pure NN（直接回归 SOC，不借 EKF）
M6: EKF + NN A2-4（K/P 独立，无 ν/NIS）

评估维度
--------
- 工况：UDDS / HWFET / US06
- 温度：25 °C / 0 °C / −20 °C
- 指标：RMSE / MAE / MAX（pp）

用法
----
    "D:/venvs/soc-paper/Scripts/python.exe" 10_代码/p5_01_baseline_compare.py
"""
from __future__ import annotations

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
from ecm_ekf import metrics  # noqa: E402

RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "04_实验", "models")
P5_DIR = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")

SEED = 7  # 主模型 seed


# --------------------------------------------------------------------------- #
# 数据加载与分组
# --------------------------------------------------------------------------- #

def load_all():
    train_hist = load_history(os.path.join(RESULTS, "ekf_train_history.npz"))
    # 优先用全量 history（P5-00 生成），含全部 OOD 文件
    full_path = os.path.join(RESULTS, "ekf_full_history.npz")
    if os.path.exists(full_path):
        test_hist = load_history(full_path)
    else:
        test_hist = load_history(os.path.join(RESULTS, "ekf_baseline_history.npz"))
    val_ids = [c for c in train_hist if "UDDS" in c]
    tr_ids = [c for c in train_hist if c not in val_ids]
    return train_hist, test_hist, tr_ids, val_ids


def target_groups(test_hist: dict) -> dict[str, list[str]]:
    """按工况 × 温度分组（P5-01 用）。

    ⚠️ 2026-09-23 修复：原 `"0degC_" in c` 会同时匹配 "10degC_"/"n10degC_"/
    "n20degC_"（子串陷阱）→ 旧 "0C" 列实为四温度混合。改为显式排除式判定。
    """
    ids = sorted(test_hist.keys())
    groups = {}

    # --- 25 °C：只评 Cycle（test_id）---
    groups["Cycle_25C"] = [c for c in ids if "25degC_Cycle" in c]

    def is_temp(c, t):
        if "trise" in c:
            return False
        if t == "10C":
            return "10degC_" in c and "n10degC" not in c and "n20degC" not in c
        if t == "0C":
            return "0degC_" in c and "10degC" not in c and "20degC" not in c
        if t == "-10C":
            return "n10degC_" in c
        if t == "-20C":
            return "n20degC_" in c
        return False

    cond_map = {
        "UDDS":  lambda c: "UDDS" in c,
        "HWFET": lambda c: "HWFET" in c or "HWFT" in c,
        "US06":  lambda c: "US06" in c,
    }
    for tname in ("10C", "0C", "-10C", "-20C"):
        for cname, cfn in cond_map.items():
            key = f"{cname}_{tname}"
            groups[key] = [c for c in ids if is_temp(c, tname) and cfn(c)]
    return groups


# --------------------------------------------------------------------------- #
# M1: Coulomb Counting（安时积分）
# --------------------------------------------------------------------------- #

def eval_coulomb_counting(hist, cids):
    """安时积分 SOC 估计（从 SOC_true 的起点开始积分，无任何校正）。
    这是最朴素的 baseline：SOC_est(t) = SOC0 - ∫I dt / Cn。
    用 EKF history 中的电流和 soc_true 起点。"""
    rows = []
    for cid in cids:
        h = hist[cid]
        n = len(h["nu"])
        if n < BURN_IN + WINDOW_L:
            continue
        soc_true = h["soc_true"]
        current = h["current_A"]
        # 安时积分：从 soc_true[0] 开始
        dt = 1.0  # 1 Hz
        Cn = 2.9  # Ah
        soc_cc = np.empty(n)
        soc_cc[0] = soc_true[0]
        for k in range(1, n):
            soc_cc[k] = soc_cc[k-1] - (current[k-1] * dt) / (3600.0 * Cn)
        soc_cc = np.clip(soc_cc, 0, 1)
        # 稳态口径
        ss = slice(BURN_IN, n)
        e = (soc_cc[ss] - soc_true[ss]) * 100.0
        rows.append({
            "cycle_id": cid,
            "RMSE": float(np.sqrt(np.mean(e**2))),
            "MAE": float(np.mean(np.abs(e))),
            "MAX": float(np.max(np.abs(e))),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# M2: EKF alone
# --------------------------------------------------------------------------- #

def eval_ekf(hist, cids):
    """从 EKF history 中直接提取 SOC 估计，算误差。"""
    rows = []
    for cid in cids:
        h = hist[cid]
        n = len(h["nu"])
        if n < BURN_IN + WINDOW_L:
            continue
        soc_est = np.array(h["soc"])
        soc_true = h["soc_true"]
        ss = slice(BURN_IN, n)
        e = (soc_est[ss] - soc_true[ss]) * 100.0
        rows.append({
            "cycle_id": cid,
            "RMSE": float(np.sqrt(np.mean(e**2))),
            "MAE": float(np.mean(np.abs(e))),
            "MAX": float(np.max(np.abs(e))),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# M3/M4/M6: EKF + NN（加载 Keras 模型评估）
# --------------------------------------------------------------------------- #

def eval_ekf_nn(model, hist, cids, feats):
    """EKF+NN 融合：SOC_fused = SOC_EKF - y_hat/100。"""
    rows = []
    for cid in cids:
        X, y, _ = build_split_arrays(hist, [cid], feats)
        if X is None:
            continue
        y_hat = model.predict(X, verbose=0).ravel()
        e_after = y - y_hat  # 融合后误差（pp）
        rows.append({
            "cycle_id": cid,
            "RMSE": float(np.sqrt(np.mean(e_after**2))),
            "MAE": float(np.mean(np.abs(e_after))),
            "MAX": float(np.max(np.abs(e_after))),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# M5: Pure NN（直接回归 SOC）
# --------------------------------------------------------------------------- #

def eval_pure_nn(model, hist, cids, feats):
    """Pure NN：直接输出 SOC 估计（不经过 EKF）。
    训练目标 = soc_true * 100（pp），输出需 /100 转回 [0,1]。"""
    rows = []
    for cid in cids:
        # 用 p3_08 的特征构造方式（只有测量量，无 EKF 信息）
        from nn_features import FEATURE_TRANSFORMS
        h = hist[cid]
        n = len(h["nu"])
        if n < BURN_IN + WINDOW_L + 1:
            continue
        F = np.stack(
            [np.clip(FEATURE_TRANSFORMS[f](h), -10.0, 10.0) for f in feats],
            axis=1).astype(np.float32)
        starts = np.arange(BURN_IN, n - WINDOW_L + 1)
        idx = starts[:, None] + np.arange(WINDOW_L)[None, :]
        X = np.take(F, idx, axis=0)
        ends = starts + WINDOW_L - 1
        soc_pred_pp = model.predict(X, verbose=0).ravel()  # pp
        soc_est = soc_pred_pp / 100.0  # 转回 [0,1]
        soc_true = h["soc_true"]
        e = (soc_est - soc_true[ends]) * 100.0  # 误差 pp
        rows.append({
            "cycle_id": cid,
            "RMSE": float(np.sqrt(np.mean(e**2))),
            "MAE": float(np.mean(np.abs(e))),
            "MAX": float(np.max(np.abs(e))),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main() -> int:
    from tensorflow import keras

    print("=" * 70)
    print("P5-01 Baseline 对比：6 方法 × 多工况 × 多温度")
    print("=" * 70)

    train_hist, test_hist, tr_ids, val_ids = load_all()
    groups = target_groups(test_hist)
    print(f"评估组: {list(groups.keys())}")
    print(f"每组文件数: {', '.join(f'{k}={len(v)}' for k, v in groups.items())}")

    # 加载模型
    # R1-M4 修复：与消融（p3_06）同族的 seed7 模型，而非 p3_04 的 L30 单模——
    # 保证 fig7 的 seed-7 切片与 fig8 的 3-seed 聚合同源可调和。
    print("\n加载模型（与消融同族 seed7）...")
    models = {}
    # EKF+NN 系列
    for group_name, model_file in [
        ("A2-0", "gru16_A2-0_seed7.keras"),
        ("A2-3", "gru16_A2-3_seed7.keras"),
        ("A2-4", "gru16_A2-4_seed7.keras"),
    ]:
        path = os.path.join(MODELS_DIR, model_file)
        if os.path.exists(path):
            models[group_name] = keras.models.load_model(path, compile=False)
            print(f"  [OK] {group_name}: {model_file}")
        else:
            print(f"  [--] {group_name}: {model_file} 不存在")
    # Pure NN
    pure_path = os.path.join(MODELS_DIR, "pureNN_seed7.keras")
    if os.path.exists(pure_path):
        models["pureNN"] = keras.models.load_model(pure_path, compile=False)
        print(f"  [OK] pureNN: pureNN_seed7.keras")
    else:
        print(f"  [--] pureNN 不存在")

    # 评估
    PURE_FEATS = ["i", "iabs", "temp", "vt", "dvt"]
    methods = [
        ("M1_Coulomb", None, None),
        ("M2_EKF", None, None),
        ("M3_A2-0", models.get("A2-0"), GROUPS["A2-0"]),
        ("M4_A2-3", models.get("A2-3"), GROUPS["A2-3"]),
        ("M5_pureNN", models.get("pureNN"), PURE_FEATS),
        ("M6_A2-4", models.get("A2-4"), GROUPS["A2-4"]),
    ]

    all_rows = []
    for gname, cids in groups.items():
        if not cids:
            continue
        print(f"\n--- {gname} ({len(cids)} 文件) ---")
        for mname, model, feats in methods:
            if model is None and mname not in ("M1_Coulomb", "M2_EKF"):
                continue
            if mname == "M1_Coulomb":
                df = eval_coulomb_counting(test_hist, cids)
            elif mname == "M2_EKF":
                df = eval_ekf(test_hist, cids)
            elif mname == "M5_pureNN":
                df = eval_pure_nn(model, test_hist, cids, feats)
            else:
                df = eval_ekf_nn(model, test_hist, cids, feats)
            if df.empty:
                continue
            row = {
                "method": mname,
                "eval_group": gname,
                "n_files": len(df),
                "RMSE": df.RMSE.mean(),
                "MAE": df.MAE.mean(),
                "MAX": df.MAX.max(),
            }
            all_rows.append(row)
            print(f"  {mname:12s}  RMSE={row['RMSE']:7.3f}  MAE={row['MAE']:7.3f}  MAX={row['MAX']:7.3f}")

    # 保存
    summary = pd.DataFrame(all_rows)
    csv_path = os.path.join(P5_DIR, "p5_01_baseline_comparison.csv")
    summary.to_csv(csv_path, index=False)

    # 打印透视表
    print("\n" + "=" * 70)
    print("RMSE 透视表（pp）")
    print("=" * 70)
    pivot = summary.pivot_table(index="eval_group", columns="method",
                                values="RMSE", aggfunc="mean")
    cols = ["M1_Coulomb", "M2_EKF", "M3_A2-0", "M4_A2-3", "M5_pureNN", "M6_A2-4"]
    pivot = pivot[[c for c in cols if c in pivot.columns]]
    print(pivot.round(3).to_string())

    print(f"\n保存: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
