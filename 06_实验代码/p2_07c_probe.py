# -*- coding: utf-8 -*-
"""P2-07 修正版可学习性探针（E4'）：按真实数据流方向

修正点（相对初版 E4）：
1. 混池 LOO → 方向化：train(5)+val(1) 训练 → test_id / 各温度 OOD 测试
   （真实场景是「25 °C 训练 → 25 °C 或 OOD 温度部署」，不是跨温度混训）
2. F3 滑窗列名重复 bug 修复
3. 特征消融分细：ν 单独 / 物理量组 / ν+物理 / 滑窗ν 增强版
   （回答 S1 的精确问题：ν 含有物理量之外的增量信息吗？）

探针
----
A（同分布）： train+val → test_id（25 °C）
B（温度迁移）：train+val → 每温度各 1 个 OOD 文件

回归目标 y = SOC_EKF − SOC_true（pp，含偏置——NN 要学的就是它）
评估：R²（相对预测 0）、RMSE_before/after（pp）、偏置消除量
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import sys

# Windows 控制台默认 GBK，而本脚本 print 含 GBK 无码位的字符 —— 不改编码会在
# **成功路径**崩掉（活干完了却退出码 1）。别靠 PYTHONIOENCODING 兜底。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
BURN_IN = 300
LAM = 1.0


def load(path: str) -> dict[str, dict[str, np.ndarray]]:
    z = np.load(path, allow_pickle=False)
    files: dict[str, dict[str, np.ndarray]] = {}
    for key in z.files:
        cid, field = key.split("::", 1)
        files.setdefault(cid, {})[field] = z[key]
    return files


def make_feats(h: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """从 history 抽特征与目标。前 300 s 收敛期剔除，末 3 s 留滑窗缓冲。"""
    n = len(h["nu"])
    sl = slice(BURN_IN, n - 3)
    nu = h["nu"]
    return {
        "nu": nu[sl],
        "nu_1": nu[sl.start - 1: sl.stop - 1],
        "nu_2": nu[sl.start - 2: sl.stop - 2],
        "nu_3": nu[sl.start - 3: sl.stop - 3],
        "iabs": np.abs(h["current_A"][sl]),
        "temp": h["temperature_C"][sl],
        "soc": h["soc"][sl],          # SOC_EKF（输入侧可用信息）
        "soc_true": h["soc_true"][sl],
        "y": (h["soc"] - h["soc_true"])[sl] * 100.0,
    }


FEAT_SETS = {
    "F0 预测0": [],
    "F1 仅ν": ["nu"],
    "F2 仅物理量(I,T,SOC_EKF)": ["iabs", "temp", "soc"],
    "F3 ν+物理量": ["nu", "iabs", "temp", "soc"],
    "F4 ν滑窗+物理量": ["nu", "nu_1", "nu_2", "nu_3",
                        "iabs", "temp", "soc"],
    # F0' 用 SOC_EKF 自身单调修正（无 ν）：检验「误差≈SOC 的函数」假说的上界
    "F5 仅SOC_EKF": ["soc"],
}


def ridge_fit(Xtr, ytr, lam=LAM):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xz = (Xtr - mu) / sd
    A = Xz.T @ Xz + lam * np.eye(Xz.shape[1])
    w = np.linalg.solve(A, Xz.T @ (ytr - ytr.mean()))
    return mu, sd, w, float(ytr.mean())


def ridge_pred(model, Xte):
    mu, sd, w, b0 = model
    return ((Xte - mu) / sd) @ w + b0


def main() -> int:
    train = load(os.path.join(RESULTS, "ekf_train_history.npz"))
    test = load(os.path.join(RESULTS, "ekf_baseline_history.npz"))

    # 训练池：train+val 全部文件
    tr_feats = [make_feats(h) for h in train.values()]
    tr_ids = sorted(test.keys())

    # 测试组
    groups: dict[str, list[str]] = {
        "A:test_id(25°C)": [c for c in test
                            if "Cycle_1" in c and "trise" not in c
                            or "25degC_Cycle" in c],
    }
    groups["A:test_id(25°C)"] = [c for c in test if "25degC_Cycle" in c]
    for tag, pat in [("B:OOD 10°C", "10degC_"), ("B:OOD 0°C", "0degC_Cycle"),
                     ("B:OOD −10°C", "n10degC_"), ("B:OOD −20°C", "n20degC_")]:
        groups[tag] = [c for c in test if pat in c and "trise" not in c]

    print("测试文件分组：")
    for k, v in groups.items():
        print(f"  {k}: {len(v)} 文件")

    for gname, cids in groups.items():
        print(f"\n===== 探针 {gname} =====")
        te_feats = [make_feats(test[c]) for c in cids if c in test]
        # 目标统计
        y_all = np.concatenate([f["y"] for f in te_feats])
        print(f"  目标（SOC 误差）: mean={y_all.mean():+.2f}pp "
              f"std={y_all.std():.2f}pp RMSE={np.sqrt((y_all**2).mean()):.2f}pp")
        for fname, cols in FEAT_SETS.items():
            if not cols:
                rmse_after = np.sqrt((y_all**2).mean())
                print(f"  {fname:<22s} R²={0.0:+.3f}  "
                      f"RMSE {np.sqrt((y_all**2).mean()):5.2f}→{rmse_after:5.2f}pp")
                continue
            Xtr = np.hstack([np.concatenate([f[c] for f in tr_feats])[:, None]
                             for c in cols])
            ytr = np.concatenate([f["y"] for f in tr_feats])
            model = ridge_fit(Xtr, ytr)
            r2s, rm_bef, rm_aft = [], [], []
            for f in te_feats:
                Xte = np.hstack([f[c][:, None] for c in cols])
                yte = f["y"]
                pred = ridge_pred(model, Xte)
                r2s.append(1.0 - np.sum((yte - pred)**2)
                           / np.sum((yte - yte.mean())**2))
                rm_bef.append(np.sqrt(np.mean(yte**2)))
                rm_aft.append(np.sqrt(np.mean((yte - pred)**2)))
            # 混池口径
            Xte_all = np.hstack([np.concatenate([f[c] for f in te_feats])[:, None]
                                 for c in cols])
            y_all2 = np.concatenate([f["y"] for f in te_feats])
            pred_all = ridge_pred(model, Xte_all)
            r2_all = 1.0 - np.sum((y_all2 - pred_all)**2) / \
                np.sum((y_all2 - y_all2.mean())**2)
            print(f"  {fname:<22s} R²={r2_all:+.3f}  "
                  f"RMSE {np.mean(rm_bef):5.2f}→{np.mean(rm_aft):5.2f}pp "
                  f"(均)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
