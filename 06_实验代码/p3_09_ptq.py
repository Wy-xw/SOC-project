# -*- coding: utf-8 -*-
"""
P3-09：训练后量化（PTQ）—— float32 Keras → **float16** TFLite

> 🔴 **2026-09-26 更正（头部与正文不符）**：本文件原自称"全整数量化 / int8"，
> **但 `ptq_convert` 实际做的是 float16**（`supported_types=[tf.float16]`，
> 见 `:128-147` 内已有的 2026-09-25 更正注记）；产物名也写作 `*_ptq_int8.tflite`。
> **真正的 int8 全整数量化在 `p3_12_int8_quant.py`**，论文的 int8 数字（0.62→0.64 pp、
> 205 kB）全部来自那里。**本脚本产出的 float16 数字对应论文的 105 kB 一档。**
> 按文件名或日志回溯论文数字的人**会被这里的命名误导**，故同步头部与打印。

量化策略
--------
- float16 权重 + float32 激活（动态范围量化），非全整数量化
- 全整数量化见 `p3_12_int8_quant.py`（论文 int8 数字的来源）

与 float32 对比
---------------
- 同一模型、同一评估集、同一口径（稳态 RMSE，pp）
- 产出 `*_ptq_compare.csv`：float32 vs **float16** 逐组对比
  ⚠️ 历史产物名为 `*_ptq_int8.tflite` / `p3_09_ptq_results.csv`，**名字带 int8 但内容是 float16**

用法
----
    python p3_09_ptq.py                    # 跑 A2-3（完整版）× seed 7
    python p3_09_ptq.py --group A2-0       # 跑指定消融组
    python p3_09_ptq.py --all-seeds        # 跑 A2-3 的 3 个 seed
    python p3_09_ptq.py --all-groups       # 跑全部 5 组 × seed 7
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "06_实验代码", "src"))

from nn_features import (  # noqa: E402
    BURN_IN, GROUPS, WINDOW_L, build_split_arrays, load_history,
)

# Windows 控制台默认 GBK，而本脚本 print 含 GBK 无码位的字符 —— 不改编码会在
# **成功路径**崩掉（活干完了却退出码 1）。别靠 PYTHONIOENCODING 兜底。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "04_实验", "models")
# TFLite 路径不能含中文，用绝对 ASCII 路径
QUANT_DIR = os.path.join(__ROOT__, "quantized")
SEEDS = [7, 13, 42]
N_CALIB = 200  # calibration 样本数


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
# 评估（与 p3_04_train.py 同口径）
# --------------------------------------------------------------------------- #

def evaluate(model, hist, cids, feats):
    rows = []
    for cid in cids:
        X, y, _ = build_split_arrays(hist, [cid], feats)
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


def evaluate_tflite(interpreter, hist, cids, feats):
    """用 TFLite interpreter 评估，逐文件算 RMSE。"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    rows = []
    for cid in cids:
        X, y, _ = build_split_arrays(hist, [cid], feats)
        if X is None:
            continue
        preds = []
        for i in range(len(X)):
            x_float = X[i:i+1].astype(np.float32)
            interpreter.set_tensor(input_details[0]["index"], x_float)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]["index"])
            preds.append(float(out[0, 0]))
        preds = np.array(preds)
        e_after = y - preds
        rows.append({
            "cycle_id": cid,
            "nn_RMSE": float(np.sqrt(np.mean(e_after**2))),
            "nn_MAE": float(np.mean(np.abs(e_after))),
            "nn_MAX": float(np.max(np.abs(e_after))),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# PTQ 核心
# --------------------------------------------------------------------------- #

def ptq_convert(keras_path: str, calib_data: np.ndarray, tflite_path: str):
    """float32 Keras → float16 TFLite（动态范围量化，权重 float16）。

    ⚠️⚠️ **2026-09-25 更正：本函数上方的旧注释「int8 灾难性失败 RMSE 0.6→106 pp」
    已被证伪，切勿再引用。**
    真跑全整数量化（见 `p3_12_int8_quant.py`，3 seed × 6 档）实测：
    **int8 只退化 +3.3 %（test_id 0.6208 → 0.6412 pp，3-seed 均值口径）；
    全部 4 个 OOD 组反而更好——但这是在 3-seed 均值下成立，seed-7 单模型上四组均略差。**
    那个「106 pp」在项目里**没有任何产物支撑**，复现 6 种路径全部失败，
    疑似早期某次非脚本化试跑的测量失误。**论文已改用真实数字并改口径。**

    改用 float16 的**真实理由不是精度，而是体积**：
    `GRU(unroll=True)` 展开 30 步、权重张量复制 30 份，int8 无法共享压缩，
    导致 **int8 模型 205 kB vs float16 105 kB（大 1.9 倍）**。
    """
    import tensorflow as tf

    model = tf.keras.models.load_model(keras_path, compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"  PTQ 模型保存: {tflite_path} ({size_kb:.1f} KB)")
    return tflite_path


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def run_one(group: str, seed: int, train_hist, test_hist, tr_ids, val_ids):
    """对一个 (group, seed) 跑 PTQ 并评估。"""
    feats = GROUPS[group]
    tag = f"gru16_{group}_seed{seed}"
    keras_path = os.path.join(MODELS_DIR, f"{tag}.keras")
    if not os.path.exists(keras_path):
        print(f"  ⚠️ 模型不存在: {keras_path}，跳过")
        return None

    # 准备 calibration 数据（train 集随机采样）
    Xtr, _, _ = build_split_arrays(train_hist, tr_ids, feats)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(Xtr), min(N_CALIB, len(Xtr)), replace=False)
    calib_data = Xtr[idx]

    # PTQ 转换
    tflite_path = os.path.join(QUANT_DIR, f"{tag}_ptq_int8.tflite")
    print(f"\n{'='*50}")
    print(f"[{group} seed={seed}] PTQ 量化")
    ptq_convert(keras_path, calib_data, tflite_path)

    # 加载 TFLite interpreter
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # float32 评估
    import tensorflow as keras_tf
    float_model = keras_tf.keras.models.load_model(keras_path, compile=False)
    groups = test_groups(test_hist)
    groups["val(25C)"] = val_ids

    float_results = {}
    quant_results = {}
    for gname, cids in groups.items():
        h = train_hist if gname == "val(25C)" else test_hist
        df_f = evaluate(float_model, h, cids, feats)
        df_q = evaluate_tflite(interpreter, h, cids, feats)
        if df_f.empty:
            continue
        float_results[gname] = {
            "RMSE": df_f.nn_RMSE.mean(),
            "MAE": df_f.nn_MAE.mean(),
            "MAX": df_f.nn_MAX.max(),
        }
        quant_results[gname] = {
            "RMSE": df_q.nn_RMSE.mean(),
            "MAE": df_q.nn_MAE.mean(),
            "MAX": df_q.nn_MAX.max(),
        }

    # 汇总
    rows = []
    for gname in float_results:
        f = float_results[gname]
        q = quant_results[gname]
        rows.append({
            "group": gname,
            "float_RMSE": f["RMSE"],
            "ptq_RMSE": q["RMSE"],
            "delta_RMSE": q["RMSE"] - f["RMSE"],
            "rel_degradation": (q["RMSE"] - f["RMSE"]) / f["RMSE"] * 100
                               if f["RMSE"] > 0 else 0,
            "float_MAE": f["MAE"],
            "ptq_MAE": q["MAE"],
            "float_MAX": f["MAX"],
            "ptq_MAX": q["MAX"],
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(QUANT_DIR, f"{tag}_ptq_compare.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n[{group} seed={seed}] float32 vs PTQ float16 (历史名 int8，实为 f16):")
    print(df[["group", "float_RMSE", "ptq_RMSE", "delta_RMSE", "rel_degradation"]]
          .to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print(f"  保存: {csv_path}")

    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="A2-3", choices=sorted(GROUPS.keys()))
    ap.add_argument("--seed", type=int, default=7, choices=SEEDS)
    ap.add_argument("--all-seeds", action="store_true",
                    help="跑当前 group 的全部 3 个 seed")
    ap.add_argument("--all-groups", action="store_true",
                    help="跑全部 5 组 × seed 7")
    args = ap.parse_args()

    train_hist, test_hist, tr_ids, val_ids = load_all()
    os.makedirs(QUANT_DIR, exist_ok=True)

    if args.all_groups:
        groups = sorted(GROUPS.keys())
        seeds = [args.seed]
    elif args.all_seeds:
        groups = [args.group]
        seeds = SEEDS
    else:
        groups = [args.group]
        seeds = [args.seed]

    all_dfs = []
    for g in groups:
        for s in seeds:
            df = run_one(g, s, train_hist, test_hist, tr_ids, val_ids)
            if df is not None:
                df.insert(0, "exp_group", g)
                df.insert(1, "seed", s)
                all_dfs.append(df)

    if all_dfs:
        summary = pd.concat(all_dfs, ignore_index=True)
        summary_path = os.path.join(QUANT_DIR, "p3_09_ptq_results.csv")
        summary.to_csv(summary_path, index=False)
        print(f"\n{'='*50}")
        print(f"全部结果保存: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
