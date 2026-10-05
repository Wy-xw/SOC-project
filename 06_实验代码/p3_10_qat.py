# -*- coding: utf-8 -*-
r"""
P3-10：量化感知训练（QAT）—— 在训练中插入伪量化节点，再转 int8 TFLite

🔴 执行状态（2026-09-25 核实，P0-25）
=====================================
**本脚本从未执行成功，无任何产物；论文不引用其任何数字。**

- 本脚本声称的 3 个产物全项目不存在：
  `gru16_<group>_seed<seed>_qat_f16.tflite`、`*_qat_compare.csv`、
  `p3_10_qat_results.csv`。输出目录 `E:\SOC_LiteProject\quantized\`
  **最后修改时间 2026-09-19 22:18**，其中只有 P3-09 的 PTQ 产物
  （`*_ptq_int8.tflite` / `*_ptq_compare.csv` / `p3_09_ptq_results.csv`）
  与 P3-11 报告 —— 说明本脚本从未跑到落盘那一步。
- **在当前环境（TF 2.21.0 + TFMOT 0.8.1）无法运行**。实测根因不在
  `quantize_annotate_layer`（那是通的），而在 `qat_train_and_convert()`
  里第 186 行：
      `float_model = tf.keras.models.load_model(keras_path, compile=False)`
  一旦先 `import tensorflow_model_optimization`，`tf.keras` 会被解析成
  `tf_keras`（Keras 2 兼容层），而 `04_实验/models/*.keras` 是 **Keras 3**
  存的（config 里 `module: keras.src.models.functional`），于是报
      TypeError: Could not deserialize class 'Functional' because its parent
      module tf_keras.src.models.functional cannot be imported.
  对照：`p3_09_ptq.py` 不导入 TFMOT，`tf.keras` 解析为 Keras 3，同样一行
  `load_model` 可以正常工作（PTQ 产物确已生成）。
- **本脚本的 docstring 第 172–177 行已自相矛盾**：注释说"从头建模型（避免加载
  兼容性问题）"，但第 186 行仍然加载 `.keras` 文件 —— 那正是失败点。
- 跳过决定（见 `09_任务清单.md` P3-10）：**float16 PTQ 已满足论文需求**
  （P3-12 int8 全整数量化另已真跑，见 `p3_12_int8_results.csv`），
  故不为凑数补跑 QAT。论文正文/图表**没有任何一处引用 QAT 数字**。
- 若日后要启用：需把第 186 行（及 254 行评估侧）的加载换成 Keras 3
  （如 `import keras; keras.models.load_model(...)`）或改用
  `tf_keras` 存/取模型，并重新验证 QAT 链路后再落盘。

与 PTQ 的区别
-------------
- PTQ：训练后直接量化，精度损失可能较大（尤其小模型）
- QAT：训练时模拟量化误差，网络学会适应 → 精度损失通常更小
- 本脚本：先 QAT 微调（低学习率），再转 int8 TFLite

流程
----
1. 加载 float32 预训练模型（A2-3 seed 7）
2. 注入 QAT 伪量化节点（TFMOT）
3. 低学习率微调 20-30 epoch
4. 转 int8 TFLite（与 PTQ 同口径）
5. 评估对比

用法
----
    python p3_10_qat.py                    # A2-3 seed 7
    python p3_10_qat.py --group A2-0       # 指定组
    python p3_10_qat.py --all-seeds        # A2-3 全 seed
"""
from __future__ import annotations

import argparse
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

# Windows 控制台默认 GBK，而本脚本 print 含 GBK 无码位的字符 —— 不改编码会在
# **成功路径**崩掉（活干完了却退出码 1）。别靠 PYTHONIOENCODING 兜底。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "04_实验", "models")
# TFLite 路径不能含中文，用绝对 ASCII 路径
QUANT_DIR = r"E:\SOC_LiteProject\quantized"
SEEDS = [7, 13, 42]
N_CALIB = 200


# --------------------------------------------------------------------------- #
# 数据（同 p3_09）
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
# 评估（同 p3_09）
# --------------------------------------------------------------------------- #

def evaluate_keras(model, hist, cids, feats):
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
# QAT 核心
# --------------------------------------------------------------------------- #

def qat_train_and_convert(keras_path: str, Xtr, ytr, Xva, yva,
                          tflite_path: str, qat_epochs: int = 30,
                          n_feats: int = 13, window_l: int = 30):
    """QAT 微调 + 转 float16 TFLite。

    由于 TFMOT 与 TF 2.21 的模型加载兼容性问题，改为：
    1. 从头建 QAT 模型（相同架构 + 伪量化注解）
    2. 加载 float32 权重做初始化
    3. 低学习率微调
    4. 转 float16 TFLite
    """
    import tensorflow as tf
    import tensorflow_model_optimization as tfmot

    # 从头建模型（避免加载兼容性问题）
    inp = tf.keras.Input(shape=(window_l, n_feats))
    x = tfmot.quantization.keras.quantize_annotate_layer(
        tf.keras.layers.GRU(16, unroll=True)
    )(inp)
    x = tfmot.quantization.keras.quantize_annotate_layer(
        tf.keras.layers.Dense(8, activation="relu")
    )(x)
    out = tfmot.quantization.keras.quantize_annotate_layer(
        tf.keras.layers.Dense(1)
    )(x)
    qat_model = tf.keras.Model(inp, out)

    # 加载 float32 权重
    float_model = tf.keras.models.load_model(keras_path, compile=False)
    qat_model.set_weights(float_model.get_weights())

    qat_model = tfmot.quantization.keras.quantize_apply(qat_model)
    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="mse",
        metrics=["mae"]
    )

    print(f"  QAT 模型参数量: {qat_model.count_params()}")

    cbs = [
        tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor="val_loss"),
    ]
    qat_model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=qat_epochs,
        batch_size=256,
        callbacks=cbs,
        verbose=2
    )

    # 转 float16 TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"  QAT 模型保存: {tflite_path} ({size_kb:.1f} KB)")
    return tflite_path


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def run_one(group: str, seed: int, train_hist, test_hist, tr_ids, val_ids):
    feats = GROUPS[group]
    tag = f"gru16_{group}_seed{seed}"
    keras_path = os.path.join(MODELS_DIR, f"{tag}.keras")
    if not os.path.exists(keras_path):
        print(f"  ⚠️ 模型不存在: {keras_path}，跳过")
        return None

    # 准备数据
    Xtr, ytr, _ = build_split_arrays(train_hist, tr_ids, feats)
    Xva, yva, _ = build_split_arrays(train_hist, val_ids, feats)

    # QAT 训练 + 转换
    tflite_path = os.path.join(QUANT_DIR, f"{tag}_qat_f16.tflite")
    print(f"\n{'='*50}")
    print(f"[{group} seed={seed}] QAT 量化")
    qat_train_and_convert(keras_path, Xtr, ytr, Xva, yva, tflite_path,
                          n_feats=len(feats), window_l=WINDOW_L)

    # 加载 TFLite interpreter
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # float32 评估
    float_model = tf.keras.models.load_model(keras_path, compile=False)
    groups = test_groups(test_hist)
    groups["val(25C)"] = val_ids

    float_results = {}
    quant_results = {}
    for gname, cids in groups.items():
        h = train_hist if gname == "val(25C)" else test_hist
        df_f = evaluate_keras(float_model, h, cids, feats)
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

    rows = []
    for gname in float_results:
        f = float_results[gname]
        q = quant_results[gname]
        rows.append({
            "group": gname,
            "float_RMSE": f["RMSE"],
            "qat_RMSE": q["RMSE"],
            "delta_RMSE": q["RMSE"] - f["RMSE"],
            "rel_degradation": (q["RMSE"] - f["RMSE"]) / f["RMSE"] * 100
                               if f["RMSE"] > 0 else 0,
            "float_MAE": f["MAE"],
            "qat_MAE": q["MAE"],
            "float_MAX": f["MAX"],
            "qat_MAX": q["MAX"],
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(QUANT_DIR, f"{tag}_qat_compare.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n[{group} seed={seed}] float32 vs QAT int8:")
    print(df[["group", "float_RMSE", "qat_RMSE", "delta_RMSE", "rel_degradation"]]
          .to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print(f"  保存: {csv_path}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="A2-3", choices=sorted(GROUPS.keys()))
    ap.add_argument("--seed", type=int, default=7, choices=SEEDS)
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--all-groups", action="store_true")
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
        summary_path = os.path.join(QUANT_DIR, "p3_10_qat_results.csv")
        summary.to_csv(summary_path, index=False)
        print(f"\n{'='*50}")
        print(f"全部结果保存: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
