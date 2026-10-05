# -*- coding: utf-8 -*-
"""
P3-12：真正的全整数量化（Full Integer Quantization）—— 三档精度实测

背景
----
论文 §4.4 声称「full int8 quantization fails ... inflating test RMSE from
0.6 pp to ≈106 pp」，但全项目没有任何 int8 量化产物或 CSV 能支撑这个数字
（唯一出处是 `p3_09_ptq.py` 的代码注释和 `02_日志/2026-09-19.md` 的散文）。
`p3_09_ptq.py::ptq_convert` 实际做的是 float16（`supported_types=[tf.float16]`），
只是文件名被写成 `_ptq_int8.tflite`。

本脚本做一次**真**的全整数量化，用同一套评估口径量出真实数字。

量化档位（同一 float32 Keras 模型、同一评估集、同一口径）
------------------------------------------------------
1. keras_fp32          —— .keras 浮点模型（参考基准）
2. tflite_fp32         —— TFLite 无优化转换（检验转换本身有无损失）
3. tflite_f16          —— float16（`p3_09_ptq.py` 实际做的那一档）
4. int8_dynamic        —— 动态范围量化（仅权重 int8，激活 float32）
5. int8_full           —— **全整数量化**：权重+激活 int8，输入/输出 int8
6. int8_full_f32io     —— 全整数量化，但输入/输出保持 float32
7. int16_act           —— int16 激活 + int8 权重（实验算符集）
8. legacy_int8_209632  —— 项目内那个来历不明的 204.7 KiB int8 模型

评估口径与 `p3_09_ptq.py` 完全一致（复用其 `evaluate` / `evaluate_tflite` 语义）：
- 稳态段（前 300 s 收敛期剔除，由 `nn_features.BURN_IN` 保证）
- 逐文件 RMSE → 分组取文件均值
- 分组：test_id(25C) / ood(10C) / ood(0C) / ood(-10C) / ood(-20C) / val(25C)

代表性数据集（calibration）
---------------------------
**用真实训练集输入特征**，取样方式与 `p3_09_ptq.py` 一致
（`RandomState(42).choice(n, 200, replace=False)`），不是随机数。
额外的 `--n-calib` 可放大校准集做敏感性检查。

中文路径说明
------------
TFLite 的 C++ 层打不开含中文的路径，因此评估时一律用 `model_content=<bytes>`
载入，产物照常写进项目内的中文路径。

用法
----
    python p3_12_int8_quant.py                  # A2-3 seed 7
    python p3_12_int8_quant.py --all-seeds      # A2-3 seed 7/13/42
    python p3_12_int8_quant.py --skip-eval      # 只转换不评估（快）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

# Windows 控制台默认 GBK：本脚本的 ✅/⇒ 是通过 `Log.append` → `print(msg)` 间接打出的，
# 静态扫 `print(...)` 的参数**发现不了** —— 2026-09-27 实测跑到一半才崩。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "06_实验代码", "src"))

from nn_features import GROUPS, build_split_arrays, load_history  # noqa: E402

RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "04_实验", "models")
# 产物落进项目内（不是 E:\SOC_LiteProject）
QUANT_DIR = os.path.join(PROJECT_ROOT, "04_实验", "quantized")

SEEDS = [7, 13, 42]
N_CALIB = 200                      # 与 p3_09_ptq.py 一致
CALIB_SEED = 42                    # 与 p3_09_ptq.py 一致

# ---- 评估档：2026-09-25 统一到 40 文件档 ----------------------------------- #
# 历史问题：本脚本与 p3_09_ptq.py 早先读 `ekf_baseline_history.npz`（8 文件：
# test_id 4 + OOD 每组仅 1 文件），而论文 §6 的 Table 2（2026-09-26 重编号前叫 Table 3）/ §6.2 / §6.3 用的是
# `ekf_full_history.npz`（40 文件：test_id 4 + OOD 每组 9 文件，见
# p5_b2_multifile_ablation.py:45 与 p3_06b_multifile_agg.csv）。
# 两批口径不同 ⇒ §4.5 的量化数字与 §6 的 OOD 数字无法互相对照（−20 °C 差 3 倍）。
# 现统一到 40 文件档：`--archive full40`（默认）。
ARCHIVES = {
    "full40":    "ekf_full_history.npz",      # 40 文件：test_id 4 + OOD 36
    "baseline8": "ekf_baseline_history.npz",  # 8 文件（历史档，仅作回溯/对照用）
}
DEFAULT_ARCHIVE = "full40"
# 产物后缀：40 文件档一律加 _full40，绝不覆盖 8 文件档的已有 CSV / 报告 / tflite
OUT_SUFFIX = {"full40": "_full40", "baseline8": ""}
TFLITE_SUFFIX = {"full40": "_full40", "baseline8": ""}

# 各量化档位对应的产物文件名后缀（文件名如实反映精度）
VARIANT_FILES = {
    "keras_fp32":      None,                                   # 不落 tflite
    "tflite_fp32":     "fp32",
    "tflite_f16":      "f16",
    "int8_dynamic":    "int8_dynamic",
    "int8_full":       "int8_full",
    "int8_full_f32io": "int8_full_f32io",
    "int16_act":       "int16act",
}
LEGACY_PATH = os.path.join(QUANT_DIR, "gru16_A2-3_seed7_ptq_int8.tflite")


# --------------------------------------------------------------------------- #
# 数据（同 p3_09_ptq.py）
# --------------------------------------------------------------------------- #

def load_all(archive: str = DEFAULT_ARCHIVE):
    train_hist = load_history(os.path.join(RESULTS, "ekf_train_history.npz"))
    test_hist = load_history(os.path.join(RESULTS, ARCHIVES[archive]))
    val_ids = [c for c in train_hist if "UDDS" in c]
    tr_ids = [c for c in train_hist if c not in val_ids]
    return train_hist, test_hist, tr_ids, val_ids


def test_groups(test_hist: dict) -> dict[str, list[str]]:
    """分组表达式与 `p5_b2_multifile_ablation.py:27-40`（已修复版）逐字一致。

    ⚠️ 2026-09-25 修复：旧写法 `"0degC_Cycle" in c` 是子串陷阱——
    `"10degC_Cycle_n"` / `"n10degC_Cycle_n"` / `"n20degC_Cycle_n"` 全都含
    `"0degC_Cycle"`，在 40 文件档上会把 16 个文件误划进 0 °C 组（正确为 9 个）。
    必须显式排除正/负 10 与 20 °C。
    """
    ids = sorted(test_hist.keys())
    return {
        "test_id(25C)": [c for c in ids if "25degC_Cycle" in c],
        "ood(10C)":  [c for c in ids if "10degC_" in c and "trise" not in c
                      and "n10degC" not in c and "n20degC" not in c],
        "ood(0C)":   [c for c in ids if "0degC_" in c and "trise" not in c
                      and "10degC" not in c and "20degC" not in c],
        "ood(-10C)": [c for c in ids if "n10degC_" in c and "trise" not in c],
        "ood(-20C)": [c for c in ids if "n20degC_" in c and "trise" not in c],
    }


# --------------------------------------------------------------------------- #
# 评估（口径与 p3_09_ptq.py 的 evaluate / evaluate_tflite 一致）
# --------------------------------------------------------------------------- #

def evaluate_keras(model, hist, cids, feats) -> pd.DataFrame:
    rows = []
    for cid in cids:
        X, y, _ = build_split_arrays(hist, [cid], feats)
        if X is None:
            continue
        y_hat = model.predict(X, verbose=0).ravel()
        e = y - y_hat
        rows.append({"cycle_id": cid, "n": len(y),
                     "RMSE": float(np.sqrt(np.mean(e ** 2))),
                     "MAE": float(np.mean(np.abs(e))),
                     "MAX": float(np.max(np.abs(e)))})
    return pd.DataFrame(rows)


def make_tflite_predictor(interp):
    """返回单样本预测函数，自动处理 int8 输入量化 / 输出反量化。"""
    ind = interp.get_input_details()[0]
    outd = interp.get_output_details()[0]
    in_dtype, out_dtype = ind["dtype"], outd["dtype"]
    in_scale, in_zp = ind["quantization"]
    out_scale, out_zp = outd["quantization"]

    def predict_one(x_float: np.ndarray) -> float:
        x = x_float.astype(np.float32)
        if in_dtype == np.int8:
            q = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
        elif in_dtype == np.int16:
            q = np.clip(np.round(x / in_scale + in_zp), -32768, 32767).astype(np.int16)
        else:
            q = x.astype(in_dtype)
        interp.set_tensor(ind["index"], q)
        interp.invoke()
        o = np.asarray(interp.get_tensor(outd["index"]),
                       dtype=np.float32).ravel()[0]
        if out_dtype in (np.int8, np.int16):
            return float((o - out_zp) * out_scale)
        return float(o)

    return predict_one


def evaluate_tflite(interp, hist, cids, feats) -> pd.DataFrame:
    predict_one = make_tflite_predictor(interp)
    rows = []
    for cid in cids:
        X, y, _ = build_split_arrays(hist, [cid], feats)
        if X is None:
            continue
        preds = np.array([predict_one(X[i:i + 1]) for i in range(len(X))])
        e = y - preds
        rows.append({"cycle_id": cid, "n": len(y),
                     "RMSE": float(np.sqrt(np.mean(e ** 2))),
                     "MAE": float(np.mean(np.abs(e))),
                     "MAX": float(np.max(np.abs(e)))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 量化转换
# --------------------------------------------------------------------------- #

def build_calib(Xtr: np.ndarray, n_calib: int) -> np.ndarray:
    """真实训练集输入特征，取样方式与 p3_09_ptq.py 完全一致（非随机数）。"""
    rng = np.random.RandomState(CALIB_SEED)
    idx = rng.choice(len(Xtr), min(n_calib, len(Xtr)), replace=False)
    return Xtr[idx]


def _rep_dataset(calib: np.ndarray):
    def gen():
        for i in range(len(calib)):
            yield [calib[i:i + 1].astype(np.float32)]
    return gen


def convert_variant(tf, model, name: str, calib: np.ndarray) -> bytes:
    """按档位名转换，返回 tflite flatbuffer 字节。转换失败抛异常（由调用方记录）。"""
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    if name == "tflite_fp32":
        pass
    elif name == "tflite_f16":
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.target_spec.supported_types = [tf.float16]
    elif name == "int8_dynamic":
        # 动态范围量化：仅权重 int8，激活在推理时动态量化
        c.optimizations = [tf.lite.Optimize.DEFAULT]
    elif name in ("int8_full", "int8_full_f32io"):
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = _rep_dataset(calib)
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if name == "int8_full":
            c.inference_input_type = tf.int8
            c.inference_output_type = tf.int8
        else:
            c.inference_input_type = tf.float32
            c.inference_output_type = tf.float32
    elif name == "int16_act":
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = _rep_dataset(calib)
        c.target_spec.supported_ops = [
            tf.lite.OpsSet.
            EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    else:
        raise ValueError(name)
    return c.convert()


def describe(interp) -> dict:
    from collections import Counter
    td = interp.get_tensor_details()
    c = Counter(np.dtype(t["dtype"]).name for t in td)
    ind = interp.get_input_details()[0]
    outd = interp.get_output_details()[0]
    return {
        "n_tensors": len(td),
        "tensor_dtypes": dict(c),
        "input_dtype": np.dtype(ind["dtype"]).name,
        "output_dtype": np.dtype(outd["dtype"]).name,
        "input_quant": [float(v) for v in ind["quantization"]],
        "output_quant": [float(v) for v in outd["quantization"]],
    }


class Log(list):
    """边追加边打印，便于长任务观察进度。"""

    def append(self, msg):
        super().append(msg)
        print(msg, flush=True)


def load_interp(tf, path: str):
    """用 model_content 载入，绕开 TFLite C++ 层对中文路径的限制。"""
    with open(path, "rb") as f:
        blob = f.read()
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    return it, len(blob)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def run_one(group: str, seed: int, train_hist, test_hist, tr_ids, val_ids,
            n_calib: int, skip_eval: bool, log: list[str],
            archive: str = DEFAULT_ARCHIVE):
    import tensorflow as tf

    feats = GROUPS[group]
    tag = f"gru16_{group}_seed{seed}"
    keras_path = os.path.join(MODELS_DIR, f"{tag}.keras")
    if not os.path.exists(keras_path):
        log.append(f"⚠️ 模型不存在: {keras_path}，跳过")
        return None, None

    Xtr, _, _ = build_split_arrays(train_hist, tr_ids, feats)
    calib = build_calib(Xtr, n_calib)
    log.append(f"[{tag}] 训练样本 {Xtr.shape}，校准样本 {calib.shape} "
               f"（真实特征，RandomState({CALIB_SEED})，与 p3_09 同法）")
    log.append(f"  校准集逐特征 min/max: "
               + ", ".join(f"{a:.3f}/{b:.3f}"
                           for a, b in zip(calib.min(axis=(0, 1)),
                                           calib.max(axis=(0, 1)))))

    model = tf.keras.models.load_model(keras_path, compile=False)
    log.append(f"  参数量 {model.count_params()}，输入 {model.input_shape}")

    os.makedirs(QUANT_DIR, exist_ok=True)
    groups = test_groups(test_hist)
    groups["val(25C)"] = val_ids

    # ---- 转换 -------------------------------------------------------------
    interps: dict[str, object] = {}
    sizes: dict[str, int] = {}
    meta: dict[str, dict] = {}
    failures: dict[str, str] = {}

    for name, suffix in VARIANT_FILES.items():
        if suffix is None:
            continue
        t0 = time.time()
        try:
            blob = convert_variant(tf, model, name, calib)
        except Exception as e:  # noqa: BLE001 —— 如实记录失败原因
            failures[name] = f"{type(e).__name__}: {str(e)[:500]}"
            log.append(f"  ❌ {name} 转换失败: {failures[name]}")
            continue
        out_path = os.path.join(
            QUANT_DIR, f"{tag}_{suffix}{TFLITE_SUFFIX[archive]}.tflite")
        with open(out_path, "wb") as f:
            f.write(blob)
        it, nbytes = load_interp(tf, out_path)
        d = describe(it)
        interps[name] = it
        sizes[name] = nbytes
        meta[name] = d
        log.append(f"  ✅ {name:16s} {nbytes:>8,} B ({nbytes/1024:7.1f} KB) "
                   f"io={d['input_dtype']}/{d['output_dtype']} "
                   f"int8张量={d['tensor_dtypes'].get('int8', 0)} "
                   f"[{time.time()-t0:.1f}s]")

    # 项目内那个来历不明的 int8 模型（只在 A2-3/seed7 有）
    if tag == "gru16_A2-3_seed7" and os.path.exists(LEGACY_PATH):
        it, nbytes = load_interp(tf, LEGACY_PATH)
        d = describe(it)
        interps["legacy_int8_209632"] = it
        sizes["legacy_int8_209632"] = nbytes
        meta["legacy_int8_209632"] = d
        log.append(f"  ✅ legacy_int8_209632 {nbytes:>8,} B "
                   f"io={d['input_dtype']}/{d['output_dtype']} "
                   f"int8张量={d['tensor_dtypes'].get('int8', 0)}")

    if skip_eval:
        return None, None

    # ---- 评估 -------------------------------------------------------------
    long_rows = []
    for name, interp in [("keras_fp32", None)] + list(interps.items()):
        for gname, cids in groups.items():
            h = train_hist if gname == "val(25C)" else test_hist
            df = (evaluate_keras(model, h, cids, feats) if interp is None
                  else evaluate_tflite(interp, h, cids, feats))
            if df.empty:
                continue
            long_rows.append({
                "exp_group": group, "seed": seed, "model": name,
                "group": gname, "n_files": len(df), "n_samples": int(df.n.sum()),
                "RMSE": df.RMSE.mean(), "MAE": df.MAE.mean(),
                "MAX": df.MAX.max(),
                "size_bytes": sizes.get(name, np.nan),
            })
        log.append(f"  评估完成: {name}")

    long_df = pd.DataFrame(long_rows)
    wide = long_df.pivot_table(index="group", columns="model", values="RMSE")
    order = [c for c in ["keras_fp32"] + list(VARIANT_FILES)
             + ["legacy_int8_209632"] if c in wide.columns]
    order = list(dict.fromkeys(order))          # 去重（keras_fp32 出现两次）
    wide = wide[order].reset_index()
    wide.insert(0, "seed", seed)
    wide.insert(0, "exp_group", group)

    extra = {"tag": tag, "sizes": sizes, "meta": meta, "failures": failures,
             "n_calib": n_calib, "archive": archive,
             "n_files_total": len(test_hist)}
    return wide, (long_df, extra, log)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="A2-3", choices=sorted(GROUPS.keys()))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--n-calib", type=int, default=N_CALIB)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--archive", default=DEFAULT_ARCHIVE,
                    choices=sorted(ARCHIVES),
                    help="评估档：full40=40 文件档（默认，与 §6 同口径）；"
                         "baseline8=8 文件档（历史档）")
    args = ap.parse_args()

    seeds = SEEDS if args.all_seeds else [args.seed]
    train_hist, test_hist, tr_ids, val_ids = load_all(args.archive)

    wides, longs, extras, log = [], [], [], Log()
    log.append(f"评估档 = {args.archive} → {ARCHIVES[args.archive]}"
               f"（{len(test_hist)} 文件）")
    for s in seeds:
        w, rest = run_one(args.group, s, train_hist, test_hist, tr_ids,
                          val_ids, args.n_calib, args.skip_eval, log,
                          archive=args.archive)
        if w is not None:
            wides.append(w)
            longs.append(rest[0])
            extras.append(rest[1])

    if not wides:
        print("\n[无评估结果]（--skip-eval 或全部失败）")
        return 0

    wide_all = pd.concat(wides, ignore_index=True)
    long_all = pd.concat(longs, ignore_index=True)

    os.makedirs(RESULTS, exist_ok=True)
    sfx = OUT_SUFFIX[args.archive]           # full40 档 → "_full40"，不覆盖旧产物
    wide_path = os.path.join(RESULTS, f"p3_12_int8_results{sfx}.csv")
    long_path = os.path.join(RESULTS, f"p3_12_int8_metrics{sfx}.csv")
    wide_all.to_csv(wide_path, index=False, encoding="utf-8-sig")
    long_all.to_csv(long_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print(f"P3-12 全整数量化 —— 逐组 RMSE（pp，稳态，文件均值）"
          f"[评估档 {args.archive} = {ARCHIVES[args.archive]}]")
    print("=" * 78)
    print(wide_all.to_string(index=False,
                             float_format=lambda v: f"{v:8.4f}"))

    print("\n模型大小 / 张量构成：")
    for e in extras:
        if e["tag"] != "gru16_A2-3_seed7":
            continue
        for k, v in e["sizes"].items():
            print(f"  {k:20s} {v:>8,} B ({v/1024:7.1f} KB)  "
                  f"{e['meta'][k]['tensor_dtypes']} io="
                  f"{e['meta'][k]['input_dtype']}/{e['meta'][k]['output_dtype']}")
        if e["failures"]:
            print("  转换失败的档位：")
            for k, v in e["failures"].items():
                print(f"    {k}: {v}")

    # 机器可读的元信息（论文/报告引用）
    meta_path = os.path.join(RESULTS, f"p3_12_int8_meta{sfx}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(extras, f, ensure_ascii=False, indent=2)

    print(f"\n保存: {wide_path}")
    print(f"      {long_path}")
    print(f"      {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
