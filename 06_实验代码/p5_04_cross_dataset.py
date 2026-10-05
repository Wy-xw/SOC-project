# -*- coding: utf-8 -*-
"""
P5-04 跨数据集泛化：McMaster 训练 → NASA PCoE 验证

设计
----
- 训练好的 EKF 参数（McMaster NCA）和 NN 模型直接用于 NASA LiCoO2 数据
- 不做任何适配/微调 → 零样本迁移，最严格的泛化测试
- 评估：EKF alone vs EKF+NN A2-3（完整特征）

数据特点差异
-----------
- McMaster: NCA（Li(NiCoAl)O₂）, 2.9 Ah, 动态工况(US06/HWFET/UDDS/LA92), 5 温度点
- NASA:     LiCoO2 钴酸锂, 2.0 Ah, 恒流 2A 放电, 25°C/4°C 室温

产出
----
- results/p5_04_cross_dataset.csv
- stdout 汇总

用法
----
    "D:/venvs/soc-paper/Scripts/python.exe" 10_代码/p5_04_cross_dataset.py
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
    BURN_IN, GROUPS, WINDOW_L, FEATURE_TRANSFORMS, SAFETY_CLIP,
    build_split_arrays,
)
from ecm_ekf import ECMParams, SOC_EKF, metrics  # noqa: E402
from ecm_tables import OCVTable, ECMParamTable, make_ekf_table_fns  # noqa: E402
from data_io import load_nasa_battery, list_nasa_discharges, build_nasa_frame  # noqa: E402

RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "04_实验", "models")
NASA_ROOT = os.path.join(PROJECT_ROOT, "04_实验", "数据", "raw", "nasa")

SOC0_BIAS = 0.10  # 与 McMaster 评估口径一致


# --------------------------------------------------------------------------- #
# 1. 在 NASA 数据上跑 EKF（用 McMaster 训练的参数）
# --------------------------------------------------------------------------- #

def run_ekf_on_nasa(df: pd.DataFrame, cycle_id: str) -> dict | None:
    """对一个 NASA 放电循环跑 EKF，返回 history dict。"""
    ocv_tab = OCVTable.from_csv()
    ptab = ECMParamTable.from_csv()
    ocv_fn, docv_fn, params_fn = make_ekf_table_fns(ocv_tab, ptab)

    i = df.current_A.to_numpy()
    v = df.voltage_V.to_numpy()
    t = df.temperature_C.to_numpy()
    s = df.soc_true.to_numpy()
    n = len(i)

    if n < 100:
        return None

    p = ECMParams(Cn=2.9, dt=1.0)  # 用 McMaster 的 Cn=2.9（故意不适配）
    soc0 = float(np.clip(s[0] + SOC0_BIAS, 0.0, 1.0))
    ekf = SOC_EKF(
        p, soc0=soc0, ocv_fn=ocv_fn, docv_fn=docv_fn, params_fn=params_fn)
    for k in range(n):
        ekf.step(float(i[k]), float(v[k]), temp_k=float(t[k]))

    h = ekf.history
    return {
        "timestamp": df.timestamp.to_numpy(),
        "soc_true": s,
        "current_A": i,
        "temperature_C": t,
        "voltage_V": v,
        "soc": np.array(h["soc"]),
        "nu": np.array(h["nu"]),
        "S": np.array(h["S"]),
        "nis": np.array(h["nis"]),
        "K0": np.array(h["K0"]),
        "K1": np.array(h["K1"]),
        "P00": np.array(h["P00"]),
        "P11": np.array(h["P11"]),
        "v1": np.array(h["v1"]),
        "vt_pred": np.array(h["vt_pred"]),
    }


# --------------------------------------------------------------------------- #
# 2. 评估
# --------------------------------------------------------------------------- #

def eval_ekf_only(hist: dict, cycle_id: str) -> dict | None:
    """评估 EKF 单独的 SOC 误差。"""
    n = len(hist["nu"])
    if n < BURN_IN + WINDOW_L:
        return None
    soc_est = hist["soc"]
    soc_true = hist["soc_true"]
    ss = slice(BURN_IN, n)
    e = (soc_est[ss] - soc_true[ss]) * 100.0
    return {
        "cycle_id": cycle_id,
        "method": "EKF",
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE": float(np.mean(np.abs(e))),
        "MAX": float(np.max(np.abs(e))),
        "n_samples": int(n - BURN_IN),
    }


def eval_ekf_nn(model, hist: dict, cycle_id: str, feats: list[str]) -> dict | None:
    """评估 EKF+NN 融合的 SOC 误差。"""
    n = len(hist["nu"])
    if n < BURN_IN + WINDOW_L + 1:
        return None
    # 构造滑窗特征
    F = np.stack(
        [np.clip(FEATURE_TRANSFORMS[f](hist), -SAFETY_CLIP, SAFETY_CLIP)
         for f in feats], axis=1).astype(np.float32)
    y_all = (hist["soc"] - hist["soc_true"]) * 100.0

    starts = np.arange(BURN_IN, n - WINDOW_L + 1)
    idx = starts[:, None] + np.arange(WINDOW_L)[None, :]
    X = np.take(F, idx, axis=0)
    ends = starts + WINDOW_L - 1
    y = y_all[ends].astype(np.float32)

    y_hat = model.predict(X, verbose=0).ravel()
    e = y - y_hat
    return {
        "cycle_id": cycle_id,
        "method": "EKF+NN_A2-3",
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE": float(np.mean(np.abs(e))),
        "MAX": float(np.max(np.abs(e))),
        "n_samples": len(y),
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main() -> int:
    from tensorflow import keras

    print("=" * 70)
    print("P5-04 跨数据集泛化：McMaster -> NASA PCoE")
    print("=" * 70)

    # 加载模型
    model_path = os.path.join(MODELS_DIR, f"gru16_A2-3_L{WINDOW_L}.keras")
    model = keras.models.load_model(model_path, compile=False)
    feats = GROUPS["A2-3"]
    print(f"模型: A2-3 ({len(feats)} 特征)")

    # 遍历 NASA 电池
    mat_files = sorted([f for f in os.listdir(NASA_ROOT) if f.endswith(".mat")])
    print(f"NASA 文件: {len(mat_files)} 个")

    # 只评估前 4 个电池（B0005-B0018，文献中最常用的）
    eval_batteries = ["B0005", "B0006", "B0007", "B0018"]
    mat_files = [f for f in mat_files
                 if any(f.startswith(b) for b in eval_batteries)]
    print(f"评估电池: {eval_batteries} ({len(mat_files)} 文件)")

    all_rows = []
    for fn in mat_files:
        path = os.path.join(NASA_ROOT, fn)
        battery = fn.replace(".mat", "")
        try:
            discharges = list_nasa_discharges(path)
        except Exception as e:
            print(f"  {battery}: 加载失败 - {e}")
            continue

        # 选 3 个有代表性的循环：早期(0)、中期(len//2)、晚期(-2)
        n_dis = len(discharges)
        if n_dis < 3:
            indices = list(range(n_dis))
        else:
            indices = [0, n_dis // 2, -2]

        for idx in indices:
            cycle_id = f"{battery}_dis{idx:03d}"
            try:
                df = build_nasa_frame(path, idx, dt=1.0)
            except Exception as e:
                print(f"  {cycle_id}: 构造失败 - {e}")
                continue

            # 跑 EKF
            hist = run_ekf_on_nasa(df, cycle_id)
            if hist is None:
                print(f"  {cycle_id}: 太短，跳过")
                continue

            # EKF 单独
            r_ekf = eval_ekf_only(hist, cycle_id)
            if r_ekf:
                r_ekf["battery"] = battery
                r_ekf["cycle_index"] = idx
                r_ekf["temp_C"] = float(discharges[idx].ambient_temperature)
                r_ekf["capacity_Ah"] = float(
                    np.asarray(discharges[idx].data.Capacity).ravel()[0])
                all_rows.append(r_ekf)

            # EKF+NN
            r_nn = eval_ekf_nn(model, hist, cycle_id, feats)
            if r_nn:
                r_nn["battery"] = battery
                r_nn["cycle_index"] = idx
                r_nn["temp_C"] = float(discharges[idx].ambient_temperature)
                r_nn["capacity_Ah"] = float(
                    np.asarray(discharges[idx].data.Capacity).ravel()[0])
                all_rows.append(r_nn)

            if r_ekf and r_nn:
                print(f"  {cycle_id}: EKF={r_ekf['RMSE']:.2f} pp -> "
                      f"NN={r_nn['RMSE']:.2f} pp "
                      f"(cap={r_ekf['capacity_Ah']:.2f} Ah, "
                      f"T={r_ekf['temp_C']:.0f}C)")

    # 保存
    df_all = pd.DataFrame(all_rows)
    csv_path = os.path.join(RESULTS, "p5_04_cross_dataset.csv")
    df_all.to_csv(csv_path, index=False)

    # 汇总
    print("\n" + "=" * 70)
    print("汇总（按电池分组）")
    print("=" * 70)
    if not df_all.empty:
        pivot = df_all.pivot_table(
            index="battery", columns="method",
            values=["RMSE", "MAE", "MAX"],
            aggfunc="mean")
        print(pivot.round(3).to_string())

        # 总体对比
        print("\n总体对比:")
        summary = df_all.groupby("method").agg(
            RMSE_mean=("RMSE", "mean"),
            MAE_mean=("MAE", "mean"),
            MAX_mean=("MAX", "mean"),
            n=("cycle_id", "count")).round(3)
        print(summary.to_string())
    else:
        print("无有效结果")

    print(f"\n保存: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
