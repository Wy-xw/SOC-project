# -*- coding: utf-8 -*-
# ⛔⛔ **2026-09-27 起：本脚本的三条输出全部作废，请勿再跑。**
#    官方生产者 = `10_代码/p5_03c_robustness_unified.py`（统一暖态口径 c′，见 09_任务清单.md P5-10）。
#    本脚本的问题（逐条）：
#      ① `p5_03_initial_bias.csv` —— 满电起步，正向半区被 [0,1] clip 吞掉（P0-24），早已作废；
#      ② `p5_03_noise.csv` / `p5_03_sampling_rate.csv` —— 满电起步且**评估全段**，
#         与偏差扫口径不同；2026-09-27 查明统一口径应为「截断 + 热态」（评估 [k0:]、
#         滤波器 V₁/P 继承收敛值），本脚本不产出该口径。
#    旧产物留档：`*.bak_20260927_preunified`。
#
# ⚠️ 2026-09-26 修：以上警告**必须用 # 注释**，不能写成裸字符串——
#    裸字符串会占用模块 docstring 的位置，使下面那句 docstring 变成普通语句，
#    于是 `from __future__ import annotations` 前出现语句 → SyntaxError。
"""
P5-03 Layer 3：鲁棒性测试

测试维度
--------
1. 噪声敏感性：在端电压上加高斯白噪声（σ = 5/10/20 mV），评估 RMSE 变化
2. 采样率降级：从 1 Hz 降采样到 0.5 Hz / 0.2 Hz，评估 RMSE 变化
3. 初始 SOC 偏差：±5/±10/±20 pp 偏差，评估收敛速度与稳态误差

基准模型：EKF+NN A2-3（完整特征，seed=7）
评估集：test_id(25°C) 4 文件 + OOD 代表性文件

用法
----
    "D:/venvs/soc-paper/Scripts/python.exe" 10_代码/p5_03_robustness.py
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
    build_split_arrays, load_history,
)
from ecm_ekf import ECMParams, SOC_EKF, metrics  # noqa: E402
from ecm_tables import OCVTable, ECMParamTable, make_ekf_table_fns  # noqa: E402

RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "04_实验", "models")
P5_DIR = RESULTS

SEED = 7
SOC0_BIAS = 0.10  # 默认初始偏差


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #

def load_full_history():
    """加载全量 EKF history。"""
    full_path = os.path.join(RESULTS, "ekf_full_history.npz")
    return load_history(full_path)


def load_model():
    from tensorflow import keras
    # 与 P5-01/fig7/fig8 同族：seed7 模型（原 L30 模型已弃用，口径统一）
    path = os.path.join(MODELS_DIR, "gru16_A2-3_seed7.keras")
    return keras.models.load_model(path, compile=False)


def eval融合(model, hist, cids, feats):
    """评估 EKF+NN 融合，返回 (mean_RMSE, mean_MAE, mean_MAX)。"""
    rmses, maes, maxs = [], [], []
    for cid in cids:
        X, y, _ = build_split_arrays(hist, [cid], feats)
        if X is None:
            continue
        y_hat = model.predict(X, verbose=0).ravel()
        e = y - y_hat
        rmses.append(float(np.sqrt(np.mean(e**2))))
        maes.append(float(np.mean(np.abs(e))))
        maxs.append(float(np.max(np.abs(e))))
    if not rmses:
        return 0, 0, 0
    return np.mean(rmses), np.mean(maes), np.max(maxs)


# --------------------------------------------------------------------------- #
# 1. 噪声敏感性
# --------------------------------------------------------------------------- #

def noise_sensitivity(model, hist, cids, feats):
    """在端电压上加高斯白噪声，重新跑 EKF + NN 融合。"""
    ocv_tab = OCVTable.from_csv()
    ptab = ECMParamTable.from_csv()
    ocv_fn, docv_fn, params_fn = make_ekf_table_fns(ocv_tab, ptab)

    noise_stds = [0.0, 0.005, 0.010, 0.020]  # V
    rows = []
    for sigma in noise_stds:
        rmses = []
        for cid in cids:
            h = hist[cid]
            n = len(h["nu"])
            if n < BURN_IN + WINDOW_L + 1:
                continue
            i_orig = h["current_A"]
            v_orig = h["voltage_V"]
            t_prof = h["temperature_C"]
            soc_true = h["soc_true"]

            # 加噪声
            rng = np.random.RandomState(42)
            v_noisy = v_orig + rng.normal(0, sigma, n)

            # 重跑 EKF
            p = ECMParams(Cn=2.9, dt=1.0)
            soc0 = float(np.clip(soc_true[0] + SOC0_BIAS, 0.0, 1.0))
            ekf = SOC_EKF(
                p, soc0=soc0, ocv_fn=ocv_fn, docv_fn=docv_fn,
                params_fn=params_fn)
            for k in range(n):
                ekf.step(float(i_orig[k]), float(v_noisy[k]),
                         temp_k=float(t_prof[k]))

            # 构造新 history（用加噪后的电压）
            new_h = dict(h)
            new_h["voltage_V"] = v_noisy
            new_h["nu"] = np.array(ekf.history["nu"])
            new_h["S"] = np.array(ekf.history["S"])
            new_h["nis"] = np.array(ekf.history["nis"])
            new_h["K0"] = np.array(ekf.history["K0"])
            new_h["K1"] = np.array(ekf.history["K1"])
            new_h["P00"] = np.array(ekf.history["P00"])
            new_h["P11"] = np.array(ekf.history["P11"])
            new_h["soc"] = np.array(ekf.history["soc"])

            X, y, _ = build_split_arrays({cid: new_h}, [cid], feats)
            if X is None:
                continue
            y_hat = model.predict(X, verbose=0).ravel()
            e = y - y_hat
            rmses.append(float(np.sqrt(np.mean(e**2))))

        rmse_mean = np.mean(rmses) if rmses else 0
        rows.append({
            "noise_std_mV": sigma * 1000,
            "RMSE_pp": rmse_mean,
        })
        print(f"  noise={sigma*1000:.0f} mV -> RMSE={rmse_mean:.3f} pp")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. 采样率降级
# --------------------------------------------------------------------------- #

def sampling_rate_test(model, hist, cids, feats):
    """降采样后评估（跳点取均值模拟低采样率）。"""
    rates = [1.0, 0.5, 0.2]  # Hz
    rows = []
    for rate in rates:
        skip = max(1, int(1.0 / rate))
        rmses = []
        for cid in cids:
            h = hist[cid]
            n = len(h["nu"])
            if n < BURN_IN + WINDOW_L + 1:
                continue
            # 降采样
            idx = np.arange(0, n, skip)
            ds_h = {k: v[idx] if isinstance(v, np.ndarray) else v
                    for k, v in h.items()}
            # 重跑 EKF（降采样后 dt 变了）
            ocv_tab = OCVTable.from_csv()
            ptab = ECMParamTable.from_csv()
            ocv_fn, docv_fn, params_fn = make_ekf_table_fns(ocv_tab, ptab)
            p = ECMParams(Cn=2.9, dt=1.0 / rate)
            soc0 = float(np.clip(h["soc_true"][0] + SOC0_BIAS, 0.0, 1.0))
            ekf = SOC_EKF(
                p, soc0=soc0, ocv_fn=ocv_fn, docv_fn=docv_fn,
                params_fn=params_fn)
            for k in range(len(idx)):
                ekf.step(float(h["current_A"][idx[k]]),
                         float(h["voltage_V"][idx[k]]),
                         temp_k=float(h["temperature_C"][idx[k]]))
            ds_h["soc"] = np.array(ekf.history["soc"])
            ds_h["nu"] = np.array(ekf.history["nu"])
            ds_h["S"] = np.array(ekf.history["S"])
            ds_h["nis"] = np.array(ekf.history["nis"])
            ds_h["K0"] = np.array(ekf.history["K0"])
            ds_h["K1"] = np.array(ekf.history["K1"])
            ds_h["P00"] = np.array(ekf.history["P00"])
            ds_h["P11"] = np.array(ekf.history["P11"])

            X, y, _ = build_split_arrays({cid: ds_h}, [cid], feats)
            if X is None:
                continue
            y_hat = model.predict(X, verbose=0).ravel()
            e = y - y_hat
            rmses.append(float(np.sqrt(np.mean(e**2))))

        rmse_mean = np.mean(rmses) if rmses else 0
        rows.append({
            "sampling_rate_Hz": rate,
            "skip_factor": skip,
            "RMSE_pp": rmse_mean,
        })
        print(f"  rate={rate} Hz (skip={skip}) -> RMSE={rmse_mean:.3f} pp")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. 初始 SOC 偏差
# --------------------------------------------------------------------------- #

def initial_soc_deviation(model, hist, cids, feats):
    """测试不同初始 SOC 偏差下的收敛性与稳态误差。"""
    ocv_tab = OCVTable.from_csv()
    ptab = ECMParamTable.from_csv()
    ocv_fn, docv_fn, params_fn = make_ekf_table_fns(ocv_tab, ptab)

    biases = [0.0, 0.05, 0.10, 0.20, -0.05, -0.10, -0.20]
    rows = []
    for bias in biases:
        rmses = []
        for cid in cids:
            h = hist[cid]
            n = len(h["nu"])
            if n < BURN_IN + WINDOW_L + 1:
                continue
            i_arr = h["current_A"]
            v_arr = h["voltage_V"]
            t_arr = h["temperature_C"]
            soc_true = h["soc_true"]

            # 重跑 EKF（不同初始偏差）
            p = ECMParams(Cn=2.9, dt=1.0)
            soc0 = float(np.clip(soc_true[0] + bias, 0.0, 1.0))
            ekf = SOC_EKF(
                p, soc0=soc0, ocv_fn=ocv_fn, docv_fn=docv_fn,
                params_fn=params_fn)
            for k in range(n):
                ekf.step(float(i_arr[k]), float(v_arr[k]),
                         temp_k=float(t_arr[k]))

            new_h = dict(h)
            new_h["soc"] = np.array(ekf.history["soc"])
            new_h["nu"] = np.array(ekf.history["nu"])
            new_h["S"] = np.array(ekf.history["S"])
            new_h["nis"] = np.array(ekf.history["nis"])
            new_h["K0"] = np.array(ekf.history["K0"])
            new_h["K1"] = np.array(ekf.history["K1"])
            new_h["P00"] = np.array(ekf.history["P00"])
            new_h["P11"] = np.array(ekf.history["P11"])

            X, y, _ = build_split_arrays({cid: new_h}, [cid], feats)
            if X is None:
                continue
            y_hat = model.predict(X, verbose=0).ravel()
            e = y - y_hat
            rmses.append(float(np.sqrt(np.mean(e**2))))

        rmse_mean = np.mean(rmses) if rmses else 0
        rows.append({
            "bias_pp": bias * 100,
            "RMSE_pp": rmse_mean,
        })
        print(f"  bias={bias*100:+.0f} pp -> RMSE={rmse_mean:.3f} pp")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main() -> int:
    print("=" * 70)
    print("P5-03 鲁棒性测试")
    print("=" * 70)

    hist = load_full_history()
    model = load_model()
    feats = GROUPS["A2-3"]

    # 选择评估文件：test_id(25°C) + 2 个 10°C 文件
    # 修正（2026-09-22）：原条件 "0degC_" in c 会误匹配 "10degC_"（子串陷阱），
    # 实际池一直含 10°C 文件；现显式选 10°C 并在论文中如实描述。
    test_cids = [c for c in sorted(hist.keys()) if "25degC_Cycle" in c]
    ood_cids = [c for c in sorted(hist.keys())
                if "10degC_" in c and "n10degC" not in c
                and "trise" not in c][:2]  # 取 2 个 10°C 文件
    eval_cids = test_cids + ood_cids
    print(f"评估文件: {len(eval_cids)} 个（test_id {len(test_cids)} + OOD {len(ood_cids)}）")

    # 1. 噪声敏感性
    print("\n--- 1. 噪声敏感性 ---")
    df_noise = noise_sensitivity(model, hist, eval_cids, feats)

    # 2. 采样率降级
    print("\n--- 2. 采样率降级 ---")
    df_rate = sampling_rate_test(model, hist, eval_cids, feats)

    # 3. 初始 SOC 偏差
    print("\n--- 3. 初始 SOC 偏差 ---")
    df_bias = initial_soc_deviation(model, hist, eval_cids, feats)

    # 保存
    df_noise.to_csv(os.path.join(P5_DIR, "p5_03_noise.csv"), index=False)
    df_rate.to_csv(os.path.join(P5_DIR, "p5_03_sampling_rate.csv"), index=False)
    df_bias.to_csv(os.path.join(P5_DIR, "p5_03_initial_bias.csv"), index=False)

    # 汇总
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print("\n噪声敏感性:")
    print(df_noise.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n采样率降级:")
    print(df_rate.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n初始 SOC 偏差:")
    print(df_bias.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\n保存: p5_03_noise.csv / p5_03_sampling_rate.csv / p5_03_initial_bias.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
