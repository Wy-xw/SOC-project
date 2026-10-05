# -*- coding: utf-8 -*-
"""
P5-10 探针：把「噪声」与「采样率」两扫改用**中段起步口径**跑一遍，**只出数字、不改任何现有产出**。

为什么写这个脚本
----------------
Fig. A.1 的三个面板口径不一：面板 (c) 初始偏差扫用**中段起步**
（取首个 `soc_true <= 0.80` 的点起跑，因为满电起步时 ±bias 会被 `[0,1]` clip 吞掉），
而 (a) 噪声、(b) 采样率两扫用的是**满电起步**。三者画在同一张图里，题注只能声明"绝对值不可比"。

用户决定：**先看数字再定**是否把 (a)(b) 也改成中段起步。
所以本脚本按 `p5_03_robustness.py` 的流程**逐字复刻**（同模型、同文件集、同噪声种子、
同降采样方式、同指标），**唯一变量 = 起步口径**，输出写到带 `_middischarge` 后缀的**新文件**。

⚠️ **本脚本不覆盖 `p5_03_noise.csv` / `p5_03_sampling_rate.csv`。**
   要不要正式替换，等看完对照再定。

用法
----
    "D:/venvs/soc-paper/Scripts/python.exe" 10_代码/p5_03c_robustness_middischarge_probe.py
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
from ecm_ekf import ECMParams, SOC_EKF  # noqa: E402
from ecm_tables import OCVTable, ECMParamTable, make_ekf_table_fns  # noqa: E402

# Windows 控制台默认 GBK，而本脚本 print 含 GBK 无码位的字符 —— 不改编码会在
# **成功路径**崩掉（活干完了却退出码 1）。别靠 PYTHONIOENCODING 兜底。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "04_实验", "models")

# 与 p5_03b_initial_bias_fixed.py 完全一致的中段起步阈值
SOC_START = 0.80
NOISE_SEED = 42          # 与 p5_03_robustness.py 的 RandomState(42) 一致
NOISE_STDS = [0.0, 0.005, 0.010, 0.020]
RATES = [1.0, 0.5, 0.2]


def load_model():
    from tensorflow import keras
    return keras.models.load_model(
        os.path.join(MODELS_DIR, "gru16_A2-3_seed7.keras"), compile=False)


def table_fns():
    return make_ekf_table_fns(OCVTable.from_csv(), ECMParamTable.from_csv())


def truncate_at_k0(h: dict, soc_start: float = SOC_START):
    """取首个 soc_true <= soc_start 的点 k0，把 h 的**全部数组字段**从 k0 截断。

    与 p5_03b_initial_bias_fixed.py 的做法一致（那边用 _CUT 白名单，这里全切更保险）。
    """
    st = np.asarray(h["soc_true"])
    hit = np.where(st <= soc_start)[0]
    if len(hit) == 0:
        return None, None
    k0 = int(hit[0])
    if len(st) - k0 < BURN_IN + WINDOW_L + 1:
        return None, None
    trunc = {k: (np.asarray(v)[k0:] if isinstance(v, np.ndarray) else v)
             for k, v in h.items()}
    return trunc, k0


def run_ekf(i_arr, v_arr, t_arr, soc0, dt):
    ocv_fn, docv_fn, params_fn = table_fns()
    ekf = SOC_EKF(ECMParams(Cn=2.9, dt=dt), soc0=soc0,
                  ocv_fn=ocv_fn, docv_fn=docv_fn, params_fn=params_fn)
    for k in range(len(i_arr)):
        ekf.step(float(i_arr[k]), float(v_arr[k]), temp_k=float(t_arr[k]))
    return ekf


def rmse_of(model, feats, cid, hist_like):
    X, y, _ = build_split_arrays({cid: hist_like}, [cid], feats)
    if X is None:
        return None
    y_hat = model.predict(X, verbose=0).ravel()
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def main() -> None:
    hist = load_history(os.path.join(RESULTS, "ekf_full_history.npz"))
    model = load_model()
    feats = GROUPS["A2-3"]

    test_cids = [c for c in sorted(hist.keys()) if "25degC_Cycle" in c]
    ood_cids = [c for c in sorted(hist.keys())
                if "10degC_" in c and "n10degC" not in c and "trise" not in c][:2]
    cids = test_cids + ood_cids
    print(f"评估文件: {len(cids)} 个（25°C {len(test_cids)} + 10°C {len(ood_cids)}）")

    # ---------------- 1. 噪声（中段起步） ----------------
    print("\n--- 1. 噪声敏感性 · 中段起步 ---")
    rows = []
    for sigma in NOISE_STDS:
        vals = []
        for cid in cids:
            tr, k0 = truncate_at_k0(hist[cid])
            if tr is None:
                continue
            rng = np.random.RandomState(NOISE_SEED)
            v_noisy = tr["voltage_V"] + rng.normal(0, sigma, len(tr["voltage_V"]))
            soc0 = float(np.clip(tr["soc_true"][0], 0.0, 1.0))   # bias = 0
            ekf = run_ekf(tr["current_A"], v_noisy, tr["temperature_C"], soc0, 1.0)
            new_h = dict(tr)
            new_h["voltage_V"] = v_noisy
            for key in ("soc", "nu", "S", "nis", "K0", "K1", "P00", "P11"):
                new_h[key] = np.array(ekf.history[key])
            r = rmse_of(model, feats, cid, new_h)
            if r is not None:
                vals.append(r)
        m = float(np.mean(vals)) if vals else float("nan")
        rows.append({"noise_std_mV": sigma * 1000, "RMSE_pp": m, "n_files": len(vals)})
        print(f"  σ={sigma*1000:>4.0f} mV → RMSE={m:.3f} pp  (n={len(vals)})")
    df_n = pd.DataFrame(rows)
    df_n.to_csv(os.path.join(RESULTS, "p5_03_noise_middischarge.csv"), index=False)

    # ---------------- 2. 采样率（中段起步） ----------------
    print("\n--- 2. 采样率降级 · 中段起步 ---")
    rows = []
    for rate in RATES:
        skip = max(1, int(1.0 / rate))
        vals = []
        for cid in cids:
            h = hist[cid]
            idx = np.arange(0, len(h["nu"]), skip)
            ds = {k: (np.asarray(v)[idx] if isinstance(v, np.ndarray) else v)
                  for k, v in h.items()}
            tr, k0 = truncate_at_k0(ds)          # 降采样后再定中段起点
            if tr is None:
                continue
            soc0 = float(np.clip(tr["soc_true"][0], 0.0, 1.0))
            ekf = run_ekf(tr["current_A"], tr["voltage_V"],
                          tr["temperature_C"], soc0, 1.0 / rate)
            new_h = dict(tr)
            for key in ("soc", "nu", "S", "nis", "K0", "K1", "P00", "P11"):
                new_h[key] = np.array(ekf.history[key])
            r = rmse_of(model, feats, cid, new_h)
            if r is not None:
                vals.append(r)
        m = float(np.mean(vals)) if vals else float("nan")
        rows.append({"sampling_rate_Hz": rate, "skip_factor": skip,
                     "RMSE_pp": m, "n_files": len(vals)})
        print(f"  {rate} Hz (skip={skip}) → RMSE={m:.3f} pp  (n={len(vals)})")
    df_r = pd.DataFrame(rows)
    df_r.to_csv(os.path.join(RESULTS, "p5_03_sampling_rate_middischarge.csv"),
                index=False)

    # ---------------- 3. 对照 ----------------
    print("\n" + "=" * 66)
    print("对照：满电起步（现盘） vs 中段起步（本次）")
    print("=" * 66)
    for tag, new_df, old_file, xcol in (
            ("噪声", df_n, "p5_03_noise.csv", "noise_std_mV"),
            ("采样率", df_r, "p5_03_sampling_rate.csv", "sampling_rate_Hz")):
        old = pd.read_csv(os.path.join(RESULTS, old_file))
        m = new_df.merge(old[[xcol, "RMSE_pp"]], on=xcol,
                         suffixes=("_mid", "_full"))
        print(f"\n【{tag}】")
        print(f"  {'x':>8}  {'满电起步':>10}  {'中段起步':>10}  {'变化':>9}")
        for _, r in m.iterrows():
            d = (r["RMSE_pp_mid"] - r["RMSE_pp_full"]) / r["RMSE_pp_full"] * 100
            print(f"  {r[xcol]:>8.3g}  {r['RMSE_pp_full']:>10.3f}  "
                  f"{r['RMSE_pp_mid']:>10.3f}  {d:>+8.1f}%")
        # 相对效应对比（这是论文真正引用的量）。
        # ⚠️ 必须按**文件里的行序**取首末（基线在首行），不能按 x 值排序：
        #    采样率的 x 是 1.0→0.5→0.2，按值升序会把 0.2 排到首行，
        #    算出的"效应"符号就反了（本脚本第一版就栽在这）。
        def rel(df, col):
            a = df["RMSE_pp"].values
            return (a[-1] - a[0]) / a[0] * 100 if len(a) > 1 else float("nan")
        print(f"  ⇒ 论文引用的**相对效应**：满电起步 {rel(old, xcol):+.1f}%  "
              f"vs 中段起步 {rel(new_df, xcol):+.1f}%")


if __name__ == "__main__":
    main()
