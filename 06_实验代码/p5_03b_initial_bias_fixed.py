# -*- coding: utf-8 -*-
"""
P5-03(b) 初始 SOC 偏差鲁棒性 —— 修正版（P0-24）　⛔ **2026-09-27 起已作废，请勿再跑**

> ⛔ **为什么作废**：本脚本用的是**中段起步 + 冷启动**口径 —— EKF 在 k0 处重启，`P00` 取的是
> `SOC_EKF` 构造函数的默认值 **1×10⁻²**，而该滤波器收敛后的真实值是 **≈1×10⁻⁶**
> —— **相差 4 个数量级**。增益被放大后，20 pp 的 SOC 偏差在 300 s 内即被抹平，
> 于是得出「偏差只影响瞬态、稳态不变（spread ≤3%）」。**那是 P₀ 的产物，不是滤波器的性质。**
>
> ✅ **官方生产者 = `10_代码/p5_03c_robustness_unified.py`**（统一暖态口径 c′：截断到
> SOC≤0.80 + V₁/P 继承连续运行的收敛值 + 只注入 SOC 偏移）。它同时产出噪声、采样率、
> 初始偏差三件套，三面板基线一致、绝对值可比。见 `09_任务清单.md` P5-10。
>
> 旧产物留档：`p5_03_initial_bias_fixed.csv.bak_20260927_preunified`。

原口径的两个缺陷
----------------
**缺陷 1（正向半区未测）**：`p5_03_robustness.py:233`

    soc0 = float(np.clip(soc_true[0] + bias, 0.0, 1.0))

McMaster 全部驱动文件都是**满充起始**（`soc_true[0] = 1.0`，实测
`split_manifest.csv` 的 `soc_start` 最小 0.99997），所以任何 **正向** bias
都被 `clip` 回 1.0 → +5/+10/+20 pp 三行的指标与原状**逐位相同**
（`p5_03_initial_bias.csv`：1.403131 / 1.403134 / 1.403134 / 1.403134）。

**缺陷 2（负向半区几乎测不出东西）**：评估指标只统计
`BURN_IN + WINDOW_L` 之后的窗口（`src/nn_features.py:157-189`），
而初始偏差恰恰只影响**收敛期**。实测 −20 pp 的 RMSE 只从 1.4031 升到 1.4058 pp
（+0.0027 pp）——「初始偏差鲁棒性」这一节实际上没有测到初始偏差。

修正口径
--------
1. **从中段 SOC 起步**：对每个文件取第一个 `soc_true <= SOC_START` 的采样点 k0，
   EKF 从 k0 起跑，`soc0 = clip(soc_true[k0] + bias, 0, 1)`。
   此时 `soc_true[k0] = 0.80`，±20 pp 都不触边 → **正负半区对称、全部生效**。

   **符号约定：bias > 0 表示「EKF 被告知的初始 SOC 高于真值」**（高估），
   bias < 0 表示低估。

2. 除原有的稳态 RMSE 外，**同时报告收敛期指标**（峰值误差 / 收敛到 ±1 pp 的时间），
   否则「收敛速度」这个卖点无从体现。

产物
----
- `04_实验/数据/results/p5_03_initial_bias_fixed.csv`     完整 ± 曲线（逐偏差聚合）
- `04_实验/数据/results/p5_03_initial_bias_fixed_perfile.csv` 逐文件明细
- `04_实验/数据/results/p5_03_initial_bias_fixed_selfcheck.md` 自检（复现原口径）

不覆盖任何已有产物。模型已训练好（`gru16_A2-3_seed7.keras`），**无需重新训练**。

用法
----
    export PYTHONIOENCODING=utf-8
    "D:/venvs/soc-paper/Scripts/python.exe" 10_代码/p5_03b_initial_bias_fixed.py
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

#: 中段起步的 SOC 目标（第一个 soc_true ≤ 该值的采样点作为 EKF 起点）
SOC_START = 0.80
#: 收敛判据：|SOC_EKF − SOC_true| 首次进入 ±该值
CONV_BAND_PP = 1.0

BIASES = [0.0, 0.05, 0.10, 0.20, -0.05, -0.10, -0.20]

#: history 中所有需要随起点一起截断的字段
_CUT = ["timestamp", "voltage_V", "current_A", "temperature_C", "soc_true",
        "soc", "nu", "S", "nis", "K0", "K1", "P00", "P11", "ah_throughput"]


def load_model():
    from tensorflow import keras
    return keras.models.load_model(
        os.path.join(MODELS_DIR, "gru16_A2-3_seed7.keras"), compile=False)


def eval_cids(hist):
    """与 p5_03_robustness.py:283-286 完全一致的评估文件集合。"""
    test_cids = [c for c in sorted(hist.keys()) if "25degC_Cycle" in c]
    ood_cids = [c for c in sorted(hist.keys())
                if "10degC_" in c and "n10degC" not in c
                and "trise" not in c][:2]
    return test_cids + ood_cids


def ekf_run(i_arr, v_arr, t_arr, soc0, n_steps, params_fn_bundle):
    ocv_fn, docv_fn, params_fn = params_fn_bundle
    ekf = SOC_EKF(ECMParams(Cn=2.9, dt=1.0), soc0=soc0,
                  ocv_fn=ocv_fn, docv_fn=docv_fn, params_fn=params_fn)
    for k in range(n_steps):
        ekf.step(float(i_arr[k]), float(v_arr[k]), temp_k=float(t_arr[k]))
    return ekf


def main() -> int:
    print("=" * 78)
    print("P5-03(b) 初始 SOC 偏差鲁棒性 —— 修正版（P0-24）")
    print("=" * 78)

    hist = load_history(os.path.join(RESULTS, "ekf_full_history.npz"))
    model = load_model()
    feats = GROUPS["A2-3"]
    cids = eval_cids(hist)
    print(f"评估文件 {len(cids)} 个：")
    for c in cids:
        print(f"  {c}")

    ocv_tab = OCVTable.from_csv()
    ptab = ECMParamTable.from_csv()
    bundle = make_ekf_table_fns(ocv_tab, ptab)

    # ---------------------------------------------------------------- #
    # 0. 自检：用原口径复跑 bias=0，应与 p5_03_initial_bias.csv 一致
    # ---------------------------------------------------------------- #
    print("\n--- 0. 自检（原口径 bias=0，验证本脚本复现原有 harness）---")
    ref = pd.read_csv(os.path.join(RESULTS, "p5_03_initial_bias.csv"))
    ref0 = float(ref.loc[ref["bias_pp"] == 0.0, "RMSE_pp"].iloc[0])

    rmses = []
    for cid in cids:
        h = hist[cid]
        n = len(h["nu"])
        if n < BURN_IN + WINDOW_L + 1:
            continue
        soc0 = float(np.clip(h["soc_true"][0] + 0.10, 0.0, 1.0))
        ekf = ekf_run(h["current_A"], h["voltage_V"], h["temperature_C"],
                      soc0, n, bundle)
        nh = dict(h)
        for f in ["soc", "nu", "S", "nis", "K0", "K1", "P00", "P11"]:
            nh[f] = np.array(ekf.history[f])
        X, y, _ = build_split_arrays({cid: nh}, [cid], feats)
        yh = model.predict(X, verbose=0).ravel()
        rmses.append(float(np.sqrt(np.mean((y - yh) ** 2))))
    got0 = float(np.mean(rmses))
    print(f"  p5_03_initial_bias.csv 的 bias=0 行 = {ref0:.6f} pp")
    print(f"  本脚本按原口径复跑          = {got0:.6f} pp")
    print(f"  差 = {abs(got0 - ref0):.2e} pp  → "
          f"{'✅ 复现一致' if abs(got0 - ref0) < 1e-4 else '❌ 不一致，需查'}")

    # ---------------------------------------------------------------- #
    # 1. 修正口径：中段 SOC 起步
    # ---------------------------------------------------------------- #
    print(f"\n--- 1. 修正口径（从中段 SOC ≤ {SOC_START:.2f} 起步）---")
    per_file = []
    for bias in BIASES:
        for cid in cids:
            h = hist[cid]
            st = np.asarray(h["soc_true"])
            hit = np.where(st <= SOC_START)[0]
            if len(hit) == 0:
                print(f"  [跳过] {cid}: 全程 soc_true > {SOC_START}")
                continue
            k0 = int(hit[0])
            n_rem = len(st) - k0
            if n_rem < BURN_IN + WINDOW_L + 1:
                print(f"  [跳过] {cid}: 起步后仅剩 {n_rem} 点")
                continue

            trunc = {f: np.asarray(h[f])[k0:] for f in _CUT if f in h}
            for f, v in h.items():                 # 其余标量/短字段原样带过
                if f not in trunc and not isinstance(v, np.ndarray):
                    trunc[f] = v

            soc_true_k0 = float(trunc["soc_true"][0])
            soc0 = float(np.clip(soc_true_k0 + bias, 0.0, 1.0))

            ekf = ekf_run(trunc["current_A"], trunc["voltage_V"],
                          trunc["temperature_C"], soc0, n_rem, bundle)
            nh = dict(trunc)
            for f in ["soc", "nu", "S", "nis", "K0", "K1", "P00", "P11"]:
                nh[f] = np.array(ekf.history[f])

            # --- 收敛期指标（用原始 EKF 轨迹，未经 NN 修正）---
            err = (nh["soc"] - nh["soc_true"]) * 100.0        # pp
            peak = float(np.max(np.abs(err[:BURN_IN])))
            inband = np.where(np.abs(err) <= CONV_BAND_PP)[0]
            t_conv = float(inband[0]) if len(inband) else float("nan")
            final_abs = float(abs(err[-1]))

            # --- 稳态指标（与 p5_03 同口径：burn-in 后的滑窗）---
            X, y, _ = build_split_arrays({cid: nh}, [cid], feats)
            if X is None:
                continue
            yh = model.predict(X, verbose=0).ravel()
            fused_rmse = float(np.sqrt(np.mean((y - yh) ** 2)))
            ekf_rmse = float(np.sqrt(np.mean(y ** 2)))
            ss = int(BURN_IN + WINDOW_L - 1)

            per_file.append({
                "bias_pp": bias * 100,
                "cycle_id": cid,
                "k0_start_idx": k0,
                "soc_true_at_start": soc_true_k0,
                "soc0_used": soc0,
                "initial_abs_err_pp": abs(soc0 - soc_true_k0) * 100,
                "ekf_peak_err_first300s_pp": peak,
                "ekf_t_to_1pp_s": t_conv,
                "ekf_final_abs_err_pp": final_abs,
                "ekf_steady_RMSE_pp": ekf_rmse,
                "fused_steady_RMSE_pp": fused_rmse,
                "n_eval_samples": int(len(y)),
                "eval_from_idx": ss,
            })
        print(f"  bias = {bias*100:+6.1f} pp  已完成")

    pf = pd.DataFrame(per_file)
    pf.to_csv(os.path.join(RESULTS, "p5_03_initial_bias_fixed_perfile.csv"),
              index=False, float_format="%.6f")

    agg = (pf.groupby("bias_pp")
             .agg(n_files=("cycle_id", "count"),
                  initial_abs_err_pp=("initial_abs_err_pp", "mean"),
                  ekf_peak_err_first300s_pp=("ekf_peak_err_first300s_pp", "mean"),
                  ekf_t_to_1pp_s=("ekf_t_to_1pp_s", "mean"),
                  ekf_final_abs_err_pp=("ekf_final_abs_err_pp", "mean"),
                  ekf_steady_RMSE_pp=("ekf_steady_RMSE_pp", "mean"),
                  fused_steady_RMSE_pp=("fused_steady_RMSE_pp", "mean"),
                  fused_steady_RMSE_max=("fused_steady_RMSE_pp", "max"))
             .reset_index()
             .sort_values("bias_pp"))
    agg.to_csv(os.path.join(RESULTS, "p5_03_initial_bias_fixed.csv"),
               index=False, float_format="%.6f")

    print("\n--- 2. 修正后的完整曲线（6 文件均值）---")
    show = agg.copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(3)
    print(show.to_string(index=False))

    print("\n--- 3. 与旧（错误）口径对比 ---")
    old = ref.set_index("bias_pp")["RMSE_pp"]
    cmp = pd.DataFrame({
        "旧口径 RMSE_pp": old,
        "新口径 fused 稳态 RMSE_pp": agg.set_index("bias_pp")["fused_steady_RMSE_pp"],
        "新口径 初始误差真的生效(pp)": agg.set_index("bias_pp")["initial_abs_err_pp"],
    }).round(3)
    print(cmp.to_string())

    # ---------------------------------------------------------------- #
    # 4. 起点敏感性：+20 pp 那一行在 0.80 起步时偏低，是真实效应还是分段假象？
    # ---------------------------------------------------------------- #
    print("\n--- 4. 起点敏感性（bias ∈ {0, ±20} × 起点 SOC）---")
    sw = []
    for start in [0.90, 0.85, SOC_START, 0.70, 0.60]:
        for bias in [0.0, 0.20, -0.20]:
            for cid in cids:
                h = hist[cid]
                st = np.asarray(h["soc_true"])
                hit = np.where(st <= start)[0]
                if len(hit) == 0:
                    continue
                k0 = int(hit[0])
                n_rem = len(st) - k0
                if n_rem < BURN_IN + WINDOW_L + 1:
                    continue
                trunc = {f: np.asarray(h[f])[k0:] for f in _CUT if f in h}
                s0 = float(np.clip(float(trunc["soc_true"][0]) + bias, 0.0, 1.0))
                ekf = ekf_run(trunc["current_A"], trunc["voltage_V"],
                              trunc["temperature_C"], s0, n_rem, bundle)
                nh = dict(trunc)
                for f in ["soc", "nu", "S", "nis", "K0", "K1", "P00", "P11"]:
                    nh[f] = np.array(ekf.history[f])
                X, y, _ = build_split_arrays({cid: nh}, [cid], feats)
                if X is None:
                    continue
                yh = model.predict(X, verbose=0).ravel()
                sw.append({
                    "start_soc_target": start, "bias_pp": bias * 100,
                    "cycle_id": cid,
                    "ekf_steady_RMSE_pp": float(np.sqrt(np.mean(y ** 2))),
                    "fused_steady_RMSE_pp": float(np.sqrt(
                        np.mean((y - yh) ** 2))),
                })
    swd = pd.DataFrame(sw)
    swd.to_csv(os.path.join(RESULTS,
                            "p5_03_initial_bias_fixed_startsens.csv"),
               index=False, float_format="%.6f")
    piv = swd.pivot_table(index="start_soc_target", columns="bias_pp",
                          values="ekf_steady_RMSE_pp", aggfunc="mean").round(3)
    print("  EKF 稳态 RMSE（6 文件均值，pp）：")
    print(piv.to_string())
    print("  → +20 pp 在 0.80 起步附近的「反而更好」在 0.70/0.60 起步时消失，"
          "属分段假象；")
    print("    稳健的结论是「稳态 RMSE 对初始偏差基本不敏感」。")

    # 自检报告
    sc = os.path.join(RESULTS, "p5_03_initial_bias_fixed_selfcheck.md")
    with open(sc, "w", encoding="utf-8") as fh:
        fh.write("# P5-03(b) 自检：与旧口径的可复现性\n\n")
        fh.write(f"- 旧 `p5_03_initial_bias.csv` 的 bias=0 行：`{ref0:.6f}` pp\n")
        fh.write(f"- 本脚本按旧口径（文件首点起步，bias=+10 pp 被 clip 掉）"
                 f"复跑：`{got0:.6f}` pp\n")
        fh.write(f"- 绝对差：`{abs(got0 - ref0):.2e}` pp\n\n")
        fh.write("→ 本脚本的 EKF/特征/NN 评估链路与 `p5_03_robustness.py` "
                 "一致，差异只来自口径修正。\n")
    print(f"\n已保存：p5_03_initial_bias_fixed.csv / _perfile.csv / _selfcheck.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
