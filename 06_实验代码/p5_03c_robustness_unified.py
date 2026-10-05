# -*- coding: utf-8 -*-
"""
P5-10 鲁棒性三扫**官方生产者**（统一口径 c′）：噪声 / 采样率 / 初始偏差。

三扫共用同一套「**截断 + 热态**」口径，取代此前"噪声·采样率走满电连续起步、初始偏差走冷启动"的分裂口径。

## 为什么是这个口径（参照依据，2026-09-27）

Fig. A.1 三面板原本口径不一：面板 (c) 初始偏差扫用**冷启动**（EKF 在循环中途以 V1=0、P=P₀ 重启），
(a)(b) 用**满电连续起步**。查明两类参照后决定统一，依据如下：

1. **参照实现**（GitHub `Muhammed-Alessa/battery-pack-soc-estimation-ekf-ukf`）的"错误初始 SOC"测试是
   **plant 与滤波器同在 t=0 起步**，只把**估计器的 SOC₀ 直接设成 0.8**（真值 0.5）——
   **滤波器不重启**。即"同点起步 + 只改 SOC 猜测值"。
2. **文献的收敛时间**：EKF 从较大初始误差收敛需 **800–1200 s**（UTwente）/ **约 2400 s**（Hou 2024），
   而本项目 `BURN_IN = 300 s`。**冷启动后只烧 300 s = 在未收敛的瞬态里做评估**，
   实测把噪声敏感度从 **+3.0%**（热态 c′）抬到 **+11.1%**（中段起步·冷启动，
   即被否掉的 (a) 路线）——**伪影**。三个口径的实测对照见 `09_任务清单.md` P5-10。
3. **RWTH Aachen**：「misinitialization errors occur almost exclusively at the beginning；
   **key figures depend strongly on profile length**」——评估段本身就会决定头条数字。

⇒ **c′ 口径**：
   - **评估段**：三扫一律只评估 `[k0, end)`，k0 = 首个 `soc_true <= SOC_START` 的点
   - **滤波器状态**：`[SOC_est, V1]` 与 `P` 在 k0 处**继承连续运行的暖值**（取自 `ekf_full_history.npz`）
   - **唯一人为注入**：SOC 猜测值 `soc_true[k0] + bias`（(a)(b) 的 bias = 0）
   ⇒ 与参照实现同构（plant 与滤波器同点起步），且 ±bias 都有余量（覆盖 Guo/Hou 都没做的正半区）。

## 2026-09-27 正式采用（用户决定 "A"）

本脚本**已从"探针"升级为官方生产者**：直接写 `p5_03_noise.csv` /
`p5_03_sampling_rate.csv` / `p5_03_initial_bias_fixed.csv`（**覆盖前自动备份**为
`*.bak_20260927_preunified`）。列契约与旧版**完全一致**，`p5_06_figures.py` 无需改动即可读新数据。

> ⚠️ 为什么必须换口径：旧 `p5_03_initial_bias_fixed.csv` 的冷启动把 `P00` 设成**默认的 1e-2**，
> 而收敛后的真实值是 **≈2e-6**（6 文件 burn-in 后中位 1.77e-6）—— **差近 4 个数量级（5 666 倍 = 10^3.75）**。增益被放大后 20 pp 偏差被迅速抹平，
> 于是得出「偏差只影响瞬态、稳态不变（spread ≤3%）」。那是 `P0` 的产物，不是滤波器的性质。
> 暖态口径下偏差**持续**到稳态（−20 pp → 8.15 pp）。

用法
----
    "D:/venvs/soc-paper/Scripts/python.exe" 10_代码/p5_03c_robustness_unified.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

# Windows 控制台默认 GBK，而本脚本末尾要打 ⇒ —— 不改编码会在**成功路径**崩掉
# （CSV 已写完，但退出码 1，会被误读成"重跑失败"）。同 `_final_check.py` 等，别再踩。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "06_实验代码", "src"))

from nn_features import (  # noqa: E402
    BURN_IN, GROUPS, WINDOW_L, build_split_arrays, load_history,
)
from ecm_ekf import ECMParams, SOC_EKF  # noqa: E402
from ecm_tables import OCVTable, ECMParamTable, make_ekf_table_fns  # noqa: E402

RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "04_实验", "models")
FIELDS = ("soc", "nu", "S", "nis", "K0", "K1", "P00", "P11")

SOC_START = 0.80        # 与 p5_03b_initial_bias_fixed.py 一致
BIASES_PP = [-20, -10, -5, 0, 5, 10, 20]
NOISE_STDS = [0.0, 0.005, 0.010, 0.020]
RATES = [1.0, 0.5, 0.2]
NOISE_SEED = 42
CONV_BAND_PP = 1.0      # 收敛判据：|err| ≤ 1 pp
OFFICIAL = ("p5_03_noise.csv", "p5_03_sampling_rate.csv",
            "p5_03_initial_bias_fixed.csv")
BACKUP_SUFFIX = ".bak_20260927_preunified"


def backup_official() -> None:
    """覆盖前把旧产物留档 —— 换口径会改变全稿引用的一串数字，必须可回溯。"""
    import shutil
    for name in OFFICIAL:
        src = os.path.join(RESULTS, name)
        if os.path.exists(src) and not os.path.exists(src + BACKUP_SUFFIX):
            shutil.copy2(src, src + BACKUP_SUFFIX)
            print(f"  已备份 {name} → {name}{BACKUP_SUFFIX}")


def main() -> None:
    backup_official()
    from tensorflow import keras
    hist = load_history(os.path.join(RESULTS, "ekf_full_history.npz"))
    model = keras.models.load_model(
        os.path.join(MODELS_DIR, "gru16_A2-3_seed7.keras"), compile=False)
    feats = GROUPS["A2-3"]
    cids = ([c for c in sorted(hist) if "25degC_Cycle" in c]
            + [c for c in sorted(hist) if "10degC_" in c
               and "n10degC" not in c and "trise" not in c][:2])
    ocv_fn, docv_fn, pf = make_ekf_table_fns(OCVTable.from_csv(),
                                             ECMParamTable.from_csv())
    print(f"评估文件 {len(cids)} 个（25°C 4 + 10°C 2）｜口径 = 截断至 SOC≤{SOC_START} + 热态起步")

    def warm_at(h, k0):
        """取 k0 处的暖值：V1 与 P 对角（归档的连续全循环运行给出）。"""
        return (float(np.asarray(h["v1"])[k0]),
                float(np.asarray(h["P00"])[k0]),
                float(np.asarray(h["P11"])[k0]))

    def build(i_arr, v_arr, t_arr, x0, p_diag, dt):
        ekf = SOC_EKF(ECMParams(Cn=2.9, dt=dt), soc0=x0[0], v1_0=x0[1],
                      ocv_fn=ocv_fn, docv_fn=docv_fn, params_fn=pf)
        ekf.P = np.diag([p_diag[0], p_diag[1]])       # 热态协方差
        for k in range(len(i_arr)):
            ekf.step(float(i_arr[k]), float(v_arr[k]), temp_k=float(t_arr[k]))
        return ekf

    def score(cid, h_src, ekf, k0=0):
        """把 h_src 与 ekf.history 都对齐到同一段后评分。

        ⚠️ 两种情形的切片方式不同，写错就长度不齐（本脚本第一版栽在这）：
          · (a)(b)：EKF 从 t=0 跑全程 → h_src 与 history 都是全长，各切 [k0:]
          · (c)   ：EKF 本身就是从 k0 起跑的 → 两者都已只剩后段，**不能再切**
        这里用 `len(ekf.history["soc"])` 判断 EKF 实际跑了多长，自动对齐。
        """
        nh = {k: (np.asarray(v)[k0:] if isinstance(v, np.ndarray) else v)
              for k, v in h_src.items()}
        for key in FIELDS:
            arr = np.array(ekf.history[key])
            if len(arr) != len(nh["soc_true"]):     # EKF 从 k0 起跑 → 取整段
                arr = arr[-len(nh["soc_true"]):]
            nh[key] = arr
        X, y, _ = build_split_arrays({cid: nh}, [cid], feats)
        if X is None:
            return None
        return float(np.sqrt(np.mean((y - model.predict(X, verbose=0).ravel()) ** 2)))

    # ---------------- 1. 噪声（连续跑，只评估后段） ----------------
    print("\n--- 1. 噪声敏感性 · c′ ---")
    rows = []
    for sigma in NOISE_STDS:
        vs = []
        for cid in cids:
            h = hist[cid]
            k0 = int(np.where(np.asarray(h["soc_true"]) <= SOC_START)[0][0])
            v = h["voltage_V"] + np.random.RandomState(NOISE_SEED).normal(
                0, sigma, len(h["voltage_V"]))
            x0 = (float(np.clip(h["soc_true"][0] + 0.10, 0, 1)), 0.0)
            ekf = build(h["current_A"], v, h["temperature_C"], x0, (1e-2, 1e-2), 1.0)
            r = score(cid, h, ekf, k0)
            if r is not None:
                vs.append(r)
        rows.append({"noise_std_mV": sigma * 1000, "RMSE_pp": float(np.mean(vs)),
                     "n_files": len(vs)})
        print(f"  σ={sigma*1000:>4.0f} mV → {np.mean(vs):.3f} pp")
    df_n = pd.DataFrame(rows)
    df_n.to_csv(os.path.join(RESULTS, "p5_03_noise.csv"), index=False)

    # ---------------- 2. 采样率（连续跑，只评估后段） ----------------
    print("\n--- 2. 采样率 · c′ ---")
    rows = []
    for rate in RATES:
        skip = max(1, int(1.0 / rate))
        vs = []
        for cid in cids:
            h = hist[cid]
            idx = np.arange(0, len(h["nu"]), skip)
            ds = {k: (np.asarray(v)[idx] if isinstance(v, np.ndarray) else v)
                  for k, v in h.items()}
            k0 = int(np.where(np.asarray(ds["soc_true"]) <= SOC_START)[0][0])
            x0 = (float(np.clip(ds["soc_true"][0] + 0.10, 0, 1)), 0.0)
            ekf = build(ds["current_A"], ds["voltage_V"], ds["temperature_C"],
                        x0, (1e-2, 1e-2), 1.0 / rate)
            r = score(cid, ds, ekf, k0)
            if r is not None:
                vs.append(r)
        rows.append({"sampling_rate_Hz": rate, "skip_factor": skip,
                     "RMSE_pp": float(np.mean(vs)), "n_files": len(vs)})
        print(f"  {rate} Hz → {np.mean(vs):.3f} pp")
    df_r = pd.DataFrame(rows)
    df_r.to_csv(os.path.join(RESULTS, "p5_03_sampling_rate.csv"), index=False)

    # ---------------- 3. 初始偏差（热态 + SOC 偏移）----------------
    # 列契约与 p5_03b_initial_bias_fixed.py **完全一致**（9 列），
    # 指标定义也逐条沿用：peak 取 err[:BURN_IN]、t_conv 取首个 |err|≤1 pp、稳态走 build_split_arrays。
    print("\n--- 3. 初始偏差 · c′（热态起步 + 注入 SOC 偏移）---")
    per = []
    for b in BIASES_PP:
        for cid in cids:
            h = hist[cid]
            st = np.asarray(h["soc_true"])
            k0 = int(np.where(st <= SOC_START)[0][0])
            v1w, p00w, p11w = warm_at(h, k0)
            # ⚠️ 起点取**连续运行到 k0 时的估计值**（h["soc"][k0]），不是真值 st[k0]：
            #    取真值等于"在 k0 处把滤波器的 SOC 悄悄重置为真值"，会让 bias=0 的基线
            #    降到 1.54 而不是与 (a)(b) 一致的 1.71 —— 三个面板就对不上（第一版栽在这）。
            #    取估计值 + 偏移，才是"给一个已在运行的滤波器注入偏差"；
            #    且 bias=0 时与 (a)(b) 的连续运行完全等价。
            soc_est_k0 = float(np.asarray(h["soc"])[k0])
            soc0 = float(np.clip(soc_est_k0 + b / 100.0, 0.0, 1.0))
            ekf = build(h["current_A"][k0:], h["voltage_V"][k0:],
                        h["temperature_C"][k0:], (soc0, v1w), (p00w, p11w), 1.0)
            nh = {k: (np.asarray(v)[k0:] if isinstance(v, np.ndarray) else v)
                  for k, v in h.items()}
            for key in FIELDS:
                nh[key] = np.array(ekf.history[key])
            err = (nh["soc"] - nh["soc_true"]) * 100.0            # pp
            peak = float(np.max(np.abs(err[:BURN_IN])))
            inband = np.where(np.abs(err) <= CONV_BAND_PP)[0]
            X, y, _ = build_split_arrays({cid: nh}, [cid], feats)
            if X is None:
                continue
            yh = model.predict(X, verbose=0).ravel()
            per.append({
                "bias_pp": float(b), "cycle_id": cid,
                "initial_abs_err_pp": abs(soc0 - float(st[k0])) * 100.0,
                "ekf_peak_err_first300s_pp": peak,
                "ekf_t_to_1pp_s": float(inband[0]) if len(inband) else float("nan"),
                "ekf_final_abs_err_pp": float(abs(err[-1])),
                "ekf_steady_RMSE_pp": float(np.sqrt(np.mean(y ** 2))),
                "fused_steady_RMSE_pp": float(np.sqrt(np.mean((y - yh) ** 2))),
            })
        print(f"  {b:>+4.0f} pp 完成")

    pf = pd.DataFrame(per)
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
             .reset_index().sort_values("bias_pp"))
    agg.to_csv(os.path.join(RESULTS, "p5_03_initial_bias_fixed.csv"),
               index=False, float_format="%.6f")
    df_b = agg.rename(columns={"fused_steady_RMSE_pp": "RMSE_pp"})
    print(agg.round(3).to_string(index=False))

    # ---------------- 4. 对照 ----------------
    print("\n" + "=" * 70)
    print("对照：现盘 vs c′（统一口径）")
    print("=" * 70)
    for tag, new, old_f, xc in (("噪声", df_n, "p5_03_noise.csv" + BACKUP_SUFFIX, "noise_std_mV"),
                                ("采样率", df_r, "p5_03_sampling_rate.csv" + BACKUP_SUFFIX, "sampling_rate_Hz"),
                                ("初始偏差", df_b, "p5_03_initial_bias_fixed.csv" + BACKUP_SUFFIX, "bias_pp")):
        try:
            old = pd.read_csv(os.path.join(RESULTS, old_f))
        except FileNotFoundError:
            continue
        ycol = "fused_steady_RMSE_pp" if "fused_steady_RMSE_pp" in old.columns else "RMSE_pp"
        # 先把旧档那列改名再 merge，避免两侧同名列被加 _u/_o 后缀后取错列
        # （原写法靠 guess 后缀，结果整列打成 nan；且相对效应那两个标签印反了）。
        m = (new.merge(old[[xc, ycol]].rename(columns={ycol: "旧口径_pp"}),
                       on=xc, how="inner")
             .sort_values(xc).reset_index(drop=True))
        print(f"\n【{tag}】")
        print(f"  {'x':>8}  {'旧口径':>10}  {'c′':>9}")
        for _, r in m.iterrows():
            print(f"  {r[xc]:>8.3g}  {r['旧口径_pp']:>10.3f}  {r['RMSE_pp']:>9.3f}")
        if len(m) > 1:
            u = (m["RMSE_pp"].iloc[-1] - m["RMSE_pp"].iloc[0]) / m["RMSE_pp"].iloc[0] * 100
            o = (m["旧口径_pp"].iloc[-1] - m["旧口径_pp"].iloc[0]) / m["旧口径_pp"].iloc[0] * 100
            print(f"  ⇒ 相对效应：旧口径 {o:+.1f}%   vs   c′ {u:+.1f}%")


if __name__ == "__main__":
    main()
