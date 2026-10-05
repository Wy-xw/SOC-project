# -*- coding: utf-8 -*-
"""P2-07：EKF 残差结构性分析 —— 止损点 S1 的证据链

问题：EKF 新息 ν 与 SOC 误差是否有**可学习的系统性结构**？
（若为白噪声 → NN 学不到东西 → 项目前提 1 不成立）

四层证据
--------
E1 白性检验   ：ν 的 ACF + Ljung-Box（H0：前 m 阶白）
E2 物理相关性 ：ν、SOC 误差分别对 |I|、温度、SOC 的互相关
     （Spearman，放电段 |I|>0.05，静置段单独报）
E3 频谱集中度 ：ν 功率谱 vs 白噪声平坦谱（1/f^α 拟合）
E4 可学习性探针：SOC 误差 e_k 对 [ν_k] / [ν_k, |I|, T, SOC_EKF] 的
     跨文件 Ridge 回归（Leave-One-File-Out CV）——
     与「预测 0」基线比，LOO-R² > 0 即存在可学的系统映射

输入：results/ekf_baseline_history.npz（P2-06 存档，8 文件）
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import signal, stats

import sys

# Windows 控制台默认 GBK，而本脚本 print 含 GBK 无码位的字符 —— 不改编码会在
# **成功路径**崩掉（活干完了却退出码 1）。别靠 PYTHONIOENCODING 兜底。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
I_REST = 0.05
BURN_IN = 300


def load_histories() -> dict[str, dict[str, np.ndarray]]:
    z = np.load(os.path.join(RESULTS, "ekf_baseline_history.npz"),
                allow_pickle=False)
    files: dict[str, dict[str, np.ndarray]] = {}
    for key in z.files:
        cid, field = key.split("::", 1)
        files.setdefault(cid, {})[field] = z[key]
    return files


def ljung_box(x: np.ndarray, m: int = 20) -> float:
    """Ljung-Box Q 统计的 p 值（H0：前 m 阶无自相关 → 白）。"""
    n = len(x)
    xc = x - x.mean()
    den = float(xc @ xc)
    if den <= 0:
        return 1.0
    acf = np.array([xc[k:] @ xc[:-k] / den for k in range(1, m + 1)])
    Q = n * (n + 2) * float(np.sum(acf**2 / (n - np.arange(1, m + 1))))
    # 卡方 survival，df=m
    from scipy.stats import chi2
    return float(chi2.sf(Q, m))


def ridge_loo_cv(files: dict[str, list[np.ndarray]], feat_sets: dict[str, int]):
    """Leave-One-File-Out Ridge：每文件 z-score（全局参数），
    留一文件训练 → 在留出文件上评估 R²（相对预测 0 基线）。"""
    # 展平
    all_X, all_y, file_idx = [], [], []
    names: list[str] = []
    for fi, (cid, (X, y)) in enumerate(files.items()):
        all_X.append(X)
        all_y.append(y)
        file_idx.append(np.full(len(y), fi))
        for j in range(X.shape[1]):
            names.append(f"feat{j}")
        break
    # 重新组织：feat_sets 定义列组
    return None  # 由 main 组装（占位说明设计，实际下面直接写）


def main() -> int:
    files = load_histories()
    print(f"载入 {len(files)} 个文件的 history\n")
    rows_e1, rows_e2, rows_e3 = [], [], []

    for cid, h in sorted(files.items()):
        n = len(h["soc_true"])
        sl = slice(BURN_IN, n)
        nu = h["nu"][sl]
        e = (h["soc"] - h["soc_true"])[sl] * 100.0   # pp
        i_abs = np.abs(h["current_A"][sl])
        temp = h["temperature_C"][sl]
        soc = h["soc_true"][sl]
        dis = i_abs > I_REST

        # E1 白性：ACF 前几阶 + Ljung-Box
        xc = nu - nu.mean()
        den = float(xc @ xc)
        acf10 = [float(xc[k:] @ xc[:-k] / den) for k in range(1, 11)]
        p_lb = ljung_box(nu, m=20)
        rows_e1.append({
            "file": cid.replace("Pan18650PF", "").strip(),
            "nu_std_mV": float(nu.std() * 1000),
            "acf1": acf10[0], "acf2": acf10[1], "acf5": acf10[4],
            "acf10": acf10[9],
            "LB_p": p_lb,
            "白噪声?": "是" if p_lb > 0.05 else "否",
        })

        # E2 物理相关性（Spearman）
        def sp(a, b):
            r, p = stats.spearmanr(a, b)
            return float(r), float(p)
        r_i, p_i = sp(i_abs[dis], nu[dis]) if dis.sum() > 50 else (np.nan, np.nan)
        r_t, p_t = sp(temp, nu)
        r_s, p_s = sp(soc[dis], nu[dis]) if dis.sum() > 50 else (np.nan, np.nan)
        r_is, _ = sp(i_abs[dis], e[dis]) if dis.sum() > 50 else (np.nan, np.nan)
        r_ts, _ = sp(temp, e)
        r_ss, _ = sp(soc[dis], e[dis]) if dis.sum() > 50 else (np.nan, np.nan)
        rows_e2.append({
            "file": rows_e1[-1]["file"],
            "ν~|I|": round(r_i, 3), "ν~T": round(r_t, 3),
            "ν~SOC": round(r_s, 3),
            "e~|I|": round(r_is, 3), "e~T": round(r_ts, 3),
            "e~SOC": round(r_ss, 3),
            "e_std_pp": float(e.std()),
            "e_bias_pp": float(e.mean()),
        })

        # E3 频谱：Welch PSD vs 白噪声平坦谱
        f, P = signal.welch(nu, fs=1.0, nperseg=min(256, len(nu) // 2))
        # 1/f^alpha 拟合（log-log）
        msk = f > 0
        alpha = float(np.polyfit(np.log(f[msk]), np.log(P[msk]), 1)[0])
        # 低频 10% 频带能量占比
        cum = np.cumsum(P)
        frac_lo = float(cum[int(0.1 * len(P))] / cum[-1])
        rows_e3.append({
            "file": rows_e1[-1]["file"],
            "alpha": round(alpha, 2),
            "低频10%能量占比": round(frac_lo, 3),
            "白噪声期望占比": 0.1,
        })

    df1, df2, df3 = map(pd.DataFrame, [rows_e1, rows_e2, rows_e3])
    print("== E1：新息白性检验（Ljung-Box, m=20）==")
    print(df1.to_string(index=False))
    print("\n== E2：物理量 Spearman 相关（放电段）==")
    print(df2.to_string(index=False))
    print("\n== E3：频谱集中度（白噪声 alpha≈0，低频占比≈0.1）==")
    print(df3.to_string(index=False))

    # ---------------- E4：可学习性探针（LOO Ridge） ---------------- #
    print("\n== E4：跨文件可学习性探针（LOO-Ridge，留一文件交叉验证）==")
    # 特征集
    FS = {
        "F0 预测0基线": [],
        "F1 仅ν": [("nu", 0)],
        "F2 ν+|I|+T+SOC": [("nu", 0), ("iabs", 0), ("temp", 0), ("soc", 0)],
        "F3 +滑窗ν(t-1..t-3)": [("nu", 0), ("iabs", 0), ("temp", 0),
                                 ("soc", 0), ("nu_1", 0), ("nu_2", 0),
                                 ("nu_3", 0)],
    }
    # 组装样本
    data = {}
    for cid, h in files.items():
        n = len(h["nu"])
        sl = slice(BURN_IN, n - 3)
        nu = h["nu"]
        data[cid] = {
            "nu": nu[sl],
            "nu_1": nu[sl.start - 1:sl.stop - 1],
            "nu_2": nu[sl.start - 2:sl.stop - 2],
            "nu_3": nu[sl.start - 3:sl.stop - 3],
            "iabs": np.abs(h["current_A"][sl]),
            "temp": h["temperature_C"][sl],
            "soc": h["soc_true"][sl],
            "y": (h["soc"] - h["soc_true"])[sl] * 100.0,
        }
    cids = sorted(data)
    for fname, feats in FS.items():
        if not feats:
            # 基线：预测 0
            r2s = []
            for c in cids:
                y = data[c]["y"]
                r2s.append(1.0 - 0.0 / float(np.sum(y**2)))  # R²=0 定义
            print(f"  {fname:<24s} R² = 0.000（定义）")
            continue
        cols = [f[0] for f in feats]
        # 全局 z-score（用其他文件的并集估计，防泄漏：LOO 时重算）
        # 简化执行：逐 LOO 折叠内 z-score（训练折数据估计参数）
        r2s = []
        for c_test in cids:
            Xtr = np.hstack([np.concatenate([data[c][col] for c in cids
                                             if c != c_test])[:, None]
                             for col in cols])
            ytr = np.concatenate([data[c]["y"] for c in cids if c != c_test])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
            lam = 1.0
            Xtr_z = (Xtr - mu) / sd
            A = Xtr_z.T @ Xtr_z + lam * np.eye(Xtr_z.shape[1])
            w = np.linalg.solve(A, Xtr_z.T @ ytr)
            b0 = ytr.mean()
            Xte = np.hstack([data[c_test][col][:, None] for col in cols])
            yte = data[c_test]["y"]
            pred = (Xte - mu) / sd @ w + b0
            r2 = 1.0 - float(np.sum((yte - pred) ** 2)) / \
                float(np.sum((yte - yte.mean()) ** 2))
            r2s.append(r2)
        print(f"  {fname:<24s} LOO-R² = {np.mean(r2s):+.3f} "
              f"(各文件: {', '.join(f'{v:+.2f}' for v in r2s)})")

    # 保存
    out = os.path.join(RESULTS, "residual_structure_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# P2-07 残差结构性分析（S1 证据）\n\n"
                "数据：ekf_baseline_history.npz（8 文件，P2-06 存档，"
                "去除前 300 s 收敛期）\n\n")
        f.write("## E1 白性检验\n\n```\n" + df1.to_string(index=False)
                + "\n```\n\n")
        f.write("## E2 物理相关性\n\n```\n" + df2.to_string(index=False)
                + "\n```\n\n")
        f.write("## E3 频谱集中度\n\n```\n" + df3.to_string(index=False)
                + "\n```\n")
    print(f"\n报告 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
