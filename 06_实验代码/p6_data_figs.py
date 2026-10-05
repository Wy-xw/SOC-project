# -*- coding: utf-8 -*-
"""
P6 图件补齐（二）：数据图两张
  Fig.6  残差结构（ν 自相关 + 频谱，双面板，带对齐门）
  Fig.10 SOC 估计时序对比（真值 vs EKF vs 融合，单面板）
数据源：ekf_full_history.npz + gru16_A2-3_seed7.keras
输出：05_论文/figures/
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "06_实验代码", "src"))
SKILL = r"C:\Users\wangyan\.claude\skills\nature-figure"
sys.path.insert(0, os.path.join(SKILL, "scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa

from nn_features import GROUPS, BURN_IN, build_split_arrays, load_history  # noqa

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
})

RESULTS = os.path.join(PROJ, "04_实验", "数据", "results")
MODELS = os.path.join(PROJ, "04_实验", "models")
OUT = os.path.join(PROJ, "05_论文", "figures")
INK, GRAY, BLUE, ACC = "#222222", "#7F8C8D", "#1F618D", "#B03A2E"


def save(fig, name, alignment=None):
    b = os.path.join(OUT, name)
    require_matplotlib_panel_alignment(
        fig, json_out=b + ".alignment.json",
        overlay_svg=b + ".alignment.svg", strict=True,
        **(alignment or {}))
    fig.savefig(b + ".svg", bbox_inches="tight")
    fig.savefig(b + ".pdf", bbox_inches="tight")
    fig.savefig(b + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("saved:", name)


def acf(x, nlags=40):
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = np.sum(x * x)
    out = np.empty(nlags + 1)
    for k in range(nlags + 1):
        out[k] = np.sum(x[:len(x) - k] * x[k:]) / denom
    return out


def fig6():
    hist = load_history(os.path.join(RESULTS, "ekf_full_history.npz"))
    k25 = next(k for k in hist if "25degC_Cycle_1" in k)
    # ⚠️ 2026-09-25 修复（P0-23）：`"0degC_Cycle_1" in k` 是子串陷阱——
    #    `"10degC_Cycle_1"` 也含该子串，而 10 °C 文件在 history 中排在前面，
    #    于是本行原先取到的是 **10 °C** 文件，面板标题却写 0 °C。
    #    同仓同类陷阱的记载见 p5_03_robustness.py:281-282、p5_01_baseline_compare.py:72。
    k0 = next(k for k in hist
              if "0degC_Cycle_1" in k and "10degC_Cycle_1" not in k
              and "n10degC_Cycle_1" not in k and "n20degC_Cycle_1" not in k)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    for ax, (k, lab, c) in [(ax1, (k25, "25 °C (ID)", BLUE)),
                            (ax2, (k0, "0 °C (OOD)", ACC))]:
        nu = np.asarray(hist[k]["nu"])[BURN_IN:]
        n = len(nu)
        a = acf(nu)
        lags = np.arange(len(a))
        if ax is ax1:
            ax.plot(lags, a, "-o", color=c, ms=2.5, lw=1.2, label=lab)
            ax.axhspan(-1.96 / np.sqrt(n), 1.96 / np.sqrt(n),
                       color=GRAY, alpha=0.25, lw=0)
            ax.axhline(0, color=GRAY, lw=0.6, ls=":")
            ax.set_xlabel("Lag (s)", fontsize=7.5)
            ax.set_ylabel("ACF of innovation $\\nu$", fontsize=7.5)
            ax.text(0.97, 0.95, "95 % white-noise band",
                    transform=ax.transAxes, fontsize=6,
                    ha="right", va="top", color=GRAY)
        else:
            ax.plot(lags, a, "-s", color=c, ms=2.5, lw=1.2, label=lab)
            f = np.fft.rfftfreq(n, d=1.0)
            P = np.abs(np.fft.rfft(nu - nu.mean())) ** 2
            P = P / P.sum()
            lo = f <= 0.05
            share = P[lo].sum() * 100
            ax.set_xlabel(f"Frequency (Hz)   (0–0.05 Hz: {share:.0f} % of power)",
                          fontsize=7.2)
            ax.set_ylabel("PSD (norm.)", fontsize=7.5)
            ax.set_yscale("log")
            ax.set_xlim(0, 0.5)
            # 🔴 2026-09-27 QA 修复（nature-figure 的 validate_figure.py 报 LOG-GUARD
            #    + 逐面板目视确认）：
            #    **原来的 `pv.min()*0.5` 是无效修复**——`P` 的最小非零值是 **4.5e-37**，
            #    而 0.01 百分位已是 5.5e-9，即**极少数近乎为 0 的 bin 把下限拽了下去**。
            #    结果 y 轴跨 **34.8 个数量级**，整条曲线（真实动态范围约 2.5 个数量级，
            #    1e-3 ~ 1e-7）全挤在顶部，**下面 85% 的版面是空的**——
            #    面板存在的理由「0.05 Hz 以下占 72 % 的功率」在图上**完全看不到**。
            #    改用 **0.5 百分位**做下限：轴跨约 5 个数量级，仅裁掉 0.33 % 的点。
            #    （技能要求：轴范围要按数据分布取，不能用极值。）
            pv = P[P > 0]
            floor = float(np.percentile(pv, 0.5) * 0.5)
            # ⚠️ **必须同时把数据本身截到 floor**，不能只设 ylim ——
            #    只设 ylim 时，低于下限的点仍会被画到坐标区外（实测：红色尖峰扎到
            #    x 轴下方、戳进刻度标签，`audit_figure_collisions.py` 判 text-stroke FAIL）。
            #    floor 以下的点相对峰值低 ~35 个数量级，是 **float64 下的数值噪声**，
            #    截断是合理的；**该规则与裁掉的比例记录在此**：
            #    floor=0.5 百分位→约 0.33 % 的点被截。
            ax.plot(f, np.maximum(P, floor), "-", color=c, lw=1.0)
            ax.set_ylim(floor, P.max() * 5)
        # 图例位置按面板分开：修完 ylim 后面板 b 的数据铺满全幅，
        # center-right 会压在噪声带上（实测被审计判为 text-stroke FAIL），
        # 而右上角（高频低功率区之上）是空的。
        ax.legend(fontsize=6.5,
                  loc=("center right" if ax is ax1 else "upper right"),
                  frameon=False)

    ax1.set_title("Innovation autocorrelation (25 °C, ID)",
                  fontsize=7.5, pad=8, loc="left")
    ax2.set_title("Innovation power spectrum (0 °C, OOD)",
                  fontsize=7.5, pad=8, loc="left")
    # 独立面板标签（置于轴外左上，避开左对齐标题；对齐门要求可检出）
    ax1.text(-0.11, 1.0, "a", transform=ax1.transAxes, fontsize=9,
             fontweight="bold", va="bottom", ha="left")
    ax2.text(-0.11, 1.0, "b", transform=ax2.transAxes, fontsize=9,
             fontweight="bold", va="bottom", ha="left")
    fig.tight_layout(pad=0.8)

    save(fig, "fig6_residual", alignment=dict(
        axes=[ax1, ax2], panel_ids=["a", "b"],
        row_groups=[["a", "b"]], require_panel_labels=True))


def fig10():
    from tensorflow import keras
    hist = load_history(os.path.join(RESULTS, "ekf_full_history.npz"))
    cid = next(k for k in hist if "25degC_Cycle_1" in k)
    feats = GROUPS["A2-3"]
    model = keras.models.load_model(
        os.path.join(MODELS, "gru16_A2-3_seed7.keras"), compile=False)

    X, y, ends_map = build_split_arrays(hist, [cid], feats)
    ends = ends_map[cid]
    yhat = model.predict(X, verbose=0).ravel()
    h = hist[cid]
    truth = np.asarray(h["soc_true"]) * 100.0
    ekf = np.asarray(h["soc"]) * 100.0
    fused = ekf[ends] - yhat

    ekf_rmse = float(np.sqrt(np.mean(y ** 2)))
    fu_rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))

    step = 5
    t = ends[::step]
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.plot(truth[ends][::step], "-", color=INK, lw=1.0,
            label="Reference (coulomb)")
    ax.plot(ekf[ends][::step], "-", color=GRAY, lw=1.2,
            label=f"EKF  ({ekf_rmse:.2f} pp, this file)")
    ax.plot(fused[::step], "-", color=BLUE, lw=1.2,
            label=f"EKF+NN  ({fu_rmse:.2f} pp, this file)")
    ax.set_xlabel(f"Time (s), 1 Hz, plotted every {step} s", fontsize=7.5)
    ax.set_ylabel("SOC (%)", fontsize=7.5)
    ax.set_ylim(0, 102)
    ax.set_title("25 °C mixed cycle, single file (test)",
                 fontsize=8, pad=4)
    ax.legend(fontsize=6.5, loc="lower left", frameon=False)
    save(fig, "fig10_soc_timeseries")


if __name__ == "__main__":
    fig6()
    fig10()
