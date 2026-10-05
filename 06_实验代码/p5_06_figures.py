# -*- coding: utf-8 -*-
"""
P5-06 论文级配图（nature-figure，Python/matplotlib）

产出 4 张数据图（SVG 主 + PDF + PNG）：
  fig7_baseline   — Baseline 对比热图（6 方法 × 13 条件）
  fig8_ablation   — 消融实验分组柱状（5 特征组 × 4 温度）
  fig9_robustness — 鲁棒性三联（噪声 / 采样率 / 初始偏差）
  fig2_ocv        — OCV-SOC 曲线

用法：
    "D:/venvs/soc-paper/Scripts/python.exe" 10_代码/p5_06_figures.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- nature-figure skill scripts ----
SKILL = r"C:\Users\wangyan\.claude\skills\nature-figure"
sys.path.insert(0, os.path.join(SKILL, "scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

# Windows 控制台默认 GBK，而本脚本 print 含 GBK 无码位的字符 —— 不改编码会在
# **成功路径**崩掉（活干完了却退出码 1）。别靠 PYTHONIOENCODING 兜底。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
OUT = os.path.join(PROJECT_ROOT, "05_论文", "figures")
os.makedirs(OUT, exist_ok=True)

# ---- MANDATORY font + SVG rules ----
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
# dict-update form: source preflight recognizes {"svg.fonttype": "none"} style
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})

# ---- Palette (nature-figure DEFAULT_COLORS) ----
PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3":        "#8BCF8B",
    "red_strong":     "#B64342",
    "neutral_light":  "#CFCECE",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#4D4D4D",
    "neutral_black":  "#272727",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
    "gold":           "#E8B800",
}
DEFAULT_COLORS = [
    PALETTE["blue_main"], PALETTE["green_3"], PALETTE["red_strong"],
    PALETTE["teal"], PALETTE["violet"], PALETTE["neutral_light"],
]

METHOD_COLORS = {
    "Coulomb": PALETTE["neutral_light"],
    "EKF":     PALETTE["neutral_mid"],
    "A2-0":    PALETTE["blue_secondary"],
    "A2-3":    PALETTE["blue_main"],
    "PureNN":  PALETTE["red_strong"],
    "A2-4":    PALETTE["teal"],
}

ABLATION_COLORS = {
    "A2-0": PALETTE["neutral_light"],
    "A2-1": PALETTE["violet"],
    "A2-2": PALETTE["teal"],
    "A2-3": PALETTE["blue_main"],
    "A2-4": PALETTE["green_3"],
}


def save_fig(fig, name, dpi=600, alignment=None):
    """Save SVG (primary) + PDF + TIFF + PNG; alignment gate for every figure
    (single-panel yields NOT APPLICABLE JSON, kept for the delivery record)."""
    base = os.path.join(OUT, name)
    opts = alignment if alignment is not None else {}
    require_matplotlib_panel_alignment(
        fig,
        json_out=base + ".alignment.json",
        overlay_svg=base + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
        **opts,
    )
    fig.savefig(base + ".svg", bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".tiff", dpi=dpi, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}.svg/.pdf/.tiff/.png")


def apply_style(font_size=9, lw=1.0):
    plt.rcParams.update({
        "font.size": font_size,
        "axes.linewidth": lw,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
    })


# =========================================================================== #
# Fig. 7 — Baseline comparison heatmap
# =========================================================================== #
def fig_baseline():
    apply_style(font_size=8, lw=0.8)
    df = pd.read_csv(os.path.join(RESULTS, "p5_01_baseline_comparison.csv"))

    # Order: methods
    method_order = ["M1_Coulomb", "M2_EKF", "M3_A2-0",
                    "M4_A2-3", "M5_pureNN", "M6_A2-4"]
    method_short = {
        "M1_Coulomb": "Coulomb", "M2_EKF": "EKF", "M3_A2-0": "A2-0",
        "M4_A2-3": "A2-3\n(ours)", "M5_pureNN": "Pure NN", "M6_A2-4": "A2-4",
    }
    # Order conditions: temperature then condition; n = files per condition
    cond_order = [
        "Cycle_25C",
        "UDDS_10C", "HWFET_10C", "US06_10C",
        "UDDS_0C", "HWFET_0C", "US06_0C",
        "UDDS_-10C", "HWFET_-10C", "US06_-10C",
        "UDDS_-20C", "HWFET_-20C", "US06_-20C",
    ]
    # ⚠️ 2026-09-26 修：n_files **真从 CSV 派生**（此前是一张写死的字典，
    #    注释却写着"从上面的表派生"——值碰巧对，但换数据就会漂）。
    #    来源：`p5_01_baseline_comparison.csv` 的 `n_files` 列，与图共用同一份数据。
    n_files = (df.drop_duplicates("eval_group")
                 .set_index("eval_group")["n_files"].to_dict())
    missing = [c for c in cond_order if c not in n_files]
    if missing:
        raise SystemExit(f"CSV 缺少这些条件的 n_files：{missing}")
    def _cond_label(c):
        wave, temp = c.split("_")
        return f"{wave}\n{temp.replace('-', '−').replace('C', '°C')}\n(n={n_files[c]})"

    cond_short = {c: _cond_label(c) for c in cond_order}

    mat = np.full((len(method_order), len(cond_order)), np.nan)
    for _, r in df.iterrows():
        i = method_order.index(r["method"])
        j = cond_order.index(r["eval_group"])
        mat[i, j] = r["RMSE"]

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    im = ax.imshow(mat, cmap="RdYlBu_r", aspect="auto",
                   vmin=0, vmax=np.nanmax(mat))
    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels([cond_short[c] for c in cond_order], fontsize=5.5)
    ax.set_yticks(range(len(method_order)))
    ax.set_yticklabels([method_short[m] for m in method_order], fontsize=7)
    ax.tick_params(length=0)

    # Annotate cells: white text only on genuinely dark cells (luminance test)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                v = mat[i, j]
                # Map value to RdYlBu_r color, decide text color by luminance
                norm_v = v / np.nanmax(mat)
                cmap = plt.get_cmap("RdYlBu_r")
                r_, g_, b_, _ = cmap(norm_v)
                lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if lum < 0.55
                        else PALETTE["neutral_black"])

    # Highlight "ours" row
    ours_i = method_order.index("M4_A2-3")
    ax.add_patch(plt.Rectangle((-0.5, ours_i - 0.5), mat.shape[1], 1,
                               fill=False, edgecolor="black",
                               linewidth=1.5, zorder=5))

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("RMSE (pp)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax.set_title("Baseline comparison across methods and conditions "
                 "(EKF+NN models: seed 7)",
                 fontsize=8, pad=6)
    save_fig(fig, "fig7_baseline", dpi=600)
    print("  Fig. 7 — Baseline heatmap done")


# =========================================================================== #
# Fig. 8 — Ablation grouped bars
# =========================================================================== #
def fig_ablation():
    apply_style(font_size=8, lw=0.8)
    # B2 升级：全量 OOD 多文件（每温度 9 文件），误差棒=文件间 SD
    df = pd.read_csv(os.path.join(RESULTS, "p3_06b_multifile_agg_seed10.csv"))

    eval_order = ["test_id(25C)", "ood(10C)", "ood(0C)",
                  "ood(-10C)", "ood(-20C)"]
    # ⚠️ 2026-09-26 修：n 改为**真从 CSV 派生**。此前写死 "n=4"/"n=9"——
    #    与 Fig.6 的同类问题同源（值碰巧对，注释却像是有来源，换数据即静默漂）。
    #    来源：本函数已经读进来的 `p3_06b_multifile_agg.csv` 的 `n_files` 列。
    n_by_grp = df.drop_duplicates("grp").set_index("grp")["n_files"].to_dict()
    _missing = [g for g in eval_order if g not in n_by_grp]
    if _missing:
        raise SystemExit(f"agg CSV 缺少这些组的 n_files：{_missing}")

    def _eval_label(g):
        n = n_by_grp[g]
        if g.startswith("test_id"):
            return f"25°C\n(ID, n={n})"
        return g[4:-1].rstrip("C").replace("-", "−") + f"°C\n(n={n})"

    eval_short = [_eval_label(g) for g in eval_order]
    exp_order = ["A2-0", "A2-1", "A2-2", "A2-3", "A2-4"]
    exp_labels = ["A2-0\n(meas.)", "A2-1\n(+ν)",
                  "A2-2\n(+NIS)", "A2-3\n(full)", "A2-4\n(K/P)"]

    n_eval = len(eval_order)
    n_exp = len(exp_order)
    fig, ax = plt.subplots(figsize=(6.5, 2.8))

    x = np.arange(n_eval)
    w = 0.15
    for k, exp in enumerate(exp_order):
        vals = []
        stds = []
        for ev in eval_order:
            row = df[(df["exp"] == exp) & (df["grp"] == ev)]
            if len(row):
                vals.append(row["RMSE_mean"].values[0])
                s = row["RMSE_std_files"].values[0]
                stds.append(0 if pd.isna(s) else s)
            else:
                vals.append(np.nan)
                stds.append(0)
        offset = (k - (n_exp - 1) / 2) * w
        ax.bar(x + offset, vals, width=w, yerr=stds,
               color=ABLATION_COLORS[exp], edgecolor="black",
               linewidth=0.6, label=exp_order[k],
               error_kw={"elinewidth": 0.8, "capsize": 1.5,
                         "capthick": 0.8})

    ekf_vals = []
    for ev in eval_order:
        row = df[df["grp"] == ev]
        if len(row):
            ekf_vals.append(row["ekf_RMSE_mean"].values[0])
        else:
            ekf_vals.append(np.nan)
    ax.plot(x, ekf_vals, "k--o", ms=3, lw=0.9, label="EKF", zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels(eval_short, fontsize=7)
    ax.set_ylabel("RMSE (pp)", fontsize=8)
    ax.legend(fontsize=6, ncol=3, loc="upper left", columnspacing=1.0,
              handlelength=1.2)
    ax.text(0.02, 0.83, "bars: mean ± SD across files (per-file over 10 seeds)",
            transform=ax.transAxes, fontsize=6, ha="left", va="top",
            color=PALETTE["neutral_dark"])
    ax.set_title("Ablation of EKF internal diagnostic features",
                 fontsize=8, pad=6)
    save_fig(fig, "fig8_ablation", dpi=600)
    print("  Fig. 8 — Ablation bars done")


# =========================================================================== #
# Fig. 9 — Robustness triptych
# =========================================================================== #
def fig_robustness():
    apply_style(font_size=7.5, lw=0.9)
    noise = pd.read_csv(os.path.join(RESULTS, "p5_03_noise.csv"))
    rate = pd.read_csv(os.path.join(RESULTS, "p5_03_sampling_rate.csv"))
    # p5_03_initial_bias.csv 已作废（满电起步 → 正向半区被 [0,1] clip 吞掉，P0-24）。
    # 现用文件 = **统一暖态口径 c′**（2026-09-27 起由 p5_03c_robustness_unified.py 产出）：
    # 截断至首个 soc_true≤0.80 + V₁/P 继承连续运行收敛值 + 只注入 SOC 偏移。
    # schema 与 noise/rate 不同：融合列叫 fused_steady_RMSE_pp（旧版叫 RMSE_pp）。
    bias = pd.read_csv(os.path.join(RESULTS, "p5_03_initial_bias_fixed.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    labels = ["a", "b", "c"]

    def _annotate_dmax(ax, base, series, x_pos=0.03, y_pos=0.97, kind="pct"):
        """QA fix: auto-scaled y axes exaggerate tiny variations — state the
        actual worst-case degradation so magnitude is readable at a glance.

        🔴 2026-10-04（意见10 P1-15）：面板 c（V 形）原标 "Δmax = +375.9%"。
        该百分比以**谷底最小值**为基线，故数值失真、且与 a/b 的百分比不同量纲。
        改为按 kind 输出：
          kind="pct"  → "+3.0%"（面板 a/b，单调上升，百分比合理）
          kind="rat"  → "4.8× baseline (+6.4 pp)"（面板 c，与正文 / 图注一致）
        """
        if kind == "rat":
            ratio = series.max() / base
            delta = series.max() - base
            txt = f"{ratio:.1f}× baseline (+{delta:.1f} pp)"
        else:
            txt = f"Δmax = +{(series.max() - base) / base * 100.0:.1f}%"
        ax.text(x_pos, y_pos, txt,
                transform=ax.transAxes, fontsize=6, ha="left", va="top",
                color=PALETTE["neutral_dark"])

    # a) Noise
    ax = axes[0]
    ax.plot(noise["noise_std_mV"], noise["RMSE_pp"], "-o",
            color=PALETTE["blue_main"], ms=4, lw=1.5)
    ax.set_xlabel("Voltage noise σ (mV)", fontsize=7.5)
    ax.set_ylabel("RMSE (pp)", fontsize=7.5)
    ax.set_title("Measurement noise", fontsize=7.5, pad=4)
    base_n = noise.loc[noise["noise_std_mV"] == 0, "RMSE_pp"].iloc[0]
    _annotate_dmax(ax, base_n, noise["RMSE_pp"])

    # b) Sampling rate
    ax = axes[1]
    ax.plot(rate["sampling_rate_Hz"], rate["RMSE_pp"], "-s",
            color=PALETTE["blue_main"], ms=4, lw=1.5)
    ax.set_xlabel("Sampling rate (Hz)", fontsize=7.5)
    ax.set_title("Sampling rate", fontsize=7.5, pad=4)
    ax.invert_xaxis()
    base_r = rate.loc[rate["sampling_rate_Hz"] == 1.0, "RMSE_pp"].iloc[0]
    _annotate_dmax(ax, base_r, rate["RMSE_pp"])

    # c) Initial bias — unified convention (c′, 2026-09-27; see
    #    10_代码/p5_03c_robustness_unified.py for the rationale).
    #    ⚠️ 2026-09-27 起：**" +20 pp 是分段伪影" 的说法已作废**。
    #    旧图把 +20 画成空心点并标 "artefact (segmentation)"，那是**冷启动口径**下的产物
    #    （EKF 在 k0 处以默认 P0=1e-2 重启 → 增益被放大 → 偏差被迅速抹平）。
    #    统一口径改用**热态起步**（P 继承收敛值 ~1e-6）后，偏差**持续**到稳态，
    #    +20 pp 读作 4.16 pp，**是真实退化，不是伪影**。标注与空心点一并删除。
    ax = axes[2]
    xb = bias["bias_pp"].to_numpy(dtype=float)
    yb = bias["fused_steady_RMSE_pp"].to_numpy(dtype=float)
    ax.plot(xb, yb, "-^", color=PALETTE["blue_main"], ms=4, lw=1.5)
    ax.set_xlabel("Initial SOC bias (pp)", fontsize=7.5)
    ax.axvline(0, color=PALETTE["neutral_mid"], ls=":", lw=0.7)
    ax.set_title("Initial SOC deviation", fontsize=7.5, pad=4)
    base_b = bias.loc[bias["bias_pp"] == 0, "fused_steady_RMSE_pp"].iloc[0]
    # 🔴 2026-09-27 QA 修复（nature-figure 的 audit_figure_collisions.py 判定
    #    **FIX BEFORE DELIVERY**）：原位置 x_pos=0.40/y_pos=0.97 时，
    #    `axvline(0)` 那条 **bias=0 的竖直虚线会从 `Δmax = +375.9%` 中间穿过**
    #    （findings: text-stroke FAIL）。
    #    V 形曲线左支很高（−20 pp → 8.15 pp），左侧没位置；**曲线右支上方、
    #    虚线右侧**才是空区 ⇒ 放到 x_pos=0.55。
    #    技能要求「从数据/边界推导标注位置，而不是写死 LABEL_Y」——
    #    这里 0.55 是由「虚线在 x=0.5、文字宽约 0.39 轴宽」推出来的。
    _annotate_dmax(ax, base_b, bias["fused_steady_RMSE_pp"],
                   x_pos=0.55, y_pos=0.80, kind="rat")

    fig.suptitle("Robustness of the EKF+NN fusion model",
                 fontsize=8, y=1.04)
    fig.tight_layout(pad=1.0)

    # alignment: 3 equal panels in one row
    alignment = dict(
        axes=list(axes),
        panel_ids=["a", "b", "c"],
        row_groups=[["a", "b", "c"]],
        require_panel_labels=True,
    )
    # add panel labels before alignment check
    from matplotlib.transforms import ScaledTranslation
    for ax, lab in zip(axes, labels):
        offset = ScaledTranslation(-4 / 72, 3 / 72, ax.figure.dpi_scale_trans)
        ax.text(0, 1, lab, transform=ax.transAxes + offset,
                fontsize=8, fontweight="bold", ha="left", va="bottom")
    fig.tight_layout(pad=1.0)

    save_fig(fig, "fig9_robustness", dpi=600, alignment=alignment)
    print("  Fig. 9 — Robustness triptych done")


# =========================================================================== #
# Fig. 2 — OCV-SOC curve
# =========================================================================== #
def fig_ocv():
    apply_style(font_size=8, lw=0.9)
    df = pd.read_csv(os.path.join(RESULTS, "ocv_soc_curve.csv"))

    # Raw C/20 discharge points as fit-quality evidence
    raw_path = os.path.join(PROJECT_ROOT, "04_实验", "数据", "processed",
                            "mcmaster_ocv_c20.csv.gz")
    raw = pd.read_csv(raw_path)
    dis = raw[raw["current_A"] > 0]          # discharge branch (positive I)
    step = max(1, len(dis) // 200)
    dis_sub = dis.iloc[::step]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    # Raw data (scatter) under the fitted polynomial (line)
    ax.scatter(dis_sub["soc_true"], dis_sub["voltage_V"], s=4,
               color=PALETTE["neutral_light"], edgecolors="none",
               zorder=1, label="C/20 discharge data")
    ax.plot(df["soc"], df["ocv_V"], "-", color=PALETTE["blue_main"],
            lw=1.8, zorder=2, label="Polynomial fit")

    ax.set_xlabel("SOC", fontsize=8)
    ax.set_ylabel("OCV (V)", fontsize=8)
    ax.set_title("OCV–SOC characteristic (25 °C)", fontsize=8, pad=5)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=6, loc="lower right", handletextpad=0.4)
    # Fit-quality evidence (from ocv_fit_report.md)
    ax.text(0.03, 0.97, "11th-order polynomial\nRMSE = 2.5 mV",
            transform=ax.transAxes, fontsize=6, ha="left", va="top",
            color=PALETTE["neutral_dark"])
    save_fig(fig, "fig2_ocv", dpi=600)
    print("  Fig. 2 — OCV curve (with raw data + fit evidence) done")


# =========================================================================== #
#: 只重出指定图时用（避免为改一张图而把另外三张的元数据时间戳一起刷新）
#:     python p5_06_figures.py robustness
BUILDERS = {
    "baseline": ("fig_baseline", "Fig. 6 baseline heatmap"),
    "ablation": ("fig_ablation", "Fig. 7 ablation bars"),
    "robustness": ("fig_robustness", "Fig. A.1 robustness triptych"),
    "ocv": ("fig_ocv", "Fig. 2 OCV curve"),
}


def main():
    print("=" * 60)
    print("P5-06 论文级配图")
    print("=" * 60)
    want = [a for a in sys.argv[1:] if not a.startswith("-")]
    todo = want or ["baseline", "ablation", "robustness", "ocv"]
    unknown = [t for t in todo if t not in BUILDERS]
    if unknown:
        print(f"未知图名 {unknown}；可选：{sorted(BUILDERS)}")
        return 2
    for name in todo:
        fn, desc = BUILDERS[name]
        print(f"\n--- {name}: {desc} ---")
        globals()[fn]()
    print(f"\n已生成 {todo}")
    print(f"全部图已保存至: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
