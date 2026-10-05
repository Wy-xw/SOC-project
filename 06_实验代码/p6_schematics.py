# -*- coding: utf-8 -*-
"""
P6 图件补齐（一）：示意图三张
  Fig.1  1RC-ECM 电路图
  Fig.3  总体框架框图（核心图）
  Fig.5  网络与特征结构（合并原 Fig.4 特征构造）
输出：05_论文/figures/（svg 主 + pdf + png 600dpi，英文标注，投稿级）
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

import sys

# Windows 控制台默认 GBK，而本脚本 print 含 GBK 无码位的字符 —— 不改编码会在
# **成功路径**崩掉（活干完了却退出码 1）。别靠 PYTHONIOENCODING 兜底。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.linewidth": 0,
})

OUT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "05_论文", "figures")
os.makedirs(OUT, exist_ok=True)

INK = "#222222"
GRAY = "#7F8C8D"
BLUE = "#1F618D"
TEAL = "#148F77"
ACCENT = "#B03A2E"
FILL_L = "#EEF3F8"
FILL_D = "#FDF2E9"


def save(fig, name):
    b = os.path.join(OUT, name)
    # 单面板图：跑对齐门并留 NOT APPLICABLE 档（规范要求保留 JSON）
    import sys as _s
    _s.path.insert(0, r"C:\Users\wangyan\.claude\skills\nature-figure\scripts")
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(
        fig, json_out=b + ".alignment.json",
        overlay_svg=b + ".alignment.svg", strict=True)
    fig.savefig(b + ".svg", bbox_inches="tight")
    fig.savefig(b + ".pdf", bbox_inches="tight")
    fig.savefig(b + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("saved:", name)


def rbox(ax, x, y, w, h, text, fc="white", ec=INK, fs=7.5, lw=1.1,
         tc=INK, z=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=fc, ec=ec, lw=lw, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=z + 1)


def arr(ax, p1, p2, color=INK, lw=1.2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, color=color,
                                 lw=lw, linestyle=ls, mutation_scale=9,
                                 zorder=6, shrinkA=0, shrinkB=0))


# ═══════════════════════════════════════════════════
# Fig. 1  1RC-ECM 电路图
# ═══════════════════════════════════════════════════
def fig1():
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    # OCV 源
    rbox(ax, 4, 15, 14, 10, "OCV\n(SOC)", fc=FILL_L, ec=BLUE, fs=7)
    # 顶线 → R0
    ax.plot([18, 26], [25, 25], color=INK, lw=1.2, zorder=3)
    rbox(ax, 26, 21, 12, 8, "$R_0$", fc="white", ec=INK, fs=8)
    ax.plot([38, 44], [25, 25], color=INK, lw=1.2, zorder=3)
    # 电流箭头
    arr(ax, (20.5, 27.4), (24.5, 27.4), color=ACCENT, lw=1.2)
    ax.text(22.5, 29, "$I$", fontsize=7.5, color=ACCENT,
            ha="center", va="bottom")

    # 并联支路
    ax.plot([44, 44], [10, 40], color=INK, lw=1.2, zorder=3)   # n1
    ax.plot([64, 64], [10, 40], color=INK, lw=1.2, zorder=3)   # n2
    ax.plot([44, 48], [36, 36], color=INK, lw=1.2)
    rbox(ax, 48, 32, 12, 8, "$R_1$", fc="white", ec=INK, fs=8)
    ax.plot([60, 64], [36, 36], color=INK, lw=1.2)
    ax.plot([44, 48], [16, 16], color=INK, lw=1.2)
    ax.plot([49.5, 49.5], [13.5, 18.5], color=INK, lw=1.6)      # 电容板
    ax.plot([51.0, 51.0], [13.5, 18.5], color=INK, lw=1.6)
    ax.plot([52.5, 64], [16, 16], color=INK, lw=1.2)
    ax.text(54.5, 13.2, "$C_1$", fontsize=8)

    # 端口
    ax.plot([64, 88], [25, 25], color=INK, lw=1.2, zorder=3)
    ax.plot([88, 88], [6, 25], color=INK, lw=1.2, zorder=3)
    ax.plot([88, 11], [6, 6], color=INK, lw=1.2, zorder=3)
    ax.plot([11, 11], [6, 15], color=INK, lw=1.2, zorder=3)
    ax.plot(88, 25, "o", color=INK, ms=3.5, zorder=6)
    ax.plot(88, 6, "o", color=INK, ms=3.5, zorder=6)
    ax.text(91.5, 25, "$+$", fontsize=8, va="center")
    ax.text(91.5, 6, "$-$", fontsize=8, va="center")
    ax.text(95, 15.5, "$V_t$", fontsize=8.5, va="center", ha="center")

    save(fig, "fig1_ecm_circuit")


# ═══════════════════════════════════════════════════
# Fig. 3  总体框架框图（核心图）
# ═══════════════════════════════════════════════════
def fig3():
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.set_xlim(0, 178)
    ax.set_ylim(0, 64)
    ax.axis("off")

    rbox(ax, 4, 27, 18, 12, "Sensors\n$V_t$, $I$, $T$", fc=FILL_L,
         ec=BLUE, fs=7.5)
    rbox(ax, 30, 25, 22, 16, "EKF\n(1RC-ECM)", fc="white", ec=INK, fs=8)
    rbox(ax, 62, 46, 26, 10, "$SOC^{EKF}$", fc="white", ec=INK, fs=8)
    rbox(ax, 62, 6, 30, 15,
         "EKF internal diagnostics\n$\\nu$, NIS, $K$, diag($P$)",
         fc=FILL_D, ec=ACCENT, fs=7)
    rbox(ax, 104, 26, 24, 14, "Feature builder\n13 ch, $L=30$",
         fc="white", ec=INK, fs=7.5)
    rbox(ax, 136, 26, 18, 14, "GRU(16)\nDense(8)", fc="white",
         ec=INK, fs=7.5)
    rbox(ax, 136, 46, 18, 10, "Fusion\n(Eq. 12)", fc=FILL_L, ec=BLUE,
         fs=7)

    arr(ax, (22, 33), (30, 33))
    arr(ax, (52, 36), (62, 49))
    arr(ax, (52, 29), (62, 14))
    arr(ax, (92, 50), (104, 36))
    arr(ax, (92, 13), (104, 30))
    arr(ax, (128, 33), (136, 33))
    ax.text(132, 36.5, "$\\hat{e}$", fontsize=8, color=ACCENT,
            fontweight="bold", ha="center")
    arr(ax, (145, 40), (145, 46))          # GRU → Fusion 底边
    # SOC_EKF → Fusion：从其右缘出发，绕上方走廊进 Fusion 顶边
    ax.plot([88, 94], [51, 51], color=INK, lw=1.2, zorder=3)
    ax.plot([94, 94], [51, 60], color=INK, lw=1.2, zorder=3)
    ax.plot([94, 145], [60, 60], color=INK, lw=1.2, zorder=3)
    arr(ax, (145, 60), (145, 56.2))
    arr(ax, (154, 51), (170, 51))
    ax.text(163, 53.5, "$SOC^{final}$", fontsize=8, ha="center")

    save(fig, "fig3_framework")


# ═══════════════════════════════════════════════════
# Fig. 5  网络与特征结构（合并 Fig.4）
# ═══════════════════════════════════════════════════
def fig5():
    fig, ax = plt.subplots(figsize=(8.8, 2.9))
    ax.set_xlim(0, 168)
    ax.set_ylim(0, 44)
    ax.axis("off")

    # ── 上排：13 通道 ──
    meas = ["SOC_EKF", "V_t", "I", "|I|", "T", "ΔV_t", "ΔSOC"]
    diag = ["ν", "NIS", "K0", "K1", "P00", "P11"]
    x0, w = 4, 12.0
    for i, t in enumerate(meas + diag):
        fc = FILL_L if i < 7 else FILL_D
        ec = BLUE if i < 7 else ACCENT
        rbox(ax, x0 + i * w, 27, w - 1.8, 8.5, t, fc=fc, ec=ec, fs=6.6,
             lw=0.9)
    ax.text(84, 40.5, "Input channels (F = 13), window L = 30 → tensor "
            "30 × 13   |   blue: measurements (7), red: EKF diagnostics "
            "(6)", fontsize=6.8, ha="center")
    ax.add_patch(FancyBboxPatch((x0, 22), 7 * w - 1.8, 3,
                                boxstyle="round,pad=0.2", fc=FILL_L,
                                ec=BLUE, lw=0.9))
    ax.add_patch(FancyBboxPatch((x0 + 7 * w, 22), 6 * w - 1.8, 3,
                                boxstyle="round,pad=0.2", fc=FILL_D,
                                ec=ACCENT, lw=0.9))

    # ── 下排：网络 ──
    rbox(ax, 6, 4, 22, 11, "GRU(16)\nunroll", fc="white", ec=INK,
         fs=6.8)
    rbox(ax, 34, 4, 20, 11, "Dense(8)\nReLU", fc="white", ec=INK,
         fs=6.8)
    rbox(ax, 60, 4, 20, 11, "Dense(1)\nresidual", fc=FILL_L,
         ec=BLUE, fs=6.8)
    arr(ax, (28, 9.5), (34, 9.5))
    arr(ax, (54, 9.5), (60, 9.5))
    arr(ax, (80, 9.5), (88, 9.5))
    arr(ax, (10, 26.5), (10, 15.4))
    ax.text(92, 9.5, "≈ 1.6 k parameters   •   float16 TFLite = 105 kB",
            fontsize=6.8, color=GRAY, va="center")

    save(fig, "fig5_network")


if __name__ == "__main__":
    fig1()
    fig3()
    fig5()
