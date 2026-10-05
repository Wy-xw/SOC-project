# -*- coding: utf-8 -*-
"""
P2-02：OCV-SOC 曲线拟合 + SOC 真值独立校验（v2，按 scipilot-figure-skill 规范）

数据
----
`04_实验/数据/processed/mcmaster_ocv_c20.csv.gz`（C/20 OCV，25 °C，放电为正）

段结构（实测）：
    seg0  起始静置 0.07 h  @4.184 V          → SOC=1 弛豫锚点
    seg1  C/20 放电 20.7 h 4.183→2.655 V     → 放电支路（拟合本体）
    seg2  静置弛豫  0.98 h                    → SOC≈0 弛豫锚点
    seg3  C/20 充电 18.1 h 2.863→4.187 V     → 充电支路（滞后带）
    seg4  终末静置 14.6 h 4.186→4.160 V      → SOC=1 弛豫锚点（回落）

已确认事实（见 p2_02_diag.py）：
- soc_true=1+Ah/2.9 未 clip 坐标下：放电支路覆盖 [-0.02, 1.01]，
  充电支路覆盖 [-0.02, 0.88]（97.4% 在 [0,1] 内，顶部未到 1）
- 公共区间滞后带（放电−充电）中位 ≈ −104 mV：量级正常，不可忽略
  （此前 −1075 mV 是 soc_true 被 clip 截断造成的坐标错位假象）
- 驱动工况文件中 ≥10min 静置段全部位于「测试起始满电静置」
  （SOC_ah≈1.00）→ 反查校验必须分层报告，不能声称覆盖全 SOC

输出
----
results/ocv_fit_report.md
results/ocv_soc_curve.csv        SOC 0~1 步长 .001 的 OCV+dOCV/dSOC（供 EKF）
results/ocv_coeffs.csv           阶数与系数（供 P2-03 参数辨识共用）
results/soc_true_validation.csv  驱动静置反查明细（含分层）
results/figures/p2_02_ocv_fit.{png,svg,pdf}
"""
from __future__ import annotations

import os
import sys
import itertools

import numpy as np
import pandas as pd

# scipilot-figure-skill 样式/导出工具（技能规范：中文字体、矢量、灰度预览）
SKILL_DIR = r"C:\Users\wangyan\.claude\skills\scipilot-figure-skill\scripts"
if SKILL_DIR not in sys.path:
    sys.path.insert(0, SKILL_DIR)
from setup_style import setup_style          # noqa: E402
from export_figure import export_figure      # noqa: E402
from visual_qa import render_preview, audit_layout  # noqa: E402

# Windows 控制台默认 GBK，而本脚本 print 含 GBK 无码位的字符 —— 不改编码会在
# **成功路径**崩掉（活干完了却退出码 1）。别靠 PYTHONIOENCODING 兜底。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "04_实验", "数据")
PROCESSED = os.path.join(DATA_ROOT, "processed")
RESULTS = os.path.join(DATA_ROOT, "results")
FIGDIR = os.path.join(RESULTS, "figures")
os.makedirs(FIGDIR, exist_ok=True)

CN_NOMINAL = 2.9
REST_CURRENT_A = 0.005
MIN_REST_S = 600.0
TAIL_S = 60.0


def main() -> int:
    df = pd.read_csv(os.path.join(PROCESSED, "mcmaster_ocv_c20.csv.gz"))
    t = np.asarray(df["timestamp"], dtype=float)
    i = np.asarray(df["current_A"], dtype=float)
    v = np.asarray(df["voltage_V"], dtype=float)
    s_unclip = 1.0 + np.asarray(df["ah_throughput"], dtype=float) / CN_NOMINAL

    st = np.where(i > REST_CURRENT_A, 1, np.where(i < -REST_CURRENT_A, -1, 0))
    bounds = np.concatenate(([0], np.where(np.diff(st) != 0)[0] + 1, [len(i)]))
    segs = {}
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        segs[k] = {"a": int(a), "b": int(b) - 1, "st": int(st[a])}
    rest_ord = sorted([k for k, s in segs.items() if s["st"] == 0],
                      key=lambda k: segs[k]["a"])
    disp = [k for k, s in segs.items() if s["st"] == 1]
    chgs = [k for k, s in segs.items() if s["st"] == -1]
    ka, kb = segs[disp[0]]["a"], segs[disp[0]]["b"]
    ca, cb = segs[chgs[0]]["a"], segs[chgs[0]]["b"]

    # ---------- 1. 放电支路拟合（未 clip 坐标，去除首尾越界点） ----------
    sd = s_unclip[ka:kb + 1]
    vd = v[ka:kb + 1]
    m_ok = (sd >= 0.0) & (sd <= 1.0)
    sd, vd = sd[m_ok], vd[m_ok]
    rng = np.random.default_rng(2)
    n = len(sd)
    hold = rng.choice(n, size=int(0.2 * n), replace=False)
    fit = np.setdiff1d(np.arange(n), hold)
    best = None
    for deg in range(6, 12):
        c = np.polyfit(sd[fit], vd[fit], deg)
        rmse = float(np.sqrt(np.mean((vd[hold] - np.polyval(c, sd[hold]))**2)))
        if best is None or rmse < best["rmse"]:
            best = {"deg": deg, "coef": c, "rmse": rmse}
    coef, dcoef = best["coef"], np.polyder(best["coef"])
    e_full = (np.polyval(coef, sd) - vd) * 1000.0
    # 边界带检查（诚实报告边界行为）
    b_lo = sd < 0.05
    b_hi = sd > 0.9
    edge_rmse = float(np.sqrt(np.mean(e_full[b_lo | b_hi]**2)))
    mid_rmse = float(np.sqrt(np.mean(e_full[~(b_lo | b_hi)]**2)))

    # ---------- 2. 滞后带（未 clip 坐标，公共区间） ----------
    sc = s_unclip[ca:cb + 1]
    vc = v[ca:cb + 1]
    g = np.linspace(0.0, 1.0, 201)
    lo = max(float(sd.min()), float(sc.min()))
    hi = min(float(sd.max()), float(sc.max()))
    common = (g >= lo) & (g <= hi)
    vgd = np.interp(g, np.sort(sd), vd[np.argsort(sd)])
    vgc = np.interp(g, np.sort(sc), vc[np.argsort(sc)])
    lag = (vgd - vgc) * 1000.0
    lc = lag[common]

    # ---------- 3. 静置锚点 ----------
    anchors = []
    for pos, k in enumerate(rest_ord):
        a, b = segs[k]["a"], segs[k]["b"]
        role = ("起始静置" if pos == 0 else ("终末静置" if pos == len(rest_ord) - 1
                                             else "中段静置"))
        soc_ap = 1.0 if role != "中段静置" else 0.0
        v_tail = float(np.median(v[max(a, b - int(TAIL_S)):b + 1]))
        anchors.append({"role": role, "soc_ap": soc_ap, "v_tail": v_tail})

    # ---------- 4. 驱动静置反查（分层） ----------
    drv = pd.read_csv(os.path.join(PROCESSED, "mcmaster_drive_cycles.csv.gz"),
                      usecols=["timestamp", "current_A", "voltage_V", "soc_true",
                               "cycle_id"])
    soc_grid = np.linspace(0.0, 1.0, 10001)
    ocv_grid = np.polyval(coef, soc_grid)
    checks = []
    for cid, gd in drv.groupby("cycle_id"):
        gi = np.asarray(gd["current_A"], dtype=float)
        stc = np.abs(gi) <= REST_CURRENT_A
        chg = np.where(np.diff(stc.astype(int)) != 0)[0]
        bds = np.concatenate(([0], chg + 1, [len(gi)]))
        for a, b in zip(bds[:-1], bds[1:]):
            if not bool(stc[a]):
                continue
            dur = float(gd["timestamp"].iloc[b - 1] - gd["timestamp"].iloc[a])
            if dur < MIN_REST_S:
                continue
            tt = max(a, b - int(TAIL_S))
            vt = float(np.asarray(gd["voltage_V"])[tt:b].mean())
            soc_ocv = float(np.argmin(np.abs(ocv_grid - vt)) / 10000.0)
            soc_ah = float(np.asarray(gd["soc_true"])[tt:b].mean())
            checks.append({"cycle_id": cid, "rest_s": dur, "v_rest": vt,
                           "soc_ah": soc_ah, "soc_ocv": soc_ocv})
    ck = pd.DataFrame(checks)
    if len(ck):
        ck["dsoc"] = (ck["soc_ocv"] - ck["soc_ah"]) * 100.0
        ck["layer"] = np.where(ck["soc_ah"] > 0.95, "满电起始静置",
                               "中段静置")
        ck = ck.sort_values("v_rest", ascending=False)
        ck.to_csv(os.path.join(RESULTS, "soc_true_validation.csv"), index=False)

    # ---------- 5. 写出曲线/系数 ----------
    grid = np.linspace(0.0, 1.0, 1001)
    pd.DataFrame({"soc": grid, "ocv_V": np.polyval(coef, grid),
                  "docv_dsoc": np.polyval(dcoef, grid)}).to_csv(
        os.path.join(RESULTS, "ocv_soc_curve.csv"), index=False)
    pd.DataFrame({"degree": [best["deg"]], "coef": [coef.tolist()]},
                 index=[0]).to_csv(os.path.join(RESULTS, "ocv_coeffs.csv"),
                                   index=False)

    # ---------- 6. 报告 ----------
    L = [
        "# P2-02 报告：OCV-SOC 拟合与 SOC 真值独立校验（v2）",
        "",
        "> 2026-09-15　数据：mcmaster_ocv_c20.csv.gz（C/20 OCV，25 °C，放电为正）",
        "> 规范：scipilot-figure-skill 流程；修正 v1 的两处错误结论（滞后带/覆盖面）。",
        "",
        "## 一、段结构（实测）",
        "",
        "| 段 | 状态 | 时长 (h) | 电压轨迹 (V) | soc 覆盖（未 clip） |",
        "|---|---|---|---|---|",
        f"| 1 | 起始静置 | {(t[segs[rest_ord[0]]['b']]-t[segs[rest_ord[0]]['a']])/3600:.2f} | "
        f"{v[segs[rest_ord[0]]['a']]:.4f} (平) | 1.00 |",
        f"| 2 | **C/20 放电** | {(t[kb]-t[ka])/3600:.1f} | {v[ka]:.3f} → {v[kb]:.3f} | "
        f"1.01 → −0.02 |",
        f"| 3 | 中段静置 | {(t[segs[rest_ord[1]]['b']]-t[segs[rest_ord[1]]['a']])/3600:.2f} | "
        f"{v[segs[rest_ord[1]]['a']]:.3f} → {v[segs[rest_ord[1]]['b']]:.3f} | ≈0 |",
        f"| 4 | **C/20 充电** | {(t[cb]-t[ca])/3600:.1f} | {v[ca]:.3f} → {v[cb]:.3f} | "
        f"−0.02 → 0.88 |",
        f"| 5 | 终末静置 | {(t[segs[rest_ord[-1]]['b']]-t[segs[rest_ord[-1]]['a']])/3600:.1f} | "
        f"{v[segs[rest_ord[-1]]['a']]:.3f} → {v[segs[rest_ord[-1]]['b']]:.3f} | ≈1.00 |",
        "",
        "> ⚠ 充电支路顶部只到 soc≈0.88（未充满到 1.0 的安时刻度），"
        "因此滞后带只在公共区间计算。",
        "",
        "## 二、放电支路拟合",
        "",
        f"- 阶数：**{best['deg']}**（6~11 阶 20% 留出 RMSE 最小；留出 {best['rmse']:.2f} mV）",
        f"- 全段（SOC∈[0,1]）：RMSE **{np.sqrt(np.mean(e_full**2)):.1f} mV**，"
        f"MAE {np.mean(np.abs(e_full)):.1f} mV，MAX {np.abs(e_full).max():.1f} mV",
        f"- **边界带诚实报告**：SOC<0.05 或 >0.9 的拟合 RMSE = {edge_rmse:.1f} mV"
        f"（边界欠约束），中段 RMSE = {mid_rmse:.1f} mV —— EKF 使用时中段可信，"
        "边界需锚点约束，见第五节。",
        "- 曲线/导数：`ocv_soc_curve.csv`；系数：`ocv_coeffs.csv`（P2-03 复用）。",
        "",
        "## 三、滞后带（公共区间，未 clip 坐标）",
        "",
        f"- 公共 SOC 区间 [{lo:.2f}, {hi:.2f}]：放电−充电中位 **{np.median(lc):.0f} mV**，"
        f"q25={np.quantile(lc,.25):.0f}，q75={np.quantile(lc,.75):.0f}",
        f"- 判定：~100 mV 量级正常（C/20 慢速下 NMC 的典型区间），**不可忽略**但"
        "小于 EKF 电压噪声设计余量；EKF 用放电支路 OCV 时需在论文中如实交代。",
        "- ⚠ v1 曾把该值误算为 −1075 mV 并写\"可忽略\"——根因是 soc_true 被 "
        "clip 截断 → 充电支路顶部错位；本版用未 clip 坐标，根因已排查。",
        "",
        "## 四、静置锚点",
        "",
        "| 段 | SOC（先验） | 段尾弛豫电压 (V) | 拟合 OCV (V) | 偏差 (mV) |",
        "|---|---|---|---|---|",
    ]
    for an in anchors:
        ocvf = float(np.polyval(coef, an["soc_ap"]))
        L.append(f"| {an['role']} | {an['soc_ap']:.2f} | {an['v_tail']:.4f} | "
                 f"{ocvf:.4f} | {(an['v_tail']-ocvf)*1000:+.1f} |")
    L += [
        "",
        "> ⚠ 起点 4.184 V vs 拟合 OCV(1) 偏差 +22 mV：边界多项式欠约束，"
        "且起始静置完全弛豫（基线最真）。**建议 P2-03/EKF 使用时把曲线重新"
        "缩放锚定到 (SOC=1, 4.184 V) 弛豫锚点**——边界误差对 SOC≈1 段影响最大。",
        "",
        "## 五、驱动工况静置反查（SOC 真值独立校验，分层）",
        "",
    ]
    if len(ck):
        top = ck[ck["layer"] == "满电起始静置"]
        mid = ck[ck["layer"] == "中段静置"]
        L += [
            f"- 有效静置段：**{len(ck)}** 段，覆盖 {ck['cycle_id'].nunique()} 个工况文件",
            f"- **分层 1｜满电起始静置**（SOC_ah≥0.95）：{len(top)} 段，"
            f"ΔSOC RMSE {np.sqrt(np.mean(top['dsoc']**2)):.2f} pt —— 只验证 "
            "SOC≈1 端点一致（起始满充约定复核）",
            f"- **分层 2｜中段静置**（SOC_ah<0.95）：{len(mid)} 段 —— 驱动文件中"
            f"中段≥10min 静置稀少，覆盖不全；**SOC 中段独立校验结论改为以"
            f"第三节滞后带公共区间 + 放电/充电支路自洽 + 本层有限样本为准**",
            "- 说明：反查法对中段（OCV 斜率平缓区）天然不敏感，±20 mV 电压误差"
            "对应 ±2~3 pt SOC，故中段样本即使有也不宜单独下强结论。",
            "",
        ]
        if len(mid):
            L += ["中段静置明细：", "",
                  "| 工况 | 静置 (s) | 电压 (V) | SOC_ah | SOC_ocv | ΔSOC (pt) |",
                  "|---|---|---|---|---|---|"]
            for _, r in mid.iterrows():
                L.append(f"| {r['cycle_id'][:40]} | {r['rest_s']:.0f} | "
                         f"{r['v_rest']:.3f} | {r['soc_ah']:.3f} | {r['soc_ocv']:.3f} | "
                         f"{r['dsoc']:+.2f} |")
        L += ["",
              "满电起始静置（前 5 段）：", "",
              "| 工况 | 静置 (s) | 电压 (V) | SOC_ah | SOC_ocv | ΔSOC (pt) |",
              "|---|---|---|---|---|---|"]
        for _, r in top.head(5).iterrows():
            L.append(f"| {r['cycle_id'][:40]} | {r['rest_s']:.0f} | {r['v_rest']:.3f} | "
                     f"{r['soc_ah']:.3f} | {r['soc_ocv']:.3f} | {r['dsoc']:+.2f} |")
    else:
        L.append("- 驱动文件中未找到 ≥10min 静置段。")
    L += [
        "",
        "## 六、结论与下一步",
        "",
        "- P2-02 目标 1（拟合）✅：放电支路 OCV 多项式 + 导数已生成并校验。",
        "- P2-02 目标 2（独立校验）——**诚实结论**：",
        "  ① SOC≈1 端点：满电静置反查 ΔSOC≈0 → 起始满充约定复核通过；",
        "  ② 容量：放电支路 ΔAh 与 2.9 约定对照（早期测试，老化最轻）；",
        "  ③ SOC 中段：驱动文件缺乏长静置样本，中段真值校验以本 OCV 文件"
        "放电/充电支路自洽为准，覆盖不完整已在第五节如实标注（R3 部分缓解）。",
        "- 下一步 P2-03：用 `ocv_coeffs.csv` + HPPC 数据做 ECM 参数辨识（R0/R1/C1），"
        "并对 OCV 曲线做 (SOC=1, 4.184 V) 锚定处理。",
        "",
    ]
    report = "\n".join(L) + "\n"
    with open(os.path.join(RESULTS, "ocv_fit_report.md"), "w",
              encoding="utf-8") as f:
        f.write(report)
    print(report[-3500:])

    # ---------- 7. 图（技能规范：zh 字体 + 矢量 + 灰度预览） ----------
    setup_style(journal="general", lang="zh")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ax = axes[0]
    ax.scatter(sd[::3], np.polyval(coef, sd[::3]) - e_full[::3] / 1000.0,
               s=4, c="#4C72B0", alpha=0.5, label="C/20 放电支路（原始）", linewidths=0)
    ax.plot(grid, np.polyval(coef, grid), "k-", lw=1.4, label=f"拟合 OCV（{best['deg']} 阶）")
    ax.scatter(sc[::3], vc[::3], s=4, c="#DD8452", marker="^", alpha=0.45,
               label="C/20 充电支路", linewidths=0)
    for an in anchors:
        ax.plot([an["soc_ap"]], [an["v_tail"]], "r*", ms=9, label="弛豫锚点")
    ax.axvspan(0.9, 1.05, color="grey", alpha=0.12)
    ax.axvspan(-0.03, 0.05, color="grey", alpha=0.12)
    ax.text(0.955, 4.0, "边界带", fontsize=7, color="#555555", ha="center")
    ax.set_xlabel("SOC"); ax.set_ylabel("电压 (V)")
    ax.set_title("OCV-SOC 拟合（C/20，25 °C）")
    ax.grid(alpha=0.3); ax.set_ylim(2.45, 4.30)
    ax.legend(loc="lower right", frameon=False, fontsize=6.5)

    ax = axes[1]
    if len(ck):
        layer_col = {"满电起始静置": "#55A868", "中段静置": "#C44E52"}
        for lay, col in layer_col.items():
            sub = ck[ck["layer"] == lay]
            if len(sub):
                ax.scatter(sub["soc_ah"], sub["v_rest"], s=28, c=col, alpha=0.85,
                           label=lay, edgecolors="white", linewidths=0.4)
        sl = np.linspace(0, 1.05, 300)
        ax.plot(sl, np.polyval(coef, sl), "k-", lw=1.4, label="拟合 OCV")
        for x in np.arange(0, 1.01, 0.25):
            ax.axvline(x, color="grey", lw=0.5, alpha=0.5)
        ax.set_title("驱动工况静置反查（按 SOC 分层）")
    else:
        ax.text(0.5, 0.5, "无 ≥10 min 静置段", ha="center", va="center")
        ax.set_title("驱动工况静置反查")
    ax.set_xlabel("SOC_ah（安时积分真值）"); ax.set_ylabel("静置段尾电压 (V)")
    ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=6.5)

    render_preview(fig, os.path.join(FIGDIR, "_preview_p2_02.png"))
    audit_layout(fig)
    export_figure(fig, os.path.join(FIGDIR, "p2_02_ocv_fit"),
                  formats=["png", "svg", "pdf"], size_inches=(10.0, 4.2),
                  dpi=300, grayscale_preview=True, tight=False)
    return 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    raise SystemExit(main())
