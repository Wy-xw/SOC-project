# -*- coding: utf-8 -*-
"""P2-07d / P0-27：复算 `results/s1_残差可学性判定.md` 的全部关键数字

背景（P0-27）
-------------
`s1_残差可学性判定.md`（止损点 S1 的判定材料）头部自称由
`p2_07_residual.py`(E1–E3) + `p2_07c_probe.py`(E4 探针) 生成，但这两个脚本
**都不写这个文件**：前者写 `residual_structure_report.md`，后者只 print。
本脚本把判定书里的每个关键数字**重新算一遍**，与书中声称值逐条比对，
落盘为 `results/s1_verdict_recompute.md`——让该判定书有可核对的复算脚本。

复算口径（严格沿用两个原脚本，未作任何改动）
--------------------------------------------
- E1/E2/E3：逻辑照抄 `p2_07_residual.py` 的 main()（BURN_IN=300、I_REST=0.05、
  Ljung-Box m=20、Welch nperseg=min(256, n//2)、放电段 |I|>0.05）
- E4 探针：逻辑照抄 `p2_07c_probe.py`（train+val 6 文件拟合 Ridge λ=1，
  特征 z-score 用训练折统计量；F2/F3 的 "soc" = **SOC_EKF**（`h["soc"]`），
  与 E1–E3 里用 `soc_true` 的口径不同——这是原脚本的行为，此处保持原样）
- 声称值：逐条抄自 `s1_残差可学性判定.md`，并标注该文件行号
  （行号对应 **2026-09-25 版的该文件**；该文件头部在 P0-27 补了溯源说明，
  若之后再改动其头部，行号需同步更新本脚本里的 loc 字符串）

输入：results/ekf_baseline_history.npz（8 文件）、results/ekf_train_history.npz（6 文件）
用法：python p2_07d_s1_verdict.py
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
OUT_MD = os.path.join(RESULTS, "s1_verdict_recompute.md")
BURN_IN = 300
I_REST = 0.05
LAM = 1.0

CHECKS: list[tuple] = []


def chk(loc: str, item: str, claimed: float, got: float, tol: float) -> None:
    ok = abs(float(got) - claimed) <= tol
    CHECKS.append((loc, item, f"{claimed:g}", f"{got:.4g}", tol, ok))


def chk_range(loc: str, item: str, lo: float, hi: float,
              glo: float, ghi: float, tol: float) -> None:
    ok = (glo >= lo - tol) and (ghi <= hi + tol)
    CHECKS.append((loc, item, f"{lo:g} ~ {hi:g}",
                   f"{glo:.3g} ~ {ghi:.3g}", tol, ok))


def condition_of(cycle_id: str) -> str:
    return cycle_id.split()[-1].replace("_Pan18650PF", "")


def load_histories(path: str) -> dict[str, dict[str, np.ndarray]]:
    z = np.load(path, allow_pickle=False)
    files: dict[str, dict[str, np.ndarray]] = {}
    for key in z.files:
        cid, field = key.split("::", 1)
        files.setdefault(cid, {})[field] = z[key]
    return files


def ljung_box(x: np.ndarray, m: int = 20) -> float:
    n = len(x)
    xc = x - x.mean()
    den = float(xc @ xc)
    if den <= 0:
        return 1.0
    acf = np.array([xc[k:] @ xc[:-k] / den for k in range(1, m + 1)])
    Q = n * (n + 2) * float(np.sum(acf**2 / (n - np.arange(1, m + 1))))
    from scipy.stats import chi2
    return float(chi2.sf(Q, m))


# --------------------------------------------------------------------------- #
# E1/E2/E3（照抄 p2_07_residual.py）
# --------------------------------------------------------------------------- #

def run_e123(files) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_e1, rows_e2, rows_e3 = [], [], []
    for cid, h in sorted(files.items()):
        n = len(h["soc_true"])
        sl = slice(BURN_IN, n)
        nu = h["nu"][sl]
        e = (h["soc"] - h["soc_true"])[sl] * 100.0
        i_abs = np.abs(h["current_A"][sl])
        temp = h["temperature_C"][sl]
        soc = h["soc_true"][sl]
        dis = i_abs > I_REST
        cond = condition_of(cid)

        xc = nu - nu.mean()
        den = float(xc @ xc)
        acf10 = [float(xc[k:] @ xc[:-k] / den) for k in range(1, 11)]
        rows_e1.append({"cond": cond, "nu_std_mV": float(nu.std() * 1000),
                        "acf1": acf10[0], "acf2": acf10[1],
                        "LB_p": ljung_box(nu, m=20)})

        def sp(a, b):
            r, _ = stats.spearmanr(a, b)
            return float(r)
        r_i = sp(i_abs[dis], nu[dis]) if dis.sum() > 50 else np.nan
        r_is = sp(i_abs[dis], e[dis]) if dis.sum() > 50 else np.nan
        r_ts = sp(temp, e)
        r_ss = sp(soc[dis], e[dis]) if dis.sum() > 50 else np.nan
        rows_e2.append({"cond": cond, "nu~|I|": r_i, "e~|I|": r_is,
                        "e~T": r_ts, "e~SOC": r_ss,
                        "e_bias_pp": float(e.mean())})

        f, P = signal.welch(nu, fs=1.0, nperseg=min(256, len(nu) // 2))
        msk = f > 0
        alpha = float(np.polyfit(np.log(f[msk]), np.log(P[msk]), 1)[0])
        cum = np.cumsum(P)
        frac_lo = float(cum[int(0.1 * len(P))] / cum[-1])
        rows_e3.append({"cond": cond, "alpha": alpha, "frac_lo": frac_lo})
    return (pd.DataFrame(rows_e1), pd.DataFrame(rows_e2), pd.DataFrame(rows_e3))


# --------------------------------------------------------------------------- #
# E4（照抄 p2_07c_probe.py）
# --------------------------------------------------------------------------- #

def make_feats(h: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = len(h["nu"])
    sl = slice(BURN_IN, n - 3)
    nu = h["nu"]
    return {
        "nu": nu[sl],
        "nu_1": nu[sl.start - 1: sl.stop - 1],
        "nu_2": nu[sl.start - 2: sl.stop - 2],
        "nu_3": nu[sl.start - 3: sl.stop - 3],
        "iabs": np.abs(h["current_A"][sl]),
        "temp": h["temperature_C"][sl],
        "soc": h["soc"][sl],
        "y": (h["soc"] - h["soc_true"])[sl] * 100.0,
    }


FEAT_SETS = {
    "F0 预测0": [],
    "F1 仅ν": ["nu"],
    "F2 仅物理量": ["iabs", "temp", "soc"],
    "F3 ν+物理量": ["nu", "iabs", "temp", "soc"],
    "F4 ν滑窗+物理量": ["nu", "nu_1", "nu_2", "nu_3", "iabs", "temp", "soc"],
    "F5 仅SOC_EKF": ["soc"],
}


def ridge_fit(Xtr, ytr, lam=LAM):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xz = (Xtr - mu) / sd
    A = Xz.T @ Xz + lam * np.eye(Xz.shape[1])
    w = np.linalg.solve(A, Xz.T @ (ytr - ytr.mean()))
    return mu, sd, w, float(ytr.mean())


def ridge_pred(model, Xte):
    mu, sd, w, b0 = model
    return ((Xte - mu) / sd) @ w + b0


def run_e4(train, test):
    tr_feats = [make_feats(h) for h in train.values()]
    groups: dict[str, list[str]] = {
        "A:test_id(25°C)": [c for c in test if "25degC_Cycle" in c]}
    for tag, pat in [("B:OOD10°C", "10degC_"), ("B:OOD0°C", "0degC_Cycle"),
                     ("B:OOD-10°C", "n10degC_"), ("B:OOD-20°C", "n20degC_")]:
        groups[tag] = [c for c in test if pat in c and "trise" not in c]

    res: dict[str, dict[str, tuple[float, float, float]]] = {}
    for gname, cids in groups.items():
        te_feats = [make_feats(test[c]) for c in cids if c in test]
        y_all = np.concatenate([f["y"] for f in te_feats])
        base_rmse = float(np.sqrt((y_all ** 2).mean()))
        res[gname] = {"_BASELINE": (base_rmse, base_rmse, 0.0),
                      "_nfile": (len(te_feats), 0.0, 0.0)}
        for fname, cols in FEAT_SETS.items():
            if not cols:
                res[gname][fname] = (base_rmse, base_rmse, 0.0)
                continue
            Xtr = np.hstack([np.concatenate([f[c] for f in tr_feats])[:, None]
                             for c in cols])
            ytr = np.concatenate([f["y"] for f in tr_feats])
            model = ridge_fit(Xtr, ytr)
            rm_bef, rm_aft = [], []
            for f in te_feats:
                Xte = np.hstack([f[c][:, None] for c in cols])
                yte = f["y"]
                pred = ridge_pred(model, Xte)
                rm_bef.append(np.sqrt(np.mean(yte ** 2)))
                rm_aft.append(np.sqrt(np.mean((yte - pred) ** 2)))
            Xte_all = np.hstack([np.concatenate([f[c] for f in te_feats])[:, None]
                                 for c in cols])
            y_all2 = np.concatenate([f["y"] for f in te_feats])
            pred_all = ridge_pred(model, Xte_all)
            r2_all = 1.0 - np.sum((y_all2 - pred_all) ** 2) / \
                np.sum((y_all2 - y_all2.mean()) ** 2)
            res[gname][fname] = (float(np.mean(rm_bef)),
                                 float(np.mean(rm_aft)), float(r2_all))
    return res


# --------------------------------------------------------------------------- #
# 与判定书声称值比对（行号对应 s1_残差可学性判定.md）
# --------------------------------------------------------------------------- #

def compare(e1, e2, e3, e4, notes: list[str]) -> None:
    g25 = e1[e1.cond.str.startswith("25degC_Cycle")]
    chk_range("L46", "25°C 4 文件 ν_std (mV)", 21, 33,
              g25.nu_std_mV.min(), g25.nu_std_mV.max(), 0.5)
    chk_range("L46", "25°C 4 文件 acf1", 0.35, 0.73,
              g25.acf1.min(), g25.acf1.max(), 0.005)
    chk_range("L46", "25°C 4 文件 acf2", 0.39, 0.69,
              g25.acf2.min(), g25.acf2.max(), 0.005)
    chk_range("L52 正文", "全体 acf1 下/上界", 0.35, 0.95,
              e1.acf1.min(), e1.acf1.max(), 0.005)

    named = {"10degC_HWFET": ("L47", "10°C HWFET", 16.7, 0.86, 0.79),
             "0degC_Cycle_1": ("L48", "0°C Cycle_1", 59.6, 0.76, 0.73),
             "n10degC_HWFET": ("L49", "−10°C HWFET", 26.0, 0.88, 0.75),
             "n20degC_HWFET": ("L50", "−20°C HWFET", 30.9, 0.95, 0.87)}
    for cond, (loc, label, s, a1, a2) in named.items():
        r = e1[e1.cond == cond].iloc[0]
        chk(loc, f"{label} ν_std", s, r.nu_std_mV, 0.05)
        chk(loc, f"{label} acf1", a1, r.acf1, 0.005)
        chk(loc, f"{label} acf2", a2, r.acf2, 0.005)

    chk("L30/L52", "8/8 文件 Ljung-Box p<0.001", 8,
        float((e1.LB_p < 0.001).sum()), 0)
    chk_range("L30", "LB p 的最大值（应≈0）", 0.0, 0.001,
              e1.LB_p.min(), e1.LB_p.max(), 0.001)

    # E2 汇总表（L57–L62）的口径：三列 = 「25 °C 4 文件」/「0 °C 及以下 3 文件」，
    # 10 °C HWFET 不在任一列内 —— 见 notes。
    m25 = e2.cond.str.startswith("25degC_Cycle")
    # 注意：不能用子串匹配 —— "0degC_" 是 "10degC_"/"n10degC_"/"n20degC_" 的子串
    # （P0-16 踩过的同一个坑），这里用 startswith 精确限定前缀。
    mcold = e2.cond.str.startswith(("0degC_", "n10degC_", "n20degC_"))
    chk_range("L59", "e~SOC ρ 25°C", 0.98, 1.00,
              e2[m25]["e~SOC"].min(), e2[m25]["e~SOC"].max(), 0.005)
    chk_range("L59", "e~SOC ρ 低温（≤0°C）", 0.97, 1.00,
              e2[mcold]["e~SOC"].min(), e2[mcold]["e~SOC"].max(), 0.005)
    chk_range("L61", "ν~|I| 25°C", -0.27, -0.18,
              e2[m25]["nu~|I|"].min(), e2[m25]["nu~|I|"].max(), 0.01)
    chk_range("L61", "ν~|I| 低温（≤0°C）", -0.68, -0.37,
              e2[mcold]["nu~|I|"].min(), e2[mcold]["nu~|I|"].max(), 0.01)
    chk_range("L62", "e_bias 25°C (pp)", -2.3, -2.0,
              e2[m25].e_bias_pp.min(), e2[m25].e_bias_pp.max(), 0.15)
    chk_range("L62", "e_bias 低温（≤0°C）(pp)", -11.4, -2.2,
              e2[mcold].e_bias_pp.min(), e2[mcold].e_bias_pp.max(), 0.15)

    # 归纳范围的两处口径问题（如实记录，不计入 pass/fail）
    ten = e2[e2.cond == "10degC_HWFET"].iloc[0]
    notes.append(
        f"L57–L62 汇总表三列口径实测为「25 °C 4 文件」+「0 °C 及以下 3 文件」，"
        f"**10 °C HWFET 未纳入任一列**（其 ν~|I| = {ten['nu~|I|']:.3f}、"
        f"e~SOC = {ten['e~SOC']:.3f}、e_bias = {ten['e_bias_pp']:.3f} pp）。"
        f"若把 10 °C 归入低温列，L61 的 ν~|I| 下界会变成 {ten['nu~|I|']:.2f}"
        f"（声称 −0.37）。该表是人工归纳区间，不是全量极值。")
    e25 = e2[m25]
    r3 = e25.loc[e25["e~T"].idxmax()]
    notes.append(
        f"L60「e ~ T | 25 °C: −0.35 ~ −0.67」未覆盖 25 °C Cycle_3"
        f"（实测 {r3['e~T']:.3f}，即 25 °C 组 e~T 实际极值为 "
        f"{e25['e~T'].min():.3f} ~ {e25['e~T'].max():.3f}）。"
        f"同样属人工筛选后的典型区间。")

    chk_range("L68", "低频 10% 能量占比", 0.37, 0.75,
              e3.frac_lo.min(), e3.frac_lo.max(), 0.005)
    chk_range("L69", "log-log 斜率 α", -2.1, -0.5,
              e3.alpha.min(), e3.alpha.max(), 0.02)

    A = e4["A:test_id(25°C)"]
    chk("L83", "探针A F2 R²", 0.885, A["F2 仅物理量"][2], 0.001)
    chk("L83", "探针A F2 RMSE 后 (pp)", 0.57, A["F2 仅物理量"][1], 0.01)
    chk("L84", "探针A F3 R²", 0.884, A["F3 ν+物理量"][2], 0.001)
    chk("L84", "探针A F3 RMSE 后 (pp)", 0.57, A["F3 ν+物理量"][1], 0.01)
    chk("L85", "探针A F5 R²", 0.923, A["F5 仅SOC_EKF"][2], 0.001)
    chk("L85", "探针A F5 RMSE 后 (pp)", 0.47, A["F5 仅SOC_EKF"][1], 0.01)
    chk("L82", "探针A F1 RMSE 后 (pp)", 1.75, A["F1 仅ν"][1], 0.01)
    chk("L78", "探针A 基线 RMSE (pp)", 2.7, A["F1 仅ν"][0], 0.05)
    chk("L87", "探针A F1 R²（混池口径）", 0.00, A["F1 仅ν"][2], 0.01)

    B = {t: e4[f"B:OOD{t}"] for t in ["10°C", "0°C", "-10°C", "-20°C"]}
    f1_bef = [6.1, 11.8, 7.4, 4.8]
    f1_aft = [5.3, 9.9, 6.6, 4.4]
    f2_r2 = [-4.9, -33.7, -5.5, -15.4]
    f5_r2 = [0.21, -8.4, 0.14, 0.25]
    for i, t in enumerate(B):
        chk("L93", f"探针B {t} F1 RMSE 前", f1_bef[i], B[t]["F1 仅ν"][0], 0.06)
        chk("L93", f"探针B {t} F1 RMSE 后", f1_aft[i], B[t]["F1 仅ν"][1], 0.06)
        chk("L94", f"探针B {t} F2 R²", f2_r2[i], B[t]["F2 仅物理量"][2], 0.06)
        chk("L95", f"探针B {t} F5 R²", f5_r2[i], B[t]["F5 仅SOC_EKF"][2], 0.02)


def main() -> int:
    test = load_histories(os.path.join(RESULTS, "ekf_baseline_history.npz"))
    train = load_histories(os.path.join(RESULTS, "ekf_train_history.npz"))
    print(f"baseline(npz) {len(test)} 文件 / train(npz) {len(train)} 文件")

    e1, e2, e3 = run_e123(test)
    print("E1–E3 复算完成，开始 E4 探针 …")
    e4 = run_e4(train, test)
    notes: list[str] = []
    compare(e1, e2, e3, e4, notes)

    n_pass = sum(1 for c in CHECKS if c[5])
    n_fail = len(CHECKS) - n_pass

    L: list[str] = []
    A = L.append
    A("# S1 判定书复算报告（P0-27）\n")
    A("> 生成脚本：`10_代码/p2_07d_s1_verdict.py`（复算 E1–E4）\n")
    A(f"> 输入：`ekf_baseline_history.npz`（{len(test)} 文件）"
      f" + `ekf_train_history.npz`（{len(train)} 文件）\n")
    A("> 比对对象：`s1_残差可学性判定.md`（行号 loc 列为该文件行号）\n")
    A(f"> **结论：{n_pass}/{len(CHECKS)} 条声称数字复算一致，"
      f"{n_fail} 条不一致。**\n")
    A("\nE1–E3 复算逻辑照抄 `p2_07_residual.py`，E4 探针照抄 "
      "`p2_07c_probe.py`；两脚本本身未改动。\n")

    A("\n## 逐条比对\n")
    A("| loc | 项目 | 判定书声称 | 本次复算 | tol | 一致 |")
    A("|---|---|---|---|---|---|")
    for loc, item, claimed, got, tol, ok in CHECKS:
        A(f"| {loc} | {item} | {claimed} | {got} | {tol:g} | "
          f"{'✅' if ok else '❌'} |")

    if notes:
        A("\n## 口径说明（数字本身无误，但汇总区间是人工归纳）\n")
        for i, n in enumerate(notes, 1):
            A(f"{i}. {n}\n")

    A("\n## 复算原始输出\n")
    A("### E1 白性检验（复算）\n")
    A("```\n" + e1.to_string(index=False, float_format=lambda v: f"{v:.4f}")
      + "\n```\n")
    A("### E2 物理相关性（复算）\n")
    A("```\n" + e2.to_string(index=False, float_format=lambda v: f"{v:.4f}")
      + "\n```\n")
    A("### E3 频谱集中度（复算）\n")
    A("```\n" + e3.to_string(index=False, float_format=lambda v: f"{v:.4f}")
      + "\n```\n")
    A("### E4 探针（复算；RMSE 为各文件均值，R² 为混池口径）\n")
    A("| 组 | 特征集 | RMSE 前 (pp) | RMSE 后 (pp) | R² |")
    A("|---|---|---|---|---|")
    for gname, d in e4.items():
        for fname in FEAT_SETS:
            bef, aft, r2 = d[fname]
            A(f"| {gname} | {fname} | {bef:.2f} | {aft:.2f} | {r2:+.3f} |")
    A("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    for loc, item, claimed, got, tol, ok in CHECKS:
        if not ok:
            print(f"  ❌ {loc} {item}: 声称 {claimed} / 复算 {got} (tol {tol:g})")
    print(f"\n{n_pass}/{len(CHECKS)} 条一致，{n_fail} 条不一致")
    print(f"报告 → {OUT_MD}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
