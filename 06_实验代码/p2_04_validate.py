# -*- coding: utf-8 -*-
"""P2-04：ECM 端电压拟合精度校验（动态工况正向仿真）

目的
----
隔离检验「OCV 曲线 + 1RC 参数表」这套模型本身的精度：
用实测电流正向仿真 RC 状态，Vt_pred = OCV(soc_true) − V1 − I·R0，
与实测端电压对比。**不涉及 EKF**（那是 P2-05/06 的事）。

方法
----
1. ecm_params_raw.csv 按 (温度 × SOC 0.1 档) 聚合中位数 → 规则网格
2. 每时刻用 (实测温度, soc_true) 双线性插值出时变 R0/R1/C1
   （驱动文件自发热明显，−20 °C 名义文件内部可升至 −3 °C，
    必须用实测温度而非名义温度）
3. V1 逐步递推：V1_{k+1} = a_k·V1_k + R1_k·(1−a_k)·I_k，a_k=exp(−dt/τ_k)
4. 误差按工况分：静置 |I|<0.05 ｜ 放电 ｜ 回馈（I<0）
   （P2-02 发现 −104 mV 滞后带，预期回馈段误差大）
5. 附「无 RC 纯 R0」基线，量化 RC 支路贡献
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(PROJECT_ROOT, "04_实验", "数据", "processed")
RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")

I_REST = 0.05
TEMPS = [-20.0, -10.0, 0.0, 10.0, 25.0]
BINS = np.arange(0.0, 1.05, 0.1)


# --------------------------------------------------------------------------- #
# 参数网格：raw → (温度 × SOC 档) 中位数
# --------------------------------------------------------------------------- #

def build_param_grid(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    """返回 {name: 2D array [n_temp, n_socbin]}，缺档用行内（同温度）填充。"""
    grids = {}
    for name in ["r0_ohm", "r1_ohm", "c1_F"]:
        g = np.full((len(TEMPS), len(BINS) - 1), np.nan)
        for ti, tp in enumerate(TEMPS):
            d = raw[raw.temperature_c == tp]
            if d.empty:
                continue
            cut = pd.cut(d.soc, BINS)
            med = d.groupby(cut, observed=True)[name].median()
            for bi, val in enumerate(med.to_numpy()):
                g[ti, bi] = val
        # 缺档回填：先 SOC 维（同温度相邻档外推），再温度维
        for ti in range(g.shape[0]):
            row = g[ti]
            idx = np.where(np.isnan(row))[0]
            if len(idx) and not np.all(np.isnan(row)):
                good = np.where(~np.isnan(row))[0]
                row[idx] = np.interp(idx, good, row[good])
        # 整行缺失（某温度某参数全空）→ 用相邻温度行
        for ti in range(g.shape[0]):
            if np.all(np.isnan(g[ti])):
                g[ti] = np.nanmedian(g, axis=0)
        grids[name] = g
    return grids


def interp_params(
    grids: dict[str, np.ndarray],
    temp: np.ndarray,
    soc: np.ndarray,
) -> dict[str, np.ndarray]:
    """逐时刻 (温度, SOC) 双线性插值。温度超 [−20,25] 端点 clamp。"""
    nb = grids["r0_ohm"].shape[1]
    n = len(temp)
    tc = np.clip(temp, TEMPS[0], TEMPS[-1])
    sc = np.clip(soc, 0.0, 1.0)
    # 温度维：下格点 index 与权重
    ti_lo = np.clip(np.searchsorted(TEMPS, tc, side="right") - 1,
                    0, len(TEMPS) - 2)
    wt = (tc - TEMPS[0][ti_lo]) / (TEMPS[1] - TEMPS[0]) \
        if False else (tc - np.asarray(TEMPS)[ti_lo]) / \
        (np.asarray(TEMPS)[ti_lo + 1] - np.asarray(TEMPS)[ti_lo])
    # SOC 维：档中点 0.05~0.95 之间的线性插值，边界 clamp
    centers = np.arange(nb) * 0.1 + 0.05
    sbc = np.clip(np.searchsorted(centers, sc, side="right") - 1, 0, nb - 2)
    wc = np.clip((sc - centers[sbc]) / 0.1, 0.0, 1.0)
    out = {}
    for name, g in grids.items():
        g_lo = g[ti_lo, sbc] + (g[ti_lo, sbc + 1] - g[ti_lo, sbc]) * wc
        g_hi = g[ti_lo + 1, sbc] + \
            (g[ti_lo + 1, sbc + 1] - g[ti_lo + 1, sbc]) * wc
        out[name] = g_lo + (g_hi - g_lo) * wt
    return out


# --------------------------------------------------------------------------- #
# 正向仿真
# --------------------------------------------------------------------------- #

def simulate_vt(
    i: np.ndarray,
    temp: np.ndarray,
    soc: np.ndarray,
    grids: dict[str, np.ndarray],
    ocv_tab: tuple[np.ndarray, np.ndarray],
    dt: float = 1.0,
    with_rc: bool = True,
) -> np.ndarray:
    """返回端电压预测序列。"""
    p = interp_params(grids, temp, soc)
    r0, r1, c1 = p["r0_ohm"], p["r1_ohm"], p["c1_F"]
    soc_g, ocv_g = ocv_tab
    v_ocv = np.interp(soc, soc_g, ocv_g)

    if not with_rc:
        return v_ocv - i * r0

    n = len(i)
    v1 = np.empty(n)
    v1_k = 0.0
    for k in range(n):
        v1[k] = v1_k
        tau = max(r1[k] * c1[k], 1e-3)
        a = np.exp(-dt / tau)
        # 参数时变时能量不严格守恒，用瞬时参数的一步离散化（工程近似）
        v1_k = a * v1_k + r1[k] * (1.0 - a) * i[k]
    return v_ocv - v1 - i * r0


def seg_mask(i: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "静置": np.abs(i) < I_REST,
        "放电": i >= I_REST,
        "回馈": i <= -I_REST,
    }


def err_stats(e_mV: np.ndarray) -> dict[str, float]:
    return {
        "RMSE_mV": float(np.sqrt(np.mean(e_mV**2))),
        "MAE_mV": float(np.mean(np.abs(e_mV))),
        "MAX_mV": float(np.max(np.abs(e_mV))),
        "bias_mV": float(np.mean(e_mV)),
        "n": int(len(e_mV)),
    }


def main() -> int:
    raw = pd.read_csv(os.path.join(RESULTS, "ecm_params_raw.csv"))
    grids = build_param_grid(raw)
    oc = pd.read_csv(os.path.join(RESULTS, "ocv_soc_curve.csv"))
    ocv_tab = (oc.soc.to_numpy(), oc.ocv_V.to_numpy())
    man = pd.read_csv(os.path.join(
        PROJECT_ROOT, "04_实验", "数据", "splits", "split_manifest.csv"))

    d = pd.read_csv(os.path.join(DATA, "mcmaster_drive_cycles.csv.gz"))

    rows = []
    detail = None   # 25 °C test_id 代表文件的逐点数据（画图用）
    for cid, g in d.groupby("cycle_id"):
        if len(g) < 100:
            continue
        i = g.current_A.to_numpy()
        v = g.voltage_V.to_numpy()
        t = g.temperature_C.to_numpy()
        s = g.soc_true.to_numpy()
        meta = man[man.cycle_id == cid]
        split = meta.split.iloc[0] if len(meta) else "?"
        ntmp = meta.nominal_temp_c.iloc[0] if len(meta) else np.nan

        res = {"cycle_id": cid, "split": split, "nominal_temp_c": ntmp,
               "n": len(g)}
        for with_rc, tag in [(True, "rc"), (False, "norc")]:
            vt_pred = simulate_vt(i, t, s, grids, ocv_tab, with_rc=with_rc)
            e = (v - vt_pred) * 1000.0
            masks = seg_mask(i)
            for name, m in masks.items():
                if m.sum() < 10:
                    continue
                st = err_stats(e[m])
                for k_, v_ in st.items():
                    res[f"{tag}_{name}_{k_}"] = v_
        rows.append(res)
        if detail is None and split == "test_id" and ntmp == 25.0:
            detail = {
                "timestamp": g.timestamp.to_numpy(),
                "vt_meas": v, "vt_pred": simulate_vt(
                    i, t, s, grids, ocv_tab, with_rc=True) * 1.0,
                "current": i, "soc": s,
            }

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "ecm_validation_per_file.csv"), index=False)

    # 汇总：split × 名义温度
    #
    # ⚠️ 2026-09-25 修正：本行原写作「（全段，mV）」，但下面算的是 `rc_放电_*`，
    #    即**放电段**而非全段。这个错标签被直接抄进了
    #    `05_论文/P3_03_网络选型报告.md`，导致论文里出现「25°C 全段 ~44 mV」
    #    （那是回馈段的值）这类错误。标签必须与所算的段一致。
    print("== 1RC-ECM 端电压误差（放电段，mV；按段内样本数加权 RMS-of-RMSE）==")
    agg = (df.groupby(["split", "nominal_temp_c"])
           .apply(lambda x: pd.Series({
               "RMSE": np.sqrt(np.average(x.rc_放电_RMSE_mV**2,
                                          weights=x.rc_放电_n)),
               "MAE": np.nanmean(x.rc_放电_MAE_mV),
               "MAX": np.nanmax(x.rc_放电_MAX_mV),
               "静置RMSE": np.sqrt(np.average(x.rc_静置_RMSE_mV**2,
                                              weights=x.rc_静置_n)),
               "回馈RMSE": (np.sqrt(np.average(x.rc_回馈_RMSE_mV**2,
                                               weights=x.rc_回馈_n))
                            if x.rc_回馈_RMSE_mV.notna().any() else np.nan),
               "无RC_RMSE": np.sqrt(np.average(x.norc_放电_RMSE_mV**2,
                                               weights=x.norc_放电_n)),
           }), include_groups=False)
           .round(1))
    print(agg.to_string())

    # ⚠️ 2026-09-25 新增：聚合值此前**只在终端 print、从不落盘**，
    #    导致论文里的 60.9 / 70.0 mV 无法从任何产物复核（连续两次审计都对不上口径）。
    #
    # 🔴 **2026-09-27 改名（原写 `ecm_validation_agg.csv`，会撞车）**：
    #    那个路径的**产出者**是 `10_代码/diag/p2_04b_ecm_agg_dump.py`
    #    （英文列 `cell/rc_disch/…` + `ALL splits @25C` 与 `test_id only @25C` 两行，
    #    `float_format="%.3f"`），**`证据表` M1 与 P0-22 引用的正是它**。
    #    本段写的是**另一种 schema**（中文列、无 ALL splits 行）。同名 → 谁后跑谁覆盖。
    #    2026-09-27 重跑本脚本时实测发生了覆盖（数值相同、格式不同），已回滚并改名。
    #    **本段不是那个文件的产出者，不要改回去。**
    agg_out = df.groupby(["split", "nominal_temp_c"]).apply(
        lambda x: pd.Series({
            "n_files": len(x),
            "rc_放电_RMSE_mV": np.sqrt(np.average(x.rc_放电_RMSE_mV**2,
                                                  weights=x.rc_放电_n)),
            "rc_静置_RMSE_mV": np.sqrt(np.average(x.rc_静置_RMSE_mV**2,
                                                  weights=x.rc_静置_n)),
            "rc_回馈_RMSE_mV": (np.sqrt(np.average(x.rc_回馈_RMSE_mV**2,
                                                   weights=x.rc_回馈_n))
                                if x.rc_回馈_RMSE_mV.notna().any() else np.nan),
            "norc_放电_RMSE_mV": np.sqrt(np.average(x.norc_放电_RMSE_mV**2,
                                                    weights=x.norc_放电_n)),
        }), include_groups=False)
    # 改名以避开 `ecm_validation_agg.csv`（那是 diag/p2_04b 的地盘，见上方说明）
    agg_out.to_csv(os.path.join(RESULTS, "ecm_validation_agg_by_split.csv"))
    print("→ ecm_validation_agg_by_split.csv（split × 温度 的聚合值，含 n_files）")

    if detail is not None:
        np.savez_compressed(
            os.path.join(RESULTS, "ecm_vt_detail_25c.npz"), **detail)
        print("\n代表文件逐点数据 → ecm_vt_detail_25c.npz")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
