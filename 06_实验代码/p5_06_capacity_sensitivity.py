# -*- coding: utf-8 -*-
"""
P5-06 容量灵敏度：SOC 真值对「容量定基」的敏感性（P0-21 重算）

背景
----
项目统一用名义容量 ``Cn = 2.9 Ah`` 计算 SOC 真值：

    soc_true(t) = clip(1 + Ah(t) / 2.9, 0, 1)          （见 src/data_io.py:280）

但 McMaster 电芯在测试周期内持续老化：1C 容量由 2017-03-09 的 **2.798 Ah**
降到 2017-07-24 的 **2.354 Ah**（本脚本直接从 `*_Dis1C_*.mat` 的 ΔAh 实测，
与 `04_实验/数据/README_数据说明.md:187-188` 给出的 2.798 / 2.354 一致）。

若某文件采集时电芯真实容量为 ``Cn_meas``，则正确的定基是

    soc_alt(t) = clip(1 + Ah(t) / Cn_meas, 0, 1)

**本脚本只量化「真值参考本身的定基不确定性」，不重训任何模型。**
信号/特征/模型全部不变，因此无需重新训练。

三种口径
--------
- ``S0_nominal`` : Cn = 2.9（现行口径，基线）
- ``S1_perfile`` : Cn = 按文件采集日期在两次 1C 实测之间线性插值的容量
- ``S2_twogroup``: Cn = 2.80（早期组，2017-05-01 前）/ 2.35（晚期组，其后）
                   —— 对应证据表 R9 的「早期 2.80 / 末期 2.35」口径

输出
----
- ``04_实验/数据/results/p5_06_capacity_sensitivity.csv``       逐文件明细
- ``04_实验/数据/results/p5_06_capacity_sensitivity_group.csv`` 按温度组聚合

用法
----
    export PYTHONIOENCODING=utf-8
    "D:/venvs/soc-paper/Scripts/python.exe" 10_代码/p5_06_capacity_sensitivity.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "06_实验代码", "src"))

from data_io import CN_MCMASTER, MCMASTER_ROOT, load_meas_mat  # noqa: E402

PROCESSED = os.path.join(PROJECT_ROOT, "04_实验", "数据", "processed")
SPLITS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "splits")
RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")

#: 证据表 R9 的两端容量口径（Ah）
CN_EARLY = 2.80
CN_LATE = 2.35
#: 早期/晚期分界（2017-05-01）——25/10 °C 集在 3–4 月，0/−10/−20 °C 集在 5–7 月
SPLIT_DATE = pd.Timestamp("2017-05-01")


# --------------------------------------------------------------------------- #
# 1. 实测 1C 容量（老化曲线锚点）
# --------------------------------------------------------------------------- #

def measure_1c_capacities() -> pd.DataFrame:
    """
    从 `*_Dis1C_*.mat` 直接测量 1C 放电容量。

    口径：文件内 ``Ah`` 计数器的**极差** ``max(Ah) − min(Ah)``，
    即该次 1C 放电实际放出的安时数（= 当时电芯的可放电容量）。
    """
    pattern = os.path.join(
        MCMASTER_ROOT, "Panasonic 18650PF Data", "25degC", "*", "*Dis1C*.mat")
    rows = []
    for p in sorted(glob.glob(pattern)):
        fn = os.path.basename(p)
        try:
            c = load_meas_mat(p)
        except Exception as exc:                      # noqa: BLE001
            print(f"  [跳过] {fn}: {exc}")
            continue
        ah = np.asarray(c["Ah"], dtype=float).ravel()
        m = re.match(r"^(\d\d-\d\d-\d\d)", fn)
        rows.append({
            "file": fn,
            "date": pd.to_datetime(m.group(1), format="%m-%d-%y"),
            "capacity_Ah": float(np.nanmax(ah) - np.nanmin(ah)),
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def fit_capacity_curve(caps: pd.DataFrame) -> tuple[pd.Timestamp,
                                                   pd.Timestamp, float, float]:
    """
    用最小二乘直线拟合「容量 ~ 日期」，返回 (t0, t1, C0, C1)。

    直线（而非端点直线）是为了让 3 月两个点和 7 月三个点共同定斜率，
    避免单个测量噪声主导。返回的两个端点值即该直线在首末测量日期的取值。
    """
    t = caps["date"].values.astype("datetime64[s]").astype(float)
    c = caps["capacity_Ah"].values.astype(float)
    slope, intercept = np.polyfit(t, c, 1)
    t0, t1 = caps["date"].min(), caps["date"].max()
    f = lambda ts: slope * float(pd.Timestamp(ts).to_datetime64().astype("datetime64[s]").astype(float)) + intercept  # noqa: E731
    return t0, t1, float(f(t0)), float(f(t1))


def capacity_at(date: pd.Timestamp, t0, t1, c0, c1) -> float:
    """按日期线性插值容量；超出实测区间则取端点值（不外推）。"""
    if date <= t0:
        return c0
    if date >= t1:
        return c1
    frac = (date - t0) / (t1 - t0)
    return float(c0 + frac * (c1 - c0))


# --------------------------------------------------------------------------- #
# 2. 逐文件灵敏度
# --------------------------------------------------------------------------- #

def analyse(processed: pd.DataFrame, caps: pd.DataFrame) -> pd.DataFrame:
    t0, t1, c0, c1 = fit_capacity_curve(caps)

    rows = []
    for cid, g in processed.groupby("cycle_id", sort=True):
        m = re.match(r"^(\d\d-\d\d-\d\d)", cid)
        if not m:
            print(f"  [跳过] 无法解析日期：{cid}")
            continue
        date = pd.to_datetime(m.group(1), format="%m-%d-%y")

        ah = g["ah_throughput"].values.astype(float)
        n = len(ah)
        if n == 0:
            continue

        cn_s1 = capacity_at(date, t0, t1, c0, c1)
        cn_s2 = CN_EARLY if date < SPLIT_DATE else CN_LATE

        soc_nom = np.clip(1.0 + ah / CN_MCMASTER, 0.0, 1.0)
        soc_s1 = np.clip(1.0 + ah / cn_s1, 0.0, 1.0)
        soc_s2 = np.clip(1.0 + ah / cn_s2, 0.0, 1.0)
        # S3/S4：全库统一容量（不区分采集日期）——用于核对证据表 R9 的口径
        soc_s3 = np.clip(1.0 + ah / CN_EARLY, 0.0, 1.0)
        soc_s4 = np.clip(1.0 + ah / CN_LATE, 0.0, 1.0)

        d1 = (soc_nom - soc_s1) * 100.0
        d2 = (soc_nom - soc_s2) * 100.0
        d3 = (soc_nom - soc_s3) * 100.0
        d4 = (soc_nom - soc_s4) * 100.0

        # 满放点 = Ah 最负处（= 该文件放电最深时刻）
        k_end = int(np.argmin(ah))
        dod = float(-ah.min())          # 该文件放电深度（Ah）

        rows.append({
            "cycle_id": cid,
            "date": date.date().isoformat(),
            "temperature_c": float(g["temperature_c"].iloc[0]),
            "test_type": str(g["test_type"].iloc[0]),
            "split": str(g["split"].iloc[0]),
            "n_samples": n,
            "dod_end_Ah": float(-ah.min()),
            "Cn_S1_Ah": round(cn_s1, 4),
            "Cn_S2_Ah": cn_s2,
            # S1：逐文件（日期插值容量）
            "S1_rmse_pp": float(np.sqrt(np.mean(d1 ** 2))),
            "S1_mae_pp": float(np.mean(np.abs(d1))),
            "S1_max_pp": float(np.max(np.abs(d1))),
            "S1_end_pp": float(d1[k_end]),
            # S2：两端容量（R9 口径）
            "S2_rmse_pp": float(np.sqrt(np.mean(d2 ** 2))),
            "S2_mae_pp": float(np.mean(np.abs(d2))),
            "S2_max_pp": float(np.max(np.abs(d2))),
            "S2_end_pp": float(d2[k_end]),
            # S3/S4：全库统一容量（R9 口径核对）
            "S3_uniform2.80_max_pp": float(np.max(np.abs(d3))),
            "S3_uniform2.80_end_pp": float(d3[k_end]),
            "S4_uniform2.35_max_pp": float(np.max(np.abs(d4))),
            "S4_uniform2.35_end_pp": float(d4[k_end]),
            # 解析式（不 clip）：Δ = DOD_end × (1/2.9 − 1/Cn)，满放点最大偏差
            "analytical_S3_pp": float(dod * (1.0 / CN_EARLY - 1.0 / CN_MCMASTER) * 100),
            "analytical_S4_pp": float(dod * (1.0 / CN_LATE - 1.0 / CN_MCMASTER) * 100),
        })
    return pd.DataFrame(rows)


def aggregate(per_file: pd.DataFrame) -> pd.DataFrame:
    out = []
    for temp, g in per_file.groupby("temperature_c"):
        out.append({
            "temperature_c": temp,
            "n_files": len(g),
            "dod_end_Ah_min": float(g["dod_end_Ah"].min()),
            "dod_end_Ah_max": float(g["dod_end_Ah"].max()),
            "Cn_S1_min": float(g["Cn_S1_Ah"].min()),
            "Cn_S1_max": float(g["Cn_S1_Ah"].max()),
            "S1_rmse_min": float(g["S1_rmse_pp"].min()),
            "S1_rmse_max": float(g["S1_rmse_pp"].max()),
            "S1_rmse_mean": float(g["S1_rmse_pp"].mean()),
            "S1_max_min": float(g["S1_max_pp"].min()),
            "S1_max_max": float(g["S1_max_pp"].max()),
            "S2_max_min": float(g["S2_max_pp"].min()),
            "S2_max_max": float(g["S2_max_pp"].max()),
        })
    return pd.DataFrame(out).sort_values("temperature_c")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main() -> int:
    print("=" * 78)
    print("P5-06 容量灵敏度（P0-21 重算）")
    print("=" * 78)

    print("\n--- 1. 实测 1C 容量（老化锚点）---")
    caps = measure_1c_capacities()
    for _, r in caps.iterrows():
        print(f"  {r['date'].date()}  {r['file']:34s}  Cn = {r['capacity_Ah']:.4f} Ah")
    t0, t1, c0, c1 = fit_capacity_curve(caps)
    print(f"\n  线性拟合：{t0.date()} ({c0:.4f} Ah) → {t1.date()} ({c1:.4f} Ah)")
    print(f"  总衰减 {c0 - c1:.4f} Ah = {100*(c0-c1)/c0:.1f} %")

    print("\n--- 2. 逐文件灵敏度 ---")
    processed = pd.read_csv(os.path.join(PROCESSED, "mcmaster_drive_cycles.csv.gz"))
    print(f"  驱动文件 {processed['cycle_id'].nunique()} 个，"
          f"{len(processed)} 个采样点")

    per_file = analyse(processed, caps)
    group = aggregate(per_file)

    out1 = os.path.join(RESULTS, "p5_06_capacity_sensitivity.csv")
    out2 = os.path.join(RESULTS, "p5_06_capacity_sensitivity_group.csv")
    per_file.to_csv(out1, index=False, float_format="%.6f")
    group.to_csv(out2, index=False, float_format="%.6f")
    print(f"  已保存 {os.path.relpath(out1, PROJECT_ROOT)}")
    print(f"  已保存 {os.path.relpath(out2, PROJECT_ROOT)}")

    print("\n--- 3. 按温度组聚合 ---")
    show = group.copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(2)
    print(show.to_string(index=False))

    print("\n--- 4. 关键结论数字 ---")
    for tag, col in [("S1 逐文件（日期插值容量）", "S1_max_pp"),
                     ("S2 两端容量（R9 口径）", "S2_max_pp")]:
        early = per_file[per_file["date"] < "2017-05-01"][col]
        late = per_file[per_file["date"] >= "2017-05-01"][col]
        print(f"\n  {tag}")
        print(f"    早期组（2017-05-01 前，{len(early)} 文件）最大偏差 "
              f"{early.min():.1f} – {early.max():.1f} pp")
        print(f"    晚期组（2017-05-01 起，{len(late)} 文件）最大偏差 "
              f"{late.min():.1f} – {late.max():.1f} pp")
        r_early = per_file[per_file["date"] < "2017-05-01"][
            col.replace("max_pp", "rmse_pp")]
        r_late = per_file[per_file["date"] >= "2017-05-01"][
            col.replace("max_pp", "rmse_pp")]
        print(f"    早期组文件级 RMSE {r_early.min():.2f} – {r_early.max():.2f} pp")
        print(f"    晚期组文件级 RMSE {r_late.min():.2f} – {r_late.max():.2f} pp")

    # 现行 test_id（25 °C 同分布集）—— 主结论所在组
    tid = per_file[per_file["split"] == "test_id"]
    print(f"\n  test_id（25 °C 同分布 4 文件）最大偏差："
          f"S1 {tid['S1_max_pp'].min():.1f} – {tid['S1_max_pp'].max():.1f} pp；"
          f"S2 {tid['S2_max_pp'].min():.1f} – {tid['S2_max_pp'].max():.1f} pp")
    print(f"  test_id 文件级 RMSE："
          f"S1 {tid['S1_rmse_pp'].min():.2f} – {tid['S1_rmse_pp'].max():.2f} pp；"
          f"S2 {tid['S2_rmse_pp'].min():.2f} – {tid['S2_rmse_pp'].max():.2f} pp")

    print("\n  --- R9 口径核对：全库统一容量（不区分采集日期）---")
    for tag, col, acol in [
            ("统一 Cn = 2.80 Ah", "S3_uniform2.80_max_pp", "analytical_S3_pp"),
            ("统一 Cn = 2.35 Ah", "S4_uniform2.35_max_pp", "analytical_S4_pp")]:
        v, a = per_file[col], per_file[acol]
        print(f"    {tag}：含 clip，逐文件 max|ΔSOC| = {v.min():.1f} – {v.max():.1f} pp")
        print(f"    {'':>16s}   解析式（不 clip）= {a.min():.1f} – {a.max():.1f} pp"
              f"   ← 证据表 R9 口径")

    whole_max = per_file["S1_max_pp"].max()
    whole_rmse_max = per_file["S1_rmse_pp"].max()
    print(f"\n  全数据集（55 文件）S1 口径：最大偏差上界 {whole_max:.1f} pp；"
          f"文件级 RMSE 上界 {whole_rmse_max:.2f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
