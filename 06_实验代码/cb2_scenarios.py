# -*- coding: utf-8 -*-
"""
CB-2 口径扫描：SOC 真值定基口径 → 消融对比的稳健性包络。

问题（审稿综合 CB-2 / R3-M2）
-----------------------------
现行真值是名义容量库仑计数 soc_true = clip(1 + Ah/2.9)。电芯在测试期内
从 ~2.78 Ah 老化到 ~2.40 Ah，故「真值本身」带一个随放电深度增长的参考偏移
Δ ≈ DOD × (1/2.9 - 1/Cn)。R3 质疑：A2-4 的增量（0.24-0.28 pp）是否只是这个
参考偏移的产物。

方法
----
**不重训。** 换参考只改标签，不改特征：e_new = e_nom - Δ。
一次推理，五种口径全部评（口径只影响 Δ）。

关键量是**配对设计的一阶抵消**：同一文件上所有臂共用同一个 Δ，
故 A2-x - A2-0 的配对差里 Δ 一阶抵消；残余是 RMSE 非线性的二阶泄漏。
本脚本量化 (a) 绝对水平随口径的漂移，(b) 配对效应随口径的漂移。

口径
----
- S0 名义 2.9（现行，参考基线，Δ ≡ 0）
- S1 逐文件：日期在两次 1C 实测间线性插值
- S2 两端：2017-05-01 前 2.80 / 其后 2.35（证据表 R9 口径）
- S3 全库统一 2.80
- S4 全库统一 2.35（激进下界）
- S5 名义 ±20% 包络的极值（2.32 = 2.9×0.8，最激进）

输出
----
``04_实验/数据/results/cb2_scenarios_perfile.csv`` 逐 (exp, seed, file, 口径) RMSE

用法
----
    "D:/venvs/soc-paper/Scripts/python.exe" 10_代码/cb2_scenarios.py
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = r"E:\SOC论文项目"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "10_代码", "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "10_代码"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from nn_features import GROUPS, BURN_IN, WINDOW_L, build_split_arrays, load_history  # noqa: E402

PROCESSED = os.path.join(PROJECT_ROOT, "04_实验", "数据", "processed")
RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
CN_NOMINAL = 2.9
CN_EARLY, CN_LATE = 2.80, 2.35
SPLIT_DATE = pd.Timestamp("2017-05-01")
SEEDS = [7, 13, 42]
EXPS = ["A2-0", "A2-1", "A2-2", "A2-3", "A2-4"]


def load_p506():
    p = os.path.join(PROJECT_ROOT, "10_代码", "p5_06_capacity_sensitivity.py")
    spec = importlib.util.spec_from_file_location("p506", p)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


def cn_of(cid: str, which: str, caps, t0, t1, c0, c1) -> float:
    m = re.match(r"^(\d\d-\d\d-\d\d)", cid)
    date = pd.to_datetime(m.group(1), format="%m-%d-%y")
    if which == "S1":
        return caps.capacity_at(date, t0, t1, c0, c1)
    if which == "S2":
        return CN_EARLY if date < SPLIT_DATE else CN_LATE
    if which == "S3":
        return CN_EARLY
    if which == "S4":
        return CN_LATE
    if which == "S5":
        return CN_NOMINAL * 0.80
    raise ValueError(which)


SCEN = ["S0", "S1", "S2", "S3", "S4", "S5"]
LABEL = {"S0": "名义 2.9（现行）", "S1": "逐文件(日期插值)",
         "S2": "两端 2.80/2.35", "S3": "统一 2.80",
         "S4": "统一 2.35", "S5": "名义×0.8 = 2.32"}


def main() -> int:
    caps = load_p506()
    meas = caps.measure_1c_capacities()
    t0, t1, c0, c1 = caps.fit_capacity_curve(meas)
    print(f"容量曲线 {t0.date()} Cn={c0:.4f} -> {t1.date()} Cn={c1:.4f} Ah")

    hist = load_history(os.path.join(RESULTS, "ekf_full_history.npz"))
    man = pd.read_csv(os.path.join(PROJECT_ROOT, "04_实验", "数据", "splits",
                                  "split_manifest.csv"))
    grp_of = dict(zip(man.cycle_id, man.split))
    nom_of = dict(zip(man.cycle_id, man.nominal_temp_c))
    NM = {25.0: "test_id(25C)", 10.0: "ood(10C)", 0.0: "ood(0C)",
          -10.0: "ood(-10C)", -20.0: "ood(-20C)"}

    d = pd.read_csv(os.path.join(PROCESSED, "mcmaster_drive_cycles.csv.gz"))
    ah_by = {c: g["ah_throughput"].values.astype(float)
             for c, g in d.groupby("cycle_id")}

    # 评估文件池
    pool = [c for c in hist
            if grp_of.get(c) in ("test_id", "test_ood") and c in ah_by
            and len(hist[c]["nu"]) >= BURN_IN + WINDOW_L + 5]
    print(f"评估池 {len(pool)} 文件")

    # 每文件评分窗内的名义误差 y 与各口径 Δ
    deltas: dict[str, dict[str, np.ndarray]] = {s: {} for s in SCEN}
    Y: dict[str, np.ndarray] = {}
    CIDS = sorted(pool)
    for cid in CIDS:
        h = hist[cid]
        n = len(h["nu"])
        ends = np.arange(BURN_IN, n - WINDOW_L + 1) + WINDOW_L - 1
        Y[cid] = ((h["soc"] - h["soc_true"]) * 100.0)[ends].astype(np.float64)
        ah = ah_by[cid]
        nom = np.clip(1.0 + ah / CN_NOMINAL, 0.0, 1.0)
        for s in SCEN:
            if s == "S0":
                deltas[s][cid] = np.zeros(len(ends))
                continue
            cn = cn_of(cid, s, caps, t0, t1, c0, c1)
            alt = np.clip(1.0 + ah / cn, 0.0, 1.0)
            deltas[s][cid] = ((alt - nom) * 100.0)[ends]

    # 一次推理，全部口径复用
    from tensorflow import keras
    print("加载模型与推理 ...")
    yhat: dict[tuple[str, int, str], np.ndarray] = {}
    for g in EXPS:
        cache = {cid: build_split_arrays(hist, [cid], GROUPS[g]) for cid in CIDS}
        for s in SEEDS:
            model = keras.models.load_model(
                os.path.join(PROJECT_ROOT, "04_实验", "models",
                             f"gru16_{g}_seed{s}.keras"), compile=False)
            for cid in CIDS:
                X, y, _ = cache[cid]
                if X is None:
                    continue
                yhat[(g, s, cid)] = model.predict(X, verbose=0).ravel().astype(np.float64)
        print(f"  {g} done")

    rows = []
    for cid in CIDS:
        y = Y[cid]
        for s in SCEN:
            dl = deltas[s][cid]
            for g in EXPS:
                for sd in SEEDS:
                    yh = yhat[(g, sd, cid)]
                    e = y - yh - dl
                    rows.append({
                        "scenario": s, "exp": g, "seed": sd, "cycle_id": cid,
                        "eval_group": NM.get(nom_of.get(cid), "?"),
                        "RMSE": float(np.sqrt(np.mean(e ** 2))),
                    })
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "cb2_scenarios_perfile.csv")
    df.to_csv(out, index=False)
    print(f"写出 {out} ({len(df)} 行)")

    print("\n" + "=" * 104)
    print("T1. 绝对水平：各档 RMSE 均值，逐口径 × 温度组（pp）")
    print("=" * 104)
    for s in SCEN:
        p = (df[df.scenario == s]
             .pivot_table(index="eval_group", columns="exp", values="RMSE",
                          aggfunc="mean")
             .reindex(["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"])
             .reindex(columns=EXPS))
        p["最优"] = p.idxmin(axis=1)
        print(f"\n[{s}] {LABEL[s]}")
        print(p.round(3).to_string())

    print("\n" + "=" * 104)
    print("T2. 配对效应随口径的漂移（每文件先跨种子平均）")
    print("=" * 104)
    print(f"{'对比':<10}{'组':<13}" + "".join(f"{s:>9}" for s in SCEN)
          + f"{'极差':>9}{'效应/SD':>10}")
    for a, c in [("A2-1", "A2-0"), ("A2-3", "A2-0"), ("A2-4", "A2-0")]:
        for gname in ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]:
            vals = []
            for s in SCEN:
                sub = df[(df.scenario == s) & (df.eval_group == gname)]
                w = sub.pivot_table(index="cycle_id", columns=["exp", "seed"],
                                    values="RMSE")
                if a not in w.columns.get_level_values(0):
                    vals.append(np.nan)
                    continue
                vals.append(float((w[a] - w[c]).mean(axis=1).mean()))
            v = np.array(vals)
            rng = np.nanmax(v) - np.nanmin(v)
            sd = np.nanstd(v)
            print(f"{a+'-'+c:<10}{gname:<13}"
                  + "".join(f"{x:>+9.4f}" for x in v)
                  + f"{rng:>9.4f}{(0 if sd == 0 else abs(v[0])/sd):>10.2f}")

    print("\n" + "=" * 104)
    print("T3. 参考偏移 Δ 的量级（pp，绝对均值；S0 恒为 0）")
    print("=" * 104)
    for s in SCEN[1:]:
        mags = [np.mean(np.abs(deltas[s][cid])) for cid in CIDS]
        ends = [abs(deltas[s][cid][-1]) for cid in CIDS]
        print(f"  {s} {LABEL[s]:<22} 全文件 |Δ| 均值 {np.mean(mags):.3f} pp"
              f"   最深点 |Δ| 均值 {np.mean(ends):.3f} pp"
              f"   最深点最大 {np.max(ends):.3f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
