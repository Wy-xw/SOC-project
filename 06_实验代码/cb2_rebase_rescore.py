# -*- coding: utf-8 -*-
"""
CB-2 重打分：把五个消融档按「逐文件容量参考」(S1) 重新计分。

背景（审稿综合 CB-2 / R3-M2）
------------------------------
现行 SOC 真值是按名义容量 Cn = 2.9 Ah 的库仑计数：

    soc_true_nom(t) = clip(1 + Ah(t) / 2.9, 0, 1)

但 McMaster 电芯在测试周期内从 2.783 Ah 老化到 2.399 Ah（`p5_06` 拟合端点）。
按采集日期插值容量后正确的参考是：

    soc_true_S1(t) = clip(1 + Ah(t) / Cn(date), 0, 1)

R3 质疑：A2-4 的增量（0.24–0.28 pp）可能只是「深放电 + 老化文件上真值系统性
偏袒」的产物，因为参考偏移 Δ 量级达数 pp。

方法
----
**不需要重训。** 模型预测 y_hat 只依赖特征（EKF 历史，不变），误差

    e = y - y_hat = (soc_ekf - soc_true_nom) * 100 - y_hat

换参考后 soc_true → soc_true_S1，故

    e_new = e - Δ,   Δ = (soc_true_S1 - soc_true_nom) * 100   (pp)

本脚本据此重算全部 (exp, seed, file) 的 RMSE，并做配对统计。
**不写回任何官方产物**，输出到独立的 cb2_*.csv 供比对。

用法
----
    "D:/venvs/soc-paper/Scripts/python.exe" 10_代码/cb2_rebase_rescore.py
"""
from __future__ import annotations

import importlib.util
import os
import re
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.join(__ROOT__, "SOC论文项目")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "10_代码", "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "10_代码"))

from nn_features import GROUPS, BURN_IN, WINDOW_L, load_history  # noqa: E402

PROCESSED = os.path.join(PROJECT_ROOT, "04_实验", "数据", "processed")
SPLITS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "splits")
RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
CN_NOMINAL = 2.9
SEEDS = [7, 13, 42]
EXPS = ["A2-0", "A2-1", "A2-2", "A2-3", "A2-4"]
ORDER = ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]


def load_capacity_module():
    """复用 p5_06 的容量曲线（与官方口径完全一致，不重新实现）。"""
    p = os.path.join(PROJECT_ROOT, "10_代码", "p5_06_capacity_sensitivity.py")
    spec = importlib.util.spec_from_file_location("p506", p)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


def build_delta(hist: dict, caps, t0, t1, c0, c1) -> dict[str, np.ndarray]:
    """逐文件 Δ = (soc_true_S1 - soc_true_nom) * 100，单位 pp。"""
    if not os.path.exists(PROCESSED):
        raise FileNotFoundError(PROCESSED)
    d = pd.read_csv(os.path.join(PROCESSED, "mcmaster_drive_cycles.csv.gz"))
    ah_by: dict[str, np.ndarray] = {}
    for cid, g in d.groupby("cycle_id"):
        ah_by[cid] = g["ah_throughput"].values.astype(float)

    out = {}
    for cid in hist:
        if cid not in ah_by:
            continue
        m = re.match(r"^(\d\d-\d\d-\d\d)", cid)
        if not m:
            continue
        date = pd.to_datetime(m.group(1), format="%m-%d-%y")
        cn = caps.capacity_at(date, t0, t1, c0, c1)
        ah = ah_by[cid]
        nom = np.clip(1.0 + ah / CN_NOMINAL, 0.0, 1.0)
        alt = np.clip(1.0 + ah / cn, 0.0, 1.0)
        out[cid] = (alt - nom) * 100.0
    return out


def main() -> int:
    caps = load_capacity_module()
    meas = caps.measure_1c_capacities()
    t0, t1, c0, c1 = caps.fit_capacity_curve(meas)
    print(f"[容量曲线] {t0.date()} Cn={c0:.4f}  →  {t1.date()} Cn={c1:.4f} Ah", flush=True)

    hist = load_history(os.path.join(RESULTS, "ekf_full_history.npz"))
    man = pd.read_csv(os.path.join(SPLITS, "split_manifest.csv"))
    grp_of = dict(zip(man.cycle_id, man.split))
    NOM = {25.0: "test_id(25C)", 10.0: "ood(10C)", 0.0: "ood(0C)",
           -10.0: "ood(-10C)", -20.0: "ood(-20C)"}
    nom_of = dict(zip(man.cycle_id, man.nominal_temp_c))

    delta = build_delta(hist, caps, t0, t1, c0, c1)
    print(f"[Δ] 覆盖 {len(delta)} 文件", flush=True)

    # 逐 (exp, file) 在评分窗内的 e 向量与 Δ 向量
    print("构建逐文件误差向量 ...", flush=True)
    vec = {}
    for g in EXPS:
        for cid, h in hist.items():
            if grp_of.get(cid) not in ("test_id", "test_ood"):
                continue
            if cid not in delta:
                continue
            n = len(h["nu"])
            if n < BURN_IN + WINDOW_L + 5:          # 评分窗至少 5 点
                continue
            ends = np.arange(BURN_IN, n - WINDOW_L + 1) + WINDOW_L - 1
            y = (h["soc"] - h["soc_true"]) * 100.0
            vec[(g, cid)] = (y[ends].astype(np.float64), delta[cid][ends].astype(np.float64))

    # 模型预测：只跑一次，缓存 y_hat
    from tensorflow import keras
    from nn_features import build_split_arrays
    print("加载模型并推理 ...", flush=True)
    yhat = {}
    for g in EXPS:
        cache = {}
        for cid in hist:
            if (g, cid) in vec:
                cache[cid] = build_split_arrays(hist, [cid], GROUPS[g])
        for s in SEEDS:
            mp = os.path.join(PROJECT_ROOT, "04_实验", "models", f"gru16_{g}_seed{s}.keras")
            model = keras.models.load_model(mp, compile=False)
            for cid in cache:
                X, y, _ = cache[cid]
                if X is None:
                    continue
                yh = model.predict(X, verbose=0).ravel().astype(np.float64)
                yhat[(g, s, cid)] = yh
            print(f"  {g} seed{s} done", flush=True)

    rows = []
    for (g, cid), (y, dl) in vec.items():
        for s in SEEDS:
            yh = yhat.get((g, s, cid))
            if yh is None:
                continue
            e_nom = y - yh
            e_s1 = e_nom - dl
            rows.append({
                "exp": g, "seed": s, "cycle_id": cid,
                "eval_group": NOM.get(nom_of.get(cid), "?"),
                "RMSE_nominal": float(np.sqrt(np.mean(e_nom ** 2))),
                "RMSE_S1": float(np.sqrt(np.mean(e_s1 ** 2))),
                "dabs_mean_pp": float(np.mean(np.abs(dl))),
                "delta_end_pp": float(dl[-1]),
            })
    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS, "cb2_rescore_perfile.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n写出 {out_csv}  ({len(df)} 行)", flush=True)

    # ---- 配对统计：A2-x - A2-0，两种口径对照 ----
    def paired(a, c, g, col):
        s = df[df.eval_group == g]
        w = s.pivot_table(index="cycle_id", columns=["exp", "seed"], values=col)
        if a not in w or c not in w:
            return None
        d = (w[a] - w[c]).stack().dropna()
        # 先按文件跨种子平均（与手稿 A.5 一致）
        f = (w[a] - w[c]).mean(axis=1).dropna()
        lo, hi = stats.t.interval(0.95, len(f) - 1, loc=f.mean(), scale=stats.sem(f))
        return f.mean(), lo, hi, len(f), int((f < 0).sum())

    def seed_pm(a, c, g, col):
        s = df[df.eval_group == g]
        w = s.pivot_table(index="cycle_id", columns=["exp", "seed"], values=col)
        if a not in w or c not in w:
            return None
        d = w[a] - w[c]
        return d.mean(axis=0)

    print("\n" + "=" * 96)
    print("逐温度配对对比：名义容量（现行） vs 逐文件容量（S1）")
    for a, c, lab in [("A2-4", "A2-0", "A2-4 − A2-0（仅置信族）"),
                      ("A2-3", "A2-0", "A2-3 − A2-0（全集）"),
                      ("A2-1", "A2-0", "A2-1 − A2-0（新息族）")]:
        print("\n" + "-" * 96)
        print(f"{lab}")
        print(f"  {'组':<13}{'名义均值':>10}{'S1 均值':>10}{'位移':>9}{'名义CI':>24}{'S1 CI':>24}{'符号':>6}")
        for g in ORDER:
            rn = paired(a, c, g, "RMSE_nominal")
            rs = paired(a, c, g, "RMSE_S1")
            if not rn or not rs:
                continue
            pn = seed_pm(a, c, g, "RMSE_nominal")
            ps = seed_pm(a, c, g, "RMSE_S1")
            sg = lambda p: "".join("+" if v > 0 else "-" for v in p.values)
            print(f"  {g:<13}{rn[0]:>10.4f}{rs[0]:>10.4f}{rs[0]-rn[0]:>+9.4f}"
                  f"   [{rn[1]:+.4f},{rn[2]:+.4f}]"
                  f"   [{rs[1]:+.4f},{rs[2]:+.4f}]"
                  f"  {sg(pn)}→{sg(ps)}")

    print("\n" + "=" * 96)
    print("[Δ 幅值分级] 逐文件 |Δ| 与「名义 → S1」的 RMSE 位移")
    gsum = df.groupby("eval_group").agg(
        n_file=("cycle_id", "nunique"), dabs=("dabs_mean_pp", "mean"))
    print(gsum.round(4).to_string())
    (df.assign(shift=df.RMSE_S1 - df.RMSE_nominal)
       .groupby("eval_group").shift.agg(["mean", "min", "max"]).round(4)
       .pipe(print))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
