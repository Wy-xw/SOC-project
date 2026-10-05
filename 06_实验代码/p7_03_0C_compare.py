# -*- coding: utf-8 -*-
"""P7-03 补证实验：0 °C 训练设定 vs 25 °C 主设定的对比。

回答审稿人 R1-M1：主设定的核心结论
「两族直接对比 A2-1 − A2-4 稳健」是否随**训练温度**改变。

口径与主文 §5.3 完全一致：
    每文件先跨种子平均 -> 文件级配对 -> 文件间双侧 t 95% CI
    效应/种子变异比 = |mean_s Δ| / SD_s Δ（ddof=0）
"""
from __future__ import annotations

import csv
import math
import os
import sys
import collections
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.stdout.reconfigure(encoding="utf-8")
RESULTS = os.path.join(__ROOT__, "04_实验", "数据", "results")
T_CRIT = {4: 3.182, 9: 2.306}

#: (标签, 文件, 同分布组名, OOD 组名)
SETTINGS = [
    ("25 °C 训练（主设定）", "p5_b2_perfile_seed10.csv",
     "test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"),
    ("0 °C 训练（补证）", "p7_02_0C_ablation_seed10.csv",
     "test_id(0C)", "ood(25C)", None, None, None),
]

PAIRS = [("A2-1", "A2-0", "新息/残差类 vs A2-0"),
         ("A2-4", "A2-0", "不确定性/增益类 vs A2-0"),
         ("A2-3", "A2-0", "完整诊断集 vs A2-0"),
         ("A2-1", "A2-4", "★ 两族直接对比")]


def load(fn):
    return list(csv.DictReader(open(os.path.join(RESULTS, fn),
                                    encoding="utf-8-sig")))


def per_file(rows, exp, group):
    """{(cycle_id, seed): RMSE}"""
    d = collections.defaultdict(list)
    for r in rows:
        if r["exp"] == exp and r["eval_group"] == group:
            d[(r["cycle_id"], r["seed"])].append(float(r["RMSE"]))
    return {k: sum(v) / len(v) for k, v in d.items()}


def paired(rows, a, b, group):
    """逐文件（先跨种子平均）配对差 a − b。"""
    pa, pb = per_file(rows, a, group), per_file(rows, b, group)
    files = sorted({k[0] for k in pa})
    out = []
    for c in files:
        va = [pa[(c, s)] for (cc, s) in pa if cc == c]
        vb = [pb[(c, s)] for (cc, s) in pb if cc == c]
        if va and vb:
            out.append(sum(va) / len(va) - sum(vb) / len(vb))
    return out


def ratio(rows, a, b, group):
    """效应/种子变异比：逐种子的组均值之差。"""
    seeds = sorted({r["seed"] for r in rows}, key=int)
    vals = []
    for s in seeds:
        sa = [float(r["RMSE"]) for r in rows
              if r["exp"] == a and r["eval_group"] == group and r["seed"] == s]
        sb = [float(r["RMSE"]) for r in rows
              if r["exp"] == b and r["eval_group"] == group and r["seed"] == s]
        if sa and sb:
            vals.append(sum(sa) / len(sa) - sum(sb) / len(sb))
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))
    ok = sum(1 for x in vals if (x > 0) == (m > 0))
    return m, sd, abs(m) / sd if sd else float("nan"), f"{ok}/{len(vals)}"


def ci(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / n)
    se = sd / math.sqrt(n - 1)
    tc = T_CRIT.get(n, 2.0)
    return n, m, m - tc * se, m + tc * se


def main() -> int:
    for label, fn, tid, *oods in SETTINGS:
        rows = load(fn)
        groups = [tid] + [g for g in oods if g]
        print("=" * 78)
        print(f"{label}   （{fn}）")
        print("=" * 78)
        for a, b, name in PAIRS:
            print(f"\n  {name}  （{a} − {b}）")
            for g in groups:
                v = paired(rows, a, b, g)
                if not v:
                    continue
                n, m, lo, hi = ci(v)
                rm, rsd, rr, sr = ratio(rows, a, b, g)
                excl = "  <<排除零" if (lo > 0 or hi < 0) else ""
                print(f"    {g:<13} n={n:2d} δ={m:+.4f} CI=[{lo:+.4f},{hi:+.4f}] "
                      f"比值={rr:.2f} 同号={sr}{excl}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
