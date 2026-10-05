# -*- coding: utf-8 -*-
"""P9-03 OCV 内核敏感性分析：族间差是否由 OCV 温度失配驱动（回应 R2-M2）。

对比三组（口径同 §5.3：逐文件先跨种子平均、再做文件级配对；
效应/种子变异比 = |mean_s Δ| / SD_s Δ，ddof=0）：

    base  : 单一 25 °C OCV 内核（论文现状）        ← `p5_b2_perfile_seed10.csv`
    ocvT  : 温度相关 OCV 内核（本次）              ← `p9_02_ocvT_ablation.csv`
    M1    : ocvT 内核 + **通道数配平**(A2-1 − A2-K0)

判据：若 A2-1 − A2-4 在 ocvT 下**方向与量级保持**，则族间差不由 OCV 失配驱动，
      R2-M2 的替代解释被排除；若**衰减/消失**，主张须收窄为"特定 OCV 失配下的族间差"。
"""
from __future__ import annotations

import collections
import csv
import io
import math
import os
import sys
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.stdout.reconfigure(encoding="utf-8")
R = os.path.join(__ROOT__, "04_实验", "数据", "results")
T_CRIT = {4: 3.182, 9: 2.306}
GROUPS = ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]
OUT = os.path.join(R, "p9_03_ocvT_compare.csv")


def load(fn):
    with io.open(os.path.join(R, fn), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def perfile(rows, exp, grp):
    d = collections.defaultdict(list)
    for r in rows:
        if r["exp"] == exp and r["eval_group"] == grp:
            d[(r["cycle_id"], r["seed"])].append(float(r["RMSE"]))
    return {k: sum(v) / len(v) for k, v in d.items()}


def paired(ra, a, rb, b, grp):
    pa, pb = perfile(ra, a, grp), perfile(rb, b, grp)
    out = []
    for c in sorted({k[0] for k in pa}):
        va = [pa[(c, s)] for (cc, s) in pa if cc == c]
        vb = [pb[(c, s)] for (cc, s) in pb if cc == c]
        if va and vb:
            out.append(sum(va) / len(va) - sum(vb) / len(vb))
    return out


def ci(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / n)
    se = sd / math.sqrt(n - 1)
    tc = T_CRIT.get(n, 2.0)
    return n, m, m - tc * se, m + tc * se


def ratio(ra, a, rb, b, grp):
    seeds = sorted({r["seed"] for r in ra + rb}, key=int)
    vals = []
    for s in seeds:
        sa = [float(r["RMSE"]) for r in ra
              if r["exp"] == a and r["eval_group"] == grp and r["seed"] == s]
        sb = [float(r["RMSE"]) for r in rb
              if r["exp"] == b and r["eval_group"] == grp and r["seed"] == s]
        if sa and sb:
            vals.append(sum(sa) / len(sa) - sum(sb) / len(sb))
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))
    ok = sum(1 for x in vals if (x > 0) == (m > 0))
    return m, (abs(m) / sd if sd else float("nan")), f"{ok}/{len(vals)}"


def main() -> int:
    base = load("p5_b2_perfile_seed10.csv")
    ocvt = load("p9_02_ocvT_ablation.csv")
    cm = load("p8_01_chanmatched.csv")

    configs = [
        ("A2-1−A2-4 | base", base, "A2-1", base, "A2-4"),
        ("A2-1−A2-4 | ocvT", ocvt, "A2-1", ocvt, "A2-4"),
        ("M1配平 | ocvT",    ocvt, "A2-1", cm,  "A2-K0"),
    ]
    rows = []
    for tag, ra, a, rb, b in configs:
        print("=" * 84)
        print(tag)
        print("=" * 84)
        for g in GROUPS:
            v = paired(ra, a, rb, b, g)
            if not v:
                continue
            n, m, lo, hi = ci(v)
            rm, rr, sr = ratio(ra, a, rb, b, g)
            excl = (lo > 0 or hi < 0)
            if n > 1:
                print(f"  {g:<13} n={n:2d} δ={m:+.4f} CI=[{lo:+.4f},{hi:+.4f}] "
                      f"比值={rr:.2f} 同号={sr}{'  <<排除零' if excl else ''}")
            rows.append(dict(config=tag, exp_a=a, exp_b=b, eval_group=g, n=n,
                             delta=m, ci_lo=lo, ci_hi=hi, ratio=rr,
                             sign_agree=sr, excl_zero=excl))
        print()

    with io.open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"已写出 {OUT}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
