# -*- coding: utf-8 -*-
"""P8-02 通道配平对照的分析与落盘（回应审稿人 R1-M2 / R2-M1）。

背景
----
核心对比 A2-1 − A2-4 同时改变族身份与通道数（8 vs 11），无法分离两种解释。
`p8_01_chanmatched.py` 新训了两条**通道数与左臂配平**的右臂：

    M1（8 通道）: A2-1 (7+ν)      −  A2-K0 (7+K₀)
    M2（9 通道）: A2-2 (7+ν+NIS)  −  A2-KK (7+K₀+K₁)

左臂取自主文 `p5_b2_perfile_seed10.csv`，右臂取自 `p8_01_chanmatched.csv`。

口径与 §5.3 一致：逐文件先跨种子平均、再做文件级配对；
效应/种子变异比 = |mean_s Δ| / SD_s Δ（ddof=0）。
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
OUT = os.path.join(R, "p8_02_chanmatched_compare.csv")


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
    """跨数据集配对：a 取自 ra，b 取自 rb。"""
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
    cm = load("p8_01_chanmatched.csv")
    m25 = load("p5_b2_perfile_seed10.csv")

    configs = [
        ("A-原(8vs11)", m25, "A2-1", m25, "A2-4", "A2-1 − A2-4（通道不配平）"),
        ("M1(8vs8)",    m25, "A2-1", cm,  "A2-K0", "A2-1 − A2-K0（8 通道配平）"),
        ("M2(9vs9)",    m25, "A2-2", cm,  "A2-KK", "A2-2 − A2-KK（9 通道配平）"),
    ]
    rows = []
    for tag, ra, a, rb, b, title in configs:
        print("=" * 84)
        print(f"{tag}  {title}")
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
            rows.append(dict(config=tag, exp_a=a, exp_b=b, eval_group=g,
                             n=n, delta=m, ci_lo=lo, ci_hi=hi,
                             ratio=rr, sign_agree=sr, excl_zero=excl))
        print()

    with io.open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"已写出 {OUT}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
