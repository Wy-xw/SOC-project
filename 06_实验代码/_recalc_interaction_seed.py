# -*- coding: utf-8 -*-
"""R1-M1 要求的种子层面证据：交互项与直接对比的跨种子符号一致率与效应/种子比。"""
import csv, math, sys, collections

sys.stdout.reconfigure(encoding="utf-8")
RES = r"E:\SOC论文项目\04_实验\数据\results"
GROUPS = ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]

rows = list(csv.DictReader(open(f"{RES}\\p5_b2_perfile_seed10.csv",
                                encoding="utf-8-sig")))
grp = {r["cycle_id"]: r["eval_group"] for r in rows}
files = {g: sorted({r["cycle_id"] for r in rows if r["eval_group"] == g})
         for g in GROUPS}
# seed-level: per[(exp, seed, cid)]
per = collections.defaultdict(list)
for r in rows:
    per[(r["exp"], r["seed"], r["cycle_id"])].append(float(r["RMSE"]))


def combine_seed(terms, g, seed):
    v = []
    for c in files[g]:
        s = 0.0
        for e, sign in terms:
            s += sign * sum(per[(e, seed, c)]) / len(per[(e, seed, c)])
        v.append(s)
    return sum(v) / len(v)


def report(title, terms):
    print(f"\n===== {title} =====")
    tot_ok = tot = 0
    for g in GROUPS:
        seeds = sorted({r["seed"] for r in rows}, key=int)
        vals = {s: combine_seed(terms, g, s) for s in seeds}
        m = sum(vals.values()) / len(vals)
        sd = math.sqrt(sum((x - m) ** 2 for x in vals.values()) / len(vals))
        sign = 1 if m > 0 else -1
        ok = sum(1 for x in vals.values() if (1 if x > 0 else -1) == sign)
        ratio = abs(m) / sd if sd else float("nan")
        print(f"  {g:14s} 组均值={m:+.4f} 跨种子sd={sd:.4f} "
              f"效应/种子比={ratio:.2f} 同号={ok}/{len(vals)}")
        if g != "test_id(25C)":
            tot_ok += ok; tot += len(vals)
    print(f"  --> OOD 合计符号一致率: {tot_ok}/{tot} ({100*tot_ok/tot:.1f}%)")


report("(a) 2×2 交互项  A2-3 − A2-2 − A2-4 + A2-0",
       [("A2-3", +1), ("A2-2", -1), ("A2-4", -1), ("A2-0", +1)])
report("(b) 两族直接对比  A2-1 − A2-4",
       [("A2-1", +1), ("A2-4", -1)])
report("(c) A2-3 − A2-2", [("A2-3", +1), ("A2-2", -1)])
report("(d) A2-3 − A2-4", [("A2-3", +1), ("A2-4", -1)])
report("参照: A2-3 − A2-0", [("A2-3", +1), ("A2-0", -1)])
report("参照: A2-1 − A2-0", [("A2-1", +1), ("A2-0", -1)])
