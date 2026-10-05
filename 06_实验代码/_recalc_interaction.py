# -*- coding: utf-8 -*-
"""按 R1-M1 的要求补算消融的**直接对比**（审稿人指出全文缺这一项）。

R1-M1 的指控：中心主张"两族不等价"要求证明两族效应**彼此**有差别，
但稿件只证明了每一族**自身**的效应落在种子噪声内，再把"两个不可分辨的效应
符号相反"当作差别。要求补：
  (a) 2×2 交互项：A2-3 − A2-2 − A2-4 + A2-0
      （在 ν 族基础上加 K/P 的增量，减去 单独加 K/P 的增量）
  (b) 直接两族对比：A2-1 − A2-4（只加 ν vs 只加 K/P）

口径与 §5.3 一致：每文件先跨种子平均 -> 文件级配对 -> 文件间双侧 t 95% CI。
"""
import csv, math, sys, collections

sys.stdout.reconfigure(encoding="utf-8")
RES = r"E:\SOC论文项目\04_实验\数据\results"
T_CRIT = {4: 3.182, 9: 2.306}
GROUPS = ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]

rows = list(csv.DictReader(open(f"{RES}\\p5_b2_perfile_seed10.csv",
                                encoding="utf-8-sig")))
# per[(exp, cid)] = 跨种子均值
cells = collections.defaultdict(list)
for r in rows:
    cells[(r["exp"], r["cycle_id"])].append(float(r["RMSE"]))
per = {k: sum(v) / len(v) for k, v in cells.items()}
grp = {r["cycle_id"]: r["eval_group"] for r in rows}
files = {g: sorted({r["cycle_id"] for r in rows if r["eval_group"] == g})
         for g in GROUPS}


def combine(terms, g):
    """terms = [(exp, +1/-1), ...]；返回该组的逐文件组合差值。"""
    out = []
    for c in files[g]:
        s = 0.0
        for e, sign in terms:
            s += sign * per[(e, c)]
        out.append(s)
    return out


def ci(v):
    n = len(v); m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / n)
    se = sd / math.sqrt(n - 1)
    tc = T_CRIT.get(n, 2.0)
    return n, m, m - tc * se, m + tc * se


def report(title, terms, note=""):
    print(f"\n===== {title} =====")
    if note:
        print(f"  {note}")
    for g in GROUPS:
        v = combine(terms, g)
        n, m, lo, hi = ci(v)
        excl = "  <<< 排除零" if (lo > 0 or hi < 0) else "  （跨零）"
        print(f"  {g:14s} n={n:2d} δ={m:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}] "
              f"改善={sum(1 for x in v if x < 0)}/{n}{excl}")
    # OOD 合并
    allv = [x for g in GROUPS[1:] for x in combine(terms, g)]
    n, m, lo, hi = ci(allv)
    print(f"  {'OOD 合并':14s} n={n:3d} δ={m:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]"
          f"{'  <<< 排除零' if (lo > 0 or hi < 0) else '  （跨零）'}")


report("(a) 2×2 交互项  A2-3 − A2-2 − A2-4 + A2-0",
       [("A2-3", +1), ("A2-2", -1), ("A2-4", -1), ("A2-0", +1)],
       "= 在 ν 族上再加 K/P 的增量，减去 单独加 K/P 的增量")

report("(b) 直接两族对比  A2-1 − A2-4",
       [("A2-1", +1), ("A2-4", -1)],
       "= 只加 ν 相对 只加 K/P")

report("(c) 直接对比  A2-3 − A2-2（在 ν 族上再加 K/P）",
       [("A2-3", +1), ("A2-2", -1)])

report("(d) 直接对比  A2-3 − A2-4（全集 vs 只 K/P）",
       [("A2-3", +1), ("A2-4", -1)])
