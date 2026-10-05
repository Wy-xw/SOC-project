# -*- coding: utf-8 -*-
"""R3-M1：中心对比 A2-1−A2-4 的稳健性机制 —— 逐种子效应的相关结构。

R3 的质疑：A2-1 − A2-4 在算术上恒等于 (A2-1 − A2-0) − (A2-4 − A2-0)。
其跨种子标准差之所以小，只可能来自两臂的**逐种子效应正相关**
（共享种子噪声在相减时抵消）。稿件从未报告该相关。
本脚本给出：逐种子配对值、相关系数、方差分解，用于判断
"稳健"是配对设计的正常收益，还是被掩蔽的不稳定。
"""
import csv, math, sys, collections
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.stdout.reconfigure(encoding="utf-8")
RES = os.path.join(__ROOT__, "04_实验", "数据", "results")
OOD = ["ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]

rows = list(csv.DictReader(open(f"{RES}\\p5_b2_perfile_seed10.csv",
                                encoding="utf-8-sig")))
grp = {r["cycle_id"]: r["eval_group"] for r in rows}
seeds = sorted({r["seed"] for r in rows}, key=int)
files = {g: sorted({r["cycle_id"] for r in rows if r["eval_group"] == g})
         for g in OOD}
per = collections.defaultdict(list)
for r in rows:
    per[(r["exp"], r["seed"], r["cycle_id"])].append(float(r["RMSE"]))
per = {k: sum(v) / len(v) for k, v in per.items()}


def seed_eff(a, b, s):
    """该种子在四个 OOD 组上的合并效应。"""
    return sum(sum(per[(a, s, c)] - per[(b, s, c)] for c in files[g])
               / len(files[g]) for g in OOD) / len(OOD)


e1 = {s: seed_eff("A2-1", "A2-0", s) for s in seeds}   # ν 族
e4 = {s: seed_eff("A2-4", "A2-0", s) for s in seeds}   # K/P 族
e14 = {s: seed_eff("A2-1", "A2-4", s) for s in seeds}  # 直接对比


def stats(d):
    m = sum(d.values()) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d.values()) / len(d))
    return m, sd


m1, s1 = stats(e1)
m4, s4 = stats(e4)
m14, s14 = stats(e14)

# 相关系数
c = sum((e1[s] - m1) * (e4[s] - m4) for s in seeds) / len(seeds)
r = c / (s1 * s4)

print("=" * 72)
print("R3-M1 验证：中心对比的逐种子相关结构")
print("=" * 72)
print(f"\n逐种子效应（pp）：")
print(f"{'seed':>6} {'A2-1−A2-0':>12} {'A2-4−A2-0':>12} {'差值':>10}")
for s in seeds:
    print(f"{s:>6} {e1[s]:>+12.4f} {e4[s]:>+12.4f} {e14[s]:>+10.4f}")

print(f"\n组均值:  ν={m1:+.4f}  K/P={m4:+.4f}  直接对比={m14:+.4f}")
print(f"跨种子 SD: ν={s1:.4f}  K/P={s4:.4f}  直接对比={s14:.4f}")
print(f"\n相关系数 r(ν效应, K/P效应) = {r:+.4f}")
var_expl = 2 * c
print(f"\n方差分解: Var(差) = Var(ν) + Var(K/P) − 2·Cov")
print(f"  = {s1**2:.4f} + {s4**2:.4f} − {2*c:.4f} = {s1**2 + s4**2 - 2*c:.4f}")
print(f"  实际 Var(差) = {s14**2:.4f}")
print(f"  若无相关(r=0)，Var(差) 会是 {s1**2 + s4**2:.4f}，")
print(f"  SD 会是 {math.sqrt(s1**2 + s4**2):.4f}（对实际 {s14:.4f}）")
