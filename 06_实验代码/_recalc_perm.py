# -*- coding: utf-8 -*-
"""为 R1-M3 / R1-m5 补**置换检验**（审稿人指出全文只有一个孤立 p 值）。

审稿人的两点：
  R1-m5  A2-4 只给了符号一致率、没给对应的方向性检验
  R2-m3  孤立的 p=0.34 没定义零假设/统计量/单双侧

做法：对每个对比，按**种子标签置换**（10! 太大 -> 用 Monte Carlo 10000 次），
统计量取「组均值效应」，检验 H0: 效应在种子间可交换（即方向由种子噪声决定）。
报告双侧 p 与置换分布，并写清零假设。
"""
import csv, math, random, sys, collections
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.stdout.reconfigure(encoding="utf-8")
random.seed(20261004)
RES = os.path.join(__ROOT__, "04_实验", "数据", "results")
GROUPS = ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]
OOD = GROUPS[1:]

rows = list(csv.DictReader(open(f"{RES}\\p5_b2_perfile_seed10.csv",
                                encoding="utf-8-sig")))
grp = {r["cycle_id"]: r["eval_group"] for r in rows}
seeds = sorted({r["seed"] for r in rows}, key=int)
files = {g: sorted({r["cycle_id"] for r in rows if r["eval_group"] == g})
         for g in GROUPS}
# per[(exp, seed, cid)]
per = collections.defaultdict(list)
for r in rows:
    per[(r["exp"], r["seed"], r["cycle_id"])].append(float(r["RMSE"]))
per = {k: sum(v) / len(v) for k, v in per.items()}


def delta_seed(a, b, g, s):
    return sum(per[(a, s, c)] - per[(b, s, c)] for c in files[g]) / len(files[g])


def perm_test(a, b, g, n=None):
    """符号翻转置换检验（**精确枚举**）。

    H0：该对比的效应在种子间可交换（方向由种子噪声决定）。
    统计量 = 组均值效应 mean_s Δ_s；置换 = 翻转每个种子的 Δ_s 符号。

    ⚠️ **精确枚举而非 Monte Carlo**：10 个种子只有 2^10 = 1024 种符号组合，
    全部可枚举，无需抽样近似（2026-10-05 按修改意见改）。
    双侧 p = 满足 |翻转后均值| ≥ |观测均值| 的组合数 / 1024。
    """
    obs = [delta_seed(a, b, g, s) for s in seeds]
    mean_obs = sum(obs) / len(obs)
    k = len(obs)
    cnt = 0
    for mask in range(1 << k):
        s = sum(-obs[i] if (mask >> i) & 1 else obs[i] for i in range(k))
        if abs(s / k) >= abs(mean_obs):
            cnt += 1
    return mean_obs, cnt / (1 << k)


PAIRS = [("A2-1", "A2-0", "A2-1 − A2-0（新息/残差类）"),
         ("A2-2", "A2-0", "A2-2 − A2-0"),
         ("A2-4", "A2-0", "A2-4 − A2-0（不确定性/增益类）"),
         ("A2-3", "A2-0", "A2-3 − A2-0（完整诊断集）"),
         ("A2-1", "A2-4", "A2-1 − A2-4（两族直接对比）"),
         ("A2-3", "A2-2", "A2-3 − A2-2"),
         ("A2-3", "A2-4", "A2-3 − A2-4")]

print("置换检验（H0：效应在种子间可交换；双侧；**精确枚举 2^10 = 1024 种符号组合**）")
print("=" * 74)
for a, b, lab in PAIRS:
    ps = []
    for g in GROUPS:
        m, p = perm_test(a, b, g)
        ps.append(p)
    # OOD 合并：按四个 OOD 的种子均值
    ood_obs = []
    for s in seeds:
        ood_obs.append(sum(delta_seed(a, b, g, s) for g in OOD) / len(OOD))
    mo = sum(ood_obs) / len(ood_obs)
    k = len(ood_obs)
    cnt = 0
    for mask in range(1 << k):
        s = sum(-ood_obs[i] if (mask >> i) & 1 else ood_obs[i] for i in range(k))
        if abs(s / k) >= abs(mo):
            cnt += 1
    p_ood = cnt / (1 << k)
    print(f"\n{lab}")
    print(f"  各温度 p: " + "  ".join(f"{g[4:-1] or '25C'}={p:.3f}"
                                     for g, p in zip(GROUPS, ps)))
    print(f"  OOD 合并: 效应={mo:+.4f}  p={p_ood:.4f}"
          f"  {'(p<0.05)' if p_ood < 0.05 else ''}")
