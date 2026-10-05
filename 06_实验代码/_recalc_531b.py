# -*- coding: utf-8 -*-
"""从 p5_b2_perfile_seed10.csv 直接算所有需要的配对统计（不依赖 l1 的 pair 列表）。

口径：
  组均值   = 每文件先跨种子平均 -> 再对文件求均值
  配对 δ   = 每文件先跨种子平均 -> 文件级配对 -> 文件间双侧 t 95% CI
  效应/种子比 = 逐 (group, seed) 组均值 -> |mean_s δ| / SD_s(ddof=0)
  符号一致率 = 每 seed 的组均值是否与合并均值同号
"""
import csv, math, sys, collections

sys.stdout.reconfigure(encoding="utf-8")
RES = r"E:\SOC论文项目\04_实验\数据\results"

T_CRIT = {4: 3.182, 9: 2.306, 12: 2.179, 40: 2.021, 90: 1.987}
GROUPS = ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]
ODD = GROUPS[1:]

rows = list(csv.DictReader(open(f"{RES}\\p5_b2_perfile_seed10.csv", encoding="utf-8-sig")))

# rmse[(exp, seed, cycle_id)] = RMSE ；并记录 group / file
rmse, meta = {}, {}
for r in rows:
    k = (r["exp"], r["seed"], r["cycle_id"])
    rmse[k] = float(r["RMSE"]) * 100.0  # 源文件是分数？先按乘 100 试，下面自检
    meta[r["cycle_id"]] = r["eval_group"]

# 自检量级
probe = [v for (e, s, c), v in rmse.items() if e == "A2-3" and meta[c] == "test_id(25C)"]
print(f"[自检] A2-3 test_id 量级样例: {sorted(probe)[:4]}")
SCALE = 1.0 if max(probe) < 5 else 0.01
if SCALE != 1.0:
    rmse = {k: v * SCALE for k, v in rmse.items()}
print(f"[自检] 采用缩放 SCALE={SCALE}")

files = {g: sorted({c for c in meta if meta[c] == g}) for g in GROUPS}
seeds = sorted({s for (_, s, _) in rmse}, key=int)


def file_mean(exp, seed, cid):
    return rmse[(exp, seed, cid)]


def group_mean(exp, g):
    """每文件跨种子平均 -> 文件均值"""
    vals = []
    for c in files[g]:
        vals.append(sum(file_mean(exp, s, c) for s in seeds) / len(seeds))
    return sum(vals) / len(vals)


def paired(a, b, g):
    """δ = a − b，文件级（先跨种子平均）"""
    return [(sum(file_mean(a, s, c) - file_mean(b, s, c) for s in seeds) / len(seeds))
            for c in files[g]]


def ci(v):
    n = len(v); m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / n)
    se = sd / math.sqrt(n - 1)
    tc = T_CRIT.get(n, 2.0)
    return n, m, m - tc * se, m + tc * se, f"{sum(1 for x in v if x < 0)}/{n}"


def ratio_and_sign(a, b, g):
    per_seed = {}
    for s in seeds:
        vals = [file_mean(a, s, c) - file_mean(b, s, c) for c in files[g]]
        per_seed[s] = sum(vals) / len(vals)
    v = list(per_seed.values()); m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
    sign = 1 if m > 0 else -1
    ok = sum(1 for x in v if (1 if x > 0 else -1) == sign)
    return abs(m) / sd if sd else float("nan"), f"{ok}/{len(v)}", per_seed


print("\n===== 组均值（RMSE pp，10 种子） =====")
exps = ["A2-0", "A2-1", "A2-2", "A2-3", "A2-4", "EKF"]
hdr = "exp   " + "".join(f"{g:>14}" for g in GROUPS)
print(hdr)
for e in exps:
    if not any(k[0] == e for k in rmse):
        continue
    print(f"{e:<6}" + "".join(f"{group_mean(e, g):>14.4f}" for g in GROUPS))

print("\n===== 配对统计 =====")
for a, b in [("A2-4", "A2-0"), ("A2-3", "A2-0"), ("A2-3", "A2-2"),
             ("A2-3", "A2-4"), ("A2-3", "A2-1"), ("A2-2", "A2-0"), ("A2-1", "A2-0")]:
    print(f"\n--- {a} − {b} ---")
    for g in GROUPS:
        v = paired(a, b, g)
        n, m, lo, hi, imp = ci(v)
        rr, sr, per = ratio_and_sign(a, b, g)
        flag = " <<排除零" if (lo > 0 or hi < 0) else ""
        print(f"  {g:14s} n={n:2d} δ={m:+.4f} CI=[{lo:+.4f},{hi:+.4f}] 改善={imp:>4s} "
              f"效应/种子比={rr:.2f} 符号={sr}{flag}")
    # OOD 合计符号一致率
    tot = ok = 0
    for g in ODD:
        _, sr, _ = ratio_and_sign(a, b, g)
        k, d = sr.split("/"); ok += int(k); tot += int(d)
    print(f"  OOD 合计符号一致率: {ok}/{tot} ({100*ok/tot:.1f}%)")
