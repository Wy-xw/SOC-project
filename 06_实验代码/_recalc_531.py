# -*- coding: utf-8 -*-
"""重算 §5.3.1 / §5.3.2 所需的 10 种子文件级配对统计。

规则（与论文口径一致）：
  1. 每文件先跨种子平均 -> 文件级配对 -> 文件间双侧 t 95% CI
  2. 效应/种子变异比 = |mean_s(delta)| / SD_s(delta)   (ddof=0)
"""
import csv, math, sys, collections

sys.stdout.reconfigure(encoding="utf-8")
RES = r"E:\SOC论文项目\04_实验\数据\results"

T_CRIT = {4: 3.182, 9: 2.306, 12: 2.179, 27: 2.052, 40: 2.021, 90: 1.987}


def load(name):
    with open(f"{RES}\\{name}", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def groups_of(cycle_id):
    """与 p5_b2_multifile_ablation.groups_of 完全一致的分组。"""
    if "25degC" in cycle_id:
        return "test_id(25C)"
    for t, tag in [(10, "10degC"), (0, "0degC"), (-10, "n10degC"), (-20, "n20degC")]:
        if tag + "_" in cycle_id and "trise" not in cycle_id:
            return f"ood({t}C)"
    return None


def paired(rows, pair):
    """-> {group: (n, mean, sd, ci_lo, ci_hi, n_improved)}"""
    by_file = collections.defaultdict(list)
    for r in rows:
        if r["pair"] != pair:
            continue
        g = r.get("eval_group") or groups_of(r["cycle_id"])
        by_file[(g, r["cycle_id"])].append(float(r["delta_pp"]))
    per_g = collections.defaultdict(list)
    for (g, cid), v in by_file.items():
        per_g[g].append(sum(v) / len(v))
    out = {}
    for g, v in sorted(per_g.items()):
        n = len(v)
        m = sum(v) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / n)
        se = sd / math.sqrt(n - 1)
        tc = T_CRIT.get(n, 2.0)
        imp = sum(1 for x in v if x < 0)
        out[g] = (n, m, sd, m - tc * se, m + tc * se, f"{imp}/{n}")
    return out


def ratio(rows, pair):
    """效应/种子变异比：逐 (group, seed) 先取组均值，再比 |mean|/SD(ddof=0)。"""
    gs = collections.defaultdict(list)
    for r in rows:
        if r["pair"] != pair:
            continue
        g = r.get("eval_group") or groups_of(r["cycle_id"])
        gs[(g, r["seed"])].append(float(r["delta_pp"]))
    per = collections.defaultdict(list)
    for (g, s), v in gs.items():
        per[g].append(sum(v) / len(v))
    out = {}
    for g, v in sorted(per.items()):
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
        out[g] = (m, sd, abs(m) / sd if sd else float("nan"))
    return out


def show(title, rows, pair):
    print(f"\n===== {title} =====")
    p = paired(rows, pair)
    r = ratio(rows, pair)
    for g in p:
        n, m, sd, lo, hi, imp = p[g]
        rm, rsd, rr = r.get(g, (0, 0, 0))
        flag = "  <<< 排除零" if (lo > 0 or hi < 0) else ""
        print(f"  {g:14s} n={n:3d} mean={m:+.4f} 95%CI=[{lo:+.4f},{hi:+.4f}] 改善={imp} "
              f"| 效应/种子比={rr:.2f} (m={rm:+.4f}, sd={rsd:.4f}){flag}")


for tag, f in [("3 种子", "l1_paired_delta_perfile.csv"),
               ("10 种子", "l1_paired_delta_perfile_seed10.csv")]:
    rows = load(f)
    print("=" * 78)
    print(f"########## {tag}  ({f}) ##########")
    for pair in ["A2-4 − A2-0", "A2-3 − A2-0", "A2-3 − A2-2", "A2-3 − A2-4",
                 "A2-2 − A2-0", "A2-1 − A2-0"]:
        show(pair, rows, pair)

# 组均值（RMSE）也需要 10 种子版本
print("\n" + "=" * 78)
print("########## 10 种子组均值（RMSE pp） ##########")
PF = load("p5_b2_perfile_seed10.csv")
print("cols:", list(PF[0].keys()))
