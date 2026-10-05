# -*- coding: utf-8 -*-
"""按温度分组重算符号一致率（与合并均值同号的种子-温度格数）。"""
import csv, sys, collections

sys.stdout.reconfigure(encoding="utf-8")
RES = r"E:\SOC论文项目\04_实验\数据\results"

rows = list(csv.DictReader(open(f"{RES}\\l1_paired_delta_perfile_seed10.csv",
                                encoding="utf-8-sig")))

groups = ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]
pairs = ["A2-1 − A2-0", "A2-2 − A2-0", "A2-3 − A2-0", "A2-4 − A2-0",
         "A2-3 − A2-1", "A2-3 − A2-4"]

print(f"{'对比':<16} " + " ".join(f"{g.split('(')[1][:-1]:>7}" for g in groups) + "   合计")
for p in pairs:
    out, tot, tot_ok = [], 0, 0
    for g in groups:
        cells = collections.defaultdict(list)
        for r in rows:
            if r["pair"] == p and r["eval_group"] == g:
                cells[r["seed"]].append(float(r["delta_pp"]))
        per_seed = {s: sum(v) / len(v) for s, v in cells.items()}
        merged = sum(per_seed.values()) / len(per_seed)
        sign = 1 if merged > 0 else -1
        ok = sum(1 for v in per_seed.values() if (1 if v > 0 else -1) == sign)
        out.append(f"{ok}/{len(per_seed)}")
        tot += len(per_seed); tot_ok += ok
    pct = 100.0 * tot_ok / tot
    print(f"{p:<16} " + " ".join(f"{o:>7}" for o in out) + f"   {tot_ok}/{tot} ({pct:.1f}%)")
