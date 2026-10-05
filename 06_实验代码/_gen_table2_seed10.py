# -*- coding: utf-8 -*-
"""生成 10 种子版表2表体（意见9 ①）。

口径与旧 3 种子版一致：逐文件先跨种子平均 → 组内取均值与文件间标准差。
"""
import sys

import pandas as pd
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RES = os.path.join(__ROOT__, "04_实验", "数据", "results")
ORDER = ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]
LAB = {"test_id(25C)": "test\\\\_id (25 \\\\textdegree C)",
       "ood(10C)": "ood (10 \\\\textdegree C)", "ood(0C)": "ood (0 \\\\textdegree C)",
       "ood(-10C)": "ood ($-$10 \\\\textdegree C)",
       "ood(-20C)": "ood ($-$20 \\\\textdegree C)"}
NFILE = {"test_id(25C)": 4, "ood(10C)": 9, "ood(0C)": 9,
         "ood(-10C)": 9, "ood(-20C)": 9}

d = pd.read_csv(f"{RES}/p5_b2_perfile_seed10.csv")
# 逐文件先跨种子平均
pf = d.groupby(["exp", "eval_group", "cycle_id"]).RMSE.mean().reset_index()
print('    ["评测组", "$n$", "EKF", "A2-0", "A2-1", "A2-2", "A2-3", "A2-4"],')
ekf = {"test_id(25C)": "2.734", "ood(10C)": "4.709", "ood(0C)": "10.137",
       "ood(-10C)": "11.979", "ood(-20C)": "14.933"}
for g in ORDER:
    sub = pf[pf.eval_group == g]
    cells = []
    for arm in ["A2-0", "A2-1", "A2-2", "A2-3", "A2-4"]:
        v = sub[sub.exp == arm].RMSE
        cells.append(f"{v.mean():.3f} $\\\\pm$ {v.std(ddof=0):.3f}")
    print(f'    ["{LAB[g]}", "{NFILE[g]}", "{ekf[g]}",')
    print('     "' + '", "'.join(cells) + '"],')
