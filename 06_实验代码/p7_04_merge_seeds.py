# -*- coding: utf-8 -*-
"""P7-04 合并 0 °C 补证实验的两批种子结果。

背景
----
补证实验分两批跑（避免单次长任务风险）：
    第一批：seeds 7, 13, 42        -> p7_02_0C_ablation_seed3.csv（已备份）
    第二批：seeds 101..707（7 个）  -> p7_02_0C_ablation.csv（p7_02 覆盖写出）
两批**无种子重叠**，合并后恰为 10 个种子，与主设定
{7,13,42,101,202,303,404,505,606,707} 完全一致。

⚠️ 本脚本做**种子集合校验**：若两批有重叠、或合并后不等于 10 个标准种子，
   直接报错退出，不写盘（避免"以为跑齐了、其实缺或重复"）。
"""
from __future__ import annotations

import csv
import io
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
RESULTS = r"E:\SOC论文项目\04_实验\数据\results"
BATCH1 = os.path.join(RESULTS, "p7_02_0C_ablation_seed3.csv")
BATCH2 = os.path.join(RESULTS, "p7_02_0C_ablation.csv")
MERGED = os.path.join(RESULTS, "p7_02_0C_ablation_seed10.csv")

EXPECT = [7, 13, 42, 101, 202, 303, 404, 505, 606, 707]


def load(p):
    with io.open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> int:
    if not os.path.exists(BATCH1):
        print(f"🔴 缺第一批：{BATCH1}")
        return 1
    a, b = load(BATCH1), load(BATCH2)
    sa = sorted({int(r["seed"]) for r in a})
    sb = sorted({int(r["seed"]) for r in b})
    print(f"第一批 {len(a)} 行，种子 {sa}")
    print(f"第二批 {len(b)} 行，种子 {sb}")

    ov = set(sa) & set(sb)
    if ov:
        print(f"🔴 两批种子重叠：{sorted(ov)} —— 中止")
        return 1

    all_seeds = sorted(set(sa) | set(sb))
    if all_seeds != EXPECT:
        print(f"🔴 合并后种子 {all_seeds} ≠ 期望 {EXPECT} —— 中止")
        return 1
    print(f"\n✅ 无重叠，合并后 10 个种子与主设定一致")

    merged = a + b
    fields = list(a[0].keys())
    # 去重键：臂 + 种子 + 评测组 + 文件
    seen, uniq = set(), []
    for r in merged:
        k = (r["exp"], r["seed"], r["eval_group"], r["cycle_id"])
        if k in seen:
            print(f"  ⚠️ 重复行已跳过：{k}")
            continue
        seen.add(k)
        uniq.append(r)
    with io.open(MERGED, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(uniq)
    print(f"已写出 {MERGED}（{len(uniq)} 行 = 5 臂 × 10 种子 × {len(uniq)//50} 文件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
