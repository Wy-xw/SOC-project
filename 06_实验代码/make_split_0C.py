# -*- coding: utf-8 -*-
"""生成 0 °C 训练的反向设定划分（补证实验用，不动原 split_manifest.csv）。

设计依据（对齐 25 °C 主设定的结构）
-----------------------------------
主设定：训练 5（HWFTa/HWFTb/LA92/NN/US06）+ 验证 1（UDDS）+ 测试 4（Cycle_1–4）
0 °C 只有单份标准工况（无 a/b），故：
    训练 4：HWFT / LA92 / NN / US06
    验证 1：UDDS
    测试 4：Cycle_1–4          ← 这 4 个成为**同分布测试**
    反向测试：25 °C 全部作为 OOD（4 个 Cycle + 5 训练 + 1 UDDS = 10 个文件）

为什么要这个设定（审稿人 R1-M1）
--------------------------------
主设定是"25 °C 训练 → 低温外推"。本设定把训练温度换到 0 °C，
检验"两族不可互换"这一结论**是否随训练温度改变**。

输出：`split_manifest_0C.csv`（与原文件同结构，只改 split 列）
"""
from __future__ import annotations

import csv
import io
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS = os.path.join(ROOT, "04_实验", "数据", "splits")
SRC = os.path.join(SPLITS, "split_manifest.csv")
DST = os.path.join(SPLITS, "split_manifest_0C.csv")

#: 0 °C 下各文件的角色
ROLE_0C = {
    "0degC_HWFET": "train",
    "0degC_LA92": "train",
    "0degC_NN": "train",
    "0degC_US06": "train",
    "0degC_UDDS": "val",
    "0degC_Cycle_1": "test_id",
    "0degC_Cycle_2": "test_id",
    "0degC_Cycle_3": "test_id",
    "0degC_Cycle_4": "test_id",
}


def role_of(cid: str, temp: str) -> str:
    """返回该文件在**新设定**下的角色。"""
    if temp == "0.0":
        for key, role in ROLE_0C.items():
            if key in cid:
                return role
        return "unused"
    # 25 °C 由训练/验证/同分布测试 -> 全部变为 OOD 测试
    if temp == "25.0":
        return "test_ood"
    return "test_ood"          # 其余低温保持 OOD


def main() -> int:
    rows = list(csv.DictReader(io.open(SRC, encoding="utf-8-sig")))
    fields = list(rows[0].keys())
    n_changed = 0
    for r in rows:
        new = role_of(r["cycle_id"], r["nominal_temp_c"])
        if new != r["split"]:
            n_changed += 1
        r["split"] = new

    with io.open(DST, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    import collections
    c = collections.Counter((r["nominal_temp_c"], r["split"]) for r in rows)
    print(f"已写出 {DST}")
    print(f"角色变更 {n_changed} 个文件\n")
    print(f"{'温度':>7} {'划分':<12} {'文件数':>6}")
    for (t, s), n in sorted(c.items(), key=lambda kv: (float(kv[0][0]), kv[0][1])):
        print(f"{t:>7} {s:<12} {n:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
