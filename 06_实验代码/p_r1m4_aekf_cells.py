# -*- coding: utf-8 -*-
"""R1-M4 / P0-26：从 r1m4_perfile.csv 重生成 r1m4_aekf_cells.csv（论文 Table 1 的 AEKF-R 列；2026-09-26 重编号前叫 Table 2）

背景（P0-26）
-------------
`results/r1m4_aekf_cells.csv`（13 行 = 1 个 25 °C 聚合格 + 12 个 OOD 单文件格）
此前**没有生成脚本**：`p_r1m4_adaptive.py` 只写 `r1m4_perfile.csv` 与
`r1m4_summary.md`，按当时流程该 13 行是人工从 perfile 表里挑/算出来的。
本脚本把这一步脚本化，使该产物**可从 perfile 表字节级复现**。

口径（与 p_r1m4_adaptive.py 的变体定义一致）
-------------------------------------------
- 变体：`variant == "AEKF_R"`（自适应 R，Table 1 第 7 列 "AEKF-R (adaptive R)"）
- 12 个 OOD 格：`condition` 精确等于 `<TEMP>degC_<DRIVE>`（每格 1 个文件）
- `Cycle_25C` 格：**不是单文件**，而是 `grp == "test_id(25C)"` 的 4 个文件
  （25degC_Cycle_1..4）的 AEKF_R RMSE 均值 —— 与 `p5_01_baseline_comparison.csv`
  中 `eval_group == "Cycle_25C"`（n_files=4）同口径，也是 p6_tables.py 消费该表的方式
- 舍入：3 位小数（与现存文件一致）

匹配方式说明
------------
不用子串匹配（`"0degC_"` 是 `"n10degC_"` 的子串，P0-16 已踩过这个坑），
而是把 cycle_id 解析成精确的 condition token 后做**等值**匹配。

用法
----
    python p_r1m4_aekf_cells.py            # 只校验：写 _regen 副本并与现存文件逐字节比对
    python p_r1m4_aekf_cells.py --write    # 校验一致后再覆盖写入 r1m4_aekf_cells.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# Windows 控制台默认 GBK，而本脚本 print 含 GBK 无码位的字符 —— 不改编码会在
# **成功路径**崩掉（活干完了却退出码 1）。别靠 PYTHONIOENCODING 兜底。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "04_实验", "数据", "results")
PERFILE = os.path.join(RES, "r1m4_perfile.csv")
CANON = os.path.join(RES, "r1m4_aekf_cells.csv")
REGEN = os.path.join(RES, "r1m4_aekf_cells_regen.csv")

VARIANT = "AEKF_R"
GROUP_25C = "test_id(25C)"
DRIVES = ["HWFET", "UDDS", "US06"]
TEMPS = {"10C": "10degC", "0C": "0degC", "-10C": "n10degC", "-20C": "n20degC"}


def condition_of(cycle_id: str) -> str:
    """'03-27-17_09.06 10degC_HWFET_Pan18650PF' -> '10degC_HWFET'"""
    return cycle_id.split()[-1].replace("_Pan18650PF", "")


def build() -> pd.DataFrame:
    df = pd.read_csv(PERFILE)
    a = df[df.variant == VARIANT].copy()
    if a.empty:
        raise SystemExit(f"perfile 中找不到 variant == {VARIANT!r}")
    a["condition"] = a.cycle_id.map(condition_of)

    cells: dict[str, float] = {}
    # 25 °C：4 文件聚合格
    g = a[a.grp == GROUP_25C]
    if len(g) == 0:
        raise SystemExit(f"perfile 中找不到 grp == {GROUP_25C!r}")
    cells["Cycle_25C"] = float(g.rmse.mean())
    n25 = len(g)

    # 12 个 OOD 单文件格（等值匹配，不做子串）
    for drive in DRIVES:
        for tag, prefix in TEMPS.items():
            want = f"{prefix}_{drive}"
            hit = a[a.condition == want]
            if len(hit) != 1:
                raise SystemExit(
                    f"condition={want!r} 命中 {len(hit)} 行（期望 1 行），"
                    "请检查 perfile 是否被改动")
            cells[f"{drive}_{tag}"] = float(hit.rmse.iloc[0])

    out = pd.DataFrame(
        {"cell": sorted(cells), "rmse": [f"{cells[c]:.3f}" for c in sorted(cells)]})
    print(f"AEKF_R 变体：{len(a)} 行；Cycle_25C 由 {n25} 个 test_id 文件求均值")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="校验一致后覆盖写入 r1m4_aekf_cells.csv")
    args = ap.parse_args()

    new = build()
    new.to_csv(REGEN, index=False)
    print(f"已写出校验副本：{REGEN}")

    if not os.path.exists(CANON):
        print(f"⚠️ 现存文件不存在：{CANON}")
        return 1

    with open(CANON, "rb") as f:
        old_bytes = f.read()
    with open(REGEN, "rb") as f:
        new_bytes = f.read()

    if old_bytes == new_bytes:
        print(f"✅ 逐字节一致（{len(new_bytes)} B）—— 现存 {os.path.basename(CANON)} "
              f"可由本脚本从 r1m4_perfile.csv 完整复现")
        if args.write:
            with open(CANON, "wb") as f:
                f.write(new_bytes)
            print(f"已按 --write 覆盖写入 {CANON}")
        else:
            print("（未加 --write，现存文件保持原样）")
        return 0

    print("❌ 不一致！逐行差异：")
    old = pd.read_csv(CANON).set_index("cell")["rmse"]
    nw = new.set_index("cell")["rmse"]
    for c in sorted(set(old.index) | set(nw.index)):
        o = old.get(c, None)
        n = nw.get(c, None)
        flag = "  " if o == n else "!!"
        print(f"  {flag} {c:<12s} 现存={o}  复算={n}")
    print(f"（现存 {len(old_bytes)} B / 复算 {len(new_bytes)} B）")
    return 2


if __name__ == "__main__":
    sys.exit(main())
