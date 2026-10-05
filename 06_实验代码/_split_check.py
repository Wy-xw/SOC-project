# -*- coding: utf-8 -*-
"""核验：论文的 OOD 温度分组是否与官方划分清单一致。

背景
----
`p3_04_train.py` 用**字符串匹配**分组：
    "ood(0C)": [c for c in ids if "0degC_Cycle" in c]
实测该模式在 `ekf_full_history.npz` 的 keys 上**命中 16 个文件**
（横跨 10/0/−10/−20 °C 四个温度）。

但产物 `p5_b2_perfile.csv` 里 `ood(0C)` 只有 **9 个、全是 0degC_**。

⇒ 两者对不上，必须查清是"训练划分错了"还是"后续脚本修正了"。

**权威依据**：`splits/split_manifest.csv` 有 `nominal_temp_c` 字段，
是数据构建阶段按**目录名**定的温度，不依赖字符串匹配。
"""
import os
import sys

import pandas as pd

ROOT = r"E:\SOC论文项目"
RESULTS = os.path.join(ROOT, "04_实验", "数据", "results")
SPLITS = os.path.join(ROOT, "04_实验", "数据", "splits")
OUT = os.path.join(ROOT, "_tmp_imgs", "_split_check.txt")


def main():
    out = []
    m = pd.read_csv(os.path.join(SPLITS, "split_manifest.csv"))
    out.append("=== split_manifest.csv 按温度×划分 的文件数 ===")
    t = m.groupby(["split", "nominal_temp_c"]).size().unstack(fill_value=0)
    out.append(t.to_string())
    out.append("")

    # 官方 OOD 各温度的文件名
    ood = m[m["split"] == "test_ood"]
    out.append("=== 官方 test_ood 按 nominal_temp_c 分组 ===")
    for temp in sorted(ood["nominal_temp_c"].unique()):
        sub = ood[ood["nominal_temp_c"] == temp]
        out.append(f"  {temp:6.1f} °C : {len(sub)} 个")
        for c in sorted(sub["cycle_id"])[:3]:
            out.append(f"        {c}")
        if len(sub) > 3:
            out.append(f"        ... 共 {len(sub)} 个")
    out.append("")

    # 产物里各组的文件数
    out.append("=== 产物 p5_b2_perfile.csv 各 eval_group 的文件数 ===")
    d = pd.read_csv(os.path.join(RESULTS, "p5_b2_perfile.csv"))
    for g in sorted(d["eval_group"].unique()):
        n = d[d["eval_group"] == g]["cycle_id"].nunique()
        out.append(f"  {g:16s} {n} 个")
    out.append("")

    # 核心比对：产物的 ood(0C) 是否都真的来自 0 °C 目录
    out.append("=== 核心比对：产物 ood(0C) 的文件在官方的 nominal_temp_c ===")
    prod0 = set(d[d["eval_group"] == "ood(0C)"]["cycle_id"].unique())
    mm = m.set_index("cycle_id")
    bad = []
    for c in sorted(prod0):
        if c in mm.index:
            tc = mm.loc[c, "nominal_temp_c"]
            flag = "" if abs(tc - 0.0) < 0.1 else "  <<< 不是 0 °C！"
            out.append(f"  {c[:44]:44s} nominal_temp={tc:6.1f}{flag}")
            if abs(tc - 0.0) > 0.1:
                bad.append(c)
        else:
            out.append(f"  {c[:44]:44s} （不在 manifest 里）")
    out.append("")
    out.append(f"⇒ 错配文件数：{len(bad)}")
    if bad:
        out.append("⇒ 🔴 OOD(0C) 组**混入了非 0 °C 文件**，划分有问题")
    else:
        out.append("⇒ ✅ OOD(0C) 组全部来自 0 °C 目录，划分正确")

    open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
