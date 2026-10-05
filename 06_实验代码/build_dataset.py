# -*- coding: utf-8 -*-
"""
构建统一格式的建模数据集并划分训练/验证/测试集 —— 对应任务 P1-03 / P1-04 / P1-05

用法
----
    "D:/venvs/soc-paper/Scripts/python.exe" build_dataset.py

产出（均在 `04_实验/数据/`）
--------------------------
    processed/mcmaster_drive_cycles.csv.gz   全部（单工况）驱动工况，统一格式
    processed/mcmaster_hppc.csv.gz            HPPC 脉冲测试（P2-03 参数辨识用）
    processed/mcmaster_ocv_c20.csv.gz         C/20 小电流充放电（P2-02 OCV-SOC 用）
    splits/split_manifest.csv                 每个工况文件的元数据与所属划分
    _tmp/rejected_files.csv                   被排除的文件及原因

划分原则（**不能随机划分**，否则数据泄漏）
--------------------------------------
- 训练 / 验证 / 测试**按「工况」与「温度」划分**，同一文件的采样点绝不跨集合。
- Cycle_1~4 是多种工况的随机混合，与 US06/HWFET/... 同分布 → 作为「同分布留出集」，
  单独成组，不混入训练集。
- 合并文件（多工况记录在同一文件）**全部排除**：其数据与拆分文件重复，
  且混有充电段，会破坏 Ah 计数器的每工况重置假设。
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import data_io as dio  # noqa: E402

# Windows 控制台默认 GBK，而本脚本 print 含 GBK 无码位的字符 —— 不改编码会在
# **成功路径**崩掉（活干完了却退出码 1）。别靠 PYTHONIOENCODING 兜底。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = os.path.join(dio.PROJECT_ROOT, "04_实验", "数据")
PROCESSED = os.path.join(DATA_DIR, "processed")
SPLITS = os.path.join(DATA_DIR, "splits")
TMP = os.path.join(DATA_DIR, "_tmp")

# --------------------------------------------------------------------------- #
# 划分定义
# --------------------------------------------------------------------------- #
#: 训练温度 = 25 °C，测试温度 = 其余全部 → 直接支撑「温度外推」实验
TRAIN_TEMP = [25.0]
OOD_TEMPS = [10.0, 0.0, -10.0, -20.0]

#: 25 °C 下的工况分工
TRAIN_TYPES = ["us06", "hwfet", "la92", "nn"]
VAL_TYPES = ["udds"]
#: Cycle_1~4：同温度、同分布，但序列不同 → 作为「同分布留出集」
ID_TEST_TYPES = ["cycle"]


def assign_split(f: dio.McMasterFile) -> str:
    """给一个驱动工况文件指定所属集合。"""
    if f.is_combined:
        return "excluded_combined"
    if f.temperature_c == 25.0 and not f.is_trise:
        if f.test_type in TRAIN_TYPES:
            return "train"
        if f.test_type in VAL_TYPES:
            return "val"
        if f.test_type in ID_TEST_TYPES:
            return "test_id"
        return "excluded_other"
    if f.temperature_c in OOD_TEMPS:
        # 变温组单独成组，用于「温度漂移」鲁棒性实验，不混入温度外推主测试
        return "test_ood_trise" if f.is_trise else "test_ood"
    return "excluded_other"


def main() -> int:
    for d in (PROCESSED, SPLITS, TMP):
        os.makedirs(d, exist_ok=True)

    files = dio.scan_mcmaster()
    drive = [f for f in files if f.is_drive_cycle]
    print(f"扫描 MAT 文件 {len(files)} 个，其中驱动工况 {len(drive)} 个")

    # -------------------- 驱动工况：转统一格式 -------------------- #
    frames, manifest, rejected = [], [], []
    for f in sorted(drive, key=lambda x: (x.temperature_c, x.test_type, x.filename)):
        split = assign_split(f)
        if split.startswith("excluded"):
            rejected.append({"relpath": f.relpath, "reason": split})
            continue
        try:
            df = dio.build_mcmaster_frame(f.path)
        except ValueError as exc:
            rejected.append({"relpath": f.relpath, "reason": f"读取失败：{exc}"})
            continue
        df["split"] = split
        df["test_file"] = f.relpath
        frames.append(df)
        manifest.append({
            "split": split, "cycle_id": df["cycle_id"].iloc[0], "relpath": f.relpath,
            "nominal_temp_c": f.temperature_c, "test_type": f.test_type,
            "is_trise": f.is_trise, "n_samples": len(df), "duration_s": float(df.timestamp.iloc[-1]),
            "soc_start": float(df.soc_true.iloc[0]), "soc_end": float(df.soc_true.iloc[-1]),
            "temp_min_c": float(df.temperature_C.min()), "temp_max_c": float(df.temperature_C.max()),
            "I_max_A": float(df.current_A.max()), "I_min_A": float(df.current_A.min()),
        })
        print(f"  [{split:15s}] {f.relpath}  n={len(df):6d}  SOC {df.soc_true.iloc[0]:.2f}→{df.soc_true.iloc[-1]:.2f}")

    if not frames:
        print("❌ 没有任何文件被纳入，请检查 raw 数据是否就位。")
        return 1

    all_df = pd.concat(frames, ignore_index=True)
    drive_path = os.path.join(PROCESSED, "mcmaster_drive_cycles.csv.gz")
    all_df.to_csv(drive_path, index=False, compression="gzip")
    print(f"\n✅ 统一格式驱动工况：{len(all_df):,} 行 → {os.path.relpath(drive_path, dio.PROJECT_ROOT)}")

    pd.DataFrame(manifest).to_csv(os.path.join(SPLITS, "split_manifest.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(rejected).to_csv(os.path.join(TMP, "rejected_files.csv"), index=False, encoding="utf-8-sig")

    # -------------------- 辅助测试：HPPC / OCV -------------------- #
    for kind, outname in [("hppc", "mcmaster_hppc.csv.gz"), ("ocv_c20", "mcmaster_ocv_c20.csv.gz")]:
        aux = [f for f in files if f.test_type == kind and not f.is_combined]
        parts = []
        for f in aux:
            try:
                d = dio.build_mcmaster_frame(f.path)
            except ValueError as exc:
                print(f"  ⚠ 跳过 {f.relpath}：{exc}")
                continue
            d["test_file"] = f.relpath
            parts.append(d)
        if parts:
            out = pd.concat(parts, ignore_index=True)
            p = os.path.join(PROCESSED, outname)
            out.to_csv(p, index=False, compression="gzip")
            print(f"✅ {kind:8s}：{len(parts)} 个文件 {len(out):,} 行 → {os.path.relpath(p, dio.PROJECT_ROOT)}")

    # -------------------- 汇总 -------------------- #
    print("\n━━━━ 划分汇总 ━━━━")
    man = pd.DataFrame(manifest)
    summary = (man.groupby("split")
               .agg(文件数=("cycle_id", "count"), 采样点=("n_samples", "sum"),
                    时长h=("duration_s", lambda s: round(s.sum() / 3600, 2)),
                    SOC起点均值=("soc_start", lambda s: round(s.mean(), 3)),
                    SOC终点均值=("soc_end", lambda s: round(s.mean(), 3)))
               .reset_index())
    print(summary.to_string(index=False))

    rej = pd.DataFrame(rejected)
    if len(rej):
        print(f"\n排除 {len(rej)} 个文件：")
        print(rej["reason"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
