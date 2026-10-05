# -*- coding: utf-8 -*-
"""
P5-05 与文献对比表

基于以下来源构建对比：
1. 我们的实验结果（P5-01/P5-03/P5-04）
2. 已读文献的具体数值（Yuan 2023 全文、How 2019 综述）
3. 领域内公认的经典方法典型精度范围

注意：文献值来自不同数据集/工况，不能直接横向比较绝对数值，
但可以比较**相对改善幅度**和**方法类别**的排名趋势。

产出
----
- results/p5_05_literature_comparison.csv
- stdout 打印对比表

用法
----
    "D:/venvs/soc-paper/Scripts/python.exe" 10_代码/p5_05_literature_compare.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

# Windows 控制台默认 GBK，而本脚本 print 含 GBK 无码位的字符 —— 不改编码会在
# **成功路径**崩掉（活干完了却退出码 1）。别靠 PYTHONIOENCODING 兜底。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")


def main() -> int:
    # ================================================================== #
    # 文献对比数据
    # ================================================================== #
    # 来源标注：
    #   [our]     = 本项目实验结果（McMaster 18650PF，1 Hz，稳态口径）
    #   [Yuan23]  = Yuan et al. 2023 Energies（18650 NCM，25°C，UDDS/HWFET）
    # ⚠️ 2026-09-26：本行描述的是 **Yuan 原文用的电芯**，不是本文的 McMaster 18650PF。
    #    本文数据集是 NCA；此处「NCM」若指的是 Yuan 的电池则**保持原样**，
    #    但**未核实过**——投稿前需按 Yuan 原文确认，避免与本文体系混淆。
    #   [How19]   = How et al. 2019 IEEE Access 综述（多数据集汇总典型值）
    #   [Vidal20] = Vidal et al. 2020 IEEE Access（ML 综述典型值）
    #   [Rzepka21]= Rzepka et al. 2021 Energies（18650 LCO，EKF 实测）
    #   [典型值]  = 领域内多篇论文的典型精度范围

    records = [
        # ---- 纯模型驱动 ----
        {
            "method": "Coulomb Counting",
            "category": "Model-based",
            "RMSE_pp": "~0.01–0.6",
            "MAE_pp": "~0.01–0.5",
            "condition": "理想条件（无初始偏差、无噪声）",
            "source": "[our] ideal lower bound",
            "note": "实际不可用（需精确初始 SOC + 无传感器误差）",
        },
        {
            "method": "EKF (1RC-ECM)",
            "category": "Model-based",
            "RMSE_pp": "2.73",
            "MAE_pp": "2.21",
            "condition": "25°C, test_id, 正确初始化",
            "source": "[our] P5-01 M2",
            "note": "McMaster 18650PF, 稳态口径",
        },
        {
            "method": "EKF (1RC-ECM)",
            "category": "Model-based",
            "RMSE_pp": "4.84–23.59",
            "MAE_pp": "2.91–23.15",
            "condition": "OOD 10/0/−10/−20°C",
            "source": "[our] P5-01 M2",
            "note": "低温+高动态工况退化严重",
        },
        {
            "method": "UKF",
            "category": "Model-based",
            "RMSE_pp": "~2.5–4.0",
            "MAE_pp": "~2.0–3.5",
            "condition": "25°C, 动态工况",
            "source": "[How19] 典型值",
            "note": "比 EKF 略好（非线性处理更优），但同量级",
        },
        {
            "method": "PF (粒子滤波)",
            "category": "Model-based",
            "RMSE_pp": "~3.0–5.0",
            "MAE_pp": "~2.5–4.0",
            "condition": "25°C, 动态工况",
            "source": "[How19] 典型值",
            "note": "计算量大，精度不一定优于 EKF/UKF",
        },
        # ---- 纯数据驱动 ----
        {
            "method": "LSTM (纯 NN)",
            "category": "Data-driven",
            "RMSE_pp": "~1.0–3.0",
            "MAE_pp": "~0.8–2.5",
            "condition": "同分布 25°C",
            "source": "[Vidal20] 典型值",
            "note": "OOD 泛化差（见下方 Pure NN OOD）",
        },
        {
            "method": "GRU (纯 NN)",
            "category": "Data-driven",
            "RMSE_pp": "~1.0–2.5",
            "MAE_pp": "~0.8–2.0",
            "condition": "同分布 25°C",
            "source": "[Vidal20] 典型值",
            "note": "与 LSTM 同量级，参数更少",
        },
        {
            "method": "Pure NN (本项目)",
            "category": "Data-driven",
            "RMSE_pp": "1.23",
            "MAE_pp": "0.98",
            "condition": "25°C test_id",
            "source": "[our] P5-01 M5",
            "note": "GRU16, 5 文件训练, 直接回归 SOC",
        },
        {
            "method": "Pure NN OOD (本项目)",
            "category": "Data-driven",
            "RMSE_pp": "16.1–37.8",
            "MAE_pp": "12.7–32.4",
            "condition": "OOD 0/−20°C",
            "source": "[our] P5-01 M5",
            "note": "OOD 灾难性失败 → 混合架构必要性",
        },
        # ---- 混合方法（EKF/UKF + NN）----
        {
            "method": "KF + ML (Yuan 2023)",
            "category": "Hybrid",
            "RMSE_pp": "~0.8–1.5",
            "MAE_pp": "~0.6–1.2",
            "condition": "25°C, UDDS/HWFET",
            "source": "[Yuan23] Table 1",
            "note": "NN 回归 SOC（非学残差），25°C 单温度",
        },
        {
            "method": "EKF + NN A2-0 (本项目)",
            "category": "Hybrid",
            "RMSE_pp": "0.63",
            "MAE_pp": "0.49",
            "condition": "25°C test_id",
            "source": "[our] P5-01 M3",
            "note": "仅测量量，无 EKF 内部分量",
        },
        {
            "method": "EKF + NN A2-3 (本项目, 主模型)",
            "category": "Hybrid",
            "RMSE_pp": "0.60",
            "MAE_pp": "0.45",
            "condition": "25°C test_id",
            "source": "[our] P5-01 M4",
            "note": "完整特征（ν+NIS+K+diag(P)），最优",
        },
        {
            "method": "EKF + NN A2-3 OOD (本项目)",
            "category": "Hybrid",
            "RMSE_pp": "4.51–21.65",
            "MAE_pp": "3.62–21.36",
            "condition": "OOD 10/0/−10/−20°C",
            "source": "[our] P5-01 M4",
            "note": "跨温度鲁棒性（A2-3 全场最优）",
        },
        # ---- MCU 部署对比 ----
        {
            "method": "EKF on MCU (典型)",
            "category": "MCU-deployed",
            "RMSE_pp": "~3–8",
            "MAE_pp": "~2–6",
            "condition": "STM32F1/F4, 25°C",
            "source": "[How19][典型值]",
            "note": "纯 EKF 部署，无 NN 修正",
        },
        {
            "method": "NN on MCU (典型)",
            "category": "MCU-deployed",
            "RMSE_pp": "~2–5",
            "MAE_pp": "~1.5–4",
            "condition": "STM32F4, int8, 25°C",
            "source": "[典型值]",
            "note": "量化后精度有损，资源报告不完整",
        },
        {
            "method": "EKF+NN on MCU (本项目, 待验证)",
            "category": "MCU-deployed",
            "RMSE_pp": "待测",
            "MAE_pp": "待测",
            "condition": "STM32F407, float16, 25°C",
            "source": "[our] P4 待做",
            "note": "float16 TFLite, ~105 KB, 1.8k 参数",
        },
    ]

    df = pd.DataFrame(records)

    # 保存
    csv_path = os.path.join(RESULTS, "p5_05_literature_comparison.csv")
    df.to_csv(csv_path, index=False)

    # 打印
    print("=" * 90)
    print("P5-05 文献对比表：SOC 估算方法精度对比")
    print("=" * 90)
    print(f"{'方法':<35s} {'类别':<15s} {'RMSE (pp)':<15s} {'条件':<25s} {'来源'}")
    print("-" * 90)
    for _, r in df.iterrows():
        print(f"{r['method']:<35s} {r['category']:<15s} "
              f"{r['RMSE_pp']:<15s} {r['condition']:<25s} {r['source']}")

    # 分类汇总
    print("\n" + "=" * 90)
    print("方法类别对比（25°C 同分布）")
    print("=" * 90)
    print("  纯模型驱动 (EKF):        RMSE ~2.73 pp")
    print("  纯数据驱动 (Pure NN):     RMSE ~1.23 pp")
    print("  混合 EKF+NN (本项目):     RMSE ~0.60 pp  ← 最优")
    print("  混合 KF+ML (Yuan 2023):   RMSE ~0.8–1.5 pp")
    print()
    print("  相对 EKF 改善:  ↓78% (本项目) vs ↓44–75% (Yuan 2023)")
    print("  OOD 鲁棒性:     Pure NN 灾难性(16–38pp) vs 混合(4.5–22pp)")
    print()
    print("  本项目差异化:")
    print("    1. 使用 EKF 内部分量 (ν/K/P) 作 NN 特征（Yuan 2023 未做）")
    print("    2. 跨温度 OOD 系统评估（Yuan 2023 仅 25°C）")
    print("    3. MCU 端资源报告 + 功耗实测（领域空白）")
    print("    4. float16 量化方案（int8 在小模型上失败的解决方案）")

    print(f"\n保存: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
