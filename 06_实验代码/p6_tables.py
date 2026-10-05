# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# ⛔ 禁止运行（DO NOT RUN）——产物 05_论文/tables_v1.md 已手工修订
#
# 2026-09-25 的三处修正已直接写入 tables_v1.md，本脚本未同步：
#   (1) OOD 单元格 n=1 口径；(2) 参数量 1.6 k；(3) 鲁棒性行按新协议更新。
# 重跑本脚本会把上述 3 处修正退回旧版（静默覆盖）。如需重跑，先同步本
# 脚本再执行，并 diff tables_v1.md 确认无回退。
# ═══════════════════════════════════════════════════════════════════════════
"""P6 图件补齐（三）：Table S1 + Table 1–2 组装 → 05_论文/tables_v1.md

⚠️ 2026-09-26：表号已重编。原 Table 2/3 → **Table 1/2**（原 Table 1 随硬件表删除
已成空号）；**原 Table 4（资源与文献对比）已删除**，本脚本改为只输出其**墓碑**以与
稿件一致。**本脚本会覆盖 `05_论文/tables_v1.md` —— 改动前先跑 `_final_check.py` 对比。**
"""
import os
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "04_实验", "数据", "results")
OUT = os.path.join(PROJ, "05_论文", "tables_v1.md")

METHODS = [("M1_Coulomb", "Coulomb (ideal init)"), ("M2_EKF", "EKF"),
           ("M3_A2-0", "EKF+NN A2-0"), ("M4_A2-3", "EKF+NN A2-3 (ours)"),
           ("M5_pureNN", "Pure NN"), ("M6_A2-4", "EKF+NN A2-4")]
CONDS = ["Cycle_25C", "UDDS_10C", "HWFET_10C", "US06_10C",
         "UDDS_0C", "HWFET_0C", "US06_0C",
         "UDDS_-10C", "HWFET_-10C", "US06_-10C",
         "UDDS_-20C", "HWFET_-20C", "US06_-20C"]
EVAL = ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]
EXPS = ["A2-0", "A2-1", "A2-2", "A2-3", "A2-4"]

t2 = pd.read_csv(os.path.join(RES, "p5_01_baseline_comparison.csv"))
t3 = pd.read_csv(os.path.join(RES, "p3_06b_multifile_agg.csv"))
try:
    aekf_cells = pd.read_csv(os.path.join(RES, "r1m4_aekf_cells.csv"),
                             index_col=0)["rmse"].to_dict()
except Exception:
    aekf_cells = {}

lines = []
L = lines.append
L("# Tables 1–3 (v1, 2026-09-22)\n")
L("> 数字口径：RMSE 单位 = 百分点 (pp)，稳态段（前 300 s 剔除）。")
L("> T1 = seed-7 同族模型（Fig. 6 同源）；T2 = 3 seeds 均值 ± SD（Fig. 7 同源）。\n")

# ---------- Table S1 (Supplementary) ----------
# ⚠️ 2026-09-26 同步记录：本脚本是**表格生成器**，会覆盖 `05_论文/tables_v1.md`。
#    此前它停留在旧状态，重跑一次就会把当天的更正全部冲掉（已修）：
#    ① 已删去 **MCU / Sensing / Cell (deployment)** 三行（随"论文不走 MCU 方向"删除，
#       现表为纯数据集概况表）—— 本处已删；
#    ② 已删去 **NCM** 行（器件为 18650PF，所在行随 ① 一并移除）；
#    ③ 已把图号 Fig.7/Fig.8 改为 **Fig. 6 / Fig. 7**（重编号后）；—— 本处在 :40
#    ④ 已把表号 **Table 2/3 → Table 1/2**、**T2/T3 → T1/T2**（见下方各节）；
#    ⑤ **原 Table 4 已整节删除**，本脚本改为只输出墓碑（见文件末）。
L("## Table S1 (Supplementary). Datasets\n")
L("| Item | Specification |")
L("|---|---|")
L("| Primary dataset | McMaster 18650PF (Kollmeyer), DOI 10.17632/wykht8y7tg.1 "
  "|")
L("| Conditions | UDDS/HWFET/US06/LA92/NN/Cycle; 25/10/0/−10/−20 °C; 1 Hz |")
L("| Splits | train 5 (25 °C), val 1, test_id 4, OOD 36, trise 9 |")
L("| Cross-dataset | NASA PCoE (LiCoO₂ 2.0 Ah), zero-shot, limitation only |\n")

# ---------- Table 1（原 Table 2，2026-09-26 重编号）----------
L("## Table 1. Baseline comparison (RMSE, pp; seed-7 models) + adaptive "
  "comparator (R1-M4)\n")
hdr = ("| Condition | " + " | ".join(n for _, n in METHODS)
       + " | AEKF-R (adaptive R) |")
L(hdr)
L("|" + "---|" * (len(METHODS) + 2))
for cond in CONDS:
    cells = []
    for m, _ in METHODS:
        r = t2[(t2["method"] == m) & (t2["eval_group"] == cond)]
        cells.append(f"{r['RMSE'].values[0]:.2f}" if len(r) else "–")
    a = f"{aekf_cells.get(cond, float('nan')):.2f}" if cond in aekf_cells \
        else "–"
    L(f"| {cond} | " + " | ".join(cells) + f" | {a} |")
L("")
L("> AEKF-R：新息协方差匹配自适应 R（窗口 60，α=0.2），与其余方法同为**正确初始化**"
  "（脚本设 +10 pp 偏置，但所有文件满电起始、`clip` 恒取 1.0，该偏置从未生效）。"
  "其相对固定 EKF 的配对增益在 0 °C "
  "（+0.82 pp）与 −20 °C（+6.46 pp）上 95% CI 排除零；同分布不显著。\n")
L("> 注（seed 与精度）：本表为 **seed-7 单 seed**；Cycle_25C 行精确值 "
  "A2-0 = 0.596 pp、A2-3 = 0.605 pp（按2位显示故同为 0.60，实际差 "
  "0.009 pp）。三 seed 聚合见 Table 2。")
L("")

# ---------- Table 2 ----------
L("## Table 2. Feature ablation over the full OOD inventory (RMSE, pp; "
  "per-file errors first averaged over 3 seeds, spread across files)\n")
L("| Eval group | n | EKF | " + " | ".join(EXPS) + " |")
L("|" + "---|" * (len(EXPS) + 3))
for ev in EVAL:
    row = t3[t3["grp"] == ev]
    ekf = row["ekf_RMSE_mean"].values[0] if len(row) else float("nan")
    n = row["n_files"].values[0] if len(row) else 0
    cells = []
    for e in EXPS:
        r = row[row["exp"] == e]
        if len(r):
            m = r["RMSE_mean"].values[0]
            s = r["RMSE_std_files"].values[0]
            cells.append("—" if pd.isna(m) else f"{m:.3f} ± {s:.3f}")
        else:
            cells.append("—")
    L(f"| {ev} | {n} | {ekf:.3f} | " + " | ".join(cells) + " |")
L("")
L("> 注：全量 OOD（每温度 9 文件；test_id 4 文件）；误差棒 = **文件间 SD**；")
L("> 配对统计（A2-3−A2-0 等，含 95% CI）见 §6.3 与源数据。\n")

# ---------- Table 4（已删除，只留墓碑）----------
# ⚠️ 2026-09-26 修复：2026-09-25 用户决定论文不走 MCU / 板级部署方向，原「资源与文献
#    对比」表整节删除。但本生成器**仍在完整重出它**（含 STM32F407ZGT6 与 ⏳ P4 占位）
#    —— 重跑一次就会把已删表"复活"。现改为输出与稿件 `tables_v1.md` 一致的**墓碑**：
#    保留删除声明 + 原表留档（路径保留、实测资源列留档为占位）。
L("## ~~Table 4. Resource and literature comparison (★ 差异化表)~~ ❌ **已删除（2026-09-25）**\n")
L("> 🔴 **删除原因**：用户决定论文不走 MCU/板级部署方向。本表以 flash/RAM/latency/power")
L("> 四指标对比为核心，其中三列（RAM/latency/power）本就为 `⏳ P4` 占位、**无任何数据**。")
L("> 删除**不涉及任何数据撤回**。")
L(">")
L("> **保留的部分**：本表的**精度列**（RMSE）信息已在 **Table 1/Table 2** 中；")
L("> 模型的**体积**（float16 105 kB、int8 205 kB）已在 **§4.5** 报告。")
L("> Naguib / Yuan / Cheng / Batool 的文献对比需求随之取消（P0-11 不再是本文阻塞项）。")
L(">")
L("> 原表内容留档如下（不含任何实测资源数据）：")
L(">")
L("> | Work | Method | Platform | RMSE (pp) | Flash | RAM | Latency | Power |")
L("> |---|---|---|---|---|---|---|---|")
L("> | ~~This work~~ | EKF+NN (float16) | ~~STM32F407ZGT6~~ | 0.62 / 0.61 "
  "| 105 kB | ⏳ | ⏳ | ⏳ |")
L("> | Naguib et al. 2022 | model-based family | automotive MCU | — "
  "| reported | reported | reported | not reported |")
L("> | Yuan et al. 2023 | KF + ML | PC | ~0.8–1.5 | n/a | n/a | n/a | n/a |")
L("")

io = open(OUT, "w", encoding="utf-8")
io.write("\n".join(lines))
io.close()
print("saved:", OUT)
print("T1 rows:", len(CONDS), "| T2 rows:", len(EVAL))
