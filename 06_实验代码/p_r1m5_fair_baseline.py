# -*- coding: utf-8 -*-
"""公平调参基线（回应意见8 §三.4：事后调优 EKF 是"稻草人"）。

问题
----
原稿的"事后调优固定 EKF"（`p_r1m4_adaptive.py`）在**测试集**上跑 9 点 Q/R 网格，
再**按分布外温度**选出网格内最优点 —— 用测试信息调参，是**乐观上界**，
不是可部署基线。以它得出"融合不占优"，有稻草人风险。

公平做法
--------
- **调参池**：`split == train | val` 的 6 个 25 °C 文件（**不含任何测试文件**）
- **评测池**：`test_ood_trise` 的 9 个温度斜坡文件（10 °C 与 −20 °C 漂移工况）
- 在调参池上按**池内平均 RMSE 最小**选一组 (R, q_soc)
- 用该组在评测池上评一次 —— **测试信息零参与**

产物
----
- results/r1m5_fair_baseline.csv    逐文件 RMSE（公平基线 + 默认基线 + 网格）
- results/r1m5_fair_baseline_summary.md
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(__ROOT__, "06_实验代码", "src"))
sys.path.insert(0, os.path.join(__ROOT__, "10_代码"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ecm_ekf import ECMParams, SOC_EKF  # noqa: E402
from ecm_tables import OCVTable, ECMParamTable, make_ekf_table_fns  # noqa: E402
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PROJ = __ROOT__
DATA = os.path.join(PROJ, "04_实验", "数据", "processed")
RES = os.path.join(PROJ, "04_实验", "数据", "results")
BURN = 300
SOC0_BIAS = 0.0          # 主实验为正确初始化
GRID = [(r, q) for r in (1e-5, 1e-4, 1e-3) for q in (1e-8, 1e-9, 1e-10)]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="trise",
                    choices=["trise", "test_ood", "test_id"],
                    help="评测池：trise(9)/test_ood(36)/test_id(4,25°C)")
    ap.add_argument("--suffix", default="",
                    help="产物文件名后缀，避免覆盖")
    args = ap.parse_args()
    EVAL_SPLIT = {"test_ood": "test_ood", "test_id": "test_id"}.get(
        args.eval_set, "test_ood_trise")
    SFX = args.suffix or ("" if args.eval_set == "trise" else f"_{args.eval_set}")

    ocv_fn, docv_fn, params_fn = make_ekf_table_fns(
        OCVTable.from_csv(), ECMParamTable.from_csv())
    man = pd.read_csv(os.path.join(PROJ, "04_实验", "数据", "splits",
                                   "split_manifest.csv"))
    d = pd.read_csv(os.path.join(DATA, "mcmaster_drive_cycles.csv.gz"))

    tune_ids = set(man[man.split.isin(["train", "val"])].cycle_id)
    eval_ids = set(man[man.split == EVAL_SPLIT].cycle_id)
    print(f"调参池 {len(tune_ids)} 文件 / 评测池 {len(eval_ids)} 文件 "
          f"(split={EVAL_SPLIT})", flush=True)

    def run(i, v, t, s, **kw):
        e = SOC_EKF(ECMParams(Cn=2.9, dt=1.0),
                    soc0=float(np.clip(s[0] + SOC0_BIAS, 0.0, 1.0)),
                    ocv_fn=ocv_fn, docv_fn=docv_fn, params_fn=params_fn, **kw)
        for k in range(len(i)):
            e.step(float(i[k]), float(v[k]), temp_k=float(t[k]))
        err = (np.array(e.history["soc"])[BURN:] - s[BURN:]) * 100
        return float(np.sqrt(np.mean(err ** 2)))

    # ---- 逐文件落盘 + 断点续跑：每个 cycle_id 算完立刻写分片，重跑自动跳过 ----
    shard_dir = os.path.join(RES, f"_shards_r1m5{SFX}")
    os.makedirs(shard_dir, exist_ok=True)

    def shard_of(cid: str) -> str:
        safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in cid)
        return os.path.join(shard_dir, f"r1m5_{safe}.csv")

    tids = [c for c in d.cycle_id.unique() if c in tune_ids or c in eval_ids]
    tids.sort(key=lambda c: (0 if c in tune_ids else 1, c))
    for n, cid in enumerate(tids, 1):
        sp = shard_of(cid)
        if os.path.exists(sp):
            print(f"  [{n}/{len(tids)}] 已完成，跳过 {cid[:46]}", flush=True)
            continue
        g = d[d.cycle_id == cid]
        if len(g) < BURN + 100:
            continue
        i = g.current_A.to_numpy(); v = g.voltage_V.to_numpy()
        t = g.temperature_C.to_numpy(); s = g.soc_true.to_numpy()
        pool = "tune" if cid in tune_ids else "eval"
        rows = [{"cycle_id": cid, "pool": pool, "variant": "default",
                 "rmse": run(i, v, t, s)}]
        for r, q in GRID:
            rows.append({"cycle_id": cid, "pool": pool,
                         "variant": f"grid_r{r:.0e}_q{q:.0e}",
                         "rmse": run(i, v, t, s, r=r, q_soc=q)})
        # 立刻落盘 —— 此刻起即使进程被杀，这个文件也不会丢
        pd.DataFrame(rows).to_csv(sp, index=False, encoding="utf-8")
        print(f"  [{n}/{len(tids)}] [{'tune' if pool=='tune' else 'eval'}] "
              f"{cid[:44]} 已落盘", flush=True)

    parts = [pd.read_csv(shard_of(c)) for c in tids if os.path.exists(shard_of(c))]
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(os.path.join(RES, f"r1m5_fair_baseline{SFX}.csv"),
              index=False, encoding="utf-8")

    # 在**调参池**上选点（仅用 tune 数据）
    tune = df[df["pool"] == "tune"]
    means = tune.groupby("variant")["rmse"].mean().sort_values()
    best = means.index[0]
    print(f"\n调参池上最优变体: {best}  (均值 {means.iloc[0]:.4f} pp)", flush=True)
    print(means.round(4).to_string(), flush=True)

    # 在**评测池**上评
    ev = df[df["pool"] == "eval"]
    ev_means = ev.groupby("variant")["rmse"].mean().sort_values()
    base = ev_means.loc["default"]
    fair = ev_means.loc[best]

    L = ["# R1-M5 公平调参基线（不泄露测试信息）\n",
         f"- 调参池：train+val 共 {len(tune_ids)} 个 25 °C 文件（**不含任何测试文件**）",
         f"- 评测池：{EVAL_SPLIT} 共 {len(eval_ids)} 个文件"
         + ("（10 / −20 °C 温度斜坡）" if EVAL_SPLIT == "test_ood_trise"
            else "（10 / 0 / −10 / −20 °C 主 OOD 格）"),
         f"- 调参判据：调参池内 9 点网格的平均 RMSE 最小 ⇒ **{best}**",
         "",
         "## 评测池上的表现（pp）",
         "",
         "| 变体 | 评测池平均 RMSE |",
         "|---|---|"]
    for k in ["default", best]:
        L.append(f"| {'默认 (R=1e-4, q=1e-9)' if k=='default' else '公平调优（调参池选出）'} | {ev_means.loc[k]:.3f} |")
    L += ["",
          f"**默认 vs 公平调优**：{base:.3f} → {fair:.3f} pp",
          f"（调参是在**与测试无关**的 6 个 25 °C 文件上做的。）",
          "",
          "## 网格在评测池上的极差（参考）",
          "",
          "| 统计 | 值 |",
          "|---|---|",
          f"| 最小 | {ev_means.min():.3f} |",
          f"| 最大 | {ev_means.max():.3f} |",
          f"| 极差 | {ev_means.max()-ev_means.min():.3f} |"]

    out = os.path.join(RES, f"r1m5_fair_baseline{SFX}_summary.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\n产物: r1m5_fair_baseline{SFX}.csv / _summary.md")
    print(f"评测池: 默认 {base:.3f} → 公平调优 {fair:.3f} pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
