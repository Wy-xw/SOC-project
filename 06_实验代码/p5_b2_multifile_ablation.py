# -*- coding: utf-8 -*-
"""
B2 补实验：全量 OOD 多文件消融 + 配对统计
- 40 文件（test_id 4 + OOD 36）× 5 特征组 × 3 seeds
- 输出：
  results/p5_b2_perfile.csv        逐 (group, seed, file) RMSE
  results/p3_06b_multifile_agg.csv 组×温度 聚合（文件间 SD + n_files + EKF 基线两列）
                                  ⚠️ 列名是下游契约：p5_06_figures.py / p6_tables.py
                                     读 `grp` 与 `ekf_RMSE_mean`，改动需同步改消费者
  results/p5_b2_paired_summary.md  配对 δ 统计（A2-3−A2-0 等）+ 方向计数
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "06_实验代码", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from nn_features import GROUPS, build_split_arrays, load_history  # noqa

# Windows 控制台默认 GBK，而本脚本 print 含 GBK 无码位的字符 —— 不改编码会在
# **成功路径**崩掉（活干完了却退出码 1）。别靠 PYTHONIOENCODING 兜底。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "04_实验", "数据", "results")
MODELS = os.path.join(PROJ, "04_实验", "models")
#: 2026-10-03 扩种子（R2-M3/R3-M3 要求给符号一致率一个零分布）。
#  用户拍板 10 种子。检查点复用 04_实验/models/gru16_{g}_seed{s}.keras。
SEEDS = [7, 13, 42, 101, 202, 303, 404, 505, 606, 707]
EXPS = ["A2-0", "A2-1", "A2-2", "A2-3", "A2-4"]


def groups_of(hist):
    ids = sorted(hist)
    g = {"test_id(25C)": [c for c in ids if "25degC_Cycle" in c]}
    g["ood(10C)"] = [c for c in ids if "10degC_" in c and "trise" not in c
                     and "n10degC" not in c and "n20degC" not in c]
    # ⚠️ 2026-09-25 修复："0degC_" 是 "10degC_" / "n10degC_" / "n20degC_" 的子串，
    #    原先这一行会把全部 36 个 OOD 文件都划进 0 °C 组（正确应为 9 个）。
    #    必须显式排除正/负 10 与 20 °C。同仓另有 6 处正确写法可参照
    #    （如 p5_01_baseline_compare.py:71-90，其注释记载该陷阱已于 2026-09-23 修复）。
    g["ood(0C)"] = [c for c in ids if "0degC_" in c and "trise" not in c
                    and "10degC" not in c and "20degC" not in c]
    g["ood(-10C)"] = [c for c in ids if "n10degC_" in c and "trise" not in c]
    g["ood(-20C)"] = [c for c in ids if "n20degC_" in c and "trise" not in c]
    return g


def main():
    from tensorflow import keras
    hist = load_history(os.path.join(RES, "ekf_full_history.npz"))
    grp = groups_of(hist)
    print({k: len(v) for k, v in grp.items()}, flush=True)

    # 预建每文件特征数组（各组一次）
    cache = {}
    for g in EXPS:
        for cid in hist:
            out = build_split_arrays(hist, [cid], GROUPS[g])
            cache[(g, cid)] = out

    # ── 🔴 2026-10-03 改为**分片续跑** ──
    #    背景：本环境下长驻后台进程会被**静默回收**（实测两次：跑到一半进程消失、
    #    无 Traceback、无退出码）。故改为**每算完一个组就落盘**，
    #    重跑时自动跳过已完成的组 —— 被杀也只丢当前组。
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只算这个组（如 A2-1）；空=按需补全")
    args = ap.parse_args()

    shard_dir = os.path.join(RES, "_shards_seed10")
    os.makedirs(shard_dir, exist_ok=True)

    def shard_path(g):
        return os.path.join(shard_dir, f"b2_seed10_{g}.csv")

    todo = [args.only] if args.only else list(EXPS)
    for g in todo:
        need = [os.path.join(shard_dir, f"b2_seed10_{g}__{ev}.csv") for ev in grp]
        if all(os.path.exists(x) for x in need):
            print(f"[skip] {g} 已完成", flush=True)
            continue
        models = {}
        for s in SEEDS:
            mp = os.path.join(MODELS, f"gru16_{g}_seed{s}.keras")
            models[s] = keras.models.load_model(mp, compile=False)
        for ev, cids in grp.items():
            evp0 = os.path.join(shard_dir, f"b2_seed10_{g}__{ev}.csv")
            if os.path.exists(evp0):
                print(f"[skip] {g} {ev}", flush=True)
                continue
            rows_ev = []
            for cid in cids:
                X, y, _ = cache[(g, cid)]
                if X is None:
                    continue
                for s in SEEDS:
                    yh = models[s].predict(X, verbose=0).ravel()
                    e = y - yh
                    rows_ev.append({"exp": g, "seed": s, "eval_group": ev,
                                    "cycle_id": cid,
                                    "RMSE": float(np.sqrt(np.mean(e ** 2)))})
            # 🔴 每温度组立即落盘（防进程被杀丢整组）
            evp = os.path.join(shard_dir, f"b2_seed10_{g}__{ev}.csv")
            pd.DataFrame(rows_ev).to_csv(evp, index=False)
            print(f"[saved] {g} {ev}", flush=True)

    # 汇总所有分片
    parts = []
    for g in EXPS:
        for ev in grp:
            fp = os.path.join(shard_dir, f"b2_seed10_{g}__{ev}.csv")
            if os.path.exists(fp):
                parts.append(pd.read_csv(fp))
    if not parts:
        print("⚠️ 无分片可汇总"); return 1
    df = pd.concat(parts, ignore_index=True)
    if len(df) < len(EXPS) * 40 * len(SEEDS):
        print(f"⚠️ 仅 {df['exp'].nunique()}/{len(EXPS)} 组完成，汇总不完整（继续跑补齐）", flush=True)
    df.to_csv(os.path.join(RES, "p5_b2_perfile_seed10.csv"), index=False)

    # 聚合：先按 (group,seed,temp) 对文件取均值，再跨 seed 汇总
    #
    # ⚠️ 2026-09-26 修复（产物↔脚本漂移）：p3_06b_multifile_agg.csv 的列名是**下游契约**——
    #    p5_06_figures.py:200 与 p6_tables.py:29 读的是 `grp` 与 `ekf_RMSE_mean`。
    #    此前脚本被改写成输出 `eval_group` 且不再输出 EKF 两列，而磁盘产物仍是旧 schema，
    #    于是"产物与脚本不一致、重跑即 KeyError"。现按产物既有 schema 复原输出：
    #    exp / grp / RMSE_mean / RMSE_std_files / n_files / ekf_RMSE_mean / ekf_RMSE_std
    #    （已逐值比对过：与本脚本复原前磁盘上的产物最大差 3.6e-15，纯浮点噪声。）
    per_seed = df.groupby(["exp", "seed", "eval_group"]).agg(
        rmse_mean=("RMSE", "mean"), n_files=("RMSE", "count")).reset_index()
    agg = per_seed.groupby(["exp", "eval_group"]).agg(
        RMSE_mean=("rmse_mean", "mean"),
        n_files=("n_files", "first")).reset_index()
    # 文件间 SD（每文件先跨 seed 平均）
    per_file = df.groupby(["exp", "eval_group", "cycle_id"]).RMSE.mean().reset_index()
    fstd = per_file.groupby(["exp", "eval_group"]).RMSE.std().rename(
        "RMSE_std_files").reset_index()
    agg = agg.merge(fstd, on=["exp", "eval_group"], how="left")

    # EKF 基线两列：取自 r1m4_perfile.csv 的 EKF_fixed（组均值 + 文件间 SD）。
    # EKF 不随特征组变化，故每个 grp 只有一个值。缺文件必须硬失败，不能静默出 NaN
    # ——静默 NaN 会让下游 p6_tables.py 把 EKF 基线列写成空值而无人察觉。
    r1m4_path = os.path.join(RES, "r1m4_perfile.csv")
    if not os.path.exists(r1m4_path):
        raise FileNotFoundError(
            f"缺少 {r1m4_path}：它提供本产物的 ekf_RMSE_mean / ekf_RMSE_std 两列。"
            f"请先运行 p_r1m4_adaptive.py。")
    r1m4 = pd.read_csv(r1m4_path, encoding="utf-8-sig")
    ekf = (r1m4[r1m4["variant"] == "EKF_fixed"]
           .groupby("grp")["rmse"]
           .agg(ekf_RMSE_mean="mean", ekf_RMSE_std="std").reset_index())
    agg = agg.merge(ekf, left_on="eval_group", right_on="grp", how="left")
    # merge 会同时带进 ekf 的 `grp` 列；必须先丢掉，否则下面 rename
    # eval_group→grp 会产生两个同名列（pandas 自动改名 grp.1），污染产物 schema
    agg = agg.drop(columns=["grp"])
    if agg["ekf_RMSE_mean"].isna().any():
        bad = agg.loc[agg["ekf_RMSE_mean"].isna(), "eval_group"].unique().tolist()
        raise ValueError(
            f"r1m4_perfile.csv 里没有这些组的 EKF_fixed 值：{bad}；"
            f"组名口径可能已漂移。")
    agg = agg.rename(columns={"eval_group": "grp"})[
        ["exp", "grp", "RMSE_mean", "RMSE_std_files", "n_files",
         "ekf_RMSE_mean", "ekf_RMSE_std"]]
    agg.to_csv(os.path.join(RES, "p3_06b_multifile_agg_seed10.csv"), index=False)

    # 配对分析
    wide = per_file.pivot_table(index=["eval_group", "cycle_id"],
                                columns="exp", values="RMSE").reset_index()
    pairs = [("A2-3", "A2-0", "A2-3 − A2-0 (全诊断 vs 恒等映射对照)"),
             ("A2-3", "A2-2", "A2-3 − A2-2 (K/P 增量)"),
             ("A2-1", "A2-0", "A2-1 − A2-0 (ν 单独)")]
    from scipy import stats as st
    lines = ["# B2 配对统计（全量 OOD，多文件）\n",
             f"文件池：test_id {len(grp['test_id(25C)'])}，OOD 每温度 "
             f"{len(grp['ood(0C)'])}；每文件先跨 3 seeds 平均，再做文件级配对。\n"]
    for hi, lo, label in pairs:
        lines.append(f"\n## {label}\n")
        lines.append("| 评估组 | n | 平均 δ (pp) | 95% CI | 改善文件数 |")
        lines.append("|---|---|---|---|---|")
        for ev in ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)",
                   "ood(-20C)"]:
            d = wide[wide.eval_group == ev]
            if not len(d):
                continue
            delta = (d[hi] - d[lo]).dropna()
            n = len(delta)
            m = delta.mean()
            se = delta.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            ci = st.t.ppf(0.975, n - 1) * se if n > 1 else np.nan
            lines.append(f"| {ev} | {n} | {m:+.3f} | "
                         f"[{m-ci:+.3f}, {m+ci:+.3f}] | "
                         f"{(delta < 0).sum()}/{n} |")
    with open(os.path.join(RES, "p5_b2_paired_summary_seed10.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("saved: p5_b2_perfile.csv / p3_06b_multifile_agg.csv / "
          "p5_b2_paired_summary.md")


if __name__ == "__main__":
    main()
