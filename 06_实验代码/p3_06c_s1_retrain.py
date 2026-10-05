# -*- coding: utf-8 -*-
"""A2：容量校正（S1）口径下**重训**残差网络，判定 RQ4「次序反转」是否仍成立。

背景（2026-10-03 三审）
----------------------
R2-M1（唯一 Blocking）/ R1-M3 / R3-M1 共同指出：
融合网络按**名义口径**残差目标 y_k 训练，换到容量校正口径**评分**时未重训，
故 0.62→3.17 pp 的「次序反转」混入了「训练目标与评分口径不一致」。

口径来源（已核验）
------------------
- `ah_throughput`（processed CSV）**能精确复现** npz 里的 soc_true（最大偏差 0.000000）
  ⇒ 权威来源是 processed CSV
- S1 容量：`cb2_scenarios.cn_of(cid, "S1", caps, t0,t1,c0,c1)`（按日期线性插值）

设计
----
- 训练/验证目标改为 S1：soc_true_S1 = clip(1 + ah_throughput / Cn_S1)
- 特征/窗口/网络/超参/种子 **完全不变**
- 重训后在 S1 口径下评分

产物（**不覆盖任何原模型**）
--------------------------
- models_s1/gru16_{g}_seed{s}.keras
- results/p3_06c_s1_retrain.csv
"""
from __future__ import annotations

import importlib.util as iu
import os
import sys
import time

sys.path.insert(0, os.path.join(__ROOT__, "06_实验代码", "src"))
sys.path.insert(0, os.path.join(__ROOT__, "06_实验代码"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tensorflow import keras  # noqa: E402
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import nn_features as NF  # noqa: E402
from nn_features import GROUPS, load_history  # noqa: E402
import p3_06_ablation as A  # noqa: E402

PROJ = os.path.join(__ROOT__, "SOC论文项目")
RESULTS = os.path.join(PROJ, "04_实验", "数据", "results")
MODELS_S1 = os.path.join(PROJ, "04_实验", "models_s1")
CN_NOMINAL = 2.9


def load_module(path: str, name: str):
    spec = iu.spec_from_file_location(name, path)
    m = iu.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


def main() -> int:
    t0 = time.time()
    os.makedirs(MODELS_S1, exist_ok=True)

    # ---------- 口径 ----------
    caps = load_module(os.path.join(PROJ, "10_代码", "p5_06_capacity_sensitivity.py"), "p506")
    cb2 = load_module(os.path.join(PROJ, "10_代码", "cb2_scenarios.py"), "cb2")
    meas = caps.measure_1c_capacities()
    tA, tB, cA, cB = caps.fit_capacity_curve(meas)
    print(f"S1 容量曲线: {tA.date()} {cA:.4f} Ah -> {tB.date()} {cB:.4f} Ah", flush=True)

    ah_by = {}
    d = pd.read_csv(os.path.join(PROJ, "04_实验", "数据", "processed",
                                 "mcmaster_drive_cycles.csv.gz"))
    for c, g in d.groupby("cycle_id"):
        ah_by[c] = g["ah_throughput"].values.astype(float)

    def soc_true_s1(cid: str, n: int) -> np.ndarray:
        cn = cb2.cn_of(cid, "S1", caps, tA, tB, cA, cB)
        return np.clip(1.0 + ah_by[cid][:n] / cn, 0.0, 1.0)

    # ---------- 数据（🔴 2026-10-03 修正：改用 **40 文件全量** OOD 库存）----------
    #   原实现走 A.load_all() → ekf_baseline_history.npz（**每 OOD 温度仅 1 文件**），
    #   导致本析出报告的低温三组只有 n=1，效果与 t 检验都不可靠。
    #   现改为 ekf_full_history.npz（25:4 / 10:9 / 0:9 / −10:9 / −20:9，共 40 文件），
    #   与表2/图6/图7 同一权威库存。
    train_hist = load_history(os.path.join(RESULTS, "ekf_train_history.npz"))
    test_hist = load_history(os.path.join(RESULTS, "ekf_full_history.npz"))
    val_ids = [c for c in train_hist if "UDDS" in c]
    tr_ids = [c for c in train_hist if c not in val_ids]

    def rescore(hist):
        out = {}
        for cid, h in hist.items():
            out[cid] = dict(h)
            out[cid]["soc_true"] = soc_true_s1(cid, len(h["soc"]))
        return out

    train_s1, test_s1 = rescore(train_hist), rescore(test_hist)

    # 🔴 2026-10-03：`A.test_groups` 的 `"0degC_Cycle"` 会**子串命中**
    #    10degC_Cycle / n10degC_Cycle / n20degC_Cycle（0 °C 组实测 16 文件）。
    #    此处**在本脚本内**修正为 exact 匹配（不改公共函数，避免影响其他管线）。
    #    照抄 `p5_b2_multifile_ablation.groups_of`（权威管线，已修子串陷阱）：
    #    0 °C 用 `"0degC_"` 全匹配（含 Cycle + UDDS/HWFET/US06/LA92/NN 共 9），
    #    并排除 `trise`/`10degC`/`20degC`。
    def test_groups_fixed(hist):
        ids = sorted(hist.keys())
        return {
            "test_id(25C)": [c for c in ids if "25degC_Cycle" in c],
            "ood(10C)":   [c for c in ids if "10degC_" in c and "trise" not in c
                           and "n10degC" not in c and "n20degC" not in c],
            "ood(0C)":    [c for c in ids if "0degC_" in c and "trise" not in c
                           and "10degC" not in c and "20degC" not in c],
            "ood(-10C)":  [c for c in ids if "n10degC_" in c and "trise" not in c],
            "ood(-20C)":  [c for c in ids if "n20degC_" in c and "trise" not in c],
        }

    groups = test_groups_fixed(test_hist)
    groups_eval = {"val(25C)": (train_s1, val_ids)}
    for k, v in groups.items():
        groups_eval[k] = (test_s1, v)
    print(f"评测组文件数: { {k: len(v[1]) for k, v in groups_eval.items()} }", flush=True)

    # ---------- 训练（S1 目标）----------
    # 🔴 2026-10-04：由 3 种子扩到 **10 种子**，与全文主口径统一
    #    （审稿人 R1-M6 指出：本条支撑 RQ4 结论，却只用了 3 种子，
    #     而全文反复示范 3 种子会出错）。
    #    已有 seed 7/13/42 的 S1 模型会被**自动复用**（见下方 os.path.exists 分支），
    #    只新训 7 个新种子 × 5 臂 = 35 个模型。
    SEEDS = [7, 13, 42, 101, 202, 303, 404, 505, 606, 707]
    # 🔴 2026-10-03 改为**逐组分片 + 续跑**（本环境后台进程会被回收，A1 时实测两次）
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑这一组，如 A2-1")
    args = ap.parse_args()

    shard_dir = os.path.join(RESULTS, "_shards_s1")
    os.makedirs(shard_dir, exist_ok=True)
    def shard(g):
        return os.path.join(shard_dir, f"s1_{g}.csv")

    todo = [args.only] if args.only else sorted(GROUPS.keys())
    records = []
    for g in todo:
        if os.path.exists(shard(g)):
            print(f"[skip] {g}", flush=True)
            records.extend(pd.read_csv(shard(g)).to_dict("records"))
            continue
        f = GROUPS[g]
        Xtr, ytr, _ = NF.build_split_arrays(train_s1, tr_ids, f)
        Xva, yva, _ = NF.build_split_arrays(train_s1, val_ids, f)
        for seed in SEEDS:
            mp = os.path.join(MODELS_S1, f"gru16_{g}_seed{seed}.keras")
            if os.path.exists(mp):
                m = keras.models.load_model(mp)
                print(f"  复用 {g} seed{seed}", flush=True)
            else:
                m = A.build_model(Xtr.shape[-1], seed=seed)
                t1 = time.time()
                m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=200, batch_size=256,
                      verbose=0, callbacks=[
                          keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                                        restore_best_weights=True),
                          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                            patience=8, min_lr=1e-5)])
                m.save(mp)
                print(f"  重训 {g} seed{seed}  {time.time()-t1:.1f}s", flush=True)

            for gk, (hist, cids) in groups_eval.items():
                for r in A.eval_one(m, hist, cids, GROUPS[g]):
                    r.update({"exp": g, "seed": seed, "eval_group": gk, "ref": "S1"})
                    records.append(r)

    # 逐组落盘（跑完一组写一组，防半途被杀全丢）
    for g in todo:
        gg = [r for r in records if r.get("exp") == g]
        if gg and not os.path.exists(shard(g)):
            pd.DataFrame(gg).to_csv(shard(g), index=False)

    parts = [pd.read_csv(shard(g)) for g in sorted(GROUPS.keys()) if os.path.exists(shard(g))]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(records)
    cols = ["exp", "seed", "eval_group", "cycle_id", "n",
            "ekf_RMSE", "nn_RMSE", "ekf_MAE", "nn_MAE", "ekf_MAX", "nn_MAX", "ref"]
    out = os.path.join(RESULTS, "p3_06c_s1_retrain.csv")
    df[cols].to_csv(out, index=False, encoding="utf-8")
    print(f"已完成组: {sorted(df['exp'].unique())}")
    print(f"\n完成：{len(df)} 行 -> {os.path.basename(out)}  耗时 {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
