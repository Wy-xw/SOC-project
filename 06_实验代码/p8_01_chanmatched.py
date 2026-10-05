# -*- coding: utf-8 -*-
"""P8-01 通道数配平对照（回应审稿人 R1-M2 / R2-M1，两位独立命中）。

问题
----
核心对比 A2-1 − A2-4 同时改变了**族身份**与**通道数**：
    A2-1 = 7 量测 + ν              =  8 通道
    A2-4 = 7 量测 + K₀,K₁,P₀₀,P₁₁  = 11 通道
故 Δ≠0 无法区分"两族功能不同"与"ν 一个通道不敌 K/P 四个通道"。

设计：把通道数**配平**，只改族身份
--------------------------------
两族内在成员数不同（新息/残差类 2 个：ν、NIS；不确定性/增益类 4 个：K₀,K₁,P₀₀,P₁₁），
故**无法配到 11 通道**；可配的只有：

    M1（8 通道）: A2-1 (7+ν)      vs  A2-K0 (7+K₀)        ← 各加 1 个
    M2（9 通道）: A2-2 (7+ν+NIS)  vs  A2-KK (7+K₀+K₁)     ← 各加 2 个

只新训右臂两条（A2-K0、A2-KK）；左臂 A2-1 / A2-2 复用既有
`p5_b2_perfile_seed10.csv` 的结果，不重训。

判据
----
若配平后 A2-1 − A2-K0 与 A2-2 − A2-KK **方向与量级保持**（比值仍 > 1），
则族主张成立（差异不能由通道数解释）；若差值大幅衰减或消失，
则标题/摘要须降为**配置级**表述（即 §6.1 已有的限定）。

用法
----
    python p8_01_chanmatched.py                 # 全量（2 臂 × 10 种子）
    python p8_01_chanmatched.py --seeds 7,13    # 探路
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "06_实验代码", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from nn_features import build_split_arrays, load_history  # noqa: E402

RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "04_实验", "模型")
OUT_CSV = os.path.join(RESULTS, "p8_01_chanmatched.csv")

EPOCHS, BATCH, PATIENCE = 200, 256, 20
SEEDS_ALL = [7, 13, 42, 101, 202, 303, 404, 505, 606, 707]

#: 7 个量测通道（与 A2-0 一致）
BASE = ["soc", "i", "iabs", "temp", "vt", "dvt", "dsoc"]

#: 本次新训的**右臂**（不确定性/增益类，通道数与同族的左臂配平）
#: 左臂 A2-1(8ch) / A2-2(9ch) 已在既有结果里，不重训。
NEW_GROUPS: dict[str, list[str]] = {
    "A2-K0": BASE + ["k0"],            # 8 通道：对 A2-1（7+ν）
    "A2-KK": BASE + ["k0", "k1"],      # 9 通道：对 A2-2（7+ν+nis）
}


#: 期望的测试库存（与主文表 2 / 图 6 同一权威库存）
#: ⚠️ 不能用 `ekf_baseline_history.npz` —— 那份**每 OOD 温度只有 1 个文件**，
#:    是 p3_06_ablation.py 时代的历史写法。本项目已因此出过一次错
#:    （见 `p3_06c_s1_retrain.py` 的修正注记、`p3_12_int8_quant.py` 的历史问题）。
EXPECT_TEST = {"test_id(25C)": 4, "ood(10C)": 9, "ood(0C)": 9,
               "ood(-10C)": 9, "ood(-20C)": 9}


def load_all():
    """训练历史用 train.npz，**测试用 full_history.npz（40 文件）**。"""
    train_hist = load_history(os.path.join(RESULTS, "ekf_train_history.npz"))
    test_hist = load_history(os.path.join(RESULTS, "ekf_full_history.npz"))
    val_ids = [c for c in train_hist if "UDDS" in c]
    tr_ids = [c for c in train_hist if c not in val_ids]
    return train_hist, test_hist, tr_ids, val_ids


def test_groups(test_hist):
    """评测分组 —— **逐字照抄主文权威实现** `p5_b2_multifile_ablation.py:groups_of`。

    ⚠️ 温度子串陷阱（本项目已踩过）：`"0degC_"` 是 `"10degC_"` / `"n10degC_"` /
       `"n20degC_"` 的**子串**。若只写 `"0degC_Cycle" in c`，会把 10/−10/−20 °C
       的 Cycle 文件全部划进 0 °C 组（实测 16 个，正确应为 9 个）。
       必须显式排除 `10degC` 与 `20degC`。见记忆 [[cohort-and-substring-traps]]。
    """
    ids = sorted(test_hist.keys())
    return {
        "test_id(25C)": [c for c in ids if "25degC_Cycle" in c],
        "ood(10C)":   [c for c in ids if "10degC_" in c and "trise" not in c
                       and "n10degC" not in c and "n20degC" not in c],
        "ood(0C)":    [c for c in ids if "0degC_" in c and "trise" not in c
                       and "10degC" not in c and "20degC" not in c],
        "ood(-10C)":  [c for c in ids if "n10degC_" in c and "trise" not in c],
        "ood(-20C)":  [c for c in ids if "n20degC_" in c and "trise" not in c],
    }


def build_model(n_feats, seed, window):
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(seed)
    inp = keras.Input(shape=(window, n_feats))
    x = keras.layers.GRU(16, unroll=True)(inp)
    x = keras.layers.Dense(8, activation="relu")(x)
    out = keras.layers.Dense(1)(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="逗号分隔；默认全部 10 个")
    args = ap.parse_args()
    seeds = ([int(s) for s in args.seeds.split(",")]
             if args.seeds else SEEDS_ALL)

    train_hist, test_hist, tr_ids, val_ids = load_all()
    groups = test_groups(test_hist)
    groups_eval = {"val(25C)": (train_hist, val_ids)}
    for k, v in groups.items():
        groups_eval[k] = (test_hist, v)

    # ⚠️ **库存自检**：不符即中止，防止误用"每温度 1 文件"的旧库存
    print(f"训练文件 {len(tr_ids)}，验证 {len(val_ids)}", flush=True)
    # ⚠️ 不用 emoji：Windows 默认 GBK 下 print emoji 会在**成功路径**抛
    #    UnicodeEncodeError（本项目已两次踩坑，见记忆 windows-gbk-stdout-trap）。
    bad = []
    for k, want in EXPECT_TEST.items():
        got = len(groups.get(k, []))
        if got != want:
            bad.append(f"{k}: 得 {got}，应 {want}")
        flag = "[OK]" if got == want else "[!!]"
        print(f"  {flag} {k}: {got} 文件（应 {want}）", flush=True)
    if bad:
        print("\n[!!] 测试库存与主文权威库存不符，终止：", flush=True)
        for b in bad:
            print("   -", b, flush=True)
        print("   请确认 using ekf_full_history.npz（40 文件）", flush=True)
        return 1

    # 特征缓存
    cache = {}
    for g, feats in NEW_GROUPS.items():
        Xtr, ytr, _ = build_split_arrays(train_hist, tr_ids, feats)
        Xva, yva, _ = build_split_arrays(train_hist, val_ids, feats)
        cache[g] = (Xtr, ytr, Xva, yva, feats)
        print(f"  {g}: {len(feats)} 通道  Xtr={Xtr.shape}", flush=True)

    from tensorflow import keras
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 分片续跑 ──────────────────────────────────────────────
    # 本环境后台长驻进程会被**静默回收**（`p5_b2_multifile_ablation.py` 记录实测两次：
    # 跑到一半进程消失、无 Traceback、无退出码）。故**每完成一个 (臂,种子) 即落盘**，
    # 重跑时自动跳过 —— 被杀只丢当前一个。
    shard_dir = os.path.join(RESULTS, "_shards_cm")
    os.makedirs(shard_dir, exist_ok=True)

    total = len(NEW_GROUPS) * len(seeds)
    n = 0
    t0 = time.time()

    for g in sorted(NEW_GROUPS.keys()):
        Xtr, ytr, Xva, yva, feats = cache[g]
        window = Xtr.shape[1]
        for seed in seeds:
            n += 1
            shard = os.path.join(shard_dir, f"{g}_seed{seed}.csv")
            if os.path.exists(shard):
                print(f"[{n}/{total}] {g} seed{seed} 已有分片，跳过", flush=True)
                continue
            mp = os.path.join(MODELS_DIR, f"gru16_cm_{g}_seed{seed}.keras")
            if os.path.exists(mp):
                model = keras.models.load_model(mp)
                tag = "载入缓存"
            else:
                model = build_model(Xtr.shape[2], seed, window)
                cbs = [
                    keras.callbacks.EarlyStopping(
                        patience=PATIENCE, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(
                        patience=8, factor=0.5, min_lr=1e-5),
                ]
                model.fit(Xtr, ytr, validation_data=(Xva, yva),
                          epochs=EPOCHS, batch_size=BATCH,
                          callbacks=cbs, verbose=0)
                model.save(mp)
                tag = "训练完成"
            rows = []
            for gname, (hist, cids) in groups_eval.items():
                for cid in cids:
                    if cid not in hist:
                        continue
                    X, y, _ = build_split_arrays(hist, [cid], feats)
                    if len(X) == 0:
                        continue
                    pred = model.predict(X, verbose=0).ravel()
                    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
                    rows.append(dict(exp=g, seed=seed, eval_group=gname,
                                     cycle_id=cid, RMSE=rmse))
            pd.DataFrame(rows).to_csv(shard, index=False)   # ← 逐个落盘
            el = time.time() - t0
            print(f"[{n}/{total}] {g} seed{seed} {tag}  "
                  f"{len(rows)} 行  已用 {el:.0f}s", flush=True)

    # ── 合并分片 ──
    import glob
    parts = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(shard_dir, "*.csv")))]
    if not parts:
        print("没有分片可合并", flush=True)
        return 1
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n已写出 {OUT_CSV}（{len(df)} 行 = "
          f"{df.exp.nunique()} 臂 × {df.seed.nunique()} 种子 × "
          f"{len(df)//max(1, df.exp.nunique()*df.seed.nunique())} 文件）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
