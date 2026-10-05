# -*- coding: utf-8 -*-
"""扩种子重跑消融 A2-0..A2-4。

背景（2026-10-03 三审结论）
--------------------------
R2-M3 / R3-M3 / 意见7 §3.1 共同指出：n=3 的「跨种子符号一致率」撑不起核心结论。
单模型训练实测 **36.5 s**（GRU 1 633 参数、CPU），扩种子成本可忽略。

设计
----
- 种子集 = 原 [7,13,42] + 新增 12 个（共 15）
- 每组×种子一个模型，**复用已有检查点**，缺的才训
- 输出 `p3_06_ablation_ext_full.csv`（与原 full.csv 同 schema，便于合并）
- 输出 `p3_06_ablation_ext_seedlist.txt`：**处置清单**（哪些复用、哪些新训）

安全
----
- 不覆盖任何已有产物（新文件名带 `_ext`）
- 不删任何文件
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, r"E:\SOC论文项目\06_实验代码\src")
sys.path.insert(0, r"E:\SOC论文项目\06_实验代码")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tensorflow import keras  # noqa: E402

from nn_features import GROUPS, build_split_arrays  # noqa: E402
import p3_06_ablation as A  # noqa: E402

RESULTS = r"E:\SOC论文项目\04_实验\数据\results"
MODELS = r"E:\SOC论文项目\04_实验\models"

#: 原 3 个种子 + 新增 7 个（共 10）。
# ⚠️ 2026-10-03：初版拟 15 种子，但**训练耗时按单次实测外推，估错 8 倍**
#    （实测 36.5s/模型是「21 epoch 早停」的幸运样本；实际种子普遍跑到 ~168 epoch
#      ⇒ 单模型约 293s）。15 种子要 4–5 h，10 种子约 2.5 h。
#    用户 2026-10-03 拍板：**10 种子**。
SEEDS_EXT = [7, 13, 42,
             101, 202, 303, 404, 505, 606, 707]
EPOCHS, BATCH, PATIENCE = 200, 256, 20


def callbacks():
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=8, min_lr=1e-5),
    ]


def main() -> int:
    t0 = time.time()
    train_hist, test_hist, tr_ids, val_ids = A.load_all()
    groups = A.test_groups(test_hist)
    groups_eval = {"val(25C)": (train_hist, val_ids)}
    for k, v in groups.items():
        groups_eval[k] = (test_hist, v)

    feats_cache = {}
    for g in GROUPS:
        f = GROUPS[g]
        Xtr, ytr, _ = build_split_arrays(train_hist, tr_ids, f)
        Xva, yva, _ = build_split_arrays(train_hist, val_ids, f)
        feats_cache[g] = (Xtr, ytr, Xva, yva)

    records, disposition, reused, trained = [], [], 0, 0

    for g in sorted(GROUPS.keys()):
        Xtr, ytr, Xva, yva = feats_cache[g]
        for seed in SEEDS_EXT:
            mp = os.path.join(MODELS, f"gru16_{g}_seed{seed}.keras")
            if os.path.exists(mp):
                m = keras.models.load_model(mp)
                reused += 1
                disposition.append(f"  复用 {g:6s} seed{seed:<5d} <- {os.path.basename(mp)}")
            else:
                m = A.build_model(Xtr.shape[-1], seed=seed)
                t1 = time.time()
                m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS,
                      batch_size=BATCH, verbose=0, callbacks=callbacks())
                m.save(mp)
                trained += 1
                dt = time.time() - t1
                disposition.append(f"  新训 {g:6s} seed{seed:<5d}   {dt:.1f}s -> {os.path.basename(mp)}")
                print(f"    {g} seed{seed}  {dt:.1f}s", flush=True)

            for gk, (hist, cids) in groups_eval.items():
                for r in A.eval_one(m, hist, cids, GROUPS[g]):
                    r.update({"exp": g, "seed": seed, "eval_group": gk})
                    records.append(r)

    df = pd.DataFrame(records)
    # 与原 full.csv 同列序
    cols = ["exp", "seed", "eval_group", "cycle_id", "n",
            "ekf_RMSE", "nn_RMSE", "ekf_MAE", "nn_MAE", "ekf_MAX", "nn_MAX"]
    df = df[cols]
    out_csv = os.path.join(RESULTS, "p3_06_ablation_ext_full.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8")

    header = [
        "=== 扩种子重跑（p3_06_ablation_ext）处置清单 ===",
        f"种子集: {SEEDS_EXT}  (共 {len(SEEDS_EXT)})",
        f"组: {sorted(GROUPS.keys())}",
        f"复用已有检查点: {reused} 个",
        f"新训: {trained} 个",
        f"总耗时: {time.time()-t0:.1f}s",
        f"产物: {os.path.basename(out_csv)}  ({len(df)} 行)",
        "",
        "逐 (组, 种子) 处置:",
    ] + disposition
    with open(os.path.join(RESULTS, "p3_06_ablation_ext_seedlist.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n")

    print("=" * 60)
    print(f"复用 {reused} / 新训 {trained} / 共 {len(SEEDS_EXT)*len(GROUPS)} 模型")
    print(f"产物: {out_csv}  ({len(df)} 行)  耗时 {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
