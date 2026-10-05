# -*- coding: utf-8 -*-
"""P9-04 在**温度相关 OCV 内核**上评估通道配平两臂（A2-K0 / A2-KK）。

为什么必须补这一步
------------------
`p9_03_compare.py` 的 "M1配平 | ocvT" 一行曾把 **A2-1(ocvT 内核评估)**
与 **A2-K0(base 内核评估)** 混着比 —— 两个数据集、两个 OCV 内核，
**不构成有效对照**（实测出现符号翻转的假象）。本脚本把配平两臂也放到
ocvT 历史上重新评估，使四臂同口径。

⚠️ 不重训：训练集是 25 °C，ΔOCV(25 °C) ≡ 0，训练特征未变（见 `p9_01` 自检），
   模型可原样复用。

产出
----
- results/p9_04_chanmatched_ocvT.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "06_实验代码", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from nn_features import build_split_arrays, load_history  # noqa: E402

RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
#: ⚠️ 配平臂的模型在**中文目录** `模型/`（P8 脚本写的），
#:    与 P9-02 用的主模型目录 `models/`（英文）**不是同一个** —— 不要混。
MODELS_DIR = os.path.join(PROJECT_ROOT, "04_实验", "模型")
OUT_CSV = os.path.join(RESULTS, "p9_04_chanmatched_ocvT.csv")

SEEDS_ALL = [7, 13, 42, 101, 202, 303, 404, 505, 606, 707]

BASE = ["soc", "i", "iabs", "temp", "vt", "dvt", "dsoc"]
NEW_GROUPS = {
    "A2-K0": BASE + ["k0"],
    "A2-KK": BASE + ["k0", "k1"],
}
EXPECT_TEST = {"test_id(25C)": 4, "ood(10C)": 9, "ood(0C)": 9,
               "ood(-10C)": 9, "ood(-20C)": 9}


def test_groups(hist):
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


def main() -> int:
    hist = load_history(os.path.join(RESULTS, "ekf_full_history_ocvT.npz"))
    groups = test_groups(hist)
    bad = [k for k, w in EXPECT_TEST.items() if len(groups.get(k, [])) != w]
    if bad:
        print("[!!] 库存不符，终止：", bad, flush=True)
        return 1

    from tensorflow import keras
    records = []
    total = len(NEW_GROUPS) * len(SEEDS_ALL)
    n = 0
    t0 = time.time()
    for g in sorted(NEW_GROUPS.keys()):
        feats = NEW_GROUPS[g]
        for seed in SEEDS_ALL:
            n += 1
            mp = os.path.join(MODELS_DIR, f"gru16_cm_{g}_seed{seed}.keras")
            if not os.path.exists(mp):
                print(f"[{n}/{total}] 缺模型 {os.path.basename(mp)}", flush=True)
                continue
            model = keras.models.load_model(mp)
            for gname, cids in groups.items():
                for cid in cids:
                    if cid not in hist:
                        continue
                    X, y, _ = build_split_arrays(hist, [cid], feats)
                    if len(X) == 0:
                        continue
                    pred = model.predict(X, verbose=0).ravel()
                    records.append(dict(exp=g, seed=seed, eval_group=gname,
                                        cycle_id=cid,
                                        RMSE=float(np.sqrt(np.mean((pred-y)**2)))))
            print(f"[{n}/{total}] {g} seed{seed}  {time.time()-t0:.0f}s", flush=True)

    pd.DataFrame(records).to_csv(OUT_CSV, index=False)
    print(f"\n已写出 {OUT_CSV}（{len(records)} 行）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
