# -*- coding: utf-8 -*-
"""
B3：自适应 EKF 核心 + NN 残差修正（R1-M4 衍生实验）
目的：检验把"自适应 R 的 EKF"作为物理核心时，残差修正能否在深低温救回/超越
      ——对比：AEKF-R 单独 vs AEKF-R+NN vs 固定EKF+NN(B2 结果)
流程：① 用 AEKF-R 重跑全部 46 文件（train5+val1+test40）生成 history
      ② 以 A2-3 特征训练 GRU16 × 3 seeds（同主流程超参）
      ③ 分组评估 + 逐文件配对（vs AEKF-R 单独）
输出：results/b3_aekf_history.npz、b3_perfile.csv、b3_summary.md
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "10_代码", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from ecm_ekf import ECMParams, SOC_EKF  # noqa
from ecm_tables import OCVTable, ECMParamTable, make_ekf_table_fns  # noqa
from nn_features import GROUPS, BURN_IN, WINDOW_L, build_split_arrays, load_history  # noqa

# Windows 控制台默认 GBK，而本脚本 print 含 GBK 无码位的字符 —— 不改编码会在
# **成功路径**崩掉（活干完了却退出码 1）。别靠 PYTHONIOENCODING 兜底。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, "04_实验", "数据", "processed")
RES = os.path.join(PROJ, "04_实验", "数据", "results")
MODELS = os.path.join(PROJ, "04_实验", "models")
SEEDS = [7, 13, 42]
SOC0_BIAS = 0.10
FIELDS = ["soc", "v1", "vt_pred", "nu", "S", "nis", "K0", "K1", "P00", "P11"]


def grp(c):
    if "trise" in c: return None
    if "25degC_Cycle" in c: return "test_id(25C)"
    if "10degC_" in c and "n10degC" not in c and "n20degC" not in c: return "ood(10C)"
    if "0degC_" in c and "10degC" not in c and "20degC" not in c: return "ood(0C)"
    if "n10degC_" in c: return "ood(-10C)"
    if "n20degC_" in c: return "ood(-20C)"
    return "train/val"


def phase1_histories():
    path = os.path.join(RES, "b3_aekf_history.npz")
    if os.path.exists(path):
        print("history 已存在，跳过 phase1")
        return
    ocv_fn, docv_fn, params_fn = make_ekf_table_fns(
        OCVTable.from_csv(), ECMParamTable.from_csv())
    man = pd.read_csv(os.path.join(PROJ, "04_实验", "数据", "splits",
                                   "split_manifest.csv"))
    d = pd.read_csv(os.path.join(DATA, "mcmaster_drive_cycles.csv.gz"))
    use = set(man[man.split.isin(["train", "val", "test_id", "test_ood"])]
              .cycle_id)
    flat = {}
    for cid, g in d.groupby("cycle_id"):
        if cid not in use or len(g) < 100:
            continue
        i = g.current_A.to_numpy(); v = g.voltage_V.to_numpy()
        t = g.temperature_C.to_numpy(); s = g.soc_true.to_numpy()
        p = ECMParams(Cn=2.9, dt=1.0)
        soc0 = float(np.clip(s[0] + SOC0_BIAS, 0.0, 1.0))
        e = SOC_EKF(p, soc0=soc0, ocv_fn=ocv_fn, docv_fn=docv_fn,
                    params_fn=params_fn, adapt_R=True)
        for k in range(len(i)):
            e.step(float(i[k]), float(v[k]), temp_k=float(t[k]))
        for k2 in FIELDS:
            flat[f"{cid}::{k2}"] = np.array(e.history[k2])
        for k2, arr in [("timestamp", g.timestamp.to_numpy()),
                        ("soc_true", s), ("current_A", i),
                        ("temperature_C", t), ("voltage_V", v)]:
            flat[f"{cid}::{k2}"] = arr
        print("  aekf:", cid[:40], flush=True)
    np.savez_compressed(path, **flat)
    print("saved:", path)


def main():
    from tensorflow import keras
    phase1_histories()
    hist = load_history(os.path.join(RES, "b3_aekf_history.npz"))
    tr = [c for c in hist if "25degC_" in c and "UDDS" not in c
          and "Cycle" not in c]
    va = [c for c in hist if "25degC_UDDS" in c]
    feats = GROUPS["A2-3"]
    Xtr, ytr, _ = build_split_arrays(hist, tr, feats)
    Xva, yva, _ = build_split_arrays(hist, va, feats)
    print(f"train {len(ytr)}, val {len(yva)}")

    report = []
    for s in SEEDS:
        mp = os.path.join(MODELS, f"gru16_A2-3_aekf_seed{s}.keras")
        if os.path.exists(mp):
            model = keras.models.load_model(mp, compile=False)
        else:
            tf = __import__("tensorflow")
            tf.random.set_seed(s)
            inp = keras.Input(shape=(WINDOW_L, Xtr.shape[2]))
            x = keras.layers.GRU(16, unroll=True)(inp)
            x = keras.layers.Dense(8, activation="relu")(x)
            out = keras.layers.Dense(1)(x)
            model = keras.Model(inp, out)
            model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
            cbs = [keras.callbacks.EarlyStopping(patience=20,
                                                 restore_best_weights=True),
                   keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5,
                                                     min_lr=1e-5)]
            model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=200,
                      batch_size=256, callbacks=cbs, verbose=0)
            model.save(mp)
            print("trained", mp)
        # 评估
        for cid in hist:
            gg = grp(cid)
            if gg is None:
                continue
            X, y, _ = build_split_arrays(hist, [cid], feats)
            if X is None:
                continue
            yh = model.predict(X, verbose=0).ravel()
            e_aekf = y                       # AEKF-R 自身残差（pp）
            e_fuse = y - yh
            report.append({"seed": s, "grp": gg, "cycle_id": cid,
                           "aekf_RMSE": float(np.sqrt(np.mean(e_aekf**2))),
                           "fused_RMSE": float(np.sqrt(np.mean(e_fuse**2)))})
    df = pd.DataFrame(report)
    df.to_csv(os.path.join(RES, "b3_perfile.csv"), index=False)
    order = ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]
    pf = df.groupby(["grp", "cycle_id"]).agg(
        aekf=("aekf_RMSE", "mean"), fused=("fused_RMSE", "mean")).reset_index()
    agg = pf.groupby("grp").agg(aekf=("aekf", "mean"),
                                fused=("fused", "mean")).loc[order]
    from scipy import stats as st
    L = ["# B3 自适应核心融合（AEKF-R + NN）\n",
         f"训练集 {len(tr)} 文件 / 验证 {len(va)}；每文件跨 3 seeds 平均。\n",
         "## 组均值（RMSE, pp）\n",
         "| 评估组 | AEKF-R 单独 | AEKF-R+NN | 变化 |",
         "|---|---|---|---|"]
    for ev in order:
        L.append(f"| {ev} | {agg.loc[ev,'aekf']:.3f} | "
                 f"{agg.loc[ev,'fused']:.3f} | "
                 f"{agg.loc[ev,'fused']-agg.loc[ev,'aekf']:+.3f} |")
    L += ["", "## 逐文件配对 δ = fused − AEKF-R（负=融合更好）\n",
          "| 评估组 | n | δ | 95% CI | 融合更好的文件 |", "|---|---|---|---|---|"]
    for ev in order:
        d = pf[pf.grp == ev]
        delta = d.fused - d.aekf
        ci = st.t.ppf(0.975, len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
        L.append(f"| {ev} | {len(d)} | {delta.mean():+.3f} | "
                 f"[{delta.mean()-ci:+.3f}, {delta.mean()+ci:+.3f}] | "
                 f"{(delta<0).sum()}/{len(d)} |")
    with open(os.path.join(RES, "b3_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("saved: b3_summary.md")


if __name__ == "__main__":
    main()
