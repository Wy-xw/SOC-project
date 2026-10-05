# -*- coding: utf-8 -*-
"""
CB-2b 独立校验：EKF 的**物理**精度，不经过任何 Δ 代数。

要验证的论断
------------
现行手稿报同分布集 EKF RMSE = 2.7342 pp。CB-2 发现其中约 74% 来自
「真值按名义容量 Cn=2.9 定基」，即度量的是 EKF 与一个**误设参考**的
分歧，不是估计误差。

但 CB-2 的「0.707 pp」是用 Δ 减法算出来的，**依赖 Δ 的具体构造**。
本脚本换一条完全独立的路径复核，且不重训：把 EKF **内部**的库仑计数
容量 Cn 从固定的 2.9 换成逐文件实测容量（物理正确），然后

  (1) 对 soc_true_nom 打分 —— 应显著**变差**（模型不再迁就名义约定）
  (2) 对 soc_true_S1  打分 —— 应**保持或改善**（模型与电芯匹配了）

若 (2) 仍落在 ~0.7 pp 量级 → 「EKF 物理上已足够准」成立，与 Δ 无关。
若 (2) 显著变坏 → CB-2 的 0.707 是 Δ 代数的算术巧合，必须撤回。

为什么这条检验独立
------------------
不涉及任何 (soc_true_alt - soc_true_nom) 的减法，直接跑 EKF、直接比真值。

⚠ 边界：OCV 表建在**新鲜电芯**的 C/20 上（`p2_02_ocv_fit.py`），
   新鲜电芯容量 ≈ 2.9，故该表对新鲜电芯是自洽的；驱动文件是老化电芯。
   本检验只改 EKF 内部 Cn，**不改 OCV 表**。

用法
----
    "D:/venvs/soc-paper/Scripts/python.exe" 10_代码/cb2b_ekf_physical_check.py
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys

import numpy as np
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd

PROJECT_ROOT = __ROOT__
sys.path.insert(0, os.path.join(PROJECT_ROOT, "10_代码", "src"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from ecm_ekf import ECMParams, SOC_EKF            # noqa: E402
from ecm_tables import OCVTable, ECMParamTable, make_ekf_table_fns  # noqa: E402

DATA = os.path.join(PROJECT_ROOT, "04_实验", "数据", "processed")
RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")
CN_NOMINAL = 2.9
SOC0_BIAS = 0.10


def load_p506():
    p = os.path.join(PROJECT_ROOT, "10_代码", "p5_06_capacity_sensitivity.py")
    spec = importlib.util.spec_from_file_location("p506", p)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


def run_ekf(i, v, t, cn: float, ocv_fn, docv_fn, params_fn, soc0: float):
    """跑一遍 EKF，返回内部 SOC 轨迹。cn 为内部库仑计数容量。"""
    ts_len = len(i)
    p = ECMParams(Cn=cn, dt=1.0)
    ekf = SOC_EKF(p, soc0=soc0, ocv_fn=ocv_fn, docv_fn=docv_fn,
                  params_fn=params_fn)
    soc = np.empty(ts_len, dtype=np.float64)
    for k in range(ts_len):
        ekf.step(float(i[k]), float(v[k]), temp_k=float(t[k]))
        soc[k] = float(ekf.x[0])
    return soc


def main() -> int:
    ocv_tab = OCVTable.from_csv()
    ptab = ECMParamTable.from_csv()
    ocv_fn, docv_fn, params_fn = make_ekf_table_fns(ocv_tab, ptab)

    caps = load_p506()
    meas = caps.measure_1c_capacities()
    t0, t1, c0, c1 = caps.fit_capacity_curve(meas)
    print(f"容量曲线 {t0.date()} Cn={c0:.4f} -> {t1.date()} Cn={c1:.4f} Ah")

    man = pd.read_csv(os.path.join(PROJECT_ROOT, "04_实验", "数据", "splits",
                                  "split_manifest.csv"))
    grp = dict(zip(man.cycle_id, man.split))
    nom = dict(zip(man.cycle_id, man.nominal_temp_c))
    NM = {25.0: "test_id(25C)", 10.0: "ood(10C)", 0.0: "ood(0C)",
          -10.0: "ood(-10C)", -20.0: "ood(-20C)"}
    d = pd.read_csv(os.path.join(DATA, "mcmaster_drive_cycles.csv.gz"))

    rows = []
    for cid, g in d.groupby("cycle_id", sort=True):
        if grp.get(cid) not in ("test_id", "test_ood"):
            continue
        if len(g) < 400:
            continue
        i = g.current_A.to_numpy(float)
        v = g.voltage_V.to_numpy(float)
        t = g.temperature_C.to_numpy(float)
        s = g.soc_true.to_numpy(float)                    # 名义口径真值
        m = re.match(r"^(\d\d-\d\d-\d\d)", cid)
        cn_s1 = caps.capacity_at(
            pd.to_datetime(m.group(1), format="%m-%d-%y"), t0, t1, c0, c1)

        # 物理参考：同一 Ah 计数，按实测容量重新定基
        ah = g.ah_throughput.to_numpy(float)
        soc_s1 = np.clip(1.0 + ah / cn_s1, 0.0, 1.0)

        soc0 = float(np.clip(s[0] + SOC0_BIAS, 0.0, 1.0))
        socA = run_ekf(i, v, t, CN_NOMINAL, ocv_fn, docv_fn, params_fn, soc0)
        socB = run_ekf(i, v, t, cn_s1, ocv_fn, docv_fn, params_fn, soc0)

        sl = slice(300, len(i))
        e = lambda x, ref: float(np.sqrt(np.mean(((x - ref) * 100.0)[sl] ** 2)))
        rows.append({
            "cycle_id": cid, "eval_group": NM.get(nom.get(cid), "?"),
            "Cn_S1": cn_s1, "Cn_nom": CN_NOMINAL,
            "A_vs_nom": e(socA, s), "A_vs_S1": e(socA, soc_s1),
            "B_vs_nom": e(socB, s), "B_vs_S1": e(socB, soc_s1),
            "EKF_published_vs_nom": e(socA, s),
        })
        print(f"  {cid[:22]:24s} Cn={cn_s1:.3f}  "
              f"A(2.9): nom {rows[-1]['A_vs_nom']:6.3f} S1 {rows[-1]['A_vs_S1']:6.3f} | "
              f"B({cn_s1:.2f}): nom {rows[-1]['B_vs_nom']:6.3f} "
              f"S1 {rows[-1]['B_vs_S1']:6.3f}")

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "cb2b_ekf_physical_check.csv")
    df.to_csv(out, index=False)
    print(f"\n写出 {out} ({len(df)} 行)")

    print("\n" + "=" * 88)
    print("EKF 精度：内部 Cn=2.9（现行） vs 内部 Cn=逐文件实测，两种打分参考")
    print("=" * 88)
    order = ["test_id(25C)", "ood(10C)", "ood(0C)", "ood(-10C)", "ood(-20C)"]
    g = df.groupby("eval_group")[["A_vs_nom", "A_vs_S1", "B_vs_nom", "B_vs_S1"]]
    agg = g.mean().reindex(order)
    cnt = df.groupby("eval_group").size().reindex(order)
    agg.insert(0, "n文件", cnt)
    agg.columns = ["n文件", "Cn2.9→名义", "Cn2.9→物理", "Cn实测→名义", "Cn实测→物理"]
    print(agg.round(3).to_string())

    print("\n[判据]")
    tid = df[df.eval_group == "test_id(25C)"]
    print(f"  同分布集 Cn=2.9 对名义 = {tid.A_vs_nom.mean():.4f} pp"
          f"  （手稿报 2.7342）")
    print(f"  同分布集 Cn=2.9 对物理 = {tid.A_vs_S1.mean():.4f} pp  （CB-2 的 0.707）")
    print(f"  同分布集 Cn=实测 对物理 = {tid.B_vs_S1.mean():.4f} pp  ← 关键：模型匹配后")
    print(f"  同分布集 Cn=实测 对名义 = {tid.B_vs_nom.mean():.4f} pp")
    print()
    ok = tid.B_vs_S1.mean() < tid.A_vs_nom.mean()
    print(f"  物理口径下 EKF 误差是否远小于名义口径: {'是' if ok else '否'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
