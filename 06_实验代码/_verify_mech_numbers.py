# -*- coding: utf-8 -*-
"""一次性：核对正文引用的两个机制性数字。

1. `K₀ ≈ 2×10⁻³`（收敛后的卡尔曼增益）
2. `P00` 收敛值 `≈1e-6` vs 构造函数默认 `1e-2`

用法：python 10_代码/_verify_mech_numbers.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"E:\SOC论文项目"
sys.path.insert(0, os.path.join(ROOT, "10_代码", "src"))
from nn_features import BURN_IN, load_history  # noqa: E402

RESULTS = os.path.join(ROOT, "04_实验", "数据", "results")

hist = load_history(os.path.join(RESULTS, "ekf_full_history.npz"))
cids = [c for c in sorted(hist.keys()) if "25degC_Cycle" in c]
cids += [c for c in sorted(hist.keys())
         if "10degC_" in c and "n10degC" not in c and "trise" not in c][:2]
print(f"评估文件 {len(cids)} 个\n")

k0_all, p00_all = [], []
for c in cids:
    h = hist[c]
    k0 = np.asarray(h["K0"], dtype=float)
    p00 = np.asarray(h["P00"], dtype=float)
    tail = slice(BURN_IN, None)          # burn-in 之后 = 收敛段
    k0_all.append(k0[tail])
    p00_all.append(p00[tail])
    print(f"  {c[:40]:42s} K0[burn-in 后] 中位 {np.median(k0[tail]):.3e}"
          f" | P00 中位 {np.median(p00[tail]):.3e}")

K0 = np.concatenate(k0_all)
P00 = np.concatenate(p00_all)
print()
print(f"K0  收敛段：中位 {np.median(K0):.3e}  均值 {K0.mean():.3e}"
      f"  [P5, P95] = [{np.percentile(K0,5):.3e}, {np.percentile(K0,95):.3e}]")
print(f"P00 收敛段：中位 {np.median(P00):.3e}  均值 {P00.mean():.3e}"
      f"  [P5, P95] = [{np.percentile(P00,5):.3e}, {np.percentile(P00,95):.3e}]")

from ecm_ekf import SOC_EKF, ECMParams  # noqa: E402
import inspect  # noqa: E402
sig = inspect.signature(SOC_EKF.__init__)
print(f"\n构造函数默认 p0 = {sig.parameters['p0'].default!r}")
print(f"→ 默认/收敛 之比 = {sig.parameters['p0'].default / np.median(P00):.1f} 倍"
      f"（即 {np.log10(sig.parameters['p0'].default / np.median(P00)):.2f} 个数量级）")
