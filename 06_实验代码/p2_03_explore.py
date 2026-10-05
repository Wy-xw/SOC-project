# -*- coding: utf-8 -*-
"""P2-03 探查：HPPC 组织结构（温度/cycle/段/SOC 覆盖）。只拿事实，不画图。"""
import pandas as pd, numpy as np
import os
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P = os.path.join(__ROOT__, "04_实验", "数据", "processed", "mcmaster_hppc.csv.gz")
d = pd.read_csv(P, usecols=["timestamp", "current_A", "voltage_V",
                            "soc_true", "cycle_id", "temperature_c"])
print("== 温度分布 ==")
print(d.groupby("temperature_c")["current_A"].count().to_string())
print("\n== 各温度 cycle 数 ==")
print(d.groupby("temperature_c")["cycle_id"].nunique().to_string())
print("\n== 各温度 SOC 覆盖 ==")
print(d.groupby("temperature_c")["soc_true"]
      .agg(["min", "max", "count"]).to_string())

cid = d["cycle_id"].iloc[0]
c = d[d["cycle_id"] == cid]
i = c["current_A"].to_numpy(); t = c["timestamp"].to_numpy()
v = c["voltage_V"].to_numpy()
st = np.where(i > 0.05, 1, np.where(i < -0.05, -1, 0))
b = np.concatenate(([0], np.where(np.diff(st) != 0)[0] + 1, [len(i)]))
print(f"\n== 抽样 cycle: {cid} (temp={c['temperature_c'].iloc[0]}) 段结构(前28) ==")
for k, (a, bb) in enumerate(zip(b[:-1], b[1:])):
    if k >= 28:
        break
    mid = a + (bb - a) // 2
    print(f"seg{k}: st={int(st[a])} dur={t[bb-1]-t[a]:6.0f}s tm={t[a]:7.0f}s "
          f"I@={i[mid]:+5.2f}A V@={v[mid]:.3f}V")
