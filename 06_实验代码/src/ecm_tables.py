# -*- coding: utf-8 -*-
"""表驱动 OCV 与 ECM 参数（P2-02 / P2-03 产出的复用封装）

供 P2-05/06（EKF 真实数据基线）、P2-07（残差分析）使用：

- OCVTable：`ocv_soc_curve.csv` → 查表 OCV / dOCV/dSOC
  （替代 `ecm_ekf.py` 里的解析占位函数）
- ECMParamTable：`ecm_params_raw.csv` → (SOC, 温度) → R0/R1/C1
  双线性插值（逻辑与 `p2_04_validate.py` 一致，沉淀为可复用模块）
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")

TEMPS = [-20.0, -10.0, 0.0, 10.0, 25.0]
BINS = np.arange(0.0, 1.05, 0.1)


class OCVTable:
    """OCV-SOC 查询表（step 0.001，线性插值，边界 clamp）。"""

    def __init__(self, soc: np.ndarray, ocv_v: np.ndarray,
                 docv: np.ndarray) -> None:
        self._soc = np.asarray(soc, dtype=float)
        self._ocv = np.asarray(ocv_v, dtype=float)
        self._docv = np.asarray(docv, dtype=float)

    @classmethod
    def from_csv(cls, path: str = os.path.join(
            RESULTS, "ocv_soc_curve.csv")) -> "OCVTable":
        d = pd.read_csv(path)
        return cls(d.soc.to_numpy(), d.ocv_V.to_numpy(),
                   d.docv_dsoc.to_numpy())

    def ocv(self, soc: float | np.ndarray) -> float | np.ndarray:
        """查表 OCV。SOC 超出 [0,1] 时用端点斜率线性延拓。

        EKF 数值发散防护（P2-06 实测根因）：起始新息 +22 mV 锚点偏差
        会把 SOC 推过 1.0，若在端点平顶（梯度 0）则观测信息丢失、
        修正链断裂 → 负增益 → 正反馈爆炸。线性延拓保证梯度恒非零，
        起虚拟势垒作用，把状态拉回 [0,1]。
        """
        s = np.asarray(soc, dtype=float)
        v = np.interp(s, self._soc, self._ocv)
        v_hi = self._ocv[-1] + (s - self._soc[-1]) * self._docv[-1]
        v_lo = self._ocv[0] + (s - self._soc[0]) * self._docv[0]
        v = np.where(s > self._soc[-1], v_hi, v)
        v = np.where(s < self._soc[0], v_lo, v)
        return float(v) if np.ndim(v) == 0 else v

    def docv_dsoc(self, soc: float | np.ndarray) -> float | np.ndarray:
        """dOCV/dSOC，端点外用端点斜率（与 ocv() 延拓一致）。"""
        d = np.interp(soc, self._soc, self._docv)
        d = np.where(np.asarray(soc, dtype=float) > self._soc[-1],
                     self._docv[-1], d)
        d = np.where(np.asarray(soc, dtype=float) < self._soc[0],
                     self._docv[0], d)
        return float(d) if np.ndim(d) == 0 else d


class ECMParamTable:
    """R0/R1/C1 查询表：(SOC, 实测温度) 双线性插值。

    网格：温度 5 档（−20~25 °C）× SOC 10 档（档中点 0.05~0.95），
    每格取该 (温度, SOC 频) 内所有 HPPC 脉冲的**中位数**。
    """

    def __init__(self, grids: dict[str, np.ndarray], temps: list[float]) -> None:
        self.grids = grids
        self.temps = np.asarray(temps, dtype=float)

    @classmethod
    def from_csv(cls, path: str = os.path.join(
            RESULTS, "ecm_params_raw.csv")) -> "ECMParamTable":
        raw = pd.read_csv(path)
        grids = cls._build_grids(raw)
        return cls(grids, TEMPS)

    # ---------------------------------------------------------------- #
    @staticmethod
    def _build_grids(raw: pd.DataFrame) -> dict[str, np.ndarray]:
        grids = {}
        for name in ["r0_ohm", "r1_ohm", "c1_F"]:
            g = np.full((len(TEMPS), len(BINS) - 1), np.nan)
            for ti, tp in enumerate(TEMPS):
                d = raw[raw.temperature_c == tp]
                if d.empty:
                    continue
                cut = pd.cut(d.soc, BINS)
                med = d.groupby(cut, observed=True)[name].median()
                for bi, val in enumerate(med.to_numpy()):
                    g[ti, bi] = val
            # 缺档回填：SOC 维（同温度相邻档外推）
            for ti in range(g.shape[0]):
                row = g[ti]
                idx = np.where(np.isnan(row))[0]
                if len(idx) and not np.all(np.isnan(row)):
                    good = np.where(~np.isnan(row))[0]
                    row[idx] = np.interp(idx, good, row[good])
            # 整行缺失 → 用相邻温度行
            for ti in range(g.shape[0]):
                if np.all(np.isnan(g[ti])):
                    g[ti] = np.nanmedian(g, axis=0)
            grids[name] = g
        return grids

    # ---------------------------------------------------------------- #
    def lookup(self, soc: float, temp: float) -> tuple[float, float, float]:
        """(SOC, 温度) → (R0, R1, C1)。插值边界 clamp。"""
        nb = self.grids["r0_ohm"].shape[1]
        tc = float(np.clip(temp, self.temps[0], self.temps[-1]))
        sc = float(np.clip(soc, 0.0, 1.0))
        # 温度维
        ti = int(np.clip(np.searchsorted(self.temps, tc, side="right") - 1,
                         0, len(self.temps) - 2))
        wt = (tc - self.temps[ti]) / (self.temps[ti + 1] - self.temps[ti])
        # SOC 维（档中点）
        centers = np.arange(nb) * 0.1 + 0.05
        si = int(np.clip(np.searchsorted(centers, sc, side="right") - 1,
                         0, nb - 2))
        ws = float(np.clip((sc - centers[si]) / 0.1, 0.0, 1.0))
        out = []
        for name in ["r0_ohm", "r1_ohm", "c1_F"]:
            g = self.grids[name]
            v = (g[ti, si] * (1 - ws) * (1 - wt)
                 + g[ti, si + 1] * ws * (1 - wt)
                 + g[ti + 1, si] * (1 - ws) * wt
                 + g[ti + 1, si + 1] * ws * wt)
            out.append(float(v))
        return out[0], out[1], out[2]


def make_ekf_table_fns(ocv_tab: OCVTable, ptab: ECMParamTable):
    """生成传给 SOC_EKF 的 (ocv_fn, docv_fn, params_fn) 三件套。

    params_fn(soc_est, temp_k) → (R0, R1, C1)，闭包内持表。
    """
    def ocv_fn(soc):
        return float(ocv_tab.ocv(soc))

    def docv_fn(soc):
        return float(ocv_tab.docv_dsoc(soc))

    def params_fn(soc_est: float, temp_k: float):
        return ptab.lookup(soc_est, temp_k)

    return ocv_fn, docv_fn, params_fn
