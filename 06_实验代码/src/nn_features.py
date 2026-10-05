# -*- coding: utf-8 -*-
"""
P3-01 特征工程：EKF history → NN 训练样本（滑窗序列 + 目标）

设计依据
--------
1. `05_论文/创新点定稿_v2.md` §5 特征向量定义：
   常规量 (SOC_EKF, V_t, I, T) + EKF 内部诊断量 (ν, NIS, K, diag(P))
   + 差分特征 (ΔV_t, ΔSOC_EKF) + 滑窗历史（长度 L）
2. `04_实验/数据/results/s1_残差可学性判定.md` 对 P3 的三条设计含义：
   - ν 的信息在序列模式里 → 滑窗输入（GRU 主场），不只是瞬时值
   - A2-0 对照基线必须能仅用测量量跑通（防"恒等映射"质疑）
   - OOD 温度如实报告 → 归一化不能放大外推失真

归一化：固定物理尺度（不用训练集 z-score）
------------------------------------------
P2-07c 探针实证：温度特征用训练集统计量归一后，OOD 温度落在 −3σ 外，
线性外推灾难性失败。NN 用**固定物理常数**缩放：
- 部署友好（MCU 端同一套常数，无需存训练统计量）
- OOD 输入只是略微出界（有界），不会被统计量放大
- 跨数量级特征（NIS, K0, P）用对数变换压缩动态范围

目标 y 与融合公式
----------------
y_k = (SOC_EKF_k − SOC_true_k) × 100   （百分点，含偏置）
SOC_fused = SOC_EKF − y_hat / 100

特征组（A2 消融，P3-06 用；定义与任务清单一致）
--------------------------------------------
A2-0 : 仅测量量（含差分）——对照基线
A2-1 : + ν
A2-2 : + ν + NIS
A2-3 : + ν + NIS + K + diag(P)   ← 完整版（主模型）
A2-4 : 测量量 + K + diag(P)（去掉 ν/NIS，检验 K/P 独立贡献）

红线
----
- `ah_throughput` 是泄漏字段（与 soc_true 相关 1.000），绝不做特征
- `soc_true` 只进目标侧
"""
from __future__ import annotations

import os

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(PROJECT_ROOT, "04_实验", "数据", "results")

BURN_IN = 300          # EKF 收敛期（秒/步），与 P2-07 一致
WINDOW_L = 30          # 滑窗长度：30 s ≈ ν 自相关主要滞后范围（acf1 0.35–0.95）
SAFETY_CLIP = 10.0     # 缩放后统一硬裁剪，防 OOD 极值


# --------------------------------------------------------------------------- #
# 单特征定义：name -> (source, transform)
#   source: history 字段名；transform: np.ndarray -> np.ndarray（逐点）
# --------------------------------------------------------------------------- #

def _t_soc(h: dict) -> np.ndarray:
    # ⚠️ soc 是 EKF 的**未裁剪**内部状态，不保证落在 [0,1]（见 ecm_ekf.py 的
    #    SOC_EKF.step：内部状态刻意不 clamp，以免破坏协方差一致性）。
    #    实测 40/40 文件的 soc[0] ∈ [1.0038, 1.1066]，首步更新后即越界。
    #    越界量很小（对同分布 RMSE 的影响 ≤0.022 pp），故未做裁剪。
    return h["soc"].astype(np.float64)


def _t_i(h: dict) -> np.ndarray:
    return h["current_A"] / 5.0                               # ±5 A → ±1


def _t_iabs(h: dict) -> np.ndarray:
    return np.abs(h["current_A"]) / 5.0


def _t_temp(h: dict) -> np.ndarray:
    return (h["temperature_C"] + 20.0) / 45.0                 # −20..25 °C → 0..1


def _t_vt(h: dict) -> np.ndarray:
    return (h["voltage_V"] - 3.0) / 1.3                       # 3.0..4.3 V → 0..1


def _t_dvt(h: dict) -> np.ndarray:
    # 电压差分 q999≈5（正常动态），采样毛刺可达 13 → tanh 软压缩到 ±5
    d = np.diff(h["voltage_V"], prepend=h["voltage_V"][:1])
    return 5.0 * np.tanh(d / 0.05)


def _t_dsoc(h: dict) -> np.ndarray:
    # SOC 差分被 EKF 增益修正项主导（I·K0/Cn 量级），绝大数 <0.9，
    # 为防收敛期可能出现的瞬态尖峰（原注归因"+10 pp 初始偏差"，但该偏置被 clip 吞掉、从未生效）→ tanh 软压缩
    # （保留 0~±5 的线性区 + 有界输出，与 SAFETY_CLIP 衔接）
    d = np.diff(h["soc"], prepend=h["soc"][:1])
    return 5.0 * np.tanh(d / 0.005)


def _t_nu(h: dict) -> np.ndarray:
    return h["nu"] / 0.1                                      # 100 mV 单位


def _t_nis(h: dict) -> np.ndarray:
    return np.log1p(np.clip(h["nis"], 0.0, None)) / 10.0      # log1p 压 12 个数量级


def _t_k0(h: dict) -> np.ndarray:
    # K0 ∈ (0, ~0.9]，中位 ~3e-3 → log10 压缩；+6/6 → 典型值 0.9 附近
    return (np.log10(np.clip(h["K0"], 1e-9, None)) + 6.0) / 6.0


def _t_k1(h: dict) -> np.ndarray:
    return h["K1"] / 1.0                                      # 有符号，线性


def _t_p00(h: dict) -> np.ndarray:
    # P00 ∈ [3e-7, 1e-3] → log10 ∈ [−6.6, −3] → (log10+7)/4 ∈ [0.1, 1]
    return (np.log10(np.clip(h["P00"], 1e-12, None)) + 7.0) / 4.0


def _t_p11(h: dict) -> np.ndarray:
    return (np.log10(np.clip(h["P11"], 1e-12, None)) + 6.0) / 4.0


FEATURE_TRANSFORMS: dict[str, callable] = {
    "soc": _t_soc, "i": _t_i, "iabs": _t_iabs, "temp": _t_temp,
    "vt": _t_vt, "dvt": _t_dvt, "dsoc": _t_dsoc,
    "nu": _t_nu, "nis": _t_nis,
    "k0": _t_k0, "k1": _t_k1, "p00": _t_p00, "p11": _t_p11,
}

#: A2 消融特征组（P3-06；A2-3 为完整主模型）
GROUPS: dict[str, list[str]] = {
    "A2-0": ["soc", "i", "iabs", "temp", "vt", "dvt", "dsoc"],
    "A2-1": ["soc", "i", "iabs", "temp", "vt", "dvt", "dsoc", "nu"],
    "A2-2": ["soc", "i", "iabs", "temp", "vt", "dvt", "dsoc", "nu", "nis"],
    "A2-3": ["soc", "i", "iabs", "temp", "vt", "dvt", "dsoc",
             "nu", "nis", "k0", "k1", "p00", "p11"],
    "A2-4": ["soc", "i", "iabs", "temp", "vt", "dvt", "dsoc",
             "k0", "k1", "p00", "p11"],
}


# --------------------------------------------------------------------------- #
# history 加载
# --------------------------------------------------------------------------- #

def load_history(path: str) -> dict[str, dict[str, np.ndarray]]:
    """读取 EKF history npz → {cycle_id: {field: array}}"""
    z = np.load(path, allow_pickle=False)
    files: dict[str, dict[str, np.ndarray]] = {}
    for key in z.files:
        cid, field = key.split("::", 1)
        files.setdefault(cid, {})[field] = z[key]
    return files


# --------------------------------------------------------------------------- #
# 滑窗构造
# --------------------------------------------------------------------------- #

def file_to_samples(
    h: dict[str, np.ndarray],
    feats: list[str],
    window_l: int = WINDOW_L,
    burn_in: int = BURN_IN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    单文件 → (X, y, t_end)

    X   : (n, L, F) float32，窗内容为 [t−L+1 .. t]，全部落在收敛期之后
    y   : (n,)    目标（pp）= (SOC_EKF − SOC_true) × 100，取窗末时刻
    t_end: (n,)   窗末时刻的文件内索引（评估与画图对齐用）

    文件太短（放不下一个收敛期外的完整窗）→ 返回 None。
    """
    n = len(h["nu"])
    if n < burn_in + window_l + 1:
        return None

    F = np.stack(
        [np.clip(FEATURE_TRANSFORMS[f](h), -SAFETY_CLIP, SAFETY_CLIP)
         for f in feats], axis=1).astype(np.float32)          # (n, F)

    y_all = (h["soc"] - h["soc_true"]) * 100.0                 # (n,) pp

    # 窗末索引 t ∈ [burn_in + L − 1, n−1]，窗内容 ≥ burn_in（全在稳态区）
    starts = np.arange(burn_in, n - window_l + 1)              # 窗首索引
    idx = starts[:, None] + np.arange(window_l)[None, :]       # (n_w, L)
    X = np.take(F, idx, axis=0)                                # (n_w, L, F)

    ends = starts + window_l - 1
    y = y_all[ends].astype(np.float32)
    return X, y, ends.astype(np.int64)


def build_split_arrays(
    files: dict[str, dict[str, np.ndarray]],
    cycle_ids: list[str],
    feats: list[str],
    window_l: int = WINDOW_L,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """多文件拼成一个大数组（训练/验证用）。返回 (X, y, ends_by_file)。"""
    xs, ys, ends = [], [], {}
    for cid in cycle_ids:
        out = file_to_samples(files[cid], feats, window_l)
        if out is None:
            continue
        X, y, e = out
        xs.append(X)
        ys.append(y)
        ends[cid] = e
    return (np.concatenate(xs), np.concatenate(ys), ends) if xs else (None, None, {})
