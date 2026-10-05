# -*- coding: utf-8 -*-
"""
数据读取与统一格式转换 —— 对应任务 P1-03 / P1-04

支持两个公开数据集
------------------
1. **McMaster 18650PF**（主数据集）
   - 来源：Kollmeyer et al., DOI 10.17632/wykht8y7tg.1
   - 本地：`04_实验/数据/raw/mcmaster/`
   - 文件：MATLAB v5/v7 `struct`，变量名 `meas`，9 个字段
   - 温度：25 / 10 / 0 / -10 / -20 °C

2. **NASA PCoE**（跨数据集泛化验证）
   - 本地：`04_实验/数据/raw/nasa/`
   - 文件：MATLAB v5/v7 `struct`，`B0005.cycle{i}.{type,data}`
   - 每个 cycle 一段，放电为 2 A 恒流

统一格式（全局约定，与 `08_数据集方案.md` 一致）
--------------------------------------------
    timestamp, voltage_V, current_A, temperature_C, soc_true, cycle_id, source

符号约定（**重要**）
------------------
- **电流：放电为正**。McMaster 原始数据放电为负，本模块读入时统一翻号。
- **soc_true ∈ [0, 1]**。
- **timestamp**：秒，从 0 开始。

SOC 真值的定义依据（P1-04 关键决策）
----------------------------------
McMaster README 规定 0 / -10 / -20 °C 的驱动工况分别在 80% / 70% / 60% DOD 处终止，
对应放电量 2.32 / 2.03 / 1.74 Ah。三者除以各自 DOD **恰好都等于 2.9 Ah**
（实测 10 °C US06 放电 2.030 Ah，与 70% × 2.9 完全吻合）。

→ 因此采用**数据集自身的容量约定 Cn = 2.9 Ah**：

    soc_true(t) = 1 + Ah(t) / Cn        （Ah ≤ 0 表示放电）

⚠️ 已知局限：电池在测试过程中老化（1C 容量由 2.80 Ah 降至 2.35 Ah），
晚期测试的 SOC 真值会因此带有系统性偏差。见 `04_实验/数据/README_数据说明.md`。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.io import loadmat

# --------------------------------------------------------------------------- #
# 全局常量
# --------------------------------------------------------------------------- #

#: McMaster 数据集自身的容量约定（Ah）。见模块 docstring。
CN_MCMASTER = 2.9

#: NASA PCoE 电池额定容量（Ah），真实容量随老化下降，逐循环由 Capacity 字段给出。
CN_NASA_NOMINAL = 2.0

#: 统一重采样周期（s）——对应端侧 1 Hz 的目标采样率
DT_DEFAULT = 1.0

UNIFIED_COLUMNS = [
    "timestamp", "voltage_V", "current_A", "temperature_C",
    "soc_true", "cycle_id", "source",
]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MCMASTER_ROOT = os.path.join(PROJECT_ROOT, "04_实验", "数据", "raw", "mcmaster")
NASA_ROOT = os.path.join(PROJECT_ROOT, "04_实验", "数据", "raw", "nasa")


# --------------------------------------------------------------------------- #
# McMaster
# --------------------------------------------------------------------------- #

#: 文件名 → (标称温度 °C, 工况类型)。工况类型用于划分训练/验证/测试。
_MCMASTER_TEMP_RE = re.compile(r"n?(\d+)degC")


#: 驱动工况的识别模式（顺序即优先级）。HWFT 是数据集中的拼写笔误（应为 HWFET）。
_DRIVE_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("us06", ("US06",)),
    ("hwfet", ("HWFET", "HWFT")),
    ("udds", ("UDDS",)),
    ("la92", ("LA92",)),
    ("nn", ("NN",)),
    ("cycle", ("CYCLE",)),
]


@dataclass
class McMasterFile:
    """McMaster 数据集中的一个 MAT 文件。"""

    path: str
    filename: str
    temperature_c: float | None      # 标称环境温度
    test_type: str                   # us06 / hwfet / udds / la92 / nn / cycle / hppc / ...
    is_drive_cycle: bool
    is_combined: bool = False        # 是否为「多工况合并文件」
    is_trise: bool = False           # 是否属于变温工况组（环境温度在工况中持续上升）

    @property
    def relpath(self) -> str:
        return os.path.relpath(self.path, MCMASTER_ROOT).replace("\\", "/")


def _descriptor(filename: str) -> str:
    """
    去掉日期/时间前缀、温度标记与 `_Pan18650PF` 后缀，只留工况描述。

    例：`03-27-17_09.06 10degC_US06_HWFET_UDDS_LA92_NN_Pan18650PF.mat`
        → `US06_HWFET_UDDS_LA92_NN`
    """
    s = filename.upper().replace(".MAT", "")
    s = re.sub(r".*DEGC[_\s]*", "", s)      # 去掉到 "degC" 为止的前缀
    s = s.replace("_PAN18650PF", "").strip("_ ")
    return s


def classify_mcmaster(filename: str) -> tuple[float | None, str, bool, bool]:
    """
    由文件名推断 (标称温度, 工况类型, 是否为驱动工况, 是否为合并文件)。

    温度
    ----
    文件名中的 `n10degC` / `n20degC` 表示 **负温度**（-10 / -20 °C）；
    25 °C 主目录下的文件不带温度前缀，默认按 25 °C 处理。

    合并文件
    --------
    数据集中有一部分文件把「多个驱动工况 + 期间的充电/暂停」记录在同一个文件里
    （见数据集 README 的 "Repeated data" 一节）。这类文件的特征是其工况描述里
    **出现两个及以上**工况名，例如 `10degC_US06_HWFET_UDDS_LA92_NN`、
    `0degC_LA92_NN`。它们的数据与拆分文件**重复**，且混有充电段，
    会使 Ah 计数器的「每工况重置」假设失效 → 默认排除，需显式识别。
    """
    m = _MCMASTER_TEMP_RE.search(filename)
    if m:
        temp = float(m.group(1))
        if re.search(r"n\d+degC", filename, re.IGNORECASE):
            temp = -temp
    else:
        temp = 25.0

    desc = _descriptor(filename)

    # 统计描述中出现的不重复工况名个数
    hits: list[str] = []
    for kind, pats in _DRIVE_PATTERNS:
        if any(p in desc for p in pats):
            hits.append(kind)
    is_combined = len(hits) >= 2

    # `..._Cycle_1to4_w_pauses_...`：多个循环 + 期间充电/暂停记录在同一文件
    # （README 第 8/10 条），性质与合并文件相同 → 一并排除。
    if "W_PAUSES" in desc or "WITH_PAUSE" in desc:
        is_combined = True

    if hits:
        return temp, hits[0], True, is_combined

    if "HPPC" in desc or "PULSE" in desc:
        return temp, "hppc", False, False
    if "OCV" in desc:
        return temp, "ocv_c20", False, False
    if "DIS1C" in desc:
        return temp, "dis_1c", False, False
    if "DIS5_10P" in desc:
        return temp, "dis_5p_10p", False, False
    if "CHARGE" in desc:
        return temp, "charge", False, False
    if "PAUSE" in desc:
        return temp, "pause", False, False
    return temp, "other", False, False


def scan_mcmaster(root: str = MCMASTER_ROOT) -> list[McMasterFile]:
    """遍历 raw/mcmaster，返回全部 MAT 文件及其分类。"""
    out: list[McMasterFile] = []
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if not fn.lower().endswith(".mat"):
                continue
            temp, kind, is_drive, is_comb = classify_mcmaster(fn)
            rel = os.path.relpath(os.path.join(dp, fn), root)
            out.append(McMasterFile(os.path.join(dp, fn), fn, temp, kind,
                                    is_drive, is_comb, "trise" in rel.lower()))
    return out


def load_meas_mat(path: str) -> dict[str, np.ndarray]:
    """
    读取一个 McMaster MAT 文件，返回原始列的 dict（**不改符号**）。

    原始字段（见数据集 README）：
        TimeStamp, Voltage, Current, Ah, Wh, Power,
        Battery_Temp_degC, Time, Chamber_Temp_degC

    Raises
    ------
    ValueError
        文件缺少 `meas` 结构，或记录为空（数据集中有 1 个空记录文件）。
    """
    try:
        raw = loadmat(path)
    except Exception as exc:                      # noqa: BLE001
        raise ValueError(f"无法解析 MAT 文件：{exc}") from exc

    if "meas" not in raw:
        raise ValueError("MAT 文件中没有 `meas` 结构")
    meas = raw["meas"]
    if meas.size == 0:
        raise ValueError("`meas` 结构为空")
    rec = meas.ravel()[0]

    out: dict[str, np.ndarray] = {}
    for name in rec.dtype.names or ():
        out[name] = np.asarray(rec[name]).ravel()
    if len(out.get("Time", [])) == 0:
        raise ValueError("记录为空（0 个采样点）")
    return out


def _uniform_resample(t: np.ndarray, channels: dict[str, np.ndarray],
                      dt: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    把非等间隔的原始采样线性插值到等间隔网格上。

    原始数据的时间步长并不统一（0.04 ~ 2.1 s，见数据集 README），
    必须先统一采样率，后续 EKF 的离散递推才有意义。
    """
    t = np.asarray(t, dtype=float)
    order = np.argsort(t, kind="stable")
    t = t[order]

    # 去掉重复时间戳（np.interp 要求严格递增）
    keep = np.concatenate(([True], np.diff(t) > 0))
    t = t[keep]

    t_grid = np.arange(t[0], t[-1] + 1e-9, dt)
    out: dict[str, np.ndarray] = {}
    for k, v in channels.items():
        v = np.asarray(v, dtype=float)[order][keep]
        out[k] = np.interp(t_grid, t, v)
    return t_grid - t[0], out


def build_mcmaster_frame(path: str, dt: float = DT_DEFAULT,
                         cn: float = CN_MCMASTER) -> pd.DataFrame:
    """
    把单个 McMaster MAT 文件转成统一格式的 DataFrame。

    处理内容
    --------
    1. 电流符号翻转：原始放电为负 → 统一为「放电为正」
    2. 线性插值到等间隔 dt
    3. 计算 SOC 真值：soc_true = 1 + Ah / cn
    """
    cols = load_meas_mat(path)
    temp, kind, _, _ = classify_mcmaster(os.path.basename(path))

    channels = {
        "voltage_V": cols["Voltage"],
        "current_A": -cols["Current"],          # ← 符号翻转
        "temperature_C": cols["Battery_Temp_degC"],
        "chamber_temp_C": cols.get("Chamber_Temp_degC", np.full(len(cols["Time"]), np.nan)),
        "ah_throughput": cols["Ah"],
    }
    t_grid, res = _uniform_resample(cols["Time"], channels, dt)

    df = pd.DataFrame({
        "timestamp": t_grid,
        "voltage_V": res["voltage_V"],
        "current_A": res["current_A"],
        "temperature_C": res["temperature_C"],
        "soc_true": np.clip(1.0 + res["ah_throughput"] / cn, 0.0, 1.0),
        "cycle_id": os.path.splitext(os.path.basename(path))[0],
        "source": "mcmaster",
    })
    df["chamber_temp_C"] = res["chamber_temp_C"]
    df["ah_throughput"] = res["ah_throughput"]
    df["test_type"] = kind
    df["temperature_c"] = temp
    return df


# --------------------------------------------------------------------------- #
# NASA PCoE
# --------------------------------------------------------------------------- #

def load_nasa_battery(path: str):
    """读取一个 NASA 电池文件，返回 `cycle` 结构数组（未做格式转换）。"""
    d = loadmat(path, squeeze_me=True, struct_as_record=False)
    key = [k for k in d if not k.startswith("__")]
    if len(key) != 1:
        raise ValueError(f"顶层变量不唯一：{key}")
    return d[key[0]]


def list_nasa_discharges(path: str) -> list:
    """返回一个 NASA 电池文件中全部放电循环。"""
    return [c for c in load_nasa_battery(path).cycle if c.type == "discharge"]


def build_nasa_frame(path: str, cycle_index: int, dt: float = DT_DEFAULT) -> pd.DataFrame:
    """
    把一个 NASA 放电循环转成统一格式。

    SOC 真值：NASA 每个放电循环前均满充，故
        soc_true(t) = 1 - Ah(t) / Capacity
    其中 Capacity 是该循环实测容量（Ah），随老化下降。
    """
    dis = list_nasa_discharges(path)
    c = dis[cycle_index]
    d = c.data

    t = np.asarray(d.Time, dtype=float)
    i_load = np.asarray(d.Current_load, dtype=float)      # 放电为正
    v = np.asarray(d.Voltage_measured, dtype=float)
    temp = np.asarray(d.Temperature_measured, dtype=float)
    cap = float(np.asarray(d.Capacity).ravel()[0])

    channels = {"voltage_V": v, "current_A": i_load, "temperature_C": temp}
    t_grid, res = _uniform_resample(t, channels, dt)

    # 由等间隔重采样后的电流累加得到放电量，避免原始不等间隔带来的积分误差
    ah = np.cumsum(res["current_A"]) * dt / 3600.0
    soc = np.clip(1.0 - ah / cap, 0.0, 1.0) if cap > 0 else np.full_like(ah, np.nan)

    battery = os.path.splitext(os.path.basename(path))[0]
    df = pd.DataFrame({
        "timestamp": t_grid,
        "voltage_V": res["voltage_V"],
        "current_A": res["current_A"],
        "temperature_C": res["temperature_C"],
        "soc_true": soc,
        "cycle_id": f"{battery}_dis{cycle_index:03d}",
        "source": "nasa",
    })
    df["ah_throughput"] = ah
    df["test_type"] = "dis_2a_cc"
    df["temperature_c"] = float(c.ambient_temperature)
    return df
