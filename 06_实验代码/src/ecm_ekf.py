# -*- coding: utf-8 -*-
"""
1RC-ECM 一阶 RC 等效电路模型 + 扩展卡尔曼滤波（EKF）SOC 估计器

对应任务：P2-01（模型）、P2-05（EKF）

状态量
------
x = [SOC, V1]^T

离散状态方程（电流 I > 0 为放电）
--------------------------------
SOC_{k+1} = SOC_k - (eta_c * dt / (3600 * Cn)) * I_k
V1_{k+1}  = exp(-dt / (R1*C1)) * V1_k
            + R1 * (1 - exp(-dt / (R1*C1))) * I_k

观测方程
--------
Vt_k = OCV(SOC_k) - V1_k - I_k * R0 + v_k

符号约定
--------
- I  > 0 : 放电（SOC 下降）
- I  < 0 : 充电（SOC 上升）
- V1     : RC 网络极化电压，放电时为正

⚠️ 关于 OCV-SOC 曲线
--------------------
本文件中的 `ocv()` / `docv_dsoc()` 是**解析占位函数**，仅用于算法开发与
自检（P2 冒烟测试）。正式的 OCV-SOC 关系必须在 P2-02 中用 McMaster
18650PF 的小电流充放电数据**拟合得到**，届时替换这两个函数即可。

⚠️ 关于合成数据
--------------
`simulate_true_system()` 产生的数据是**合成数据**，只允许用于验证算法
代码是否正确，**不得作为实验测量结果写入论文**。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# OCV-SOC 关系（占位解析式，P2-02 将用实测数据拟合替换）
# --------------------------------------------------------------------------- #

def ocv(soc: np.ndarray | float) -> np.ndarray | float:
    """
    开路电压 OCV(SOC)，单位 V。

    形状刻意做成 NCA 电池的典型形态：
    - 两端陡（SOC < 0.15 / > 0.85）→ dOCV/dSOC 大 → EKF 可观测性好
    - 中段平（0.2 < SOC < 0.8）   → dOCV/dSOC 小 → EKF 可观测性差

    中段可观测性差正是本课题要解决的问题：EKF 在中段收敛慢、误差大，
    给「NN 学残差」留出了发挥空间。
    """
    s = np.asarray(soc, dtype=float)
    return (3.72 + 0.35 * s - 0.12 * s**2
            - 0.35 * np.exp(-20.0 * s)
            + 0.25 * np.exp(20.0 * (s - 1.0)))


def docv_dsoc(soc: np.ndarray | float) -> np.ndarray | float:
    """dOCV/dSOC，EKF 观测雅可比 H 的第一个元素。"""
    s = np.asarray(soc, dtype=float)
    return (0.35 - 0.24 * s
            + 7.0 * np.exp(-20.0 * s)
            + 5.0 * np.exp(20.0 * (s - 1.0)))


# --------------------------------------------------------------------------- #
# 模型参数
# --------------------------------------------------------------------------- #

@dataclass
class ECMParams:
    """1RC-ECM 参数。默认值参考 Panasonic NCR18650PF 量级，P2-03 重新辨识。"""

    Cn: float = 2.9          # 额定容量 (Ah)
    R0: float = 0.025        # 欧姆内阻 (Ohm)
    R1: float = 0.015        # 极化电阻 (Ohm)
    C1: float = 2000.0       # 极化电容 (F)
    eta_c: float = 1.0       # 库仑效率
    dt: float = 1.0          # 采样周期 (s)

    @property
    def tau(self) -> float:
        """RC 时间常数 (s)。"""
        return self.R1 * self.C1


# --------------------------------------------------------------------------- #
# EKF
# --------------------------------------------------------------------------- #

class SOC_EKF:
    """
    1RC-ECM 的扩展卡尔曼滤波器。

    状态 x = [SOC, V1]^T，观测 z = Vt（端电压）。

    Parameters
    ----------
    params : ECMParams
    soc0 : float
        初始 SOC 猜测值。刻意给偏，用于验证收敛性。
    p0 : float
        初始协方差对角元。
    q_soc, q_v1 : float
        过程噪声方差（状态两维）。
    r : float
        电压量测噪声方差 (V^2)。
    """

    def __init__(
        self,
        params: ECMParams,
        soc0: float = 1.0,
        v1_0: float = 0.0,
        p0: float = 1e-2,
        q_soc: float = 1e-9,
        q_v1: float = 1e-6,
        r: float = 1e-4,
        ocv_fn=None,
        docv_fn=None,
        params_fn=None,
        adapt_R: bool = False,
        adapt_Q: bool = False,
        adapt_win: int = 60,
        adapt_alpha: float = 0.2,
    ) -> None:
        """
        ocv_fn / docv_fn : callable，SOC → OCV / dOCV/dSOC。
            缺省用本模块解析占位函数（合成自检用）；
            真实数据用 `src/ecm_tables.py` 的 OCVTable 查表函数。

        params_fn : callable (soc_est, temp_k) -> (R0, R1, C1)。
            每步按（SOC 估计, 实测温度）插值 ECM 参数（真实数据用）。
            缺省 None → 用 params 的常数（合成自检路径，行为与旧版一致）。
        """
        self.p = params
        self.x = np.array([soc0, v1_0], dtype=float)
        self.P = np.diag([p0, p0])
        self.Q = np.diag([q_soc, q_v1])
        self.R = np.array([[r]])
        self.ocv_fn = ocv_fn if ocv_fn is not None else ocv
        self.docv_fn = docv_fn if docv_fn is not None else docv_dsoc
        self.params_fn = params_fn

        # 自适应扩展（R1-M4 对照实验用；默认关闭 → 行为与旧版完全一致）
        # adapt_R: 新息协方差匹配 R̂ = mean(ν²)_win − H P⁻ Hᵀ（窗口均值）
        # adapt_Q: 由矫正量驱动的过程噪声跟踪 q_soc ← (K0 ν)²
        # 两者均用指数滑动平均（alpha）平滑，并设下限防塌缩
        self.adapt_R = adapt_R
        self.adapt_Q = adapt_Q
        self.adapt_win = int(adapt_win)
        self.adapt_alpha = float(adapt_alpha)
        self._r_est = float(r)
        self._q_soc_est = float(q_soc)
        self._nu_buf: list[float] = []
        self._q_v1_init = float(q_v1)

        # 逐步记录，供 P2-07 残差分析与 P2-09 特征导出使用
        self.history: dict[str, list] = {
            "soc": [], "v1": [], "vt_pred": [], "nu": [], "S": [], "nis": [],
            "K0": [], "K1": [], "P00": [], "P11": [],
        }

    # ---------------------------------------------------------------- #
    def _step_params(self, temp_k: float | None) -> tuple[float, float, float]:
        """本步 ECM 参数（欧姆内阻, 极化电阻, 极化电容）。"""
        if self.params_fn is not None and temp_k is not None:
            return self.params_fn(float(self.x[0]), float(temp_k))
        return self.p.R0, self.p.R1, self.p.C1

    def _f(self, x: np.ndarray, i_k: float, r1: float, c1: float) -> np.ndarray:
        """状态转移 f(x, u)。"""
        soc, v1 = x
        a = np.exp(-self.p.dt / max(r1 * c1, 1e-3))
        soc_next = soc - (self.p.eta_c * self.p.dt / (3600.0 * self.p.Cn)) * i_k
        v1_next = a * v1 + r1 * (1.0 - a) * i_k
        return np.array([soc_next, v1_next])

    def _A(self, r1: float, c1: float) -> np.ndarray:
        """状态雅可比 A = df/dx（本模型为常矩阵，仅 V1 行衰减系数）。"""
        a = np.exp(-self.p.dt / max(r1 * c1, 1e-3))
        return np.array([[1.0, 0.0], [0.0, a]])

    def _h(self, x: np.ndarray, i_k: float, r0: float) -> float:
        """观测函数 h(x, u)。"""
        soc, v1 = x
        return float(self.ocv_fn(soc) - v1 - i_k * r0)

    def _H(self, x: np.ndarray) -> np.ndarray:
        """观测雅可比 H = dh/dx。"""
        return np.array([[float(self.docv_fn(x[0])), -1.0]])

    # ---------------------------------------------------------------- #
    def step(self, i_k: float, vt_k: float, temp_k: float | None = None) -> float:
        """
        推进一个采样步：预测 → 更新。

        temp_k：实测温度（°C）。给了且构造时设置了 params_fn，
        则本步参数按 (SOC 估计, temp_k) 插值（真实数据路径）；
        否则用常数参数（合成自检路径，行为与旧版一致）。

        Returns
        -------
        float
            更新后的 SOC 估计值。
        """
        r0, r1, c1 = self._step_params(temp_k)
        A = self._A(r1, c1)
        x_pred = self._f(self.x, i_k, r1, c1)

        # 自适应 Q：用上一步的矫正量 (K0 ν)² 跟踪过程噪声（延迟 1 步）
        if self.adapt_Q and len(self._nu_buf) > 0:
            last_k0 = self.history["K0"][-1] if self.history["K0"] else 0.0
            last_nu = self._nu_buf[-1]
            q_new = (last_k0 * last_nu) ** 2
            self._q_soc_est = ((1 - self.adapt_alpha) * self._q_soc_est
                               + self.adapt_alpha * max(q_new, 1e-12))
            self.Q = np.diag([self._q_soc_est, self._q_v1_init])

        P_pred = A @ self.P @ A.T + self.Q

        H = self._H(x_pred)
        z_pred = self._h(x_pred, i_k, r0)

        nu = vt_k - z_pred                      # 新息 ν

        # 自适应 R：窗口新息协方差匹配（R̂ = mean(ν²) − H P⁻ Hᵀ）
        if self.adapt_R and len(self._nu_buf) >= 10:
            w = self._nu_buf[-self.adapt_win:]
            s_hat = float(np.mean(np.square(w)))
            r_new = s_hat - (H @ P_pred @ H.T).ravel()[0]
            self._r_est = ((1 - self.adapt_alpha) * self._r_est
                           + self.adapt_alpha * max(r_new, 1e-6))
            self.R = np.array([[self._r_est]])
        self._nu_buf.append(float(nu))

        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)     # 卡尔曼增益

        self.x = x_pred + (K @ np.array([[nu]])).ravel()
        # Joseph 形式协方差更新（数值等价但保证 PSD，
        # 防真实数据上 P 失对称/负定 → 负增益 → 正反馈发散）
        IKH = np.eye(2) - K @ H
        self.P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T

        # 记录内部分量（P2-09 的 NN 特征来源）
        self.history["soc"].append(float(self.x[0]))
        self.history["v1"].append(float(self.x[1]))
        self.history["vt_pred"].append(float(z_pred))
        self.history["nu"].append(float(nu))
        self.history["S"].append(float(S[0, 0]))
        self.history["nis"].append(float(nu * nu / S[0, 0]))
        self.history["K0"].append(float(K[0, 0]))
        self.history["K1"].append(float(K[1, 0]))
        self.history["P00"].append(float(self.P[0, 0]))
        self.history["P11"].append(float(self.P[1, 1]))

        # 内部状态刻意不做 clamp：裁剪是非线性的，会破坏协方差一致性。
        # ⚠️ 因此本返回值（以及 history["soc"]）**不保证**落在 [0,1]。
        #    实测首步更新后即越界到约 1.10。机理：初值 soc0 被 clip 到 1.0，
        #    叠加一个偏大的首步新息，再经 K0 放大所致。
        #    （注：预测步 _f() **确实使用** x[0]，此处旧注释曾误写成"丢掉初值"。）
        #    越界量小：对同分布 RMSE 的影响 ≤0.022 pp。
        # 🔧 soc_clipped() 是留给消费者按需裁剪的工具，**当前无调用点**——
        #    所有下游（nn_features、评估脚本）用的都是未裁剪值。
        return float(self.x[0])

    def soc_clipped(self) -> float:
        """对外输出的 SOC，裁剪到 [0, 1]（当前无调用点，见 step() 的说明）。"""
        return float(np.clip(self.x[0], 0.0, 1.0))


# --------------------------------------------------------------------------- #
# 合成数据发生器（仅供算法自检，不得写入论文）
# --------------------------------------------------------------------------- #

def make_current_profile(
    n: int = 3600,
    dt: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """
    生成一条「类 US06」的电流曲线：恒流段 + 脉冲段 + 随机波动。

    目的只是让 EKF 面对一个不平凡的工况，不是真实工况复现。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt

    i_base = 0.4 + 0.3 * np.sin(2 * np.pi * t / 600.0)          # 慢变基底
    i_pulse = np.where((t % 120.0) < 20.0, 1.5, 0.0)            # 周期脉冲
    i_noise = 0.1 * rng.standard_normal(n)                      # 随机波动

    i = i_base + i_pulse + i_noise
    return np.clip(i, -2.0, 3.0)


def simulate_true_system(
    params: ECMParams,
    i_profile: np.ndarray,
    soc_init: float = 1.0,
    v_noise_std: float = 0.005,
    seed: int = 1,
) -> dict[str, np.ndarray]:
    """
    用真实参数正向仿真，产生「真值」与端电压量测。

    ⚠️ 合成数据，仅用于算法自检。
    """
    rng = np.random.default_rng(seed)
    n = len(i_profile)
    dt = params.dt
    a = np.exp(-dt / params.tau)

    soc = np.empty(n)
    v1 = np.empty(n)
    vt = np.empty(n)

    soc_k, v1_k = soc_init, 0.0
    for k in range(n):
        vt_k = float(ocv(soc_k)) - v1_k - i_profile[k] * params.R0
        vt[k] = vt_k + rng.normal(0.0, v_noise_std)

        soc[k], v1[k] = soc_k, v1_k

        soc_k = soc_k - (params.eta_c * dt / (3600.0 * params.Cn)) * i_profile[k]
        v1_k = a * v1_k + params.R1 * (1.0 - a) * i_profile[k]

    return {"soc_true": soc, "v1_true": v1, "vt_meas": vt}


def run_ekf(
    params: ECMParams,
    i_profile: np.ndarray,
    vt_meas: np.ndarray,
    soc0: float = 1.0,
    temp_profile: np.ndarray | None = None,
    **ekf_kwargs,
) -> SOC_EKF:
    """在整段数据上跑 EKF，返回滤波器（含 history）。

    temp_profile 给定且滤波器配置了 params_fn 时按实测温度插值参数。
    """
    ekf = SOC_EKF(params, soc0=soc0, **ekf_kwargs)
    if temp_profile is None:
        for i_k, vt_k in zip(i_profile, vt_meas):
            ekf.step(i_k, vt_k)
    else:
        for i_k, vt_k, tp_k in zip(i_profile, vt_meas, temp_profile):
            ekf.step(float(i_k), float(vt_k), temp_k=float(tp_k))
    return ekf


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #

def metrics(soc_est: np.ndarray, soc_true: np.ndarray) -> dict[str, float]:
    """SOC 估计精度指标（单位：百分点 %）。"""
    e = (np.asarray(soc_est) - np.asarray(soc_true)) * 100.0
    return {
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE": float(np.mean(np.abs(e))),
        "MAX": float(np.max(np.abs(e))),
    }
