# 数据探索性分析报告（McMaster 18650PF）

- 数据文件：`04_实验/数据/processed/mcmaster_drive_cycles.csv.gz`
- 总采样点：**531,570**（1 Hz 重采样）
- 工况文件：**55** 个；排除合并文件 8 个

## 1. 划分总览

| 集合 | 文件数 | 采样点 | 时长 (h) | 温度 (标称) | 工况 |
|---|---|---|---|---|---|
| `train` | 5 | 45,868 | 12.7 | +25 °C | hwfet, la92, nn, us06 |
| `val` | 1 | 22,448 | 6.2 | +25 °C | udds |
| `test_id` | 4 | 44,504 | 12.4 | +25 °C | cycle |
| `test_ood` | 36 | 348,471 | 96.8 | -20/-10/+0/+10 °C | cycle, hwfet, la92, nn, udds, us06 |
| `test_ood_trise` | 9 | 70,279 | 19.5 | -20/+10 °C | cycle, us06 |

## 2. SOC 真值覆盖（决定模型在哪些区间被训练/评估）

| 集合 | SOC min | SOC max | SOC<0.2 占比 | 0.2–0.8 占比 | SOC>0.8 占比 |
|---|---|---|---|---|---|
| `train` | 0.066 | 1.000 | 13.0% | 64.4% | 22.6% |
| `val` | 0.068 | 1.000 | 13.6% | 64.2% | 22.3% |
| `test_id` | 0.035 | 1.000 | 14.6% | 63.4% | 22.0% |
| `test_ood` | 0.100 | 1.000 | 3.2% | 52.0% | 44.7% |
| `test_ood_trise` | 0.200 | 1.000 | 3.8% | 71.7% | 24.5% |

## 3. 电流分布

约定：**放电为正**。负电流 = 回馈制动（regen）。README 说明 10 °C 以下无 regen。

| 集合 | I 均值 | I 最大(放电) | I 最小(充电/回馈) | 回馈占比 | 活跃驾驶占比 | P99 |
|---|---|---|---|---|---|---|
| `train` | +1.031 | 19.80 | -9.43 | 16.4% | 93.1% | 7.98 |
| `val` | +0.431 | 7.33 | -4.11 | 19.8% | 94.6% | 3.54 |
| `test_id` | +0.867 | 18.85 | -10.28 | 18.1% | 93.0% | 7.33 |
| `test_ood` | +0.793 | 21.00 | -9.28 | 5.0% | 61.4% | 7.05 |
| `test_ood_trise` | +1.065 | 13.85 | -9.29 | 10.2% | 84.7% | 7.88 |

> **活跃驾驶占比** = |I| > 0.05 A 的采样点比例。
> `test_ood` 内部差异极大：最低 21.3%，最高 93.7%。
> ⚠️ 这意味着**温度外推集与训练集不止温度不同**，还同时存在
> ① 回馈制动缺失（≤0 °C 为 0%）② 激励不足（活跃占比低至 21%）。
> 详见 `README_数据说明.md` 第八节第 2 条。

## 4. 实测电池温度（`Battery_Temp_degC`，电芯中部热电偶）

| 集合 | min | 均值 | max | 组内标准差(中位) |
|---|---|---|---|---|
| `train` | 25.4 | 27.1 | 32.8 | 0.66 |
| `val` | 25.6 | 26.0 | 27.1 | 0.23 |
| `test_id` | 21.8 | 26.7 | 30.0 | 0.88 |
| `test_ood` | -20.3 | -0.4 | 23.9 | 2.39 |
| `test_ood_trise` | -20.1 | 11.0 | 27.3 | 7.14 |

## 5. 数据质量检查

- `voltage_V`：缺失 0 个
- `current_A`：缺失 0 个
- `temperature_C`：缺失 0 个
- `soc_true`：缺失 0 个
- 电压越界（<2.0 V 或 >4.25 V）：0 个
- `chamber_temp_C` 缺失率：50.8%（该通道在部分文件全为 NaN，**不作为特征**）

## 6. 各工况文件速查

| 集合 | 文件 | 时长 s | SOC 起→止 | T min→max | I max |
|---|---|---|---|---|---|
| test_id | 25degC_Cycle_1_Pan18650PF | 10983 | 1.00→0.07 | 22→30 | 18.9 |
| test_id | 25degC_Cycle_2_Pan18650PF | 11147 | 1.00→0.07 | 26→29 | 18.0 |
| test_id | 25degC_Cycle_3_Pan18650PF | 10264 | 1.00→0.13 | 25→29 | 17.1 |
| test_id | 25degC_Cycle_4_Pan18650PF | 12106 | 1.00→0.04 | 25→29 | 17.5 |
| test_ood | n20degC_Cycle_1_Pan18650PF | 5080 | 1.00→0.40 | -20→-3 | 14.8 |
| test_ood | n20degC_Cycle_2_Pan18650PF | 5046 | 1.00→0.40 | -20→-4 | 14.1 |
| test_ood | n20degC_Cycle_3_Pan18650PF | 5023 | 1.00→0.40 | -20→-7 | 15.4 |
| test_ood | n20degC_Cycle_4_Pan18650PF | 5043 | 1.00→0.40 | -20→-5 | 14.2 |
| test_ood | n20degC_HWFET_Pan18650PF | 11370 | 1.00→0.40 | -20→16 | 5.8 |
| test_ood | n20degC_LA92_Pan18650PF | 12849 | 1.00→0.40 | -20→16 | 10.1 |
| test_ood | n20degC_NN_Pan18650PF | 11678 | 1.00→0.40 | -20→17 | 15.3 |
| test_ood | n20degC_UDDS_Pan18650PF | 16081 | 1.00→0.40 | -20→16 | 7.3 |
| test_ood | n20degC_US06_Pan18650PF | 2661 | 1.00→0.40 | -20→-0 | 12.4 |
| test_ood | n10degC_Cycle_1_Pan18650PF | 6034 | 1.00→0.30 | -10→1 | 14.2 |
| test_ood | n10degC_Cycle_2_Pan18650PF | 5982 | 1.00→0.30 | -10→0 | 14.0 |
| test_ood | n10degC_Cycle_3_Pan18650PF | 5696 | 1.00→0.30 | -10→3 | 13.4 |
| test_ood | n10degC_Cycle_4_Pan18650PF | 6119 | 1.00→0.30 | -10→-0 | 14.4 |
| test_ood | n10degC_HWFET_Pan18650PF | 12279 | 1.00→0.30 | -10→17 | 5.2 |
| test_ood | n10degC_LA92_Pan18650PF | 14093 | 1.00→0.30 | -10→17 | 9.9 |
| test_ood | n10degC_NN_Pan18650PF | 5267 | 1.00→0.30 | -10→1 | 14.9 |
| test_ood | n10degC_UDDS_Pan18650PF | 18114 | 1.00→0.30 | -10→17 | 7.1 |
| test_ood | n10degC_US06_Pan18650PF | 10257 | 1.00→0.30 | -10→17 | 13.0 |
| test_ood | 0degC_Cycle_1_Pan18650PF | 8815 | 1.00→0.10 | 0→8 | 12.4 |
| test_ood | 0degC_Cycle_2_Pan18650PF | 8388 | 1.00→0.10 | 1→9 | 12.4 |
| test_ood | 0degC_Cycle_3_Pan18650PF | 6259 | 1.00→0.20 | 1→13 | 12.6 |
| test_ood | 0degC_Cycle_4_Pan18650PF | 7717 | 1.00→0.20 | 1→7 | 12.8 |
| test_ood | 0degC_HWFET_Pan18650PF | 5998 | 1.00→0.20 | 1→6 | 5.9 |
| test_ood | 0degC_LA92_Pan18650PF | 15405 | 1.00→0.20 | 0→18 | 9.4 |
| test_ood | 0degC_NN_Pan18650PF | 13476 | 1.00→0.20 | 0→18 | 15.1 |
| test_ood | 0degC_UDDS_Pan18650PF | 12868 | 1.00→0.20 | 1→3 | 6.9 |
| test_ood | 0degC_US06_Pan18650PF | 3672 | 1.00→0.20 | 1→14 | 13.4 |
| test_ood | 10degC_Cycle_1_Pan18650PF | 9395 | 1.00→0.24 | 11→15 | 17.3 |
| test_ood | 10degC_Cycle_2_Pan18650PF | 8123 | 1.00→0.27 | 11→17 | 21.0 |
| test_ood | 10degC_Cycle_3_Pan18650PF | 10097 | 1.00→0.12 | 11→16 | 18.1 |
| test_ood | 10degC_Cycle_4_Pan18650PF | 9917 | 1.00→0.14 | 11→16 | 19.3 |
| test_ood | 10degC_HWFET_Pan18650PF | 10591 | 1.00→0.12 | 11→24 | 5.3 |
| test_ood | 10degC_LA92_Pan18650PF | 16145 | 1.00→0.18 | 10→24 | 10.1 |
| test_ood | 10degC_NN_Pan18650PF | 14078 | 1.00→0.19 | 11→23 | 15.0 |
| test_ood | 10degC_UDDS_Pan18650PF | 24609 | 1.00→0.11 | 11→24 | 6.6 |
| test_ood | 10degC_US06_Pan18650PF | 4210 | 1.00→0.21 | 11→19 | 19.8 |
| test_ood_trise | n20degC_trise_Cycle_1_Pan18650PF | 6947 | 1.00→0.20 | -20→13 | 13.5 |
| test_ood_trise | n20degC_trise_Cycle_2_Pan18650PF | 6967 | 1.00→0.20 | -20→12 | 13.5 |
| test_ood_trise | n20degC_trise_Cycle_3_Pan18650PF | 6088 | 1.00→0.20 | -20→17 | 13.9 |
| test_ood_trise | n20degC_trise_Cycle_4_Pan18650PF | 7595 | 1.00→0.20 | -20→13 | 13.5 |
| test_ood_trise | n20degC_trise_US06_Pan18650PF | 3533 | 1.00→0.20 | -20→13 | 13.4 |
| test_ood_trise | 10degC_trise_Cycle_1_Pan18650PF | 9818 | 1.00→0.20 | 10→27 | 12.7 |
| test_ood_trise | 10degC_trise_Cycle_2_Pan18650PF | 9834 | 1.00→0.20 | 10→25 | 12.7 |
| test_ood_trise | 10degC_trise_Cycle_3_Pan18650PF | 9826 | 1.00→0.20 | 10→24 | 12.7 |
| test_ood_trise | 10degC_trise_Cycle_4_Pan18650PF | 9662 | 1.00→0.20 | 10→25 | 13.3 |
| train | 25degC_HWFTa_Pan18650PF | 7612 | 1.00→0.07 | 26→30 | 5.5 |
| train | 25degC_HWFTb_Pan18650PF | 7597 | 1.00→0.07 | 26→30 | 5.6 |
| train | 25degC_LA92_Pan18650PF | 14103 | 1.00→0.11 | 26→28 | 9.8 |
| train | 25degC_NN_Pan18650PF | 11733 | 1.00→0.12 | 25→30 | 14.6 |
| train | 25degC_US06_Pan18650PF | 4818 | 1.00→0.11 | 26→33 | 19.8 |
| val | 25degC_UDDS_Pan18650PF | 22447 | 1.00→0.07 | 26→27 | 7.3 |
