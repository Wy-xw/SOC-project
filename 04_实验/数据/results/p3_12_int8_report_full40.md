# P3-12 全整数量化 —— 40 文件档（与 §6 同口径）实测报告

日期：2026-09-25
脚本：`10_代码/p3_12_int8_quant.py`（`--archive full40`，默认档）
模型：`gru16_A2-3_seed7 / seed13 / seed42`（16 特征组 A2-3，GRU(16) unroll=True，L=30）
评估档：`04_实验/数据/results/ekf_full_history.npz` —— **40 文件**
（test_id(25C) 4 + ood(10C) 9 + ood(0C) 9 + ood(-10C) 9 + ood(-20C) 9）

产物（全部新文件，未覆盖既有任何产物）：

| 文件 | 说明 |
|---|---|
| `04_实验/数据/results/p3_12_int8_results_full40.csv` | 主表：seed × group × 各档位 RMSE（宽表） |
| `04_实验/数据/results/p3_12_int8_metrics_full40.csv` | 长表：含 MAE / MAX / n_files / n_samples / 文件字节数 |
| `04_实验/数据/results/p3_12_int8_meta_full40.json` | 元信息：张量构成、量化参数、大小、失败记录、archive 标记 |
| `04_实验/quantized/*_full40.tflite` | 18 个模型（3 seed × 6 档），文件名加 `_full40` |

---

## 0. 结论（先讲结果）

1. **本报告取代 `p3_12_int8_report.md`（8 文件档）作为论文 §4.5 的证据来源。**
   8 文件档 = 40 文件档的**真子集**（4 个 25 °C Cycle 完全相同，OOD 由每组 1 文件扩到 9 文件）。
2. **§4.5 的论文数字不需要改。** 论文 §4.5 只引用两句话：
   test RMSE 0.62 → 0.64 pp（+3.3 %）、int8 205 kB vs float16 105 kB（1.9×）。
   这两组数字**在 40 文件档下逐位不变**——test_id(25C) 4 个文件两档就是同一批，
   模型字节也完全相同（见 §3 sha256 校验）。
3. **真正被修掉的是「证据链口径」**：旧 8 文件档下 §4.5 隐含的 OOD 数字
   （−20 °C 4.72 pp）与 §6 / Table 2 的 −20 °C 14.63 pp 差 3 倍。
   统一后两组数字出自**同一个 40 文件评估集**，审稿人可逐组对照。
4. int8 的精度结论不变且更干净：40 文件档下 **4 个 OOD 组全部略优于 float32**
   （−0.01 % ~ −0.70 %），test_id(25C) 退化 +3.29 %。

---

## 1. 原来怎么选文件（问题定位）

| 位置 | 旧行为 |
|---|---|
| `10_代码/p3_12_int8_quant.py:95-96`（改前） | `load_history(RESULTS/"ekf_baseline_history.npz")` → **8 文件档** |
| `10_代码/p3_12_int8_quant.py:102-111`（改前） | `ood(0C)` 用 `"0degC_Cycle" in c` → 子串陷阱 |
| `10_代码/p3_09_ptq.py:57` | 同样读 `ekf_baseline_history.npz`（8 文件档） |
| `10_代码/p3_09_ptq.py:63-72` | 同一套 `test_groups` 表达式（同样的子串陷阱） |
| 对照：`10_代码/p5_b2_multifile_ablation.py:45` | 读 `ekf_full_history.npz` → **40 文件档**（论文 §6 口径） |
| 对照：`10_代码/p5_b2_multifile_ablation.py:27-40` | 已修复的分组表达式（显式排除 10/20 °C） |

两个档位的文件构成（本报告实测枚举）：

| 档 | 总文件 | 25 °C | 10 °C | 0 °C | −10 °C | −20 °C |
|---|---|---|---|---|---|---|
| `ekf_baseline_history.npz` | 8 | 4 | 1 | 1 | 1 | 1 |
| `ekf_full_history.npz` | 40 | 4 | 9 | 9 | 9 | 9 |

**注意**：任务派发单上 baseline 档写的「10 °C = 2」与实测不符，实测 OOD 每组各 1 文件
（8 = 4 + 1 + 1 + 1 + 1）。这不影响结论，但引用时应以本表实测值为准。

**子串陷阱的量级**（为什么必须换表达式）：在 40 文件档上，旧写法
`"0degC_Cycle" in c` 会命中 `10degC_Cycle_1..4`、`n10degC_Cycle_1..4`、
`n20degC_Cycle_1..4`，把 **16 个文件**误划进 0 °C 组（正确为 9 个）。

---

## 2. 改成了什么

`10_代码/p3_12_int8_quant.py` 三处改动：

1. **评估档参数化**（新增 `ARCHIVES` / `DEFAULT_ARCHIVE`，`load_all(archive)`）：
   默认 `full40` → `ekf_full_history.npz`；保留 `--archive baseline8` 供回溯。
2. **分组表达式逐字对齐 `p5_b2_multifile_ablation.py:27-40`**（`test_groups`）：
   ```python
   "ood(10C)":  [c for c in ids if "10degC_" in c and "trise" not in c
                 and "n10degC" not in c and "n20degC" not in c],
   "ood(0C)":   [c for c in ids if "0degC_" in c and "trise" not in c
                 and "10degC" not in c and "20degC" not in c],
   "ood(-10C)": [c for c in ids if "n10degC_" in c and "trise" not in c],
   "ood(-20C)": [c for c in ids if "n20degC_" in c and "trise" not in c],
   ```
3. **产物改名隔离**（`OUT_SUFFIX` / `TFLITE_SUFFIX`）：40 文件档一律加 `_full40`，
   8 文件档保留原名。**原有 CSV / 报告 / tflite 一个都没动**（见 §6 产物清单）。

未改动：训练集 `ekf_train_history.npz`、`val(25C)` 定义（train 集 UDDS 文件）、
校准集取样（`RandomState(42).choice(n, 200)`）、评估口径（`BURN_IN=300`，逐文件 RMSE → 文件均值）。

---

## 3. 硬校验：新 float32 列 vs `p3_06b_multifile_agg.csv`

**校验 A —— 逐 seed（15/15 全部逐位吻合）**

参考值由 `p5_b2_perfile.csv` 按**文件名**重算分组（该 CSV 自带的 `eval_group` 列是
子串陷阱污染版，不能直接用，见 §5），脚本：`p5_b2_a24_paired.py:125` 亦记载此坑。

| seed | 分组 | 新 `keras_fp32` | 参考 RMSE | 差值 |
|---|---|---|---|---|
| 7 | test_id(25C) | 0.604726 | 0.604726 | 0 |
| 7 | ood(10C) | 3.191445 | 3.191445 | 0 |
| 7 | ood(0C) | 8.631808 | 8.631808 | 0 |
| 7 | ood(-10C) | 10.952947 | 10.952947 | 0 |
| 7 | ood(-20C) | 14.116489 | 14.116489 | 1.8e-15 |
| 13 | test_id(25C) | 0.621899 | 0.621899 | 0 |
| 13 | ood(10C) | 3.444671 | 3.444671 | 0 |
| 13 | ood(0C) | 9.009449 | 9.009449 | 0 |
| 13 | ood(-10C) | 11.641766 | 11.641766 | 0 |
| 13 | ood(-20C) | 15.081352 | 15.081352 | 0 |
| 42 | test_id(25C) | 0.635761 | 0.635761 | 0 |
| 42 | ood(10C) | 3.349718 | 3.349718 | 0 |
| 42 | ood(0C) | 8.830005 | 8.830005 | 0 |
| 42 | ood(-10C) | 11.391429 | 11.391429 | 0 |
| 42 | ood(-20C) | 14.706223 | 14.706223 | 0 |

**校验 B —— 3 seed 均值 vs `p3_06b_multifile_agg.csv`（A2-3）**

| 分组 | n_files | 新 `keras_fp32`（3 seed 均值） | `p3_06b_multifile_agg.csv` | 差值 |
|---|---|---|---|---|
| test_id(25C) | 4 | 0.620795 | 0.620795 | 0 |
| ood(10C) | 9 | 3.328611 | 3.328611 | 0 |
| ood(0C) | 9 | 8.823754 | 8.823754 | 0 |
| ood(-10C) | 9 | 11.328714 | 11.328714 | 3.6e-15 |
| ood(-20C) | 9 | 14.634688 | 14.634688 | 0 |

**校验 C —— 模型字节未变**：18 个 `_full40.tflite` 与其 8 文件档同名文件
**sha256 逐字节相同**（例：`gru16_A2-3_seed7_int8_full{,_full40}.tflite`
= `7173124edd25cc31...`，且与 legacy `gru16_A2-3_seed7_ptq_int8.tflite` 也相同）。
⇒ **本次唯一变量就是评估集**，量化转换路径未被扰动。

**结论：两批口径已完全对齐，无需如任务单预留的那样"如实报告差异"。**

---

## 4. 新表：40 文件档逐组 RMSE（pp，3 seed 均值，稳态，文件均值）

| 分组 | n_files | float32 (keras) | tflite_fp32 | float16 | int8 动态 | **int8 全整数** | int8(全,f32 io) | int16 激活 | legacy int8 |
|---|---|---|---|---|---|---|---|---|---|
| test_id(25C) | 4 | 0.6208 | 0.6208 | 0.6206 | 0.6208 | **0.6412** | 0.6412 | 1.5044 | 0.6213 |
| ood(10C) | 9 | 3.3286 | 3.3286 | 3.3287 | 3.3286 | **3.3054** | 3.3054 | 3.6377 | 3.1980 |
| ood(0C) | 9 | 8.8238 | 8.8238 | 8.8238 | 8.8238 | **8.8233** | 8.8233 | 9.0049 | 8.6729 |
| ood(-10C) | 9 | 11.3287 | 11.3287 | 11.3287 | 11.3287 | **11.2831** | 11.2831 | 11.1275 | 10.9685 |
| ood(-20C) | 9 | 14.6347 | 14.6347 | 14.6347 | 14.6347 | **14.5557** | 14.5557 | 14.2558 | 14.1333 |
| val(25C) | 1 | 1.0089 | 1.0089 | 1.0086 | 1.0089 | **1.0216** | 1.0216 | 1.0047 | 0.8937 |

（legacy / int16 列只有 seed 7 有值，这里是 seed 7 单值，不是 3 seed 均值。）

**int8 相对 float32 的变化**

| 分组 | float32 → int8 | 相对变化 |
|---|---|---|
| test_id(25C) | 0.6208 → 0.6412 | **+3.29 %** |
| ood(10C) | 3.3286 → 3.3054 | −0.70 % |
| ood(0C) | 8.8238 → 8.8233 | −0.01 % |
| ood(-10C) | 11.3287 → 11.2831 | −0.40 % |
| ood(-20C) | 14.6347 → 14.5557 | −0.54 % |
| val(25C) | 1.0089 → 1.0216 | +1.26 % |

**逐 seed（test_id(25C)，唯一退化的组）**

| seed | float32 | float16 | int8 全整数 | int8 退化 |
|---|---|---|---|---|
| 7 | 0.6047 | 0.6043 | 0.6213 | +2.74 % |
| 13 | 0.6219 | 0.6212 | 0.6366 | +2.36 % |
| 42 | 0.6358 | 0.6363 | 0.6657 | +4.71 % |

**读法**：与 8 文件档结论一致——int8 在最难的 in-distribution 组退化 +2.4 ~ +4.7 %，
全部 4 个 OOD 温度组略优于 float32。`int8_full` 与 `int8_full_f32io` 在全部
18 个 (group, seed) 组合上 RMSE 逐位相同（误差来自权重/激活量化，与输入输出 dtype 无关）。
`int16_act` 档仍显著异常（test_id 1.5044 pp，float32 的 2.4 倍），**不建议采用、不作为论文结论**。

### 4.1 新旧口径逐组对照（本报告的核心）

同为 A2-3、同 3 seed（7/13/42）、同一批 .keras 模型，**只换评估集**：

| 分组 | float32 8 文件档 | float32 40 文件档 | 比值 | int8 8 文件档 | int8 40 文件档 |
|---|---|---|---|---|---|
| test_id(25C) | 0.6208 | 0.6208 | **1.00×**（同一批 4 文件） | 0.6412 | 0.6412 |
| ood(10C) | 3.0280 | 3.3286 | 1.10× | 2.9977 | 3.3054 |
| ood(0C) | 9.6073 | 8.8238 | 0.92× | 9.6301 | 8.8233 |
| ood(-10C) | 6.7327 | 11.3287 | 1.68× | 6.6783 | 11.2831 |
| **ood(-20C)** | **4.7249** | **14.6347** | **3.10×** | 4.5445 | 14.5557 |
| val(25C) | 1.0089 | 1.0089 | 1.00× | 1.0216 | 1.0216 |

⇒ 这就是「§4.5 与 §6 数字对不上」的机制：**不是计算错误，是评估子集不同**。
旧 8 文件档每组 OOD 只有 1 个文件（如 −20 °C 只有 `n20degC_HWFET` 一个工况），
40 文件档每组 9 个文件（4 个 Cycle + 5 个标准工况），难度分布完全不同。

---

## 5. 8 文件档为何存在（是否可行统一）

**判定：40 文件档重跑完全可行，已直接统一。** 依据：

1. `p5_b2_multifile_ablation.py:53-61` 用**同一批** `gru16_A2-3_seed{7,13,42}.keras`
   在 `ekf_full_history.npz` 上评估，`p3_06b_multifile_agg.csv` 即由此产出 ⇒ 模型与 40 文件档**天然兼容**。
2. 40 文件档是 8 文件档的**严格超集**：4 个 25 °C Cycle 文件在两档中完全相同
   （逐组 test_id RMSE 0.620795 两档一致可证），额外 36 个 OOD 文件特征齐全
   （A2-3 的 13 个特征在全部 40 文件上均可构造，无缺字段、无 `None` 返回）。
3. 本次实测 3 seed × 6 档 × 5 组**全部成功**，无转换失败、无文件被跳过
   （`p3_12_int8_meta_full40.json` 的 `failures` 为空）。

**8 文件档的历史来源**：`p2_05_06_ekf_baseline.py` 时代的基线 EKF 只跑了
1 个代表工况/温度，`ekf_baseline_history.npz` 是其产物；`p3_09_ptq.py` 与
`p3_12_int8_quant.py` 因此继承了 8 文件档。而 `p5_00_full_history.py` 后来补跑了
全部 40 文件（`ekf_full_history.npz`）供 §6 多文件消融使用。**两批从此分叉，无人对齐。**

⇒ 不需要「在论文里显式标注 §4.5 用另一个评估子集」这一替代方案。
`--archive baseline8` 分支保留，仅供回溯旧报告用，**不再用于论文**。

**遗留观察（非本任务范围，建议单独立项）**

同一「8 文件 / 40 文件」分叉在项目里**不止 p3_12 一处**，共发现 3 个 8 文件档遗留产出：

| 产物 / 脚本 | 状态 |
|---|---|
| `p3_09_ptq.py:57` → `E:\SOC_LiteProject\quantized\*_ptq_compare.csv` | **仍读 8 文件档**，其 float32 基准列与 §6 不同口径；本任务未改动 |
| `10_代码/p3_06_ablation.py:52` → `p3_06_ablation_agg.csv`（09-19 20:19） | 8 文件档旧跑。其 A2-3 行（10C 3.028 / 0C 9.607 / −10C 6.733 / −20C **4.725**）与本次测得的 8 文件档数值**逐位一致**，确证同源 |
| `p3_04_train.py:62`、`p3_05_hparam_search.py:71`、`p3_10_qat.py:89` | 同样读 `ekf_baseline_history.npz`（其中 p3_10 QAT 未产出论文数字） |

`p3_06_ablation_agg.csv` 这条论文侧**已经处理过**：`05_论文/drafts/§6_results_discussion_EN_v1.md:310-311`
明确记载「原指针写 `p3_06_ablation_agg.csv`，那是**已作废的单文件/温度旧跑**（数值与 Table 3 差最多 3 倍）；
正确来源是 `p3_06b_multifile_agg.csv`」，`10_代码/p6_tables.py:21` 也确实读 40 文件档。
**⇒ §6 侧口径是对的；§4.5 侧（p3_12）是唯一还没对齐的，本次补齐。**

---

## 6. 对论文的具体影响

### 6.1 §4.5 正文数字：**不需要改**

论文 §4.5 现有的全部数字在 40 文件档下逐位成立：

| 论文 §4.5 原句 | 40 文件档实测 | 是否需要改 |
|---|---|---|
| test RMSE 0.62 → 0.64 pp，+3.3 % | 0.6208 → 0.6412，+3.29 % | **否** |
| "four of the five ... groups are marginally *better* under int8" | 4 个 OOD 组（10/0/−10/−20 °C）全部略优；test_id 是唯一变差组 | **否**（表述仍准确） |
| int8 205 kB vs float16 105 kB，1.9× | 209 632 B vs 107 888 B，1.94× | **否** |
| float16 精度损失 < 0.1 % | 见 §4 表：最大**相对**偏差 0.028 %（val 0.03 % / test_id −0.03 %） | **否** |

### 6.2 需要改的是**证据出处**（4 处文件引用）

这些位置目前指向 8 文件档产物，应改指 `_full40`：

| 位置 | 现文 | 应改为 |
|---|---|---|
| `05_论文/drafts/§4_proposed_method_EN_v1.md:258` | `p3_12_int8_results.csv` + `p3_12_int8_metrics.csv` | `p3_12_int8_results_full40.csv` + `p3_12_int8_metrics_full40.csv` |
| `05_论文/drafts/§5_experimental_setup_EN_v1.md:181` | 同上 | 同上 |
| `05_论文/drafts/§6_results_discussion_EN_v1.md:323` | `p3_12_int8_results.csv` | `p3_12_int8_results_full40.csv` |
| `05_论文/drafts/§0_abstract_title_EN_v1.md:141` | `p3_12_int8_results.csv` | `p3_12_int8_results_full40.csv` |

**建议加一句口径声明**（任何一处即可，例如 §4.5 末）：

> Quantization metrics are evaluated on the same 40-file cohort used for the
> ablation in Section 6 (4 in-distribution + 36 out-of-distribution files,
> 9 files per OOD temperature), so that the two sections are directly comparable.

### 6.3 本报告取代旧报告

`p3_12_int8_report.md`（8 文件档，2026-09-25 11:46）中 §2 主结果表的 OOD 四行
（3.0280 / 9.6073 / 6.7327 / 4.7249）**已过时**，应以本报告 §4 表为准。
旧报告其余章节（§3 模型大小、§4 「106 pp」取证、§5 legacy 文件来历）**结论仍然有效**，
因为那些都是模型侧事实，与评估集无关。
旧报告未删除、未改动，保留备查。

---

## 7. 产物清单与「未覆盖」核对

**新增**

| 文件 | 说明 |
|---|---|
| `p3_12_int8_results_full40.csv` | 主表（3 seed × 6 分组 × 8 档位） |
| `p3_12_int8_metrics_full40.csv` | 长表（含 n_files / n_samples / MAE / MAX / size_bytes） |
| `p3_12_int8_meta_full40.json` | 元信息（含 `archive: "full40"`、`n_files_total: 40`、张量构成、failures） |
| `04_实验/quantized/gru16_A2-3_seed{7,13,42}_{fp32,f16,int8_dynamic,int8_full,int8_full_f32io,int16act}_full40.tflite` | 18 个模型，与旧档 sha256 相同 |

**未覆盖（mtime 已核对未变）**

`p3_12_int8_results.csv`(11:44) / `p3_12_int8_metrics.csv`(11:44) /
`p3_12_int8_meta.json`(11:44) / `p3_12_int8_report.md`(11:46) /
`04_实验/quantized/*.tflite`（非 full40，11:37–11:42）/
`gru16_A2-3_seed7_ptq_int8.tflite`（09-19 22:06，legacy）/
`p3_06b_multifile_agg.csv`(09-23 22:41) / `p5_b2_perfile.csv`(09-23 22:40)。
`p3_09_ptq.py` 未改动。

---

## 8. 复现命令

```bash
# 40 文件档（默认，与 §6 同口径）
"D:/venvs/soc-paper/Scripts/python.exe" 10_代码/p3_12_int8_quant.py \
    --group A2-3 --all-seeds --archive full40

# 回溯旧 8 文件档（写回原名，慎用）
"D:/venvs/soc-paper/Scripts/python.exe" 10_代码/p3_12_int8_quant.py \
    --group A2-3 --all-seeds --archive baseline8
```

---

## 9. 未解决项

1. **`p3_09_ptq.py`（float16 PTQ）仍用 8 文件档**，其
   `E:\SOC_LiteProject\quantized\*_ptq_compare.csv` 的 float32 基准列与 §6 不同口径。
   本任务范围只到 `p3_12`，未改动该脚本 —— 若论文任何位置引用该 CSV 的 OOD 数字，需同样处理。
2. **`p5_b2_perfile.csv` 自带的 `eval_group` 列仍是子串陷阱污染版**
   （`ood(0C)` 含 36 文件）。`p3_06b_multifile_agg.csv` 用的是修复后分组、数值已独立复核正确，
   但 `p5_b2_perfile.csv` 本身未重生成。下游若直接读该列会重现错误
   （`p5_b2_a24_paired.py:125`、`10_代码/diag/p5_b2_paired_fixed.py` 均已绕开）。
3. **`p3_12_int8_report.md` 未加过时提示**。为避免动既有产物，未在其顶部插标记；
   建议由用户决定是否加一行指向本报告。
4. 全 int8 模型仍未在 STM32F407 + TFLite Micro / CMSIS-NN 上实测（论文已不走 MCU 方向，非阻塞）。
5. 论文草稿 4 处文件引用（§6.2 表）尚未改动 —— 本次只落盘实验产物，未编辑论文。
