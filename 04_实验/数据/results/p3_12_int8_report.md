# P3-12 全整数量化（Full Integer Quantization）实测报告

日期：2026-09-25
脚本：`10_代码/p3_12_int8_quant.py`
模型：`gru16_A2-3_seed7 / seed13 / seed42`（与论文头条一致，16 特征组 A2-3，L=30）
产物：`04_实验/quantized/`、`04_实验/数据/results/p3_12_int8_*.csv|json`

---

## 0. 结论（先讲结果）

1. **全 int8 量化没有失败。** 转换成功、可推理、精度几乎无损。
2. **论文 §4.4 的「int8 把 test RMSE 从 0.6 pp 抬到 ≈106 pp」被证伪。**
   三 seed 实测 test_id(25C)：float32 **0.6208 pp** → int8 **0.6412 pp**
   （+3.3 %，绝对值 +0.02 pp）。**不是 106 pp 量级，差 170 倍。**
3. **项目内那个 204.7 KiB 的 `gru16_A2-3_seed7_ptq_int8.tflite` 是真的 int8 模型**，
   而且它的**精度就是 0.62 pp**——与本次新转的模型**字节完全相同**（sha256 一致）。
   它是被遗忘的真结果，不是失败产物。
4. 因此论文的量化叙事需要改写：不再是「int8 失败所以退到 float16」，
   而应是「int8 可行且几乎无损，代价是模型文件反而更大（见 §4）」。

---

## 1. 方法与配置（与 p3_09 同一口径）

| 项 | 设置 |
|---|---|
| 基准模型 | `gru16_{group}_seed{seed}.keras`（GRU(16, unroll=True)+Dense(8)+Dense(1)，1 633 参数） |
| 校准集 | **真实训练集输入特征**，`RandomState(42).choice(n, 200, replace=False)`，与 `p3_09_ptq.py` 取样方式完全一致（不是随机数） |
| 全 int8 转换 | `Optimize.DEFAULT` + `representative_dataset` + `supported_ops=[TFLITE_BUILTINS_INT8]` + `inference_input_type=tf.int8` + `inference_output_type=tf.int8` |
| 评估口径 | 稳态段（前 300 s 收敛期剔除，`nn_features.BURN_IN`），逐文件 RMSE → 分组取文件均值 |
| 评估分组 | test_id(25C) 4 文件 / ood(10C) / ood(0C) / ood(-10C) / ood(-20C) / val(25C) |
| 推理 | 逐样本 `set_tensor`+`invoke`；输入按模型自带 scale/zero_point 量化，输出反量化 |
| 载入方式 | `Interpreter(model_content=<bytes>)` —— TFLite C++ 层打不开含中文的路径 |

---

## 2. 主结果：逐组 RMSE（pp，3 seed 均值）

| 分组 | float32 (keras) | float16 | int8 动态范围 | **int8 全整数量化** | int8 相对 float32 |
|---|---|---|---|---|---|
| test_id(25C) | 0.6208 | 0.6206 | 0.6208 | **0.6412** | **+3.29 %** |
| ood(10C) | 3.0280 | 3.0281 | 3.0280 | **2.9977** | −1.00 % |
| ood(0C) | 9.6073 | 9.6074 | 9.6073 | **9.6301** | +0.24 % |
| ood(-10C) | 6.7327 | 6.7327 | 6.7327 | **6.6783** | −0.81 % |
| ood(-20C) | 4.7249 | 4.7249 | 4.7249 | **4.5445** | −3.82 % |
| val(25C) | 1.0089 | 1.0086 | 1.0089 | **1.0216** | +1.26 % |

逐 seed（test_id(25C)）：

| seed | float32 | float16 | int8 全整数 | int8 退化 |
|---|---|---|---|---|
| 7 | 0.6047 | 0.6043 | 0.6213 | +2.74 % |
| 13 | 0.6219 | 0.6212 | 0.6366 | +2.36 % |
| 42 | 0.6358 | 0.6363 | 0.6657 | +4.71 % |

**读法**：全 int8 在最难的 in-distribution 组上退化 +2.4 ~ +4.7 %，
在四个 OOD 温度组上**有的还略好于 float32**（量化噪声相当于正则）。
最坏情况是 test_id 上的 +0.04 pp 绝对值。

参考：`int8_full` 与 `int8_full_f32io`（全 int8 但输入输出保持 float32）
在全部 18 个 (group, seed) 组合上 RMSE **逐位相同**，说明误差来自权重/激活量化本身，
与输入输出是否 int8 无关。

### 附：int16 激活档（异常，如实记录）

`EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8` 反而比 int8 差得多：
test_id 3 seed 均值 **1.5044 pp**（float32 的 2.4 倍），ood(0C) 10.39 vs 9.61。
但 ood(-20C) 4.41 vs 4.72 又更好。**该档位不建议采用**，
机制未查明（推测与 30 步 unroll 的 GRU 循环状态在 int16 逐张量 scale 下的动力学改变有关）。
不作为论文结论使用。

---

## 3. 模型大小与张量构成（seed 7）

| 档位 | 文件 | 大小 | 张量构成 | 输入/输出 |
|---|---|---|---|---|
| float32 | `gru16_A2-3_seed7_fp32.tflite` | 109 856 B (107.3 KiB) | 615 float32 + 12 int32 | fp32/fp32 |
| float16 | `gru16_A2-3_seed7_f16.tflite` | 107 888 B (105.4 KiB) | 615 fp32 + 10 fp16 + 12 int32 | fp32/fp32 |
| 动态范围 | `gru16_A2-3_seed7_int8_dynamic.tflite` | 109 856 B (107.3 KiB) | 与 float32 完全相同 | fp32/fp32 |
| **全 int8** | `gru16_A2-3_seed7_int8_full.tflite` | **209 632 B (204.7 KiB)** | **612 int8 + 74 int32** | **int8/int8** |
| 全 int8 (fp32 io) | `gru16_A2-3_seed7_int8_full_f32io.tflite` | 209 968 B (205.0 KiB) | 612 int8 + 2 fp32 + 74 int32 | fp32/fp32 |
| int16 激活 | `gru16_A2-3_seed7_int16act.tflite` | 143 152 B (139.8 KiB) | 607 int16 + 4 int64 + 4 int8 + 12 int32 | fp32/fp32 |

三个必须说清的事实：

1. **「int8 动态范围量化」这一档在 TFLite 里等价于 float32**：
   本模型只跑一次 `invoke`、没有多次调用，动态量化省不到任何东西，
   产物与 float32 逐字节同尺寸、RMSE 逐位相同。**它不是有效选项。**
2. **全 int8 的产物比 float16 大 1.9 倍**（204.7 KiB vs 105.4 KiB）。
   原因：`GRU(unroll=True)` 把 30 个时间步展开，权重张量在图上被复制 30 份，
   而 int8 权重**不能被去重共享**，于是 612 个 int8 张量的缓冲 + flatbuffer 元数据
   反而超过 float16 的权重压缩收益。**在 flash 占用这一点上 int8 并不优于 float16。**
3. 「1.8 k 参数 × 2 B = 3.6 KiB」与「105 KiB」之间的差额是展开图与元数据，
   与量化档位无关（评审批注 R2-M7 的质疑成立）。

---

## 4. 「106 pp」取证结论：证伪，且无法复现

### 4.1 事实链

- 「106 pp」在全项目的唯一出处是 `10_代码/p3_09_ptq.py:131` 的代码注释与
  `02_日志/2026-09-19.md:89` 的散文。**没有任何 CSV、日志或产物记录过这个数字。**
- `p3_09_ptq.py::ptq_convert` 实际做的是 float16（`supported_types=[tf.float16]`），
  文件名却写成 `_ptq_int8.tflite`。
- 本次真跑全 int8：**3 seed × 6 分组全部成功，最差退化 +4.71 %。**

### 4.2 复现尝试（均未得到 106 pp）

| 尝试 | 结果 |
|---|---|
| 真实特征校准（200 样本，本次主实验） | test_id **0.6213 pp** |
| 随机均匀 [0,1] 校准 | 0.609 pp（首文件前 3 000 样本） |
| 标准正态校准 | 1.019 pp |
| 均匀 [−5,5] 校准 | 1.144 pp |
| 2026-09-18 那一代模型 `gru16_A2-3_L30` | float32 0.6001 → int8 **0.5972 pp**（反而更好） |
| 把未量化 float32 直接塞进 int8 输入模型（旧 `evaluate_tflite` 的写法） | **直接抛异常**：`ValueError: Cannot set tensor: Got value of type FLOAT32 but expected type INT8` —— 不会静默产生数字 |
| 故意不做输出反量化（把原始 int8 码当预测值） | test_id 69.6 / ood(10C) 91.5 / ood(0C) 76.6 / ood(−10C) 110.6 / ood(−20C) 110.4 pp |

**最后一行是唯一能凑出「100 pp 量级」的路径**：如果评估代码忘记写
`(o − zero_point) × scale`，误差就会跳到 70–110 pp。
但 test_id 组只能得到 69.6 pp，**与 106 不一致**。

**结论：106 pp 无法被任何已尝试的配置复现。它很可能是早期项目状态下的一次
测量/记录失误（不做输出反量化是最接近的机制，但未被证实）。论文不应继续引用它。**

---

## 5. 那个 204.7 KiB 模型的来历与精度

**来历（已查清）**

- 路径：`04_实验/quantized/gru16_A2-3_seed7_ptq_int8.tflite`
- 大小 209 632 B，mtime **2026-09-19 22:06**（早于 `p3_09_ptq.py` 的最后修改时间 22:09，
  也早于 `E:\SOC_LiteProject\` 里 float16 产物的 22:11）
- 全项目**没有任何脚本写这个路径**（`grep -rn "quantized"` 只命中 `E:\SOC_LiteProject\`），
  也没有对应的 .pyc 残留

**决定性证据**：本次 P3-12 用「A2-3 seed7 + 真实特征 200 样本校准 +
`TFLITE_BUILTINS_INT8` + int8 输入输出」转出来的新文件
`04_实验/quantized/gru16_A2-3_seed7_int8_full.tflite`，
与这个 legacy 文件

```
sha256 = 7173124edd25cc313d4ee1fc...   （两者完全相同）
```

**逐字节相同**（大小 209 632 B 也相同，输入量化参数 (0.039213091135025024, 0)、
输出量化参数 (0.03693313151597977, 103) 完全一致，同为 612 个 int8 张量 + 74 个 int32 张量）。

**判定**：它就是 2026-09-19 那次 int8 尝试的**真实产物**——用与本次 P3-12 完全相同的
转换配置跑出来的全 int8 模型，之后 `p3_09_ptq.py` 被改成 float16、输出目录被换到
`E:\SOC_LiteProject\`，这个文件就被落在了项目里、无人引用。它不是失败品，是**被遗忘的正确结果**。

**精度（本次实测，全 6 分组，seed 7）**

| 分组 | legacy int8 RMSE (pp) | 同模型 float32 (pp) |
|---|---|---|
| test_id(25C) | **0.6213** | 0.6047 |
| ood(10C) | 3.0536 | 3.0537 |
| ood(0C) | 9.7646 | 9.7100 |
| ood(-10C) | 6.5633 | 6.5331 |
| ood(-20C) | 4.3167 | 4.3119 |
| val(25C) | 0.8937 | 0.8922 |

**该文件已保留原样，未被覆盖。**（新产物一律用新文件名）

---

## 6. 对论文的具体影响

| 位置 | 现状 | 建议 |
|---|---|---|
| 摘要 | "after full int8 quantization fails at this size" | 删掉「失败」，改为「int8 与 float16 均几乎无损；float16 体积更小故被采用」 |
| §1 贡献句 | "full int8 quantization failing" 作为 negative result | 负结果不存在，改为体积-精度权衡 |
| §4.4 | "inflating test RMSE from 0.6 pp to ≈106 pp" | 改为实测：test_id 0.621 → 0.641 pp（3 seed，+3.3 %）；并说明 unroll 展开导致 int8 产物反而更大（204.7 KiB vs float16 105.4 KiB） |
| §5.4 | "Full int8 ... (0.6 → ≈106 pp); retained as a reported negative result" | 同上改写 |
| R1/R2/R3 审稿意见 B3 | 「量化证据链断裂」 | 本报告 + `p3_12_int8_*.csv` 即为补上的证据链 |
| `p3_09_ptq.py` 注释 L130-133、`_ptq_int8` 文件名 | 与代码实际行为不符 | 建议改注释（未改动，避免影响已有产物） |

---

## 7. 产物清单

| 文件 | 说明 |
|---|---|
| `04_实验/数据/results/p3_12_int8_results.csv` | 主表：seed × group × 各档位 RMSE（宽表） |
| `04_实验/数据/results/p3_12_int8_metrics.csv` | 长表：含 MAE / MAX / 文件数 / 样本数 / 文件字节数 |
| `04_实验/数据/results/p3_12_int8_meta.json` | 机器可读元信息：张量构成、量化参数、大小、失败记录 |
| `04_实验/quantized/gru16_A2-3_seed7_int8_full.tflite` | 全 int8（int8 输入输出），204.7 KiB |
| `04_实验/quantized/gru16_A2-3_seed7_int8_full_f32io.tflite` | 全 int8，float32 输入输出，205.0 KiB |
| `04_实验/quantized/gru16_A2-3_seed7_f16.tflite` | float16，105.4 KiB |
| `04_实验/quantized/gru16_A2-3_seed7_fp32.tflite` | 无优化 float32，107.3 KiB |
| `04_实验/quantized/gru16_A2-3_seed7_int8_dynamic.tflite` | 动态范围（实测等价 float32） |
| `04_实验/quantized/gru16_A2-3_seed7_int16act.tflite` | int16 激活（异常，不建议采用） |
| `04_实验/quantized/gru16_A2-3_seed7_{f16,int8_full,int8_full_f32io,int16act}.tflite`（seed13/42） | 同上，其余 seed |
| `04_实验/quantized/gru16_A2-3_seed7_ptq_int8.tflite` | **原有 legacy 文件，保持原样未动** |

未覆盖/未删除：`p3_09_ptq.py`、`p3_11_quant_compare.py` 及其在 `E:\SOC_LiteProject\quantized\` 的全部产物。

## 8. 复现命令

```bash
"D:/venvs/soc-paper/Scripts/python.exe" 10_代码/p3_12_int8_quant.py --group A2-3 --all-seeds
```

单 seed：`--seed 7`；只转换不评估：`--skip-eval`；改校准集大小：`--n-calib 500`。

---

## 9. 未解决项

1. **106 pp 的确切来源仍未证实**（无 CSV/日志/脚本证据）。最接近的可复现机制是
   「输出未反量化」得到 70–110 pp 量级，但 test_id 具体值对不上。建议论文直接删除该数字。
2. **int16 激活档为何显著变差**未查明，本报告只做记录，不做解释性主张。
3. 全 int8 模型尚未在 STM32F407 + TFLite Micro / CMSIS-NN 上实测
   （flash / RAM / 延迟 / 功耗四项仍是 §6.5 的待测项）。
4. `04_实验/quantized/` 里 legacy 文件与新产物 sha256 相同、内容重复，
   为避免动用户已有数据**未做删除**，建议由用户决定是否合并/改名。
