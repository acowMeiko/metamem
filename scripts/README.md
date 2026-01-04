# Scripts 目录说明

本目录包含项目所有的 Bash 脚本，按功能分类如下：

---

## 📊 数据集处理脚本（DPO 数据生成）

这些脚本用于处理不同数据集，生成 DPO (Direct Preference Optimization) 训练数据。

### 单数据集处理
- **`run_bbh.sh`** - 处理 BBH 数据集（27个任务）
  - GPU: 0,1
  - 输出: `output/bbh/dpo_*.jsonl`
  - 每个子任务独立输出文件

- **`run_mmlu.sh`** - 处理 MMLU 数据集（4个文件）
  - GPU: 6,7
  - 输出: `output/mmlu/dpo_*.jsonl`
  - 每个子文件独立输出

- **`run_gsm8k.sh`** - 处理 GSM8K 数学数据集
  - GPU: 2,3
  - 输出: `output/gsm8k/dpo_gsm8k.jsonl`

- **`run_math.sh`** - 处理 MATH 数学数据集
  - GPU: 4,5
  - 输出: `output/math/dpo_math.jsonl`

- **`run_svamp.sh`** - 处理 SVAMP 数学应用题数据集
  - GPU: 4,5
  - 输出: `output/svamp/dpo_svamp.jsonl`

### 批量处理
- **`run_all_datasets.sh`** - 顺序处理所有5个数据集
  - 适用场景：单GPU或顺序执行
  - 执行顺序：BBH → MMLU → GSM8K → MATH → SVAMP

- **`run_parallel_all.sh`** - 并行处理所有5个数据集
  - 适用场景：多GPU并行加速
  - GPU分配：
    - BBH: 0,1
    - GSM8K: 2,3
    - MATH: 4,5
    - MMLU: 6,7
    - SVAMP: 与MATH共享4,5（MATH完成后启动）

- **`run_bbh_mmlu.sh`** - 仅处理 BBH 和 MMLU（最大数据集）
  - 适用场景：快速测试或重点处理

---

## 🧪 测试与验证脚本

- **`test_small_batch.sh`** - 小批量测试脚本
  - 功能：使用每个数据集的少量样本测试整个流程
  - 用途：验证配置、调试问题、快速迭代

- **`check_datasets.sh`** - 数据集文件检查脚本
  - 功能：验证所有数据集文件是否存在
  - 用途：部署前检查、故障排查

---

## 🚀 推理脚本

- **`infer.sh`** - Level1 数据集推理脚本
  - 功能：使用训练好的模型对 `level1.jsonl` 进行推理
  - GPU: 4,5
  - 输入: `data/dpo_by_level/level1.jsonl`
  - 输出: `output/local_inference_with_accuracy.json`
  - 日志: `logs/stage_infer_level1.log`

---

## 📁 目录结构约定

### 输入目录
- `dataset/` - 原始数据集
  - `dataset/bbh/` - BBH 任务文件
  - `dataset/mmlu/` - MMLU 数据文件
  - `dataset/gsm8k/` - GSM8K 训练/测试集
  - `dataset/math/` - MATH 测试集
  - `dataset/svamp/` - SVAMP 训练/测试集

### 输出目录
- `output/` - 所有生成结果的根目录
  - `output/bbh/` - BBH 生成的 DPO 数据
  - `output/mmlu/` - MMLU 生成的 DPO 数据
  - `output/gsm8k/` - GSM8K 生成的 DPO 数据
  - `output/math/` - MATH 生成的 DPO 数据
  - `output/svamp/` - SVAMP 生成的 DPO 数据
  - `output/*.json` - 推理结果与统计数据

### 日志目录
- `logs/` - 所有运行日志
  - `logs/bbh_*.log` - BBH 处理日志
  - `logs/mmlu_*.log` - MMLU 处理日志
  - `logs/stage_infer_*.log` - 推理日志
  - `logs/stage_first.log` - DPO 生成主日志

---

## 🎯 使用建议

### 首次运行
1. 检查数据集：`bash scripts/check_datasets.sh`
2. 小批量测试：`bash scripts/test_small_batch.sh`
3. 单数据集试运行：`bash scripts/run_bbh.sh`

### 生产环境
- **单GPU机器**：`bash scripts/run_all_datasets.sh`
- **多GPU机器**：`bash scripts/run_parallel_all.sh`

### 重点数据集
- **BBH + MMLU**：`bash scripts/run_bbh_mmlu.sh`

### 推理评估
- **Level1 推理**：`bash scripts/infer.sh`

---

## ⚙️ GPU 分配策略

| 数据集 | GPU | 说明 |
|--------|-----|------|
| BBH    | 0,1 | 27个任务，数据量大 |
| GSM8K  | 2,3 | 中等数据量 |
| MATH   | 4,5 | 中等数据量 |
| MMLU   | 6,7 | 4个文件，数据量大 |
| SVAMP  | 4,5 | 小数据量，与MATH共享 |

**注意**：并行运行时确保 GPU 分配不重叠，避免显存冲突。

---

## 📝 日志与输出

### 查看实时日志
```bash
# 查看特定数据集日志
tail -f logs/bbh_*.log
tail -f logs/mmlu_*.log

# 查看并行运行日志
tail -f run_bbh_parallel.out
tail -f run_mmlu_parallel.out
```

### 检查输出结果
```bash
# 查看生成的 DPO 文件
ls -lh output/bbh/
ls -lh output/mmlu/

# 统计生成数据量
wc -l output/*/dpo_*.jsonl
```

---

## 🔧 故障排查

### 常见问题
1. **显存不足**：减少 `BATCH_SIZE` 或增加 GPU 数量
2. **进程残留**：检查并杀死旧的 vLLM 进程
3. **输出为空**：检查日志中的 ERROR 信息
4. **GPU 冲突**：确保并行脚本的 GPU 分配不重叠

### 清理与重启
```bash
# 查看 GPU 使用情况
nvidia-smi

# 杀死残留进程
pkill -f vllm
pkill -f stage_first

# 清理输出（谨慎操作）
rm -rf output/*/dpo_*.jsonl
```

---

## 📌 版本信息

- **创建日期**：2025-12-27
- **最后更新**：2025-12-27
- **适用版本**：metanew2 v1.0
