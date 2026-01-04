# GPU 配置说明 - 2张80G A800

## 🚀 快速开始

### **方式一：单独处理一个数据集（推荐调试）**

```powershell
# 处理 BBH 的 boolean_expressions 任务
.\run_single_dataset.ps1 -DatasetName "bbh" -DatasetFile "boolean_expressions"

# 处理 MMLU 的 test 文件
.\run_single_dataset.ps1 -DatasetName "mmlu" -DatasetFile "test"
```

### **方式二：批量处理所有数据集（自动化）**

```powershell
# 自动处理所有 BBH(27个) + MMLU(4个) 数据集
.\batch_run_datasets.ps1
```

### **方式三：自定义 GPU 和参数**

```powershell
# 使用 GPU 5,6，调整批次大小
.\run_single_dataset.ps1 `
    -DatasetName "bbh" `
    -DatasetFile "boolean_expressions" `
    -GPU "5,6" `
    -BatchSize 96 `
    -MaxWorkers 25
```

---

## ⚙️ 2张A800 GPU 优化参数

### **当前配置（已优化）**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **CUDA_VISIBLE_DEVICES** | `3,4` | 使用 GPU 3 和 4 |
| **BATCH_SIZE** | `128` | vLLM批处理大小（2张80G A800可支持） |
| **MAX_WORKERS** | `30` | API并发线程数（2张GPU并行优化） |
| **tensor_parallel_size** | `2` | vLLM张量并行（固定2张GPU） |
| **gpu_memory_utilization** | `0.9` | 显存利用率（80G大显存可提高） |

### **为什么这样配置？**

1. **BATCH_SIZE = 128**
   - 单张A800 (80GB): 可处理 batch_size 64-96
   - 2张A800并行: **可提升至 128-192**
   - 推荐 128（稳定且高效）

2. **MAX_WORKERS = 30**
   - API调用与GPU推理并行
   - 2张GPU可同时处理更多请求
   - 避免过高导致API限流

3. **tensor_parallel_size = 2**
   - 固定值，对应2张GPU
   - 自动在 `inference/local_inference.py` 中配置

---

## 📊 性能预估

### **单个数据集处理时间**

| 数据集 | 数据量 | 预计耗时 (2×A800) |
|--------|--------|-------------------|
| BBH 单任务 | ~100-250条 | 5-15分钟 |
| MMLU 单文件 | ~500-3000条 | 15-60分钟 |
| GSM8K | ~7473条 | 1-2小时 |
| MATH | ~5000条 | 1-1.5小时 |

### **批量处理总耗时**

- **BBH 全部27任务**: ~3-6小时
- **MMLU 全部4文件**: ~1-3小时
- **总计**: 约 **4-9小时**

---

## 🔧 根据实际情况调整参数

### **如果显存不足 (OOM)**

```powershell
# 降低批次大小和显存利用率
$env:BATCH_SIZE = "64"
# 修改 inference/local_inference.py 中的 gpu_memory_utilization=0.8
```

### **如果API频繁超时**

```powershell
# 降低并发数
.\run_single_dataset.ps1 -DatasetName "bbh" -DatasetFile "xxx" -MaxWorkers 15
```

### **如果想更快处理**

```powershell
# 提高批次大小（需确保显存足够）
.\run_single_dataset.ps1 -DatasetName "bbh" -DatasetFile "xxx" -BatchSize 192
```

---

## 📁 输出目录结构

```
output/
├── bbh/
│   ├── dpo_boolean_expressions.jsonl
│   ├── dpo_causal_judgement.jsonl
│   └── ... (27个文件)
├── mmlu/
│   ├── dpo_auxiliary_train.jsonl
│   ├── dpo_dev.jsonl
│   ├── dpo_test.jsonl
│   └── dpo_validation.jsonl
└── dpo_final.jsonl (临时文件，会被移动)

logs/
├── bbh_boolean_expressions_20251223_143022.log
├── mmlu_test_20251223_150145.log
└── ... (详细运行日志)
```

---

## 🎯 推荐工作流程

### **1. 小规模测试（验证配置）**

```powershell
# 测试单个BBH任务
.\run_single_dataset.ps1 -DatasetName "bbh" -DatasetFile "boolean_expressions"
```

### **2. 查看日志确认无误**

```powershell
# 实时监控最新日志
Get-Content logs\*.log -Tail 50 -Wait
```

### **3. 批量处理全部数据集**

```powershell
# 自动化运行所有数据集
.\batch_run_datasets.ps1
```

### **4. 监控GPU使用情况**

```powershell
# 另开终端监控GPU
nvidia-smi -l 5
```

---

## 🐛 常见问题

### **Q: 如何切换到其他GPU？**

```powershell
# 使用GPU 5和6
.\run_single_dataset.ps1 -DatasetName "bbh" -DatasetFile "xxx" -GPU "5,6"
```

### **Q: 如何暂停和恢复？**

脚本支持断点续传：
- 进度保存在 `checkpoints/dpo_progress.json`
- 中断后重新运行相同命令会自动续传

### **Q: 如何只处理部分BBH任务？**

修改 `batch_run_datasets.ps1`，注释掉不需要的任务：

```powershell
$BBH_TASKS = @(
    "boolean_expressions",
    "causal_judgement"
    # "date_understanding",  # 注释掉不需要的
)
```

### **Q: API调用失败怎么办？**

检查 `config.py` 中的配置：
- `STRONG_MODEL_API_URL`
- `STRONG_MODEL_KEY`
- API配额是否充足

---

## 📈 关键代码位置

- **GPU配置**: `inference/local_inference.py` 第40-56行
- **批处理配置**: `config.py` 第33-36行
- **数据集适配**: `stage_first.py` 第66-260行
- **单数据集脚本**: `run_single_dataset.ps1`
- **批量运行脚本**: `batch_run_datasets.ps1`

---

## ✅ 当前已优化项

- ✅ 修复 DPO 格式（`rejected` 字段）
- ✅ 3层验证机制（防止空chosen）
- ✅ 重试机制（3次指数退避）
- ✅ API超时配置（60秒）
- ✅ 2张A800 GPU并行优化
- ✅ 断点续传支持
- ✅ 详细日志记录

现在可以直接开始推理了！🚀
