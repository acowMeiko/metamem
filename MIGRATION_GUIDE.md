# 文件迁移对照表

## 已废弃的文件 (可删除)

### 业务逻辑文件
| 旧文件 | 新文件 | 状态 | 说明 |
|--------|--------|------|------|
| `stage_first.py` | `core/stages.py::StageOneAgent` | ❌ 已替代 | DPO数据生成逻辑已完全迁移 |
| `stage_second.py` | `core/stages.py::StageTwoAgent` | ❌ 已替代 | Memory更新逻辑已完全迁移 |
| `stage_infer.py` | `core/stages.py::InferenceAgent` | ❌ 已替代 | 推理逻辑已完全迁移 |
| `main.py` | `run_experiments.py` | ⚠️ 保留参考 | 旧主入口，建议逐步废弃 |

### 配置和模板文件
| 旧文件 | 新文件 | 状态 | 说明 |
|--------|--------|------|------|
| `config.py` | `core/config.py` | ⚠️ 保留参考 | 旧配置，新架构已完全重写 |
| `template/prompt_template.py` | `templates/prompts.py` | ❌ 已替代 | Prompt管理已迁移 |

### 临时和测试文件
| 文件 | 状态 | 说明 |
|------|------|------|
| `check_princiles.py` | ❌ 可删除 | 临时检查脚本 |
| `test.json` | ❌ 可删除 | 测试数据文件 |
| `REFACTORING_SUMMARY.md` | ❌ 可删除 | 已被新文档替代 |
| `__pycache__/` | ❌ 可删除 | Python缓存目录 |
| `template/` | ❌ 可删除 | 整个目录已被 `templates/` 替代 |

---

## 保留的文件 (仍在使用)

### 核心业务文件
| 文件/目录 | 用途 | 说明 |
|-----------|------|------|
| `module/memory_module.py` | Memory管理 | 新架构仍在使用 |
| `module/execute_module.py` | ⚠️ 部分使用 | 可逐步废弃 |
| `module/plan_module.py` | ⚠️ 部分使用 | 可逐步废弃 |
| `inference/local_inference.py` | vLLM推理 | 新架构依赖 |
| `inference/api_inference.py` | API推理 | 新架构依赖 |

### 数据和输出
| 文件/目录 | 用途 |
|-----------|------|
| `data/` | 数据集存储 |
| `output/` | 输出结果 |
| `memory/` | Memory文件 |
| `logs/` | 日志文件 |
| `checkpoints/` | 检查点 |

### 文档
| 文件 | 状态 | 说明 |
|------|------|------|
| `README_NEW.md` | ✓ 保留 | 新版README |
| `REFACTORING_GUIDE.md` | ✓ 保留 | 重构指南 |
| `REFACTORING_COMPLETE.md` | ✓ 保留 | 完成报告 |
| `docs/ARCHITECTURE_COMPARISON.md` | ✓ 保留 | 架构对比 |
| `docs/ARCHITECTURE_DESIGN.md` | ✓ 保留 | 架构设计 |
| `README_GPU_CONFIG.md` | ✓ 保留 | GPU配置文档 |

---

## 清理步骤

### 自动清理 (推荐)

```powershell
# 运行清理脚本
.\cleanup_old_files.ps1
```

### 手动清理

```powershell
# 删除已废弃的文件
Remove-Item stage_first.py
Remove-Item stage_second.py
Remove-Item stage_infer.py
Remove-Item check_princiles.py
Remove-Item test.json
Remove-Item REFACTORING_SUMMARY.md

# 删除已废弃的目录
Remove-Item template -Recurse
Remove-Item __pycache__ -Recurse

# 清理Python缓存
Get-ChildItem -Path . -Filter "__pycache__" -Recurse | Remove-Item -Recurse -Force
Get-ChildItem -Path . -Filter "*.pyc" -Recurse | Remove-Item -Force
```

---

## 迁移后的目录结构

```
metanew3/
├── core/                    # ✓ 新增 - 核心模块
│   ├── __init__.py
│   ├── base.py
│   ├── stages.py
│   └── config.py
│
├── data/                    # ✓ 新增 - 数据处理
│   ├── __init__.py
│   └── processor.py
│
├── templates/               # ✓ 新增 - Prompt管理
│   ├── __init__.py
│   └── prompts.py
│
├── inference/               # ✓ 保留 - 推理引擎
│   ├── engine.py           # ✓ 新增
│   ├── local_inference.py
│   └── api_inference.py
│
├── module/                  # ⚠️ 部分使用
│   ├── memory_module.py    # ✓ 保留
│   ├── execute_module.py   # ⚠️ 可废弃
│   └── plan_module.py      # ⚠️ 可废弃
│
├── examples/                # ✓ 新增 - 示例代码
│   └── quick_start.py
│
├── docs/                    # ✓ 新增 - 文档
│   ├── ARCHITECTURE_COMPARISON.md
│   └── ARCHITECTURE_DESIGN.md
│
├── run_experiments.py       # ✓ 新增 - 新主入口
├── cleanup_old_files.ps1    # ✓ 新增 - 清理脚本
│
├── main.py                  # ⚠️ 旧主入口 (保留参考)
├── config.py                # ⚠️ 旧配置 (保留参考)
│
└── [数据和输出目录保持不变]
```

---

## 注意事项

### ⚠️ 删除前确认

1. **确保新架构测试通过**: 先验证 `run_experiments.py` 可正常运行
2. **备份重要数据**: 确保 `data/`, `output/`, `memory/` 已备份
3. **Git提交**: 删除前确保所有更改已提交

### ✓ 安全删除的文件

以下文件可以**安全删除**，不会影响新架构:
- `stage_first.py`
- `stage_second.py`
- `stage_infer.py`
- `check_princiles.py`
- `test.json`
- `template/` (整个目录)
- `__pycache__/` (所有Python缓存)

### ⚠️ 暂时保留的文件

以下文件建议**暂时保留**，用于参考或兼容:
- `main.py` - 可能有用户仍在使用
- `config.py` - 用于理解旧配置
- `module/execute_module.py` - 部分功能仍被使用
- `module/plan_module.py` - 部分功能仍被使用

---

## 清理后的验证

清理后，运行以下命令验证新架构:

```powershell
# 验证Stage 1
python run_experiments.py --stage 1 --dataset gsm8k --dataset-path dataset/gsm8k/test.jsonl --debug

# 验证Stage 2
python run_experiments.py --stage 2 --debug

# 验证Stage 3
python run_experiments.py --stage 3 --dataset gsm8k --dataset-path dataset/gsm8k/test.jsonl --debug
```

---

## 回滚方案

如果删除后发现问题，可以通过Git恢复:

```powershell
# 查看删除的文件
git status

# 恢复单个文件
git checkout <filename>

# 恢复所有删除
git reset --hard HEAD
```

---

**建议**: 先运行 `cleanup_old_files.ps1` 进行交互式清理，确认无误后再手动删除其他文件。
