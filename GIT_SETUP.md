# 重新设置 Git 仓库

## ✅ 已完成的清理

- ✓ 已删除旧的 `.git` 目录（从 metanew2 仓库）
- ✓ 项目现在是一个干净的目录，可以关联到新仓库

---

## 📝 连接到新 GitHub 仓库的步骤

### 方法 1: 使用 Git 命令行

```bash
# 1. 初始化新的 Git 仓库
git init

# 2. 添加所有文件
git add .

# 3. 创建初始提交
git commit -m "Initial commit: MetaEvo refactored architecture"

# 4. 添加远程仓库（替换为你的新仓库地址）
git remote add origin https://github.com/YOUR_USERNAME/YOUR_NEW_REPO.git

# 5. 推送到远程仓库
git push -u origin main
```

### 方法 2: 从 GitHub 创建仓库开始

1. **在 GitHub 上创建新仓库**
   - 访问 https://github.com/new
   - 输入仓库名称（例如：`metanew3` 或 `metaevo-refactored`）
   - 选择 Public 或 Private
   - **不要**勾选"Initialize this repository with a README"
   - 点击"Create repository"

2. **GitHub 会显示连接命令**，类似：
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_NEW_REPO.git
   git push -u origin main
   ```

3. **按照 GitHub 显示的命令执行**

---

## 📦 推荐的 .gitignore

在提交前，建议创建 `.gitignore` 文件：

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
logs/
*.log

# Output files
output/
checkpoints/
*.jsonl
*.json

# Model files (如果很大)
em_model/
*.bin
*.safetensors
*.pt

# Data files (如果很大)
data/original_data/*.json
dataset/

# Memory files
memory/*.json

# OS
.DS_Store
Thumbs.db
```

---

## 🎯 推荐的首次提交结构

你可以分批提交，使历史更清晰：

### Commit 1: 核心架构
```bash
git add core/ inference/engine.py data/ templates/
git commit -m "feat: Add core architecture (base classes, stages, config)"
```

### Commit 2: 主入口和文档
```bash
git add run_experiments.py examples/ docs/ *.md
git commit -m "feat: Add main entry point and documentation"
```

### Commit 3: 保留的旧模块
```bash
git add module/ inference/local_inference.py inference/api_inference.py
git commit -m "feat: Retain legacy modules for compatibility"
```

### Commit 4: 配置和其他
```bash
git add config.py main.py stage_first.py stage_second.py
git commit -m "chore: Keep old files for reference"
```

---

## 🔄 如果需要更改远程仓库

如果将来需要更改远程仓库地址：

```bash
# 查看当前远程仓库
git remote -v

# 删除现有远程仓库
git remote remove origin

# 添加新的远程仓库
git remote add origin https://github.com/YOUR_USERNAME/NEW_REPO.git

# 推送
git push -u origin main
```

---

## 📋 检查清单

- [ ] 在 GitHub 创建新仓库
- [ ] 创建 `.gitignore` 文件
- [ ] 运行 `git init`
- [ ] 运行 `git add .`
- [ ] 运行 `git commit -m "Initial commit"`
- [ ] 添加远程仓库 `git remote add origin ...`
- [ ] 推送 `git push -u origin main`

---

## 💡 提示

- **仓库名称建议**: `metaevo-refactored`, `metanew3`, 或 `metaevo-framework`
- **描述建议**: "MetaEvo: A modular meta-reasoning framework with memr3-style architecture"
- **主题标签**: `machine-learning`, `nlp`, `dpo`, `meta-learning`, `reasoning`

---

## ⚠️ 注意事项

1. **大文件**: 如果 `data/` 或 `em_model/` 中有大文件，考虑使用 Git LFS 或不提交
2. **敏感信息**: 确保 API keys 等敏感信息不在代码中（已使用环境变量，应该没问题）
3. **模型文件**: 建议不提交模型权重文件，在 README 中说明如何下载

---

## 🎉 完成后

完成设置后，你的新仓库将包含完整的重构后架构，可以：
- 分享给团队成员
- 作为独立项目发展
- 保留完整的开发历史

如有问题，可以参考 GitHub 文档：https://docs.github.com/
