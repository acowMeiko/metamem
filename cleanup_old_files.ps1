# 清理旧架构文件

Write-Host "=" * 60
Write-Host "MetaEvo 架构重构 - 清理旧文件"
Write-Host "=" * 60

# 定义要删除的文件列表
$filesToDelete = @(
    "stage_first.py",           # 已被 core/stages.py::StageOneAgent 替代
    "stage_second.py",          # 已被 core/stages.py::StageTwoAgent 替代
    "stage_infer.py",           # 已被 core/stages.py::InferenceAgent 替代
    "check_princiles.py",       # 临时检查脚本，不再需要
    "test.json",                # 测试文件
    "REFACTORING_SUMMARY.md"    # 旧的重构总结，已被新文档替代
)

# 定义要删除的目录列表
$dirsToDelete = @(
    "template",                 # 已被 templates/ 替代
    "__pycache__"               # Python 缓存
)

Write-Host "`n准备删除以下文件:"
foreach ($file in $filesToDelete) {
    if (Test-Path $file) {
        Write-Host "  - $file" -ForegroundColor Yellow
    }
}

Write-Host "`n准备删除以下目录:"
foreach ($dir in $dirsToDelete) {
    if (Test-Path $dir) {
        Write-Host "  - $dir" -ForegroundColor Yellow
    }
}

# 询问确认
Write-Host "`n这些文件已经被新架构替代，可以安全删除。"
$confirm = Read-Host "是否继续删除? (y/n)"

if ($confirm -eq 'y' -or $confirm -eq 'Y') {
    Write-Host "`n开始清理..." -ForegroundColor Green
    
    # 删除文件
    foreach ($file in $filesToDelete) {
        if (Test-Path $file) {
            Remove-Item $file -Force
            Write-Host "  ✓ 已删除: $file" -ForegroundColor Green
        }
    }
    
    # 删除目录
    foreach ($dir in $dirsToDelete) {
        if (Test-Path $dir) {
            Remove-Item $dir -Recurse -Force
            Write-Host "  ✓ 已删除: $dir" -ForegroundColor Green
        }
    }
    
    Write-Host "`n✓ 清理完成!" -ForegroundColor Green
    Write-Host "`n保留的旧文件 (供参考):"
    Write-Host "  - main.py (旧主入口，供兼容性参考)"
    Write-Host "  - config.py (旧配置，已被 core/config.py 替代)"
    Write-Host "  - module/ (旧模块，部分仍被新架构使用)"
    Write-Host "`n新架构文件:"
    Write-Host "  - run_experiments.py (新主入口) ✓"
    Write-Host "  - core/ (核心模块) ✓"
    Write-Host "  - data/ (数据处理) ✓"
    Write-Host "  - templates/ (Prompt管理) ✓"
    
} else {
    Write-Host "`n取消清理操作。" -ForegroundColor Yellow
}

Write-Host "`n" + "=" * 60
