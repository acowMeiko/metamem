# ==================== 批量运行数据集生成脚本 ====================
# 功能：按顺序处理多个数据集，每个数据集独立运行
# 使用2张A800 GPU (编号3,4)
# =================================================================

param(
    [string]$GPU = "3,4",  # GPU编号
    [int]$BatchSize = 128,  # vLLM批处理大小（2张80G A800优化）
    [int]$MaxWorkers = 30   # API并发数（2张A800优化）
)

$ErrorActionPreference = "Continue"  # 单个失败不影响后续

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "批量处理数据集 - 2张A800 GPU优化配置" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "GPU设备: $GPU"
Write-Host "vLLM批次大小: $BatchSize (2张A800优化)"
Write-Host "API并发数: $MaxWorkers (2张A800优化)"
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# ==================== BBH 数据集列表 ====================
$BBH_TASKS = @(
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting"
)

# ==================== MMLU 数据集列表 ====================
$MMLU_FILES = @(
    "auxiliary_train",
    "dev",
    "test",
    "validation"
)

# ==================== 统计变量 ====================
$total_success = 0
$total_failed = 0
$start_time = Get-Date

# ==================== 处理 BBH 数据集 ====================
Write-Host "==================== 处理 BBH 数据集 ====================" -ForegroundColor Green
Write-Host "共 $($BBH_TASKS.Count) 个任务"
Write-Host ""

$bbh_success = 0
foreach ($task in $BBH_TASKS) {
    Write-Host "[$($bbh_success + 1)/$($BBH_TASKS.Count)] 处理: $task" -ForegroundColor Yellow
    
    try {
        .\run_single_dataset.ps1 `
            -DatasetName "bbh" `
            -DatasetFile $task `
            -GPU $GPU `
            -BatchSize $BatchSize `
            -MaxWorkers $MaxWorkers
        
        if ($LASTEXITCODE -eq 0) {
            $bbh_success++
            $total_success++
            Write-Host "✅ $task 完成" -ForegroundColor Green
        } else {
            $total_failed++
            Write-Host "❌ $task 失败" -ForegroundColor Red
        }
    } catch {
        $total_failed++
        Write-Host "❌ $task 异常: $_" -ForegroundColor Red
    }
    
    Write-Host ""
}

Write-Host "BBH 数据集处理完成: 成功 $bbh_success/$($BBH_TASKS.Count)" -ForegroundColor Cyan
Write-Host ""

# ==================== 处理 MMLU 数据集 ====================
Write-Host "==================== 处理 MMLU 数据集 ====================" -ForegroundColor Green
Write-Host "共 $($MMLU_FILES.Count) 个文件"
Write-Host ""

$mmlu_success = 0
foreach ($file in $MMLU_FILES) {
    Write-Host "[$($mmlu_success + 1)/$($MMLU_FILES.Count)] 处理: $file" -ForegroundColor Yellow
    
    try {
        .\run_single_dataset.ps1 `
            -DatasetName "mmlu" `
            -DatasetFile $file `
            -GPU $GPU `
            -BatchSize $BatchSize `
            -MaxWorkers $MaxWorkers
        
        if ($LASTEXITCODE -eq 0) {
            $mmlu_success++
            $total_success++
            Write-Host "✅ $file 完成" -ForegroundColor Green
        } else {
            $total_failed++
            Write-Host "❌ $file 失败" -ForegroundColor Red
        }
    } catch {
        $total_failed++
        Write-Host "❌ $file 异常: $_" -ForegroundColor Red
    }
    
    Write-Host ""
}

Write-Host "MMLU 数据集处理完成: 成功 $mmlu_success/$($MMLU_FILES.Count)" -ForegroundColor Cyan
Write-Host ""

# ==================== 最终汇总 ====================
$end_time = Get-Date
$duration = $end_time - $start_time

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "批量处理完成！" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "BBH: 成功 $bbh_success/$($BBH_TASKS.Count) 个任务"
Write-Host "MMLU: 成功 $mmlu_success/$($MMLU_FILES.Count) 个文件"
Write-Host ""
Write-Host "总计: 成功 $total_success, 失败 $total_failed"
Write-Host "耗时: $($duration.Hours)小时 $($duration.Minutes)分钟 $($duration.Seconds)秒"
Write-Host ""
Write-Host "输出目录:"
Write-Host "  - output/bbh/"
Write-Host "  - output/mmlu/"
Write-Host ""
Write-Host "日志目录: logs/"
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
