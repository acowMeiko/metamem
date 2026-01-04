import os
from pathlib import Path
# 地址参数
Project_ROOT = Path(__file__).parent.absolute()#项目根路径，运行代码的父目录的绝对路径

check_root = os.path.join(Project_ROOT, 'checkpoints')  #模型检查点目录
output_dir = os.path.join(Project_ROOT, 'output')  #输出目录
data_dir = os.path.join(Project_ROOT, 'data')  #数据集目录
BASE_MODEL_NAME = os.getenv('BASE_MODEL_NAME', "/home/share/hcz/qwen2.5-14b-awq")
lora_model_path = os.getenv('LORA_MODEL_PATH', "/home/models/qwen_dpo4_lora")  #LoRA模型路径
MAX_MODEL_LEN = 32768  # 提升到 32K，充分利用 80G A800 显存
MEMORY_FILE =os.path.join(Project_ROOT, 'memory', 'memory_round4.json')  #Memory文件路径

# ==================== 强模型API配置（用于生成Chosen） ====================
STRONG_MODEL_NAME = os.getenv('STRONG_MODEL_NAME', 'DeepSeek-R1')
STRONG_MODEL_API_URL = os.getenv('STRONG_MODEL_API_URL', 'https://llmapi.paratera.com/v1/')
STRONG_MODEL_KEY = os.getenv('STRONG_MODEL_KEY', 'sk-0tKGY03c9OJPODlWGzAGPw')

DEFAULT_TEMPERATURE = float(os.getenv('DEFAULT_TEMPERATURE', '0'))
DEFAULT_TOP_P = float(os.getenv('DEFAULT_TOP_P', '0.95  '))  # top_p 必须在 (0, 1] 区间

# ==================== 生成长度限制配置 ====================
DEFAULT_MAX_TOKENS = int(os.getenv('DEFAULT_MAX_TOKENS', '4096'))  # 从8192降到2048，避免重复生成
# 专用生成长度限制（用于特定任务）
TASK_DESC_MAX_TOKENS = 2560     # 任务描述生成（level3需要更长，增加到2560）
DIFF_ANALYSIS_MAX_TOKENS = 1024  # 差异分析生成  
PRINCIPLE_MAX_TOKENS = 2560     # 原则生成（level3需要更长，增加到2560）
ANSWER_MAX_TOKENS = 2048         # 答案生成

# ==================== Memory配置 ====================
 
SAVE_FREQUENCY = int(os.getenv('SAVE_FREQUENCY', '50'))  # 保存频率
# ==================== 日志配置 ====================
LOG_DIR = Project_ROOT / "logs"
LOG_FILE = LOG_DIR / "stage_first.log"  # 默认日志文件
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ==================== 批处理和并发配置 ====================
# vLLM批处理大小（针对大数据集优化）
# 4×80GB A800 + 14B模型 + LoRA，tensor_parallel_size=4
# 实际生成长度512-2048，显存充足，可支持更大batch
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '256'))  # 4卡配置（保守），可尝试256甚至更大

# API并发线程数
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '20'))  # 增加并发数

original_data_file = os.path.join(data_dir, 'original_data', 'merged_all_levels.json')  #原始数据集文件路径
dpo_progress_file = os.path.join(check_root, 'dpo_progress.json')  #进度文件路径
memory_checkpoint_file = os.path.join(check_root, 'memory_progress.json')  #Memory断点文件路径
# 支持通过环境变量自定义输出文件名
dpo_final_file = os.getenv('DPO_OUTPUT_FILE', os.path.join(output_dir, 'dpo_final.jsonl'))  #最终DPO数据文件路径
data_levels_file = os.path.join(data_dir, 'dpo_llamafactory', 'dpo_level_level4_llamafactory.json')  #数据级别文件路径
test_file = os.path.join(data_dir,"test", "test_memory.json")  #测试数据文件路径
# 超参数
batch_size = 64  #本地推理批次大小
output_dir = os.path.join(Project_ROOT, 'output')  #输出目录

