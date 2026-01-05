"""
一次性脚本：为 DPO 数据添加 task_description 字段

用途：
    读取 DPO 数据文件，为每个样本生成 task_description，并保存回文件。
    避免在 Stage 2 中重复生成 task_description。

使用方法：
    python add_task_descriptions.py --input data/dpo_llamafactory/dpo_level1.json --output data/dpo_llamafactory/dpo_level1_with_taskdesc.json
    python add_task_descriptions.py --input data/dpo_llamafactory/dpo_level1.json  # 原地更新

注意：
    - 使用完毕后可以删除此脚本
    - 会自动创建备份文件（.bak）
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import shutil

# 导入必要的模块
from core.config import MetaConfig, initialize_config
from inference.engine import InferenceEngineBuilder
from templates.prompts import PromptTemplate

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_markdown(text: str) -> str:
    """清理 markdown 代码块标记"""
    text_clean = text.strip()
    if text_clean.startswith('```json'):
        text_clean = text_clean[7:]
    elif text_clean.startswith('```'):
        text_clean = text_clean[3:]
    if text_clean.endswith('```'):
        text_clean = text_clean[:-3]
    return text_clean.strip()


def setup_inference_engine(config: MetaConfig):
    """设置推理引擎"""
    builder = InferenceEngineBuilder()
    
    # 配置 weak model
    builder.set_weak_model(
        model_type=config.models.weak_model_type,
        name=config.models.weak_model_name,
        url=config.models.weak_model_url,
        api_key=config.models.weak_model_key
    )
    
    # 配置 strong model
    builder.set_strong_model(
        model_type=config.models.strong_model_type,
        name=config.models.strong_model_name,
        url=config.models.strong_model_url,
        api_key=config.models.strong_model_key
    )
    
    # 设置默认参数
    builder.set_defaults(
        temperature=config.inference.default_temperature,
        top_p=config.inference.default_top_p,
        max_tokens=config.inference.default_max_tokens,
        repetition_penalty=config.inference.default_repetition_penalty,
        frequency_penalty=config.inference.default_frequency_penalty
    )
    
    return builder.build()


def add_task_descriptions(
    input_file: Path,
    output_file: Path,
    batch_size: int = 32,
    create_backup: bool = True
) -> None:
    """
    为 DPO 数据添加 task_description 字段
    
    Args:
        input_file: 输入的 DPO JSON 文件路径
        output_file: 输出的 JSON 文件路径
        batch_size: 批处理大小
        create_backup: 是否创建备份文件
    """
    logger.info("=" * 80)
    logger.info("开始为 DPO 数据添加 task_description")
    logger.info("=" * 80)
    
    # 1. 加载数据
    logger.info(f"加载数据: {input_file}")
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        dpo_data = json.load(f)
    
    if not isinstance(dpo_data, list):
        logger.error(f"数据格式错误: 期望列表格式，实际为 {type(dpo_data)}")
        return
    
    logger.info(f"加载了 {len(dpo_data)} 条数据")
    
    # 2. 检查是否已有 task_description
    has_taskdesc = all('task_description' in item for item in dpo_data)
    if has_taskdesc:
        logger.warning("所有数据已包含 task_description 字段")
        user_input = input("是否要重新生成所有 task_description? (y/N): ")
        if user_input.lower() != 'y':
            logger.info("操作取消")
            return
    
    # 3. 创建备份
    if create_backup and output_file.exists():
        backup_file = output_file.with_suffix('.bak')
        logger.info(f"创建备份文件: {backup_file}")
        shutil.copy2(output_file, backup_file)
    
    # 4. 初始化配置和组件
    logger.info("初始化推理引擎...")
    config = MetaConfig.from_env()
    initialize_config(config)
    
    engine = setup_inference_engine(config)
    prompts = PromptTemplate()
    
    # 5. 提取所有问题
    logger.info("提取问题...")
    questions = []
    valid_indices = []
    
    for i, item in enumerate(dpo_data):
        question = item.get('question')
        if not question:
            logger.warning(f"第 {i} 项缺少 question 字段，跳过")
            continue
        questions.append(question)
        valid_indices.append(i)
    
    logger.info(f"有效问题数量: {len(questions)}")
    
    if not questions:
        logger.error("没有找到有效的问题，退出")
        return
    
    # 6. 批量生成 task_description
    logger.info(f"批量生成 task_description (batch_size={batch_size})...")
    task_desc_prompts = [prompts.get_task_description_prompt(q) for q in questions]
    
    task_descs_raw = engine.batch_inference(
        prompts=task_desc_prompts,
        model_type='weak',
        batch_size=batch_size,
        max_tokens=config.inference.task_desc_max_tokens,
        temperature=0.1,
        repetition_penalty=1.1
    )
    
    # 7. 清理并添加到数据中
    logger.info("处理生成结果...")
    success_count = 0
    
    for idx, task_desc_raw in zip(valid_indices, task_descs_raw):
        task_desc = clean_markdown(task_desc_raw)
        if task_desc:
            dpo_data[idx]['task_description'] = task_desc
            success_count += 1
        else:
            logger.warning(f"第 {idx} 项 task_description 生成失败或为空")
            dpo_data[idx]['task_description'] = ""
    
    logger.info(f"成功生成 {success_count}/{len(questions)} 个 task_description")
    
    # 8. 保存结果
    logger.info(f"保存结果到: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)
    
    logger.info("=" * 80)
    logger.info("✓ 完成!")
    logger.info(f"✓ 输出文件: {output_file}")
    logger.info(f"✓ 成功处理: {success_count}/{len(dpo_data)} 条数据")
    
    if create_backup and output_file.exists():
        logger.info(f"✓ 备份文件: {output_file.with_suffix('.bak')}")
    
    logger.info("=" * 80)
    
    # 9. 显示示例
    if success_count > 0:
        logger.info("\n示例数据 (前 3 条):")
        for i, item in enumerate(dpo_data[:3]):
            if 'task_description' in item:
                logger.info(f"\n[{i}] Question: {item['question'][:80]}...")
                logger.info(f"    Task Desc: {item['task_description'][:100]}...")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='为 DPO 数据添加 task_description 字段',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 保存到新文件
  python add_task_descriptions.py --input data/dpo_llamafactory/dpo_level1.json --output data/dpo_llamafactory/dpo_level1_with_taskdesc.json
  
  # 原地更新（会自动创建 .bak 备份）
  python add_task_descriptions.py --input data/dpo_llamafactory/dpo_level1.json
  
  # 不创建备份
  python add_task_descriptions.py --input data/dpo_llamafactory/dpo_level1.json --no-backup
  
  # 自定义批处理大小
  python add_task_descriptions.py --input data/dpo_llamafactory/dpo_level1.json --batch-size 64
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入的 DPO JSON 文件路径'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='输出的 JSON 文件路径（不指定则原地更新）'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批处理大小（默认: 32）'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='不创建备份文件'
    )
    
    args = parser.parse_args()
    
    # 处理路径
    input_file = Path(args.input)
    
    if args.output:
        output_file = Path(args.output)
    else:
        # 原地更新
        output_file = input_file
    
    try:
        add_task_descriptions(
            input_file=input_file,
            output_file=output_file,
            batch_size=args.batch_size,
            create_backup=not args.no_backup
        )
    except KeyboardInterrupt:
        logger.info("\n用户中断")
    except Exception as e:
        logger.error(f"错误: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
