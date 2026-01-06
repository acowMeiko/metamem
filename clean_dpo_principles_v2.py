"""
一次性脚本：清洗 DPO 数据中的 chosen 和 rejected 字段
只保留 Principle 字段，移除 Application、Explanation 等额外信息

用途:
    处理 DPO 数据，使 chosen 和 rejected 字段只包含 principle 内容，但保留 JSON 格式

使用方法:
    python clean_dpo_principles_v2.py --input data/dpo_llamafactory/dpo_level1.json
    python clean_dpo_principles_v2.py --all  # 处理所有 level 文件

注意: 使用完毕后可以删除此脚本
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_principles(text: str) -> str:
    """
    从 JSON 字符串中提取所有 Principle 字段，保留 JSON 格式
    
    Args:
        text: 包含 JSON 的字符串
        
    Returns:
        只包含 Principle 字段的 JSON 字符串
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 清理可能的 markdown 代码块标记
    text_clean = text.strip()
    if text_clean.startswith('```json'):
        text_clean = text_clean[7:]
    elif text_clean.startswith('```'):
        text_clean = text_clean[3:]
    if text_clean.endswith('```'):
        text_clean = text_clean[:-3]
    text_clean = text_clean.strip()
    
    try:
        # 尝试解析为 JSON
        data = json.loads(text_clean)
        
        # 提取并重构只包含 Principle 的数据
        cleaned_output = []
        
        if isinstance(data, dict):
            # 检查 output 字段
            if 'output' in data and isinstance(data['output'], list):
                for item in data['output']:
                    if isinstance(item, dict) and 'Principle' in item:
                        # 只保留 Principle 字段
                        cleaned_output.append({"Principle": item['Principle']})
                
                if cleaned_output:
                    # 返回格式化的 JSON
                    result = {"output": cleaned_output}
                    return json.dumps(result, ensure_ascii=False)
                else:
                    # 没有找到 Principle，返回原文本
                    logger.debug("未找到 Principle 字段，保留原文本")
                    return text
            else:
                # 没有 output 字段，返回原文本
                logger.debug("未找到 output 字段，保留原文本")
                return text
        else:
            # 不是字典格式，返回原文本
            logger.debug("数据格式不是字典，保留原文本")
            return text
            
    except json.JSONDecodeError as e:
        # JSON 解析失败，返回原文本
        logger.debug(f"JSON 解析失败: {e}，保留原文本")
        return text


def clean_dpo_data(input_file: Path, output_file: Path, create_backup: bool = True) -> Dict[str, int]:
    """
    清洗 DPO 数据文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        create_backup: 是否创建备份
        
    Returns:
        统计信息字典
    """
    logger.info("=" * 80)
    logger.info(f"清洗 DPO 数据: {input_file.name}")
    logger.info("=" * 80)
    
    # 加载数据
    if not input_file.exists():
        logger.error(f"文件不存在: {input_file}")
        return {}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        dpo_data = json.load(f)
    
    if not isinstance(dpo_data, list):
        logger.error(f"数据格式错误: 期望列表，实际为 {type(dpo_data)}")
        return {}
    
    logger.info(f"加载了 {len(dpo_data)} 条数据")
    
    # 创建备份
    if create_backup and output_file.exists():
        backup_file = output_file.with_suffix('.bak')
        logger.info(f"创建备份: {backup_file}")
        shutil.copy2(output_file, backup_file)
    
    # 清洗数据
    stats = {
        'total': len(dpo_data),
        'chosen_cleaned': 0,
        'rejected_cleaned': 0,
        'chosen_unchanged': 0,
        'rejected_unchanged': 0,
        'errors': 0
    }
    
    for i, item in enumerate(dpo_data):
        try:
            # 处理 chosen 字段
            if 'chosen' in item:
                original_chosen = item['chosen']
                cleaned_chosen = extract_principles(original_chosen)
                
                if cleaned_chosen != original_chosen:
                    item['chosen'] = cleaned_chosen
                    stats['chosen_cleaned'] += 1
                else:
                    stats['chosen_unchanged'] += 1
            
            # 处理 rejected 字段
            if 'rejected' in item:
                original_rejected = item['rejected']
                cleaned_rejected = extract_principles(original_rejected)
                
                if cleaned_rejected != original_rejected:
                    item['rejected'] = cleaned_rejected
                    stats['rejected_cleaned'] += 1
                else:
                    stats['rejected_unchanged'] += 1
                    
        except Exception as e:
            logger.error(f"处理第 {i} 项时出错: {e}")
            stats['errors'] += 1
    
    # 保存结果
    logger.info(f"保存清洗后的数据到: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)
    
    # 显示统计
    logger.info("=" * 80)
    logger.info("清洗统计:")
    logger.info(f"  总数据量: {stats['total']}")
    logger.info(f"  chosen 已清洗: {stats['chosen_cleaned']}")
    logger.info(f"  chosen 未变化: {stats['chosen_unchanged']}")
    logger.info(f"  rejected 已清洗: {stats['rejected_cleaned']}")
    logger.info(f"  rejected 未变化: {stats['rejected_unchanged']}")
    logger.info(f"  错误数: {stats['errors']}")
    logger.info("=" * 80)
    
    # 显示示例
    if stats['chosen_cleaned'] > 0 or stats['rejected_cleaned'] > 0:
        logger.info("\n示例数据 (前 2 条):")
        for i, item in enumerate(dpo_data[:2]):
            logger.info(f"\n[{i}]")
            if 'chosen' in item:
                preview = item['chosen'][:150] + "..." if len(item['chosen']) > 150 else item['chosen']
                logger.info(f"  chosen: {preview}")
            if 'rejected' in item:
                preview = item['rejected'][:150] + "..." if len(item['rejected']) > 150 else item['rejected']
                logger.info(f"  rejected: {preview}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='清洗 DPO 数据，只保留 Principle 字段（保留 JSON 格式）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文件
  python clean_dpo_principles_v2.py --input data/dpo_llamafactory/dpo_level1.json
  
  # 保存到新文件
  python clean_dpo_principles_v2.py --input data/dpo_llamafactory/dpo_level1.json --output data/dpo_llamafactory/dpo_level1_cleaned.json
  
  # 处理所有 level 文件
  python clean_dpo_principles_v2.py --all
  
  # 不创建备份
  python clean_dpo_principles_v2.py --input data/dpo_llamafactory/dpo_level1.json --no-backup
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='输入的 DPO JSON 文件路径'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='输出的 JSON 文件路径（不指定则原地更新）'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='处理所有 dpo_level*.json 文件'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='不创建备份文件'
    )
    
    args = parser.parse_args()
    
    try:
        if args.all:
            # 处理所有 level 文件
            data_dir = Path('data/dpo_llamafactory')
            if not data_dir.exists():
                logger.error(f"目录不存在: {data_dir}")
                return
            
            level_files = sorted(data_dir.glob('dpo_level*.json'))
            # 排除 .bak 文件
            level_files = [f for f in level_files if not f.name.endswith('.bak')]
            
            if not level_files:
                logger.error("未找到 dpo_level*.json 文件")
                return
            
            logger.info(f"找到 {len(level_files)} 个文件")
            
            total_stats = {
                'total': 0,
                'chosen_cleaned': 0,
                'rejected_cleaned': 0,
                'chosen_unchanged': 0,
                'rejected_unchanged': 0,
                'errors': 0
            }
            
            for input_file in level_files:
                output_file = input_file
                stats = clean_dpo_data(
                    input_file=input_file,
                    output_file=output_file,
                    create_backup=not args.no_backup
                )
                
                for key in total_stats:
                    total_stats[key] += stats.get(key, 0)
                
                logger.info("")
            
            # 显示总统计
            logger.info("=" * 80)
            logger.info("总体统计:")
            logger.info(f"  总数据量: {total_stats['total']}")
            logger.info(f"  chosen 已清洗: {total_stats['chosen_cleaned']}")
            logger.info(f"  rejected 已清洗: {total_stats['rejected_cleaned']}")
            logger.info(f"  错误数: {total_stats['errors']}")
            logger.info("=" * 80)
            
        else:
            # 处理单个文件
            if not args.input:
                parser.error("需要指定 --input 或使用 --all")
            
            input_file = Path(args.input)
            output_file = Path(args.output) if args.output else input_file
            
            clean_dpo_data(
                input_file=input_file,
                output_file=output_file,
                create_backup=not args.no_backup
            )
    
    except KeyboardInterrupt:
        logger.info("\n用户中断")
    except Exception as e:
        logger.error(f"错误: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
