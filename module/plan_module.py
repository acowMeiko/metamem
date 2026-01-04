from template.prompt_template import TASK_DESC_PROMPT, DIRECT_ANSWER_PROMPT, GUIDED_ANSWER_PROMPT, PRINCIPLE_PROMPT, DIFF_PROMPT
from inference.api_inference import gpt_call
from inference.local_inference import batch_inference, single_inference
from typing import List
import logging
import config

logger = logging.getLogger(__name__)

# ==================== 单条推理函数（保持兼容） ====================

# CDE process.
def generate_difference_list(question: str, pred: str, label: str, use_local: bool = True) -> str:
    """
    生成差异列表
    
    Args:
        question: 问题
        pred: 预测答案
        label: 标准答案
        use_local: 是否使用本地模型（默认True）
    """
    prompt = DIFF_PROMPT.substitute(question=question, pred=pred, label=label)
    if use_local:
        from inference.local_inference import single_inference
        return single_inference(prompt)
    else:
        return gpt_call(user=prompt)


def generate_principles(question: str, diff_list: str, model="weak", use_local: bool = True) -> str:
    """
    生成原则
    
    Args:
        question: 问题
        diff_list: 差异列表
        model: 模型类型 ("weak" 或 "strong")
        use_local: weak模型是否使用本地，strong始终使用API
    """
    prompt = PRINCIPLE_PROMPT.substitute(question=question, diff_list=diff_list)
    
    if model == "strong":
        # 强模型始终使用API
        return gpt_call(
            user=prompt, 
            model=config.STRONG_MODEL_NAME, 
            url=config.STRONG_MODEL_API_URL, 
            api_key=config.STRONG_MODEL_KEY
        )
    else:
        # 弱模型根据参数决定
        if use_local:
            from inference.local_inference import single_inference
            return single_inference(prompt)
        else:
            return gpt_call(user=prompt, model=config.BASE_MODEL_NAME)


# ==================== 批量推理函数（新增） ====================

def batch_generate_difference_list(questions: List[str], preds: List[str], labels: List[str]) -> List[str]:
    """
    批量生成差异列表
    
    Args:
        questions: 问题列表
        preds: 预测答案列表
        labels: 标准答案列表
        
    Returns:
        差异列表的列表
    """
    prompts = [
        DIFF_PROMPT.substitute(question=q, pred=p, label=l)
        for q, p, l in zip(questions, preds, labels)
    ]
    return batch_inference(prompts)


def batch_generate_principles(questions: List[str], diff_lists: List[str], model="weak") -> List[str]:
    """
    批量生成原则
    
    Args:
        questions: 问题列表
        diff_lists: 差异列表的列表
        model: 模型类型 ("weak" 使用本地, "strong" 使用API)
        
    Returns:
        原则列表
    """
    prompts = [
        PRINCIPLE_PROMPT.substitute(question=q, diff_list=d)
        for q, d in zip(questions, diff_lists)
    ]
    
    if model == "weak":
        # 弱模型使用本地批量推理
        # 使用更多stop序列防止重复生成
        stop_sequences = [
            "```\n\n",  # JSON代码块结束后的空行
            "}\n}\n```",  # 嵌套JSON结束
            "Final Answer",
            "**Final",
            "\n\n\n\n",  # 多个空行
        ]
        results = batch_inference(
            prompts, 
            max_tokens=config.PRINCIPLE_MAX_TOKENS,
            stop=stop_sequences,
            temperature=0.1  # 降低随机性
        )
        
        # 后处理：截断重复内容
        cleaned_results = []
        for result in results:
            # 如果发现重复的JSON块，只保留第一个
            if result.count('```json') > 1:
                # 找到第一个完整的 ```json...``` 块
                first_end = result.find('```', result.find('```json') + 7)
                if first_end > 0:
                    result = result[:first_end + 3]
            cleaned_results.append(result)
        
        return cleaned_results
    else:
        # 强模型使用API（在main_pipeline.py中并发调用）
        # 这里不应该被调用，如果需要应该使用concurrent_generate_chosen
        raise ValueError("强模型应使用concurrent_generate_chosen进行并发调用")

