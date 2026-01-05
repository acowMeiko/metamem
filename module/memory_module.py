from template.prompt_template import TASK_DESC_PROMPT, DIRECT_ANSWER_PROMPT, GUIDED_ANSWER_PROMPT, PRINCIPLE_MATCH_PROMPT
from inference.api_inference import gpt_call
from inference.local_inference import batch_inference, single_inference
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List
import logging
import config
import json
import numpy as np
import re
import os

class MemoryManager:
    def __init__(self, path=None):
        # 如果环境变量未设置，则默认使用 0,1,2,3（4卡）
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        
        self.path = path or str(config.MEMORY_FILE)
        self.memory = self.load()
        # 使用 vLLM 模型进行语义匹配

    def load(self):
        """加载内存数据，确保返回字典类型"""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
            if isinstance(loaded_data, dict):
                return loaded_data
            else:
                print(f"警告：{self.path} 中数据不是字典类型，已初始化为空字典")
                return {}
        except FileNotFoundError:
            print(f"提示：{self.path} 不存在，已初始化为空字典")
            return {}
        except json.JSONDecodeError:
            print(f"错误：{self.path} 中JSON格式无效，已初始化为空字典")
            return {}

    def save(self):
        """保存内存字典到JSON文件"""
        if not isinstance(self.memory, dict):
            raise TypeError("self.memory 必须是字典类型，无法保存")

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)

    def _semantic_match_with_llm(self, query_task: str, candidate_task: str) -> bool:
        """
        使用 vLLM 判断两个任务描述是否是同一个意思
        返回 True (匹配) 或 False (不匹配)
        """
        prompt = f"""请判断以下两个任务描述是否表达相同的意思。只回答"是"或"否"，不要有任何其他文字。

任务1: {query_task}
任务2: {candidate_task}

是否相同:"""
        
        try:
            response = single_inference(prompt, temperature=0.0, max_tokens=5)
            response_clean = response.strip().lower()
            # 判断是否包含肯定词
            if '是' in response_clean or 'yes' in response_clean or '相同' in response_clean:
                return True
            else:
                return False
        except Exception as e:
            logging.error(f"LLM 语义匹配失败: {e}")
            return False

    def retrieve(self, task_desc: str):
        """使用 vLLM 进行语义匹配"""
        normalized_task = task_desc.strip()

        # 如果 memory 为空，直接返回
        if not self.memory:
            return None, []

        # 遍历所有已存储的任务，使用 LLM 判断是否匹配
        for task, principles in self.memory.items():
            if self._semantic_match_with_llm(normalized_task, task):
                return task, principles

        # 没有找到匹配的任务
        return None, []

    def add_task(self, task_desc: str, principles: list):
        # 类型验证：防止字符串被当作列表存储导致字符级迭代
        if not isinstance(principles, list):
            raise TypeError(
                f"principles 必须是 list 类型，当前类型: {type(principles).__name__}。"
                f"如果传入字符串，Python 会将其按字符迭代，导致数据损坏。"
            )
        self.memory[task_desc] = principles

    def merge_principles(self, task_desc: str, new_principles: list):
        old_principles = self.memory.get(task_desc, [])
        filtered = self._resolve_conflicts(old_principles, new_principles)
        self.memory[task_desc] = filtered

    def _resolve_conflicts(self, old: list, new: list) -> list:
        if not old:
            return new
        if not new:
            return old

        prompt = PRINCIPLE_MATCH_PROMPT.substitute(
            old="\n".join(old),
            new="\n".join(new)
        )
        result = single_inference(prompt)

        pattern = r'(\{\s*"comparisons"\s*:\s*\[.*?\]\s*\})'
        match = re.search(pattern, result, flags=re.DOTALL)
        if match:
            try:
                # 提取出来的字符串
                comparisons_json_str = match.group(1)
                # 修复常见的JSON格式错误：双花括号 {{ }} -> { }
                comparisons_json_str = comparisons_json_str.replace('{{', '{').replace('}}', '}')
                # 加载为 Python 对象
                relations_dict = json.loads(comparisons_json_str)
                relations = relations_dict.get("comparisons", [])
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON decode failed: {e}\nRaw string:\n{comparisons_json_str}")
                relations = []
        else:
            print("[WARN] No valid 'comparisons' JSON found in model output:\n", result[:500])
            relations = []

        retained_old = set(old)
        retained_new = []

        # ✅ 安全检查，防止 NoneType 报错
        if not relations:
            print("[WARN] relations is empty, skipping conflict resolution.")
            return list(retained_old)

        for match in relations:
            old_rule = match.get("old", "").strip()
            new_rule = match.get("new", "").strip()
            relation = match.get("relation", "").strip()

            if relation == "Redundant":
                if old_rule in retained_old:
                    retained_old.remove(old_rule)
                retained_new.append(new_rule)
            elif relation == "Conflicting":
                if old_rule in retained_old:
                    retained_old.remove(old_rule)
                retained_new.append(new_rule)
            elif relation == "Irrelevant":
                retained_new.append(new_rule)

        return list(retained_old.union(retained_new))


