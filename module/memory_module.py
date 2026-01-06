from templates.prompts import PromptTemplate
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
        self.prompts = PromptTemplate()
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
        
        根据严格的语义判断标准：
        - 数学问题本质一致（已知条件、求解目标、约束限制）
        - 数学对象、运算关系、推理方向无实质性差异
        - 仅表述顺序或用词差异视为相同
        - 有条件增减、目标变化、约束修改则视为不同
        
        返回 True (匹配) 或 False (不匹配)
        """
        prompt = self.prompts.get_semantic_match_prompt(query_task, candidate_task)
        
        try:
            response = single_inference(prompt, temperature=0.0, max_tokens=5)
            response_clean = response.strip().lower()
            # 新 prompt 要求严格输出 "Yes" 或 "No"
            # 判断是否为肯定回答
            if 'yes' in response_clean:
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

    def batch_retrieve(self, task_descs: List[str]):
        """批量语义匹配：使用 vLLM 批量推理
        
        Args:
            task_descs: 待匹配的任务描述列表
            
        Returns:
            List[Tuple[str, Optional[str], list]]: 每个元素为 (input_task, matched_task, principles)
                - input_task: 输入的任务描述
                - matched_task: 匹配到的任务描述（None 表示未匹配）
                - principles: 匹配到的原则列表（空列表表示未匹配）
        """
        # 如果 memory 为空，全部返回未匹配
        if not self.memory:
            return [(task, None, []) for task in task_descs]
        
        memory_tasks = list(self.memory.keys())
        results = []
        
        # 为每个输入任务构造匹配 prompts
        all_prompts = []
        task_prompt_map = []  # 记录每个 prompt 对应的 (input_idx, memory_idx)
        
        for i, input_task in enumerate(task_descs):
            for j, memory_task in enumerate(memory_tasks):
                prompt = self.prompts.get_semantic_match_prompt(input_task.strip(), memory_task)
                all_prompts.append(prompt)
                task_prompt_map.append((i, j))
        
        # 批量推理
        try:
            responses = batch_inference(all_prompts, temperature=0.0, max_tokens=5, use_tqdm=False)
            
            # 解析结果，为每个 input_task 找到第一个匹配的 memory_task
            matched = [None] * len(task_descs)  # 记录每个输入任务是否已匹配
            
            for idx, response in enumerate(responses):
                input_idx, memory_idx = task_prompt_map[idx]
                
                # 如果该输入任务已经匹配过，跳过
                if matched[input_idx] is not None:
                    continue
                
                # 判断是否匹配（新 prompt 严格输出 "Yes" 或 "No"）
                response_clean = response.strip().lower()
                if 'yes' in response_clean:
                    memory_task = memory_tasks[memory_idx]
                    matched[input_idx] = memory_task
            
            # 构造返回结果
            for i, input_task in enumerate(task_descs):
                if matched[i] is not None:
                    results.append((input_task, matched[i], self.memory[matched[i]]))
                else:
                    results.append((input_task, None, []))
                    
        except Exception as e:
            logging.error(f"批量语义匹配失败: {e}")
            # 失败时返回全部未匹配
            results = [(task, None, []) for task in task_descs]
        
        return results

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

    def batch_resolve_conflicts(self, conflict_pairs: List[tuple]) -> List[list]:
        """批量解决原则冲突
        
        Args:
            conflict_pairs: List[(old_principles, new_principles)] 冲突对列表
            
        Returns:
            List[list]: 解决后的原则列表
        """
        if not conflict_pairs:
            return []
        
        # 过滤掉空的情况
        valid_pairs = []
        valid_indices = []
        results = [None] * len(conflict_pairs)
        
        for i, (old, new) in enumerate(conflict_pairs):
            if not old:
                results[i] = new
            elif not new:
                results[i] = old
            else:
                valid_pairs.append((old, new))
                valid_indices.append(i)
        
        # 如果没有需要处理的冲突
        if not valid_pairs:
            return results
        
        # 构造批量 prompts
        batch_prompts = []
        for old, new in valid_pairs:
            prompt = self.prompts.get_principle_match_prompt(
                new_principle="\n".join(new),
                old_principle="\n".join(old)
            )
            batch_prompts.append(prompt)
        
        # 批量推理
        try:
            responses = batch_inference(batch_prompts, use_tqdm=False)
            
            # 解析每个响应
            for idx, result in enumerate(responses):
                original_idx = valid_indices[idx]
                old, new = valid_pairs[idx]
                
                pattern = r'(\{\s*"comparisons"\s*:\s*\[.*?\]\s*\})'
                match = re.search(pattern, result, flags=re.DOTALL)
                
                if match:
                    try:
                        comparisons_json_str = match.group(1)
                        comparisons_json_str = comparisons_json_str.replace('{{', '{').replace('}}', '}')
                        relations_dict = json.loads(comparisons_json_str)
                        relations = relations_dict.get("comparisons", [])
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] JSON decode failed: {e}")
                        relations = []
                else:
                    print(f"[WARN] No valid 'comparisons' JSON found in model output")
                    relations = []
                
                retained_old = set(old)
                retained_new = []
                
                if not relations:
                    results[original_idx] = list(retained_old)
                else:
                    for match_item in relations:
                        old_rule = match_item.get("old", "").strip()
                        new_rule = match_item.get("new", "").strip()
                        relation = match_item.get("relation", "").strip()

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
                    
                    results[original_idx] = list(retained_old.union(retained_new))
        
        except Exception as e:
            logging.error(f"批量冲突解决失败: {e}")
            # 失败时保留旧原则
            for idx in valid_indices:
                if results[idx] is None:
                    results[idx] = conflict_pairs[idx][0]
        
        return results

    def _resolve_conflicts(self, old: list, new: list) -> list:
        if not old:
            return new
        if not new:
            return old

        prompt = self.prompts.get_principle_match_prompt(
            new_principle="\n".join(new),
            old_principle="\n".join(old)
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


