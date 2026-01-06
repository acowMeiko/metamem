"""
Centralized prompt template management.

All prompts are stored here using the Template Registry Pattern.
No prompts should be hardcoded in business logic.

Usage:
    prompts = PromptTemplate()
    prompt = prompts.get_direct_answer_prompt(question)
"""

from string import Template
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class PromptTemplate:
    """
    Centralized prompt template manager.
    
    All prompts are defined as class attributes and accessed via methods.
    This ensures:
    - Single source of truth for all prompts
    - Easy maintenance and versioning
    - No hardcoded prompts in business logic
    """
    
    # ==================== System Prompts ====================
    
    SYSTEM_EXPERT = "You are an expert in solving problems."
    
    # ==================== Task Description Prompt ====================
    
    TASK_DESC_TEMPLATE = Template('''
Your task is to identify and extract the main task description from a given question. First, analyze the domain of the question, categorize it into a relevant subcategory, and then generate a concise, clear, and abstract task description that reflects the core objective.

**Steps to Perform Structured Analysis:**
1. **Analyze the domain of the question:** Determine the field or category the question belongs to
2. **Categorize the task:** Identify the specific type of problem within that domain
3. **Generate the task description:** Based on the identified domain and subcategory, create a task description that is:
   - Concise and clear
   - Abstract and general
   - Focused on the core objective
   - Free of unnecessary details or background information

**Question:** 
$question

**Output Format (JSON only, no extra text):**
{
  "taskDescription": {
    "description": "Clear, abstract, and specific description of the task, focusing on the core action or objective."
  }
}
''')
    
    # ==================== Answer Generation Prompts ====================
    
    DIRECT_ANSWER_TEMPLATE = Template('''
<system>
You are an expert in solving problems. Please answer the question below. Let's think step by step.
</system>
$question
''')
    
    GUIDED_ANSWER_TEMPLATE = Template(r'''
<system>
You are an expert in solving problems. Provide a clear step-by-step solution.
After your step-by-step reasoning, you MUST output the final answer in EXACTLY this format:

#### <final_answer>

Rules for <final_answer>:
- For numeric answers: output the number directly (e.g., #### 42 or #### 3.14 or #### -5)
- For expressions: output the expression (e.g., #### \frac{3}{4} or #### 2\pi)
- For text answers: output the text (e.g., #### Yes or #### \text{Evelyn})
- The answer should be on a single line after ####
- No extra text, quotes, or tags after the #### line

Examples:
#### 14400
#### \frac{14}{3}
#### 3.1415
#### \text{Yes}
#### -2

Now solve the question below step by step, and finish by outputting the final answer in the format specified above.
</system>

Please ensure your response adheres to the specified principles.

<input>
Question: ${question}
Principles to follow:
${principles}
</input>
''')
    
    # ==================== Analysis Prompts ====================
    
    DIFF_ANALYSIS_TEMPLATE = Template('''
You are an expert in analyzing and comparing task responses to identify *fine-grained*, *task-relevant*, and *impactful* differences between answers that affect quality.

**Task:**
Given a high-quality and a low-quality answer to the same task, identify detailed differences that reflect meaningful changes in correctness, reasoning, completeness, or clarity.

**Steps:**
1. **Understand the task type:** Identify whether the task involves reasoning, generation, factual recall, explanation, etc.
2. **Perform targeted comparison:** Compare the answers component by component (sentence by sentence, step by step, or idea by idea).
3. **Identify key differences:** For each meaningful difference:
   - Quote or paraphrase the specific content from both answers
   - Indicate the aspect being affected
   - Explain why this difference matters
4. **You MUST internally perform domain analysis and task categorization, but you MUST NOT output any analysis, reasoning steps, or explanations.

**Important guidelines:**
- Avoid vague language like "clearer" or "more logical" unless supported by concrete details
- Specify missing steps, incorrect reasoning, unsupported claims, or structural flaws
- Use task-specific language

**Input:**
Question: $question
Low-quality Answer: $pred
High-quality Answer: $label

**Output (JSON only, no extra text):**
{{
  "differences": [
    {{
      "Aspect": "Aspect being evaluated",
      "HighQuality": "Quoted or paraphrased content from the HQ answer",
      "LowQuality": "Quoted or paraphrased content from the LQ answer",
      "Differences": "Detailed explanation of why this difference affects answer quality"
    }}
  ]
}}
''')
    
    PRINCIPLE_TEMPLATE = Template('''
You are a prompt engineering expert skilled in deriving precise and generalizable principles that improve language model outputs. Your task is to formulate principles based on observed differences between high- and low-quality answers, ensuring each principle reflects a specific failure pattern and offers guidance for correction.

**Task:**
Generate reusable and insightful improvement principles based on observed differences between two answers.

**Steps:**
1. Carefully examine each identified difference and explain how it impacts the answer quality (e.g., logical structure, factual accuracy, clarity, completeness, or relevance).
2. For each difference, derive a principle that captures the core insight and helps guide future answer generation.
3. Ensure each principle is general enough to be reused across similar tasks, yet clearly grounded in the specific difference observed.
4. Respond strictly in the following JSON format.

**Input:**
Question: $question
Difference: $diff_list

**Output (JSON only, no extra text):**
{{
  "output": [
    {{
      "Principle": "State a clear and generalizable insight derived from the difference."
    }}
  ]
}}
''')
    
    # ==================== Memory Management Prompts ====================
    
    PRINCIPLE_MATCH_TEMPLATE = Template('''
Your goal is to compare the new_principle against each of the existing_principles, and decide one of the following for each:
1. Redundant: if the new and old principle express essentially the same idea. Prefer the newer one.
2. Conflicting: if the two principles provide contradictory guidance. Keep the one that is more general or correct.
3. Irrelevant: if the existing principle is not applicable to the current task anymore. Suggest deletion.

Please return your evaluation in the following JSON format exactly:
{{
    "comparisons": [
        {{
            "old": "<text of the existing principle>",
            "new": "<text of the new principle>",
            "relation": "Redundant | Conflicting | Irrelevant"
        }}
    ]
}}

<input>
New_principle: $new
Existing_principle: $old
</input>
''')
    
    SEMANTIC_MATCH_TEMPLATE = Template('''
Please strictly compare the core semantics of the following two mathematical reasoning task descriptions and determine whether they express exactly the same meaning.

Core judgment criteria:
1.  The **essence of the mathematical problem stem** is consistent (including known conditions, solving objectives, and constraint limitations);
2.  There are no substantive differences in the **mathematical objects, operational relationships, and reasoning directions** involved;
3.  Only differences in expression order or wording (e.g., "Find the value of A" vs "Calculate the result of A") without changing the core solving objective are considered the same;
4.  If there are additions/reductions of known conditions, changes in solving objectives, or modifications of constraint conditions, they are considered different.

Task 1: $task1
Task 2: $task2

Notes:
1.  Only judge the semantic consistency of the task descriptions, without performing specific mathematical calculations;
2.  The output result can only be "Yes" or "No", and no additional explanations, punctuation, or supplementary content are allowed.

Are they the same:'''
    )
    
    # ==================== Dialogue Format Template ====================
    
    DIALOGUE_FORMAT = Template("""<|im_start|>user\n$query<|im_end|>\n<|im_start|>assistant\n""")
    
    # ==================== Public Methods ====================
    
    def get_task_description_prompt(self, question: str) -> str:
        """
        Get prompt for generating task description.
        
        Args:
            question: The question to analyze
            
        Returns:
            Formatted prompt
        """
        return self.TASK_DESC_TEMPLATE.substitute(question=question)
    
    def get_direct_answer_prompt(self, question: str) -> str:
        """
        Get prompt for direct answer generation (no guidance).
        
        Args:
            question: The question to answer
            
        Returns:
            Formatted prompt
        """
        return self.DIRECT_ANSWER_TEMPLATE.substitute(question=question)
    
    def get_guided_answer_prompt(self, question: str, principles: str) -> str:
        """
        Get prompt for guided answer generation (with principles).
        
        Args:
            question: The question to answer
            principles: Principles to follow (formatted string or list joined by newlines)
            
        Returns:
            Formatted prompt
        """
        return self.GUIDED_ANSWER_TEMPLATE.substitute(
            question=question,
            principles=principles
        )
    
    def get_diff_analysis_prompt(self, question: str, pred: str, label: str) -> str:
        """
        Get prompt for difference analysis.
        
        Args:
            question: The original question
            pred: Predicted (low-quality) answer
            label: Ground truth (high-quality) answer
            
        Returns:
            Formatted prompt
        """
        return self.DIFF_ANALYSIS_TEMPLATE.substitute(
            question=question,
            pred=pred,
            label=label
        )
    
    def get_principle_prompt(self, question: str, diff_list: str) -> str:
        """
        Get prompt for principle extraction.
        
        Args:
            question: The original question
            diff_list: Difference analysis text
            
        Returns:
            Formatted prompt
        """
        return self.PRINCIPLE_TEMPLATE.substitute(
            question=question,
            diff_list=diff_list
        )
    
    def get_principle_match_prompt(self, new_principle: str, old_principle: str) -> str:
        """
        Get prompt for principle matching/conflict resolution.
        
        Args:
            new_principle: New principle to compare
            old_principle: Existing principle to compare against
            
        Returns:
            Formatted prompt
        """
        return self.PRINCIPLE_MATCH_TEMPLATE.substitute(
            new=new_principle,
            old=old_principle
        )
    
    def get_semantic_match_prompt(self, task1: str, task2: str) -> str:
        """
        Get prompt for semantic task matching.
        
        Args:
            task1: First task description
            task2: Second task description
            
        Returns:
            Formatted prompt
        """
        return self.SEMANTIC_MATCH_TEMPLATE.substitute(
            task1=task1,
            task2=task2
        )
    
    def format_dialogue(self, query: str) -> str:
        """
        Format a query in dialogue format (for certain models).
        
        Args:
            query: The query/prompt to format
            
        Returns:
            Dialogue-formatted string
        """
        return self.DIALOGUE_FORMAT.substitute(query=query)
    
    def get_all_templates(self) -> Dict[str, Template]:
        """
        Get all templates for inspection or custom usage.
        
        Returns:
            Dictionary of template name -> Template object
        """
        return {
            'task_description': self.TASK_DESC_TEMPLATE,
            'direct_answer': self.DIRECT_ANSWER_TEMPLATE,
            'guided_answer': self.GUIDED_ANSWER_TEMPLATE,
            'diff_analysis': self.DIFF_ANALYSIS_TEMPLATE,
            'principle': self.PRINCIPLE_TEMPLATE,
            'principle_match': self.PRINCIPLE_MATCH_TEMPLATE,
            'semantic_match': self.SEMANTIC_MATCH_TEMPLATE,
            'dialogue_format': self.DIALOGUE_FORMAT
        }


# Global instance for easy access
_prompt_template = PromptTemplate()


def get_prompt_template() -> PromptTemplate:
    """Get the global PromptTemplate instance."""
    return _prompt_template
