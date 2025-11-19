from typing import List, Dict, Union
from collections import defaultdict

import ray
import numpy as np
from bs4 import BeautifulSoup

from marti.worlds.workflows.base import MultiAgentWorkflow

# --- robust score parsing utils ---
import re, logging
LOG = logging.getLogger(__name__)
_TAG_RE = re.compile(r"<\s*score\s*>\s*([0-9]+)\s*<\s*/\s*score\s*>", re.I)
_DIGIT_RE = re.compile(r"\b([1-3])\b")  # 只接受 1/2/3

def _extract_text(resp):
    # 统一提文本的辅助函数（None/不同结构都处理）
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    t = getattr(resp, "text", None)
    if isinstance(t, str) and t:
        return t
    outs = getattr(resp, "outputs", None)
    if isinstance(outs, (list, tuple)) and outs:
        t = getattr(outs[0], "text", None)
        if isinstance(t, str) and t:
            return t
    if isinstance(resp, dict):
        if isinstance(resp.get("text"), str):
            return resp["text"]
        ch = resp.get("choices") or resp.get("outputs")
        if isinstance(ch, list) and ch:
            t = ch[0].get("text") if isinstance(ch[0], dict) else getattr(ch[0], "text", None)
            if isinstance(t, str):
                return t
    if isinstance(resp, (list, tuple)) and resp:
        return _extract_text(resp[0])
    return ""

def _safe_extract_int(text, default=None, lo=0, hi=10):
    if not text:
        return default
    last = str(text).strip().splitlines()[-1]
    m = _SCORE_RE.search(last)
    if not m:
        return default
    try:
        v = int(m.group(1))
    except Exception:
        return default
    return max(lo, min(hi, v))
# --- end utils ---

solver_prompt = {
    "math": """
The problem is presented as follows:
{problem}

Current discussion on the problem is presented as follows for your reference:
{discussion}

Provide your step-by-step solution to the problem. The final answer should be enclosed within \\boxed{{}}, e.g. \\boxed{{n}}.
""",
    "coding": """
The problem is presented as follows:
{problem}

Current discussion on the problem is presented as follows for your reference:
{discussion}

First analyze the requirements and form a step-by-step plan to implement the solution. Then provide your Python code to solve the problem. The code should be enclosed within ```python and ``` tags, e.g.
```python
def function():
    pass
```
Apart from your analysis and plan, only one snippet of code is allowed in your solution.
""",
    "science": """
The problem is presented as follows:
{problem}

Current discussion on the problem is presented as follows for your reference:
{discussion}

Provide your step-by-step solution to the problem. The final answer should be a decimal number enclosed within \\boxed{{}}, e.g. \\boxed{{1}}, \\boxed{{0.1}}, or \\boxed{{0.01}}. The unit part given in the problem should not be enclosed.
""",
}


evaluator_prompt = {
    "math": """
The problem is presented as follows:
{problem}

You are required to evaluate the following solution:
{solution}

You should point out every possible error and defect in the solution. Provide your evaluation by listing all the mistakes you find in the solution, specifying what is wrong and why. Keep your evaluation concise and clear. Avoid using a lot of words to retell the reasoning process.
""",
    "coding": """
The problem is presented as follows:
{problem}

You are required to evaluate the following solution:
{solution}

You should point out every possible error and defect in the solution. Provide your evaluation by listing test cases that cannot be passed and explaining the underlying reasons. Keep your evaluation concise and clear. Avoid using a lot of words to retell the solution code.
""",
    "science": """
The problem is presented as follows:
{problem}

You are required to evaluate the following solution:
{solution}

You should point out every possible error and defect in the solution. Provide your evaluation by listing all the mistakes you find in the solution, specifying what is wrong and why. Keep your evaluation concise and clear. Avoid using a lot of words to retell the reasoning process.
"""
}


scorer_prompt = {
    "math": """
The problem is presented as follows:
{problem}

You are required to score the following solution:
{solution}

The evaluation on the solution is presented as follows:
{evaluation}

You should consider the rationality of the evaluation and score the solution. The score should be an integer between 1 and 3 with the following standards:
3: The solution is completely correct, and none of the mistakes mentioned in the evaluation is effective.
2: Some minor mistakes mentioned in the evaluation do exist, but they do not affect the overall correctness.
1: Some of the mistakes mentioned in the evaluation are fatal, which directly lead to an incorrect answer.

Your score should be enclosed within "<score>" and "</score>" tags. You should also briefly explain the reasons before providing your score. Keep your scoring concise and clear. Avoid using a lot of words to retell the reasoning process.

For example:
The calculation error mentioned in the evaluation cannot be ignored and leads to an incorrect answer.
<score>1</score>
""",
    "coding": """
The problem is presented as follows:
{problem}

You are required to score the following solution:
{solution}

The evaluation on the solution is presented as follows:
{evaluation}

You should consider the rationality of the evaluation and score the solution. The score should be an integer between 1 and 3 with the following standards:
3: The solution is completely correct, and none of the mistakes mentioned in the evaluation is effective.
2: Some minor mistakes mentioned in the evaluation do exist, but they do not affect the overall correctness.
1: Some of the mistakes mentioned in the evaluation are fatal, which directly lead to an incorrect answer.

Your score should be enclosed within "<score>" and "</score>" tags. You should also briefly explain the reasons before providing your score. Keep your scoring concise and clear. Avoid using a lot of words to retell the reasoning process.

For example:
As the evaluation points out, the last test case will actually lead to index out of bounds.
<score>1</score>
""",
    "science": """
The problem is presented as follows:
{problem}

You are required to score the following solution:
{solution}

The evaluation on the solution is presented as follows:
{evaluation}

You should consider the rationality of the evaluation and score the solution. The score should be an integer between 1 and 3 with the following standards:
3: The solution is completely correct, and none of the mistakes mentioned in the evaluation is effective.
2: Some minor mistakes mentioned in the evaluation do exist, but they do not affect the overall correctness.
1: Some of the mistakes mentioned in the evaluation are fatal, which directly lead to an incorrect answer.

Your score should be enclosed within "<score>" and "</score>" tags. You should also briefly explain the reasons before providing your score. Keep your scoring concise and clear. Avoid using a lot of words to retell the reasoning process.

For example:
The calculation error mentioned in the evaluation cannot be ignored and leads to an incorrect answer.
<score>1</score>
"""
}


class CoMASEnvironment(MultiAgentWorkflow):
    def __init__(self, agent_list, template_id, sampling_params, *args, **kwargs):
        super().__init__(agent_list=agent_list, template_id=template_id, sampling_params=sampling_params)
        self.num_rounds = kwargs.get("num_rounds", 1)
        self.num_references = kwargs.get("num_references", 1)
        self.num_agents = len(agent_list)
        self.comas_history = None

    def _render_discussion(self, discussion_list: List[Dict]) -> str:
        discussion_content = ""
        if len(discussion_list) == 0:
            discussion_content += "(Empty discussion)\n\n"
        else:
            if self.num_references > len(discussion_list):
                num_references = len(discussion_list)
            else:
                num_references = self.num_references
            discussion_list = np.random.choice(
                discussion_list,
                size=num_references,
                replace=False
            ) # randomly select a subset of discussions
            for discussion_id, discussion in enumerate(discussion_list):
                solver_step = discussion["solver_step"]
                evaluator_step = discussion["evaluator_step"]
                discussion_content += f"=== Discussion {discussion_id + 1} ===\n"
                discussion_content += f"Solution: {solver_step['solution_content'].strip()}\n"
                discussion_content += f"Evaluation: {evaluator_step['evaluation_content'].strip()}\n\n"
        return discussion_content.strip()

    def _extract_score(self, scorer_content: str) -> int:
        """
        尽量解析 <score>n</score>；不行就从文本里找最后一个 1/2/3；
        还不行就返回 2（中性）作为兜底，确保永远返回 int。
        """
        txt = (scorer_content or "").strip()
        if not txt:
            LOG.warning("scorer content empty; fallback to 2")
            return 2

        m = _TAG_RE.search(txt)
        if m:
            try:
                val = int(m.group(1))
                return 1 if val <= 1 else 3 if val >= 3 else 2
            except Exception:
                pass

        # 兼容被裁剪成 '... <score>1</' 的情况：从全文里找“单个 1-3”
        m2 = _DIGIT_RE.findall(txt)
        if m2:
            try:
                val = int(m2[-1])
                return 1 if val <= 1 else 3 if val >= 3 else 2
            except Exception:
                pass

        LOG.warning("Failed to parse scorer content; fallback to 2 | raw=%r", txt[:2000])
        return 2

    def _parallel_inference(self, requests: List[Dict]) -> List[Dict]:
        # Collect requests by agent
        collected_requests = defaultdict(list)
        for request_id, request in enumerate(requests):
            collected_requests[request["agent_id"]].append({
                "request_id": request_id,
                "prompt": request["prompt"],
                "appendix": request["appendix"]
            })

        # Conduct inference in parallel
        indices_list = []
        references_list = []
        for agent_id, agent_requests in collected_requests.items():
            llms = self.agent_list[agent_id]["llms"]
            tokenizer = self.agent_list[agent_id]["tokenizer"]

            # Convert prompts to chat format
            indices = []
            prompts = []
            for request in agent_requests:
                indices.append({
                    "request_id": request["request_id"],
                    "agent_id": agent_id,
                    "prompt": request["prompt"],
                    "appendix": request["appendix"]
                })
                chat_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": request["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(chat_prompt)

            # Distribute requests to LLMs
            batch_size = (len(prompts) + len(llms) - 1) // len(llms)
            for llm_id, llm in enumerate(llms):
                batch_indices = indices[llm_id * batch_size: (llm_id + 1) * batch_size]
                batch_prompts = prompts[llm_id * batch_size: (llm_id + 1) * batch_size]
                if batch_prompts:
                    indices_list.append(batch_indices)
                    references = llm.generate.remote(
                        batch_prompts,
                        sampling_params=self.sampling_params
                    )
                    references_list.append(references)

        # Collect all results
        responses = [None] * len(requests)

        completions_list = ray.get(references_list)
        for indices, completions in zip(indices_list, completions_list):
            for index, completion in zip(indices, completions):
                safe_text = _extract_text(completion)
                if safe_text is None:
                    # 给空串并写一条限流日志（不阻断训练）
                    if LOG.isEnabledFor(logging.WARNING):
                        LOG.warning("LLM completion is None; request_id=%s agent=%s", index["request_id"], index["agent_id"])
                    safe_text = ""
                responses[index["request_id"]] = {
                    "agent_id": index["agent_id"],
                    "prompt": index["prompt"],
                    "completion": safe_text,
                    "appendix": index["appendix"]
                }

        # 3) 回收结果并按 request_id 写回
        responses = [None] * len(requests)
        if references_list:
            completions_list = ray.get(references_list)
            for batch_indices, completions in zip(indices_list, completions_list):
                for index, completion in zip(batch_indices, completions):
                    text = _extract_text(completion) or ""
                    responses[index["request_id"]] = {
                        "agent_id": index["agent_id"],
                        "prompt": index["prompt"],
                        "completion": text,
                        "appendix": index["appendix"],
                    }
        return responses

    def run(self, problems: List[str], tasks: List[str]):
        num_diss_per_round = len(problems) * self.num_agents
        num_acts_per_agent = num_diss_per_round // self.num_agents
        agent_order = list(range(self.num_agents)) * num_acts_per_agent

        # initialize comas history for each problem
        comas_history = {
            "problem_list": []
        }
        for problem, task in zip(problems, tasks):
            comas_history["problem_list"].append({
                "problem_content": problem,
                "task_name": task,
                "round_list": []
            })

        for round_id in range(self.num_rounds):
            # prepare the round for each problem
            print(f"Simulating round {round_id + 1} out of {self.num_rounds}...")
            for problem_id, problem in enumerate(comas_history["problem_list"]):
                problem["round_list"].append({
                    "discussion_list": [{
                        "solver_step": {},
                        "evaluator_step": {},
                        "scorer_step": {}
                    } for _ in range(self.num_agents)]
                })

            # solver step - parallel inference across all agents
            # for each problem, each agent will generate a solution
            print("Starting solver step...")
            solver_requests = []

            for problem_id, problem in enumerate(comas_history["problem_list"]):
                round = problem["round_list"][round_id]
                discussion_list = []
                if round_id > 0:
                    last_round = problem["round_list"][round_id - 1]
                    discussion_list = last_round["discussion_list"]
                prompt = solver_prompt[problem["task_name"]].format(
                    problem=problem["problem_content"].strip(),
                    discussion=self._render_discussion(discussion_list)
                ).strip()
                for agent_id in range(self.num_agents):
                    discussion_id = agent_id
                    solver_requests.append({
                        "agent_id": agent_id,
                        "prompt": prompt,
                        "appendix": {
                            "problem_id": problem_id,
                            "round_id": round_id,
                            "discussion_id": discussion_id
                        }
                    })

            solver_responses = self._parallel_inference(solver_requests)

            for response in solver_responses:
                problem_id = response["appendix"]["problem_id"]
                round_id = response["appendix"]["round_id"]
                discussion_id = response["appendix"]["discussion_id"]
                problem = comas_history["problem_list"][problem_id]
                round = problem["round_list"][round_id]
                discussion = round["discussion_list"][discussion_id]
                discussion["solver_step"] = {
                    "solution_content": response["completion"],
                    "chat_history": {
                        "type": "solution",
                        "agent_id": response["agent_id"],
                        "prompt": response["prompt"],
                        "completion": response["completion"]
                    }
                }

            # evaluator step - parallel inference across all agents
            # for each problem, for each solution, each agent will generate an evaluation
            print("Starting evaluator step...")
            np.random.shuffle(agent_order)
            agent_selector = iter(agent_order)
            evaluator_requests = []

            for problem_id, problem in enumerate(comas_history["problem_list"]):
                round = problem["round_list"][round_id]
                for discussion_id, discussion in enumerate(round["discussion_list"]):
                    agent_id = next(agent_selector)
                    solver_step = discussion["solver_step"]
                    prompt = evaluator_prompt[problem["task_name"]].format(
                        problem=problem["problem_content"].strip(),
                        solution=solver_step["solution_content"].strip()
                    ).strip()
                    evaluator_requests.append({
                        "agent_id": agent_id,
                        "prompt": prompt,
                        "appendix": {
                            "problem_id": problem_id,
                            "round_id": round_id,
                            "discussion_id": discussion_id
                        }
                    })

            evaluator_responses = self._parallel_inference(evaluator_requests)

            for response in evaluator_responses:
                problem_id = response["appendix"]["problem_id"]
                round_id = response["appendix"]["round_id"]
                discussion_id = response["appendix"]["discussion_id"]
                problem = comas_history["problem_list"][problem_id]
                round = problem["round_list"][round_id]
                discussion = round["discussion_list"][discussion_id]
                discussion["evaluator_step"] = {
                    "evaluation_content": response["completion"],
                    "chat_history": {
                        "type": "evaluator",
                        "agent_id": response["agent_id"],
                        "prompt": response["prompt"],
                        "completion": response["completion"]
                    }
                }

            # scorer step - parallel inference across all agents
            # for each problem, for each solution in the current round, each agent will generate a score
            print("Starting scorer step...")
            np.random.shuffle(agent_order)
            agent_selector = iter(agent_order)
            scorer_requests = []

            for problem_id, problem in enumerate(comas_history["problem_list"]):
                round = problem["round_list"][round_id]
                for discussion_id, discussion in enumerate(round["discussion_list"]):
                    agent_id = next(agent_selector)
                    solver_step = discussion["solver_step"]
                    evaluator_step = discussion["evaluator_step"]
                    prompt = scorer_prompt[problem["task_name"]].format(
                        problem=problem["problem_content"].strip(),
                        solution=solver_step["solution_content"].strip(),
                        evaluation=evaluator_step["evaluation_content"].strip(),
                    ).strip()
                    scorer_requests.append({
                        "agent_id": agent_id,
                        "prompt": prompt,
                        "appendix": {
                            "problem_id": problem_id,
                            "round_id": round_id,
                            "discussion_id": discussion_id
                        }
                    })

            scorer_responses = self._parallel_inference(scorer_requests)

            for response in scorer_responses:
                generated_score = self._extract_score(response["completion"])
                if generated_score is None:
                    generated_score = 1  # 或 0；建议 1，偏保守
                problem_id = response["appendix"]["problem_id"]
                round_id = response["appendix"]["round_id"]
                discussion_id = response["appendix"]["discussion_id"]
                problem = comas_history["problem_list"][problem_id]
                round = problem["round_list"][round_id]
                discussion = round["discussion_list"][discussion_id]
                discussion["scorer_step"] = {
                    "score_content": response["completion"],
                    "generated_score": generated_score,
                    "chat_history": {
                        "type": "score",
                        "agent_id": response["agent_id"],
                        "prompt": response["prompt"],
                        "completion": response["completion"]
                    }
                }

        # update the comas history
        self.comas_history = comas_history

        # print an example solution with the new pipeline
        problem = comas_history["problem_list"][0]
        round = problem["round_list"][0]
        discussion = round["discussion_list"][0]
        solver_step = discussion["solver_step"]
        evaluator_step = discussion["evaluator_step"]
        scorer_step = discussion["scorer_step"]
        print(f"--- Problem ---")
        print(f"{problem['problem_content']}\n")
        print(f"--- Solution ---")
        print(f"{solver_step['solution_content']}\n")
        print(f"--- Evaluation ---")
        print(f"{evaluator_step['evaluation_content']}\n")
        print(f"--- Score ---")
        print(f"{scorer_step['score_content']}\n")

    def get_history(self):
        return self.comas_history
