import random
from typing import List
from collections import defaultdict

import ray

from marti.worlds.workflows.base import MultiAgentWorkflow


initial_prompt_template = {
    "math": """
Give an initial answer to the following question:
{question}

The final answer should be enclosed within \\boxed{{}}, e.g. \\boxed{{n}}.
""",
    "coding": """
Give an initial answer to the following question:
{question}

Provide your Python code to solve the problem. The code should be enclosed within ```python and ``` tags, e.g.
```python
def function():
    pass
```
""",
    "science": """
Give an initial answer to the following question:
{question}

The final answer should be a decimal number enclosed within \\boxed{{}}, e.g. \\boxed{{1}}, \\boxed{{0.1}}, or \\boxed{{0.01}}. The unit part given in the problem should not be enclosed.
"""
}


def construct_initial_prompt(question: str, task: str) -> str:
    return initial_prompt_template[task].format(question=question)


debate_prompt_template = {
    "math": """
Here are solutions from other agents:
{responses_str}

Using these responses as additional advice, give an updated answer to the following question:
{question}

The final answer should be enclosed within \\boxed{{}}, e.g. \\boxed{{n}}.
""",
    "coding": """
Here are solutions from other agents:
{responses_str}

Using these responses as additional advice, give an updated answer to the following question:
{question}

Provide your Python code to solve the problem. The code should be enclosed within ```python and ``` tags, e.g.
```python
def function():
    pass
```
""",
    "science": """
Here are solutions from other agents:
{responses_str}

Using these responses as additional advice, give an updated answer to the following question:
{question}

The final answer should be a decimal number enclosed within \\boxed{{}}, e.g. \\boxed{{1}}, \\boxed{{0.1}}, or \\boxed{{0.01}}. The unit part given in the problem should not be enclosed.
"""
}


def construct_debate_prompt(question: str, previous_responses: List[str], task: str) -> str:
    responses_str = "\n\n".join([f"Agent {i+1}: {resp}" for i, resp in enumerate(previous_responses)])
    return debate_prompt_template[task].format(responses_str=responses_str, question=question)


class MultiAgentDebate(MultiAgentWorkflow):
    def __init__(
        self,
        agent_list,
        template_id,
        sampling_params,
        *args, **kwargs
    ):
        super().__init__(agent_list=agent_list, template_id=template_id, sampling_params=sampling_params)
        self.num_rounds = kwargs.get("num_rounds", kwargs.get("num_rounds", len(self.agent_list)))
        self.max_others = kwargs.get("max_others", kwargs.get("max_others", 5))
        self.contain_self = kwargs.get("contain_self", kwargs.get("contain_self", True))
        self.shuffle_responses = kwargs.get("shuffle_responses", kwargs.get("shuffle_responses", True))
        self.debate_histories = []

    def process_prompt_for_thinking_model(self, solution, agent_id):
        if self.agent_list[agent_id]["is_reasoning_model"]:
            solution = solution.split("</think>")[-1].strip().strip("</answer>").strip("<answer>")
        return solution

    def prepare_debate_round(self, problems: List[str], prev_responses: List[List[str]], round_num: int):
        all_prompts = []
        for prob_idx, (question, problem_responses) in enumerate(zip(problems, prev_responses)):
            # For each agent in each problem
            for agent_idx in range(self.num_agents):
                if round_num == 0:
                    task = self.tasks[prob_idx]
                    prompt = construct_initial_prompt(question, task)
                else:
                    # Get other agents' responses and extract answer spans
                    if self.contain_self:
                        other_responses = problem_responses
                    else:
                        other_responses = problem_responses[:agent_idx] + problem_responses[agent_idx+1:]
                    other_responses = [self.process_prompt_for_thinking_model(resp, aid) for aid, resp in enumerate(other_responses)]
                    if self.shuffle_responses:
                        random.shuffle(other_responses)
                    other_responses = other_responses[:min(self.max_others, len(other_responses))]
                    task = self.tasks[prob_idx]
                    prompt = construct_debate_prompt(question, other_responses, task)
                all_prompts.append([agent_idx, prompt])
        return all_prompts

    def distribute_prompts(self, prompts) -> List[str]:
        """Distribute prompts to agents and collect responses"""
        indexed_prompts = [(i, agent_id, text) for i, (agent_id, text) in enumerate(prompts)]

        agent_prompt_map = defaultdict(list)
        for idx, agent_id, text in indexed_prompts:
            agent_prompt_map[agent_id].append((idx, text))

        final_results = [None] * len(prompts)

        remote_refs, idx_lists = [], []
        for agent_id, indexed_texts in agent_prompt_map.items():
            llms = self.agent_list[agent_id]["llms"]
            tokenizer = self.agent_list[agent_id]["tokenizer"]
            is_reasoning_model = self.agent_list[agent_id]["is_reasoning_model"]

            processed_prompts = []
            for idx, text in indexed_texts:
                chat_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                if is_reasoning_model:
                    chat_prompt += "<think>"
                processed_prompts.append([idx, chat_prompt])

            n_llms = len(llms)
            total_prompts = len(processed_prompts)
            batch_size = (total_prompts + n_llms - 1) // n_llms

            for i, llm in enumerate(llms):
                chunk = processed_prompts[i * batch_size: (i + 1) * batch_size]

                chunk_indices = [idx for idx, _ in chunk]
                chunk_prompts = [prompt for _, prompt in chunk]

                ref = llm.generate.remote(
                    chunk_prompts,
                    sampling_params=self.sampling_params
                )
                remote_refs.append(ref)
                idx_lists.append(chunk_indices)

        results = ray.get(remote_refs)
        for chunk_indices, outputs in zip(idx_lists, results):
            for idx, output in zip(chunk_indices, outputs):
                final_results[idx] = output.outputs[0].text

        assert None not in final_results
        # TODO: Add thinking token to the response
        # if self.forced_thinking:
        #     final_results = ["<think>" + text for text in final_results]

        for idx in random.sample(list(range(len(prompts))), 2):
            print(
                f"Prompt >>>> (Agent {prompts[idx][0]}) " + repr(prompts[idx][1]))
            print("Output >>>> " + repr(final_results[idx]))

        return final_results

    def organize_responses_by_problem_and_agents(self, round_num, num_problems, all_prompts, all_responses):
        for prob_idx in range(num_problems):
            prob_prompts = all_prompts[
                prob_idx * self.num_agents: (prob_idx + 1) * self.num_agents
            ]
            prob_responses = all_responses[
                prob_idx * self.num_agents: (prob_idx + 1) * self.num_agents
            ]

            for agent_idx in range(self.num_agents):
                # assert prob_prompts[agent_idx][0] == agent_idx, f"{prob_prompts[agent_idx][0]} vs {agent_idx}"
                self.debate_histories[prob_idx][agent_idx].append({
                    "user": prob_prompts[agent_idx][1],  # [idx, prompt]
                    "assistant": prob_responses[agent_idx],
                    "agent_id": agent_idx,
                    "turn_id": round_num
                })

    def run(self, problems: List[str], tasks: List[str] = None):
        self.tasks = tasks
        num_problems = len(problems)
        """
        [
            [{}, {}, {}] # turn 1 - agent 1, agent 2, agent 3
            [{}] # turn 2 - agent 1
            [{}, {}] # turn 3 - agent 1, agent 2
            ...
        ]
        """
        self.debate_histories = [
                [
                    [] for _ in range(self.num_agents)
                ] for _ in range(num_problems)
            ]

        # Initial round
        initial_prompts = self.prepare_debate_round(
            problems=problems,
            prev_responses=[[] for _ in range(num_problems)],
            round_num=0
        )

        # Get initial responses
        all_responses = self.distribute_prompts(initial_prompts)

        self.organize_responses_by_problem_and_agents(
            round_num=0,
            num_problems=num_problems,
            all_prompts=initial_prompts,
            all_responses=all_responses
        )

        for round_num in range(1, self.num_rounds):
            # Get previous round's responses for each problem
            prev_responses = [
                    [agent[-1]["assistant"] for agent in agents]
                    for agents in self.debate_histories
            ]

            # Prepare all prompts for this round
            round_prompts = self.prepare_debate_round(
                problems=problems,
                prev_responses=prev_responses,
                round_num=round_num
            )

            # Get responses
            all_responses = self.distribute_prompts(round_prompts)

            self.organize_responses_by_problem_and_agents(
                round_num=round_num,
                num_problems=num_problems,
                all_prompts=round_prompts,
                all_responses=all_responses
            )

    def get_history(self):
        return self.debate_histories
