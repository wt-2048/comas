import numpy as np
from collections import Counter
from copy import deepcopy
from marti.verifiers.qwen.qwen_eval import qwen_reward_fn, qwen_reward_fn_format, extract_answer, majority_vote
from math_verify import parse, verify
from marti.verifiers.verify_coding import verify_answer as verify_coding_answer

# def jaccard_sim(ans1, ans2):
#     set1, set2 = set(ans1), set(ans2)
#     return len(set1 & set2) / len(set1 | set2)

def jaccard_sim(ans1, ans2):
    set1, set2 = set(ans1), set(ans2)
    union_len = len(set1 | set2)
    if union_len == 0:
        return 0.0
    return len(set1 & set2) / union_len

class MultiAgentRewardAllocation:
    def __init__(self,
                 name=None,
                 alpha=0.5, 
                 beta=0.5,
                 *args, **kwargs):
        print("Multi-agent Reward Allocation", locals())
        self.name = name
        self.alpha = alpha
        self.beta = beta
        self.use_ttrl = kwargs.get("ttrl", False)

    def _math_reward_fn(self, answer, golden):
        try:
            parsed_answer = parse(answer)
            parsed_golden = parse(golden)
            return 1.0 if verify(parsed_golden, parsed_answer) else 0.0
        except Exception:
            return 0.0

    def _coding_reward_fn(self, answer, golden):
        try:
            checker = golden
            verify_result = verify_coding_answer(answer, checker)
            return 1.0 if verify_result['correct'] else 0.0
        except Exception as e:
            print(f"Coding verification failed: {e}")
            return 0.0

    def _science_reward_fn(self, generated_text: str, golden_answer: str) -> float:
        try:
            parsed_result = parse(generated_text)[1]
            float_result = float(parsed_result)
            float_answer = float(golden_answer)
            return 1.0 if abs(float_result - float_answer) < 0.05 * abs(float_answer) else 0.0
        except Exception:
            return 0.0

    def _dummy_reward_fn(self, generated_text: str, golden_answer: str) -> float:
        return 0.5

    def _score(self, answer, golden, reward_fn):
        if isinstance(answer, str):
            return reward_fn(answer, golden)
        elif isinstance(answer, list):
            return [reward_fn(sub_answer, golden) for sub_answer in answer]
        else:
            raise ValueError(f"Invalid answer type: {type(answer)}")

    def assign_rewards(self, all_answers, golden_answers, task_names, turn_ids):
        local_rewards = np.zeros(np.asarray(all_answers).shape, dtype=np.float32)

        for pid, prob_answers in enumerate(all_answers):
            for aid, agent_answers in enumerate(prob_answers):
                # route per problem by task when provided
                task = task_names[pid]
                if self.use_ttrl:
                    # only math answers can be voted
                    reward_fn = self._dummy_reward_fn
                    if task == "math":
                        reward_fn = self._math_reward_fn
                    elif task == "coding":
                        reward_fn = self._dummy_reward_fn
                    elif task == "science":
                        reward_fn = self._dummy_reward_fn
                    else:
                        raise ValueError(f"Invalid task name: {task}")
                    local_rewards[pid][aid] = self._score(agent_answers, golden_answers[pid], reward_fn)
                else:
                    if task == "math":
                        reward_fn = self._math_reward_fn
                    elif task == "coding":
                        reward_fn = self._coding_reward_fn
                    elif task == "science":
                        reward_fn = self._science_reward_fn
                    else:
                        raise ValueError(f"Invalid task name: {task}")
                    local_rewards[pid][aid] = self._score(agent_answers, golden_answers[pid], reward_fn)

        # if isinstance(local_rewards[0][0], float):
        if np.issubdtype(type(local_rewards[0][0]), np.floating):
            # compute accuracy of final answer for chain / mixture
            outcome_rewards = []
            for pid, prob_answers in enumerate(all_answers):
                task = task_names[pid]
                if task == "math":
                    reward_fn = self._math_reward_fn
                elif task == "coding":
                    reward_fn = self._coding_reward_fn
                elif task == "science":
                    reward_fn = self._science_reward_fn
                else:
                    raise ValueError(f"Invalid task name: {task}")
                outcome = np.mean([reward_fn(agent[-1], golden_answers[pid]) for agent in prob_answers])
                outcome_rewards.append(outcome)
            local_rewards = self._reward_shaping(local_rewards, turn_ids)
        else:
            outcome_rewards = []
            for pid, prob_answers in enumerate(all_answers):
                task = task_names[pid]
                finals = [agent[-1] for agent in prob_answers]
                if task == "math":
                    reward_fn = self._math_reward_fn
                elif task == "coding":
                    reward_fn = self._coding_reward_fn
                elif task == "science":
                    reward_fn = self._science_reward_fn
                else:
                    raise ValueError(f"Invalid task name: {task}")
                outcome = np.mean([reward_fn(final, golden_answers[pid]) for final in finals])
                outcome_rewards.append(outcome)
            for pid, prob_rewards in enumerate(local_rewards):
                local_rewards[pid] = self._reward_shaping(prob_rewards)

        return local_rewards, outcome_rewards

    def _reward_shaping(self, local_rewards, turn_ids=None):
        if self.name in ["quality", "margin", "conditional_quality", "quality_with_outcome", "margin_with_outcome"]:
            from copy import deepcopy
            raw_rewards = deepcopy(local_rewards)
            M, T = raw_rewards.shape

            for m in range(M):
                # We need to start from turn 1, especially for mixture of agents
                # For chain / debate, turn is equal to turn_id
                start = 1
                if turn_ids is not None:
                    for turn, turn_id in enumerate(turn_ids[m]):
                        if turn_id == 1:
                            start = turn
                            break

                for t in range(start, T):
                    if turn_ids is None:
                        # compute all previous rewards for debate
                        Q = np.mean(raw_rewards[:, :t])
                    else:
                        Q = np.mean(raw_rewards[m][:t])

                    R_final = raw_rewards[m][t]
                    if "quality" in self.name:
                        dynamic_term = Q * R_final - (1-Q) * (1-R_final)
                        # print(f"{m} - {t}", Q, R_final, dynamic_term)
                        shaped_reward = R_final + self.alpha * dynamic_term
                    elif "margin" in self.name:
                        baseline_term = R_final - Q
                        shaped_reward = R_final + self.alpha * baseline_term
                    else:
                        raise NotImplementedError

                    local_rewards[m][t] = shaped_reward

            if "outcome" in self.name:
                for m in range(M):
                    for t in range(T):
                        local_rewards[m][t] += self.beta * raw_rewards[m][-1]

        return local_rewards

    def init_golden_answers_from_majority_voting(self, histories, task_names, batch_size):
        n_prompts = len(histories) // batch_size
        all_golden_answers = []
        for prompt_idx in range(n_prompts):
            all_outputs = histories[batch_size*prompt_idx: batch_size*(prompt_idx+1)]
            all_tasks = task_names[batch_size*prompt_idx: batch_size*(prompt_idx+1)]
            candidate_answers = []
            # (num_problems, num_agents, num_turns)
            for group_outputs, task_name in zip(all_outputs, all_tasks):
                for agent_outputs in group_outputs:
                    # only math answers can be voted
                    if isinstance(agent_outputs, str):
                        if task_name == "math":
                            try:
                                extracted_answer = parse(agent_outputs)[1]
                            except Exception:
                                extracted_answer = agent_outputs
                        else:
                            extracted_answer = agent_outputs
                        candidate_answers.append(extracted_answer)
                    else:
                        for agent_output in agent_outputs:
                            assert isinstance(agent_output, str)
                            if task_name == "math":
                                try:
                                    extracted_answer = parse(agent_output)[1]
                                except Exception:
                                    extracted_answer = agent_output
                            else:
                                extracted_answer = agent_output
                            candidate_answers.append(extracted_answer)
            model_answers = [answer for answer in candidate_answers if answer is not None]
            counter = Counter(model_answers)
            majority_answer, _ = counter.most_common(1)[0]
            all_golden_answers.extend([majority_answer for _ in range(batch_size)])
        assert len(all_golden_answers) == len(histories)
        return all_golden_answers

    def run(self, histories, golden_answers, task_names=None, n_samples_per_prompt=None):
        """
        MoA & CoA: [[agent1, agent2, agent3], [...]]
        MAD: [[[Agent11, Agent12, Agent13], [Agent21, Agent22, Agent23], [...]], [...]]
        """
        if self.use_ttrl:
            assert n_samples_per_prompt is not None, "`n_samples_per_prompt` should be provided for test-time RL"
            golden_answers = self.init_golden_answers_from_majority_voting(histories, task_names, n_samples_per_prompt)

        if isinstance(histories[0][0], dict):
            all_answers = [
                [agent["assistant"] for agent in agent_data]
                for agent_data in histories
            ]
            turn_ids = [
                [agent["turn_id"] for agent in agent_data]
                for agent_data in histories
            ]
        elif isinstance(histories[0][0], list):
            all_answers = [
                [[turn["assistant"] for turn in agent] for agent in agent_data]
                for agent_data in histories
            ]
            turn_ids = [
                [[turn["turn_id"] for turn in agent] for agent in agent_data]
                for agent_data in histories
            ]
        else:
            raise ValueError

        return self.assign_rewards(all_answers, golden_answers, task_names, turn_ids)


def flatten(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


class MADRewardAllocation:
    def __init__(self,
                 alpha: float = 0.5,  # weight for local advantage
                 beta: float = 0.5,   # weight for final reward
                 gamma: float = 0.9,  # discount factor for future rewards
                 subtract_baseline: bool = True,
                 use_majority_vote: bool = True,
                 reward_shaping_type: str = "quality",
                 reward_shaping_alpha: float = 0.5,
                 reward_shaping_beta: float = 0.5,
                 ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.subtract_baseline = subtract_baseline
        self.use_majority_vote = use_majority_vote
        self.reward_shaping_type = reward_shaping_type
        self.reward_shaping_alpha = reward_shaping_alpha 
        self.reward_shaping_beta = reward_shaping_beta

    def assign_rewards(self, all_answers, golden_answer):
        """
        all_answers: (M, T) where each all_answers[m][t] is the answer of the m-th agent at turn t.
        """
        M, T = len(all_answers), len(all_answers[0])

        # if self.use_majority_vote:
        # Compute final reward using the final round answers via majority vote.
        final_round_answers = [all_answers[m][-1] for m in range(M)]
        R_final = majority_vote(final_round_answers, golden_answer)
        # else:
        #     R_final = 0.0

        # Compute local rewards for every turn.
        local_rewards = np.zeros((M, T), dtype=np.float32)
        for m in range(M):
            for t in range(T):
                local_rewards[m][t] = qwen_reward_fn(
                    all_answers[m][t], golden_answer)
        # Compute the final reward for each agent.  
        adjusted_rewards = self.shaped_rewards_by_type(local_rewards, R_final)
        
        return adjusted_rewards, R_final
    
    def shaped_rewards_by_type(self, local_rewards, R_final):
        if self.reward_shaping_type in ["quality", "margin", "conditional_quality", "quality_with_outcome", "margin_with_outcome"]:
            from copy import deepcopy
            raw_rewards = deepcopy(local_rewards)
            M, T = raw_rewards.shape

            for m in range(M):
                for t in range(1, T):
                    Q = np.mean(raw_rewards[m][:t])
                    R_final = raw_rewards[m][t]
                    if "quality" in self.reward_shaping_type:
                        dynamic_term = Q * R_final - (1-Q) * (1-R_final)
                        # print(f"{m} - {t}", Q, R_final, dynamic_term)
                        shaped_reward = R_final + self.reward_shaping_alpha * dynamic_term
                    elif "margin" in self.reward_shaping_type:
                        baseline_term = R_final - Q
                        shaped_reward = R_final + self.reward_shaping_alpha * baseline_term
                    else:
                        raise NotImplementedError

                    local_rewards[m][t] = shaped_reward

            if "outcome" in self.reward_shaping_type:
                for m in range(M):
                    for t in range(T):
                        local_rewards[m][t] += self.reward_shaping_beta * raw_rewards[m][-1]

        return local_rewards

    def run(self, debate_histories, golden_answers):
        total_rewards = []
        final_rewards = []
        for history, golden_answer in zip(debate_histories, golden_answers):
            # normalized_answers = [
            #     [extract_answer_span(turn["assistant"]) for turn in agent] for agent in history
            # ]

            # !!! We need to input all the response with <think> into reward function
            normalized_answers = [
                [turn["assistant"] for turn in agent] for agent in history
            ]

            rewards, R_final = self.assign_rewards(
                normalized_answers, golden_answer)
            total_rewards.append(rewards)
            final_rewards.append(R_final)
        return total_rewards, final_rewards


class ChainRewardAllocation:
    def __init__(self, reward_shaping_type="quality", reward_shaping_alpha=0.5, reward_shaping_beta=0.5):
        self.reward_shaping_type = reward_shaping_type
        self.reward_shaping_alpha = reward_shaping_alpha
        self.reward_shaping_beta = reward_shaping_beta

    def assign_rewards(self, all_answers, golden_answers):
        # (num_question, num_agents)
        M, T = len(all_answers), len(all_answers[0])

        local_rewards = np.zeros((M, T), dtype=np.float32)
        for m in range(M):
            for t in range(T):
                local_rewards[m][t] = qwen_reward_fn(
                    all_answers[m][t], golden_answers[m])

        if self.reward_shaping_type == "conditional_quality":
            # we need to give reward for verifier
            for m in range(M):
                judgement = extract_answer(all_answers[m][1], "math")
                if judgement in ["0", "1"]:
                    judgement = int(judgement)
                    verifier_reward = int(judgement == local_rewards[m][0])
                else:
                    verifier_reward = 0
                local_rewards[m][1] = verifier_reward

        if self.reward_shaping_type is None:
            pass
        elif self.reward_shaping_type in ["quality", "margin", "conditional_quality", "quality_with_outcome", "margin_with_outcome"]:
            raw_rewards = deepcopy(local_rewards)
            for m in range(M):
                for t in range(1, T):
                    Q = np.mean(raw_rewards[m][:t])
                    R_final = raw_rewards[m][t]
                    if "quality" in self.reward_shaping_type:
                        dynamic_term = Q * R_final - (1-Q) * (1-R_final)
                        shaped_reward = R_final + self.reward_shaping_alpha * dynamic_term
                    elif "margin" in self.reward_shaping_type:
                        baseline_term = R_final - Q
                        shaped_reward = R_final + self.reward_shaping_alpha * baseline_term
                    else:
                        raise NotImplementedError

                    local_rewards[m][t] = shaped_reward

            if "outcome" in self.reward_shaping_type:
                for m in range(M):
                    for t in range(T):
                        local_rewards[m][t] += self.reward_shaping_beta * raw_rewards[m][-1]
        else:
            raise NotImplementedError

        return local_rewards

    def run(self, histories, golden_answers):
        all_answers = [[agent["assistant"] for agent in history]
                       for history in histories]
        rewards = self.assign_rewards(all_answers, golden_answers)

        return rewards


class AggRewardAllocation:
    def __init__(self, reward_shaping_type="quality", reward_shaping_alpha=0.5):
        self.reward_shaping_type = reward_shaping_type
        self.reward_shaping_alpha = reward_shaping_alpha

    def assign_rewards(self, all_answers, golden_answers):
        M, T = len(all_answers), len(all_answers[0])

        local_rewards = np.zeros((M, T), dtype=np.float32)
        for m in range(M):
            for t in range(T):
                local_rewards[m][t] = qwen_reward_fn(
                    all_answers[m][t], golden_answers[m])

        if self.reward_shaping_type is None:
            pass
        elif self.reward_shaping_type in ["quality", "margin"]:
            for m in range(M):
                Q = np.mean(local_rewards[m][:-1])
                R_final = local_rewards[m][-1]
                if self.reward_shaping_type == "quality":
                    dynamic_term = Q * R_final - (1-Q) * (1-R_final)
                    shaped_reward = R_final + self.reward_shaping_alpha * dynamic_term
                elif self.reward_shaping_type == "margin":
                    baseline_term = R_final - Q
                    shaped_reward = R_final + self.reward_shaping_alpha * baseline_term
                else:
                    raise NotImplementedError
                local_rewards[m][-1] = shaped_reward
        else:
            raise NotImplementedError

        return local_rewards

    def run(self, histories, golden_answers):
        all_answers = [[agent["assistant"] for agent in history]
                       for history in histories]
        rewards = self.assign_rewards(all_answers, golden_answers)

        return rewards

    
if __name__ == "__main__":
    master_chain = ChainRewardAllocation()
    master_mad = MADRewardAllocation()
    master_mix = AggRewardAllocation()
    preview_rm = MultiAgentRewardAllocation(name="quality")


    import srsly
    ####### master - chain
    data = srsly.read_json("/root/kyzhang/llms/MARTI/outputs/test/0419-chain10.json")["metadata"]
    histories = [d["history"] for d in data]
    golds = [d["label"] for d in data]
    rewards = master_chain.run(histories, golds)
    print(rewards[:3])
    # for aid in range(len(rewards[0])):
    #     print(aid, np.mean([rwd[aid] for rwd in rewards]))
    print()

    rewards, outcomes = preview_rm.run(histories, golds)
    print(rewards[:3])
    print(outcomes)
    # for aid in range(len(rewards[0])):
    #     print(aid, np.mean([rwd[aid] for rwd in rewards]))
    print()

    print("="*25)
    ######## preview - mad
    data = srsly.read_json("/root/kyzhang/llms/MARTI/outputs/test/0419-mad10.json")["metadata"]
    histories = [d["history"] for d in data]
    golds = [d["label"] for d in data]
    rewards, outcomes = master_mad.run(histories, golds)
    print(rewards[:3])
    print(outcomes)
    # for aid in range(len(rewards[0])):
    #     print(aid, np.mean([rwd[aid] for rwd in rewards]))
    print()

    rewards, outcomes = preview_rm.run(histories, golds)
    print(rewards[:3])
    print(outcomes)
    # for aid in range(len(rewards[0])):
    #     print(aid, np.mean([rwd[aid] for rwd in rewards]))

    print("="*25)
    ######## preview - mad
    data = srsly.read_json("/root/kyzhang/llms/MARTI/outputs/test/0419-mix10.json")["metadata"]
    histories = [d["history"] for d in data]
    golds = [d["label"] for d in data]
    rewards = master_mix.run(histories, golds)
    print(rewards[:3])

    # for aid in range(len(rewards[0])):
    #     print(aid, np.mean([rwd[aid] for rwd in rewards]))
    print()

    rewards, outcomes = preview_rm.run(histories, golds)
    print(rewards[:3])
    print(outcomes)

    # for aid in range(len(rewards[0])):
    #     print(aid, np.mean([rwd[aid] for rwd in rewards]))
    
    print("="*25)
