from math_verify import parse, verify

from marti.verifiers.verify_coding import verify_answer as verify_coding_answer


example_history = {
    "problem_list": [
        {
            "problem_content": "...",
            "task_name": "...",
            "round_list": [
                {
                    "discussion_list": [
                        {
                            "solver_step": {
                                "solution_content": "...",
                                "chat_history": {
                                    "type": "solution",
                                    "agent_id": 0,
                                    "prompt": "...",
                                    "completion": "..."
                                }
                            },
                            "evaluator_step": {
                                "evaluation_content": "...",
                                "chat_history": {
                                    "type": "evaluator",
                                    "agent_id": 0,
                                    "prompt": "...",
                                    "completion": "..."
                                }
                            },
                            "scorer_step": {
                                "score_content": "...",
                                "generated_score": 1,
                                "chat_history": {
                                    "type": "score",
                                    "agent_id": 0,
                                    "prompt": "...",
                                    "completion": "..."
                                }
                            }
                        },
                        ...
                    ]
                },
                ...
            ]
        },
        ...
    ]
}


class CoMASReward(object):
    def __init__(self, *args, **kwargs):
        pass

    def _verify_solution(self, result, answer, task):
        try:
            if task == "math":
                parsed_result = parse(result)
                parsed_answer = parse(answer)
                return verify(parsed_answer, parsed_result)
            elif task == "coding":
                checker = answer
                verify_result = verify_coding_answer(result, checker)
                return verify_result['correct']
            elif task == "science":
                parsed_result = parse(result)[1]
                float_result = float(parsed_result)
                float_answer = float(answer)
                return abs(float_result - float_answer) < 0.05 * abs(float_answer)
            else:
                raise ValueError(f"Unsupported task: {task}")
        except Exception as error:
            return False

    def run(self, comas_histories, golden_answers, task_names):
        # verify all the solutions
        for problem, answer, task in zip(comas_histories["problem_list"], golden_answers, task_names):
            for round in problem["round_list"]:
                for discussion in round["discussion_list"]:
                    solver_step = discussion["solver_step"]
                    solution = solver_step["solution_content"]
                    discussion["is_correct"] = self._verify_solution(solution, answer, task)

        # allocate rewards
        for problem in comas_histories["problem_list"]:
            for round in problem["round_list"]:
                for discussion in round["discussion_list"]:
                    solver_step = discussion["solver_step"]
                    evaluator_step = discussion["evaluator_step"]
                    scorer_step = discussion["scorer_step"]
                    generated_score = scorer_step["generated_score"]
                    if generated_score is None:
                        solver_step["chat_history"]["reward"] = 0.0
                        evaluator_step["chat_history"]["reward"] = 0.0
                        scorer_step["chat_history"]["reward"] = -1.0
                    else:
                        normalized_score = (generated_score - 1.0) / 2.0        
                        solver_step["chat_history"]["reward"] = normalized_score
                        evaluator_step["chat_history"]["reward"] = 1 - normalized_score
                        scorer_step["chat_history"]["reward"] = 0.0

        # calculate final accuracy
        final_accuracy = []
        for problem in comas_histories["problem_list"]:
            num_discussions = 0
            num_correct_discussions = 0
            round = problem["round_list"][-1]
            for discussion in round["discussion_list"]:
                num_discussions += 1
                if discussion["is_correct"]:
                    num_correct_discussions += 1
            final_accuracy.append(num_correct_discussions / num_discussions)

        return comas_histories, final_accuracy
