import concurrent.futures
from typing import List
from collections import Counter
from marti.verifiers.qwen.qwen_math_parser import extract_answer
from marti.verifiers.qwen.math_grade import grade_answer
from marti.verifiers.qwen.grader import math_equal as qwen_math_equal
from marti.verifiers.verify_coding import verify_answer as verify_coding_answer
import re


def _hotpotqa_f1_score(predicted_answer, golden_answer):
    if predicted_answer is None or golden_answer is None:
        return 0.0   
    def normalize_answer(answer):
        answer = str(answer).strip().lower()
        answer = re.sub(r'[^\w\s]', '', answer)
        tokens = answer.split()
        return tokens
    pred_tokens = normalize_answer(predicted_answer)
    gold_tokens = normalize_answer(golden_answer)
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    common_tokens = pred_set & gold_set
    if len(common_tokens) == 0:
        return 0.0
    precision = len(common_tokens) / len(pred_set)
    recall = len(common_tokens) / len(gold_set)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def qwen_reward_fn_format(generated_text, golden_answer, task="math"):
    model_answer = extract_answer(generated_text, task)
    accuracy = 1.0 if grade_answer(model_answer, golden_answer) else -0.5 #0.0
    if "boxed" not in generated_text:
        accuracy = -1.0
    return accuracy

def qwen_reward_fn(generated_text, golden_answer, task="math"):
    model_answer = extract_answer(generated_text, task)
    accuracy = 1.0 if grade_answer(model_answer, golden_answer) else 0.0 #-0.5 #0.0
    # if "boxed" not in generated_text:
    #     accuracy = -1.0
    return accuracy

def majority_vote(
    solutions: List[str],
    ground_truth: str,
    task="math"
):
    if task in ["math", "science"]:
        model_answers = [extract_answer(generated_text, task) for generated_text in solutions]
        model_answers = [answer for answer in model_answers if answer is not None]

        if len(model_answers) == 0:
            return 0.0

        counter = Counter(model_answers)
        
        majority_answer, _ = counter.most_common(1)[0]
        accuracy = 1.0 if grade_answer(majority_answer, ground_truth) else 0.0

        return accuracy

    elif task == "coding":
        # majority voting not available, directly compute the accuracy
        results = [verify_coding_answer(sol, ground_truth)["correct"] for sol in solutions]
        accuracy = sum(results) / len(results)
        return accuracy

    else:
        raise ValueError(f"Invalid task name: {task}")

def test_time_train(
    solutions: List[str],
    ground_truth: str,
    task="math"):
    model_answers = [extract_answer(generated_text, task) for generated_text in solutions]
    counter = Counter([answer for answer in model_answers if answer is not None])
    
    majority_answer, majority_count = counter.most_common(1)[0]

    # if majority_count / len(solutions) > 0.0 and majority_count > 1:
    rewards = [float(grade_answer(majority_answer, model_answer)) for model_answer in model_answers]
    # else:
    #     rewards = [0.0] * len(solutions)

    assert len(rewards) == len(solutions), f"{len(rewards)} vs {len(solutions)}"
    
    return rewards


def qwen_math_equal_subprocess(prediction, reference, timeout_seconds=10):
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(qwen_math_equal, prediction=prediction, reference=reference, timeout=False)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            return False

def simplerl_reward_fn(generated_text, golden_answer):
    model_answer = extract_answer(generated_text, "math")
    accuracy = 1.0 if qwen_math_equal_subprocess(prediction=model_answer, reference=golden_answer) else -0.5
    if "boxed" not in generated_text:
        accuracy = -1.0
    return accuracy