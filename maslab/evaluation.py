import os
import re
import json
import math
import argparse

from math_verify import parse, verify
from utils.coding import verify_answer


def main(args):
    results = []
    assert os.path.exists(args.result_file)
    with open(args.result_file, 'r') as file:
        for line in file:
            if line.strip():
                results.append(json.loads(line))
    print(f"Loaded {len(results)} results")

    outputs = {
        "dataset": args.dataset,
        "num_total_results": 0,
        "num_correct_results": 0,
        "accuracy": 0.0,
        "results": []
    }
    num_total_results = len(results)
    num_correct_results = 0

    for index, result in enumerate(results):
        question = result.get("query", "")
        answer = result.get("response", "")
        ground_truth = result.get("gt", "")

        # math-based datasets
        if args.dataset in ["GSM8K", "MATH-500", "AMC-2023", "AIME-2024"]:
            parsed_ground_truth = parse(ground_truth)
            parsed_answer = parse(answer)
            is_correct = verify(parsed_ground_truth, parsed_answer)

        # choice-based datasets
        elif args.dataset in ["GPQA", "MedQA", "MMLU"]:
            parsed_ground_truth = ground_truth.strip().upper()
            pattern = re.compile(r'\\boxed\{([A-D])\}', re.IGNORECASE)
            match = pattern.findall(answer)
            parsed_answer = match[-1].strip().upper() if match else ""
            is_correct = (parsed_answer == parsed_ground_truth)

        # value-based datasets
        elif args.dataset in ["SciBench"]:
            if ground_truth.startswith("+"):
                ground_truth = ground_truth[1:]
            parsed_ground_truth = parse(ground_truth)
            parsed_answer = parse(answer)
            try:
                parsed_ground_truth = float(parsed_ground_truth[0])
                parsed_answer = float(parsed_answer[0])
                is_correct = math.isclose(parsed_answer, parsed_ground_truth, rel_tol=0.05)
            except:
                is_correct = False

        # code-based datasets
        elif args.dataset in ['HumanEval', 'MBPP']:
            if args.dataset == 'HumanEval':
                test = ground_truth["test"]
                entry_point = ground_truth["entry_point"]
                ground_truth = f"{test}\ncheck({entry_point})"
                checker = ground_truth
                result_dict = verify_answer(answer, checker, timeout=3.0)
                is_correct = result_dict['correct']
                parsed_ground_truth = f"Entry point: {entry_point}"
                parsed_answer = f"Formatted: {result_dict['formatted']}, Status: {result_dict['status']}"
            elif args.dataset == 'MBPP':
                test_cases = ground_truth
                checker = '\n'.join(test_cases)
                result_dict = verify_answer(answer, checker, timeout=3.0)
                is_correct = result_dict['correct']
                parsed_ground_truth = f"Test cases: {len(test_cases)} cases"
                parsed_answer = f"Formatted: {result_dict['formatted']}, Status: {result_dict['status']}"

        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        if is_correct:
            num_correct_results += 1
        outputs["results"].append({
            "index": index,
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "parsed_ground_truth": str(parsed_ground_truth),
            "parsed_answer": str(parsed_answer),
            "is_correct": is_correct
        })

    accuracy = num_correct_results / num_total_results
    outputs["num_total_results"] = num_total_results
    outputs["num_correct_results"] = num_correct_results
    outputs["accuracy"] = accuracy
    print(f"Number of total results: {num_total_results}")
    print(f"Number of correct results: {num_correct_results}")
    print(f"Accuracy: {accuracy}")

    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as file:
            json.dump(outputs, file, indent=4)
        print(f"Results saved to {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the inference results')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--result_file', type=str,required=True,help='Path to the inference results file')
    parser.add_argument('--output_file', type=str, default=None, help='Path to save the evaluation results')
    args = parser.parse_args()
    main(args)
