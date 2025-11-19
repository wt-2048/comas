import os
import json


def write_to_jsonl(lock, file_name, data):
    with lock:
        with open(file_name, 'a') as f:
            json.dump(data, f)
            f.write('\n')


def reserve_unprocessed_queries(output_path, test_dataset):
    processed_queries = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                infered_sample = json.loads(line)
                processed_queries.add(infered_sample["query"])

    test_dataset = [sample for sample in test_dataset if sample["query"] not in processed_queries]
    return test_dataset
