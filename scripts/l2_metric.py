import json
from statistics import median

id = "814"

FILE_PATH = f"results/score/result_{id}_unfaithful_l2.jsonl"

def median_total_effect(file_path):
    with open(file_path, "r") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    return median(row["full"] for row in rows)


print(median_total_effect(FILE_PATH))