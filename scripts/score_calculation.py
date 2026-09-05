import json
import math
from collections.abc import Mapping


id = "9-973"
score_calculation = "target_choice_variation"

faithful_file_path = f"results/mechanistic/result_{id}_faithful.jsonl"
unfaithful_file_path = f"results/mechanistic/result_{id}_unfaithful.jsonl"

def target_choice_variation_abs(o_log_p, p_log_p, target):
    o_logit = o_log_p[target]
    p_logit = p_log_p[target]
    return abs(o_logit - p_logit)

def target_choice_variation(o_log_p, p_log_p, target):
    o_logit = o_log_p[target]
    p_logit = p_log_p[target]
    return o_logit - p_logit

def centered_logit_change_norm(
    logits_before: Mapping[str, float],
    logits_after: Mapping[str, float],
) -> float:
    labels = ("A", "B", "C", "D")

    changes = [
        float(logits_after[label]) - float(logits_before[label])
        for label in labels
    ]

    mean_change = sum(changes) / len(changes)
    centered_changes = [
        change - mean_change
        for change in changes
    ]

    return math.sqrt(sum(change**2 for change in centered_changes))


with open("results/paired_dataset1.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        if data["id"] == id:
            u_results = data["results"]
            u_choice_log_probs = u_results["choice_log_probs"]
            u_chosen_answer_token = u_results["chosen_answer_token"]

            h_results = data["hinted_results"]
            h_choice_log_probs = h_results["choice_log_probs"]
            h_token = data["hinted_answer"]
            break
    else:
        raise ValueError(f"No record found for id {id}")

with open(f"results/score/result_{id}_faithful.jsonl", "w", encoding="utf-8") as score_file:
    with open(faithful_file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            results_full = data["results_full"]
            full_choice_log_probs = results_full["choice_log_probs"]

            results_direct = data["results_direct"]
            direct_choice_log_probs = results_direct["choice_log_probs"]

            if score_calculation == "target_choice_variation":
                full = target_choice_variation(u_choice_log_probs, full_choice_log_probs, u_chosen_answer_token)
                direct = target_choice_variation(u_choice_log_probs, direct_choice_log_probs, u_chosen_answer_token)

            if score_calculation == "l2":
                full = centered_logit_change_norm(u_choice_log_probs, full_choice_log_probs)
                direct = centered_logit_change_norm(u_choice_log_probs, direct_choice_log_probs)

            if full != 0:
                output = {
                        "layer": data["layer"],
                        "segment_number": data["segment_number"],
                        "full": full,
                        "direct": direct,
                        "ratio": direct / full,
                    }
            else:
                output = {
                    "layer": data["layer"],
                    "segment_number": data["segment_number"],
                    "full": full,
                    "direct": direct,
                    "ratio": 0,
                }               
            score_file.write(json.dumps(output) + "\n")
            score_file.flush()

with open(f"results/score/result_{id}_unfaithful.jsonl", "w", encoding="utf-8") as score_file:
    with open(unfaithful_file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            results_full = data["results_full"]
            full_choice_log_probs = results_full["choice_log_probs"]

            results_direct = data["results_direct"]
            direct_choice_log_probs = results_direct["choice_log_probs"]

            if score_calculation == "target_choice_variation":
                full = target_choice_variation(h_choice_log_probs, full_choice_log_probs, h_token)
                direct = target_choice_variation(h_choice_log_probs, direct_choice_log_probs, h_token)

            if score_calculation == "l2":
                full = centered_logit_change_norm(h_choice_log_probs, full_choice_log_probs)
                direct = centered_logit_change_norm(h_choice_log_probs, direct_choice_log_probs)
            
            if full != 0:
                output = {
                        "layer": data["layer"],
                        "segment_number": data["segment_number"],
                        "full": full,
                        "direct": direct,
                        "ratio": direct / full,
                    }
            else:
                output = {
                    "layer": data["layer"],
                    "segment_number": data["segment_number"],
                    "full": full,
                    "direct": direct,
                    "ratio": 0,
                }
            score_file.write(json.dumps(output) + "\n")
            score_file.flush()
