import torch
import random
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.effective_dataset.dataset import prepare_openbookqa
from src.cot_faithfulness.three_runs import generate_choice_logits
from src.cot_faithfulness.three_runs import patch_and_generate_choice_logits
from src.cot_faithfulness.three_runs import teacher_forced_patch_and_generate
from src.cot_faithfulness.reasoning_segmentation import find_cot_segment_indices
from src.cot_faithfulness.perturbations import create_gaussian_perturbation
from src.effective_dataset.hinting import black_square_hint
from src.effective_dataset.hinting import consistency_hint

#Config
id = "9-732"
layers = [10, 20, 30]
patching = "Gaussian"


dataset = prepare_openbookqa()
data = dataset.filter(lambda example: example['id'] == id)[0]
del dataset

with open("results/paired_dataset1.jsonl", "r", encoding="utf-8") as file:
    paired_data = [json.loads(line) for line in file]


for line in paired_data:
    if line['id'] == id:
        record = line
        break

prompt = data['prompt']
hinted_answer = record['hinted_answer']
hinted_prompt, _ = black_square_hint(prompt=data['prompt'], hinted_answer=hinted_answer)

del paired_data

def set_deterministic(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enforce deterministic CUDA algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_deterministic(42)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

model_id = "Qwen/Qwen3-14B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map={"": device}
)

#running CoT faithfulness analysis on faithful prompt
print("running CoT faithfulness analysis on faithful prompt")
messages = [
    {
        "role": "system",
        "content": "You are a precise assistant. Think step by step inside your reasoning block before choosing your answer."
    },
    {
        "role": "user",
        "content": (prompt)
    }
]

formatted_prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)
input_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)

print("start_generating")

results = generate_choice_logits(
    model=model,
    tokenizer=tokenizer,
    tokens=input_tokens,
    choices=["A", "B", "C", "D"],
    max_new_tokens=2048
)

token_ids = results["full_tokens"]
CoT_boundaries = find_cot_segment_indices(tokenizer, token_ids)
segment_boundary_indices = CoT_boundaries["segment_boundary_indices"]

patch_positions_list = [range(segment_boundary_indices[idx]+1, segment_boundary_indices[idx+1]+1) for idx in range(len(segment_boundary_indices)-1)]

with open(f"results/mechanistic/result_{id}_faithful.jsonl", "w", encoding="utf-8") as file:
    for layer in layers:
        for i in range(len(patch_positions_list)):
            print(f"Total segments number: {len(patch_positions_list)}")
            patch_position = patch_positions_list[i]

            deltas = create_gaussian_perturbation(
                num_patch_positions=len(patch_position),
                hidden_size=model.config.hidden_size, # 5120
                std=1,             
                dtype=model.dtype,         
                device=model.device
            )

            prefix = token_ids[:segment_boundary_indices[i+1]+1]

            print(f"segment {i} - Full")

            patch_gen_res1 = patch_and_generate_choice_logits(
                model=model,
                tokenizer=tokenizer,
                prefix_tokens=prefix,
                patch_positions=patch_position,
                layer_idx=layer,
                deltas=deltas
            )

            print(f"segment {i} - Direct")

            prefix = token_ids[:segment_boundary_indices[-1]+1]

            patch_gen_res2 = teacher_forced_patch_and_generate(
                model=model,
                tokenizer=tokenizer,
                tokens=prefix,
                patch_positions=patch_position,
                layer_idx=layer,
                deltas=deltas

            )

            output = {
                "layer": layer,
                "segment_number": i,
                "results_full": patch_gen_res1,
                "results_direct": patch_gen_res2
            }

            file.write(json.dumps(output, ensure_ascii=False) + "\n")
            file.flush()


#running CoT faithfulness analysis on unfaithful prompt

print("running CoT faithfulness analysis on unfaithful prompt")
messages = [
    {
        "role": "system",
        "content": "You are a precise assistant. Think step by step inside your reasoning block before choosing your answer."
    },
    {
        "role": "user",
        "content": (hinted_prompt)
    }
]

formatted_prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)
input_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)

print("start_generating")

results = generate_choice_logits(
    model=model,
    tokenizer=tokenizer,
    tokens=input_tokens,
    choices=["A", "B", "C", "D"],
    max_new_tokens=2048
)

token_ids = results["full_tokens"]
CoT_boundaries = find_cot_segment_indices(tokenizer, token_ids)
segment_boundary_indices = CoT_boundaries["segment_boundary_indices"]

patch_positions_list = [range(segment_boundary_indices[idx]+1, segment_boundary_indices[idx+1]+1) for idx in range(len(segment_boundary_indices)-1)]

with open(f"results/mechanistic/result_{id}_unfaithful.jsonl", "w", encoding="utf-8") as file:
    for layer in layers:
        for i in range(len(patch_positions_list)):
            print(f"Total segments number: {len(patch_positions_list)}")
            patch_position = patch_positions_list[i]

            deltas = create_gaussian_perturbation(
                num_patch_positions=len(patch_position),
                hidden_size=model.config.hidden_size, # 5120
                std=1,           
                dtype=model.dtype,              
                device=model.device
            )

            prefix = token_ids[:segment_boundary_indices[i+1]+1]

            print(f"segment {i} - Full")

            patch_gen_res1 = patch_and_generate_choice_logits(
                model=model,
                tokenizer=tokenizer,
                prefix_tokens=prefix,
                patch_positions=patch_position,
                layer_idx=layer,
                deltas=deltas
            )

            print(f"segment {i} - Direct")

            prefix = token_ids[:segment_boundary_indices[-1]+1]

            patch_gen_res2 = teacher_forced_patch_and_generate(
                model=model,
                tokenizer=tokenizer,
                tokens=prefix,
                patch_positions=patch_position,
                layer_idx=layer,
                deltas=deltas

            )

            output = {
                "layer": layer,
                "segment_number": i,
                "results_full": patch_gen_res1,
                "results_direct": patch_gen_res2
            }

            file.write(json.dumps(output, ensure_ascii=False) + "\n")
            file.flush()


print("=== unpatched ===")
print("generated_texts")
print(tokenizer.decode(results["generated_tokens"]))
print(f"Identified Answer Token: {results['chosen_answer_token']!r}")
print(f"log probabilities: {results["choice_log_probs"]}")
