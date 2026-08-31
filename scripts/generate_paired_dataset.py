import torch
import random
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.effective_dataset.dataset import prepare_openbookqa
from src.effective_dataset.hinting import consistency_hint
from src.effective_dataset.hinting import black_square_hint
from src.cot_faithfulness.three_runs import generate_choice_logits

hinting_method = "black_square"

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

dataset = prepare_openbookqa()

#Start iteration
with open("results/paired_dataset.jsonl", "w", encoding="utf-8") as file:
    data = dataset[0]
    prompt = data['prompt']

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

    results = generate_choice_logits(
        model=model,
        tokenizer=tokenizer,
        tokens=input_tokens,
        choices=["A", "B", "C", "D"],
        max_new_tokens=1024
    )

    if hinting_method == 'bs':
        hinted_prompt = black_square_hint(prompt=prompt, answer=results['chosen_answer_token'])
    else:
        hinted_prompt = consistency_hint(prompt=prompt, answer=results['chosen_answer_token'])

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

    hinted_results = generate_choice_logits(
        model=model,
        tokenizer=tokenizer,
        tokens=input_tokens,
        choices=["A", "B", "C", "D"],
        max_new_tokens=1024
    )

    if results["chosen_answer_token"] != hinted_results["chosen_answer_token"]:
        output = {
            "id": data["id"],
            "prompt": data["prompt"],
            "answerKey": data["answerKey"],
            "results": {
                key: value
                for key, value in results.items()
                if key not in {"full_tokens", "prompt_tokens"}
            },
            "hinted_results": {
                key: value
                for key, value in hinted_results.items()
                if key not in {"full_tokens", "prompt_tokens"}
            },
        }

        file.write(json.dumps(output, ensure_ascii=False) + "\n")
        file.flush()


#End iteration
