import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from effective_dataset.dataset import prepare_openbookqa
from src.cot_faithfulness.three_runs import generate_choice_logits

def set_deterministic(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enforce deterministic CUDA algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_deterministic(42)

model_id = "Qwen/Qwen3-14B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

dataset = prepare_openbookqa()

#Start iteration
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

print(results['generated_text'])

print("=" * 50)

print(results['chosen_answer_token'])

print("=" * 50)

print(results['choice_log_probs'])



#End iteration
