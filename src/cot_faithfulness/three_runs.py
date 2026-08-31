import re
import torch
import torch.nn.functional as F
from typing import List, Dict, Any

def patch_and_generate_choice_logits(
    model,
    tokenizer,
    prefix_tokens: List[int],
    patch_positions: List[int],
    layer_idx: int,
    deltas: torch.Tensor,
    choices: List[str] = ["A", "B", "C", "D"],
    max_new_tokens: int = 512
) -> Dict[str, Any]:


    device = model.device
    input_ids = torch.tensor([prefix_tokens], dtype=torch.long, device=device)
    prompt_len = input_ids.shape[1]

    # Pre-build token mapping for each choice (handling space variants)
    choice_map = {}
    all_choice_token_ids = []
    for c in choices:
        ids = list({
            tokenizer.encode(c, add_special_tokens=False)[-1],
            tokenizer.encode(f" {c}", add_special_tokens=False)[-1],
            tokenizer.encode(f"({c})", add_special_tokens=False)[-1],
        })
        choice_map[c] = ids
        all_choice_token_ids.extend(ids)
    
    all_choice_token_ids = list(set(all_choice_token_ids))

    # Single-pass hook for prefill phase patching
    patched = False
    def patch_hook(module, input, output):
        nonlocal patched
        if not patched:
            hidden_states = output[0] if isinstance(output, tuple) else output
            modified = hidden_states.clone()
            for pos_idx, pos in enumerate(patch_positions):
                modified[:, pos, :] += deltas[pos_idx].to(device)
            patched = True
            return (modified,) + output[1:] if isinstance(output, tuple) else modified
        return output

    target_layer = model.model.layers[layer_idx]
    hook_handle = target_layer.register_forward_hook(patch_hook)

    past_key_values = None
    current_input_ids = input_ids
    
    # Store history of generated token IDs and their step-wise logits
    step_logits_history = []
    generated_token_ids = []

    try:
        with torch.no_grad():
            for step in range(max_new_tokens):
                if past_key_values is None:
                    outputs = model(input_ids=current_input_ids, use_cache=True)
                else:
                    outputs = model(
                        input_ids=current_input_ids[:, -1:], 
                        past_key_values=past_key_values, 
                        use_cache=True
                    )

                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[0, -1, :] # [vocab_size]
                
                # Keep step logits on CPU to prevent GPU memory buildup
                step_logits_history.append(next_token_logits.cpu())

                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                generated_token_ids.append(next_token_id)
                current_input_ids = torch.cat(
                    [current_input_ids, torch.tensor([[next_token_id]], device=device)], dim=-1
                )

                # Stop standard generation at EOS
                if next_token_id == tokenizer.eos_token_id:
                    break

    finally:
        hook_handle.remove()

    # Identify the step corresponding to the final decision
    target_step_idx = None
    
    # Strategy 1: Find the last generated token that maps to a choice option
    for idx in reversed(range(len(generated_token_ids))):
        if generated_token_ids[idx] in all_choice_token_ids:
            target_step_idx = idx
            break

    # If no choice token was found, fall back to the very last step
    if target_step_idx is None:
        target_step_idx = len(step_logits_history) - 1

    # Retrieve logits and compute log probabilities for choices at that specific step
    target_logits = step_logits_history[target_step_idx]
    target_log_probs = F.log_softmax(target_logits, dim=-1)

    choice_log_probs = {}
    for choice, token_ids in choice_map.items():
        # Get maximum log prob across whitespace/formatting variants of the token
        choice_log_probs[choice] = max(target_log_probs[t_id].item() for t_id in token_ids)

    return {
        "decision_step": target_step_idx,
        "target_logits": target_logits,
        "choice_log_probs": choice_log_probs,
        "generated_text": tokenizer.decode(generated_token_ids),
        "chosen_answer_token": tokenizer.decode([generated_token_ids[target_step_idx]])
    }

def teacher_forced_patch_and_generate(
    model,
    tokenizer,
    tokens: List[int],
    patch_positions: List[int],
    layer_idx: int,
    deltas: torch.Tensor,
    choices: List[str] = ["A", "B", "C", "D"],
    max_new_tokens: int = 512
) -> Dict[str, Any]:
    """
    Teacher-forces the model on prefix `tokens` [t1, ..., tn] with residual stream 
    patching at `patch_positions` in layer `layer_idx`. Then continues generating 
    autoregressively from tn until an answer choice ('A', 'B', 'C', 'D') is produced, 
    backtracking to extract choice log probabilities at the final decision step.
    """
    device = model.device
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    prompt_len = input_ids.shape[1]

    # Pre-build choice token mapping (handles "A", " A", "(A)")
    choice_map = {}
    all_choice_token_ids = []
    for c in choices:
        ids = list({
            tokenizer.encode(c, add_special_tokens=False)[-1],
            tokenizer.encode(f" {c}", add_special_tokens=False)[-1],
            tokenizer.encode(f"({c})", add_special_tokens=False)[-1],
        })
        choice_map[c] = ids
        all_choice_token_ids.extend(ids)
    
    all_choice_token_ids = list(set(all_choice_token_ids))

    # Single-pass prefill hook to apply perturbation ONLY during initial sequence evaluation [t1...tn]
    patched = False
    def patch_hook(module, input, output):
        nonlocal patched
        if not patched:
            hidden_states = output[0] if isinstance(output, tuple) else output
            modified = hidden_states.clone()
            for pos_idx, pos in enumerate(patch_positions):
                modified[:, pos, :] += deltas[pos_idx].to(device)
            patched = True
            return (modified,) + output[1:] if isinstance(output, tuple) else modified
        return output

    target_layer = model.model.layers[layer_idx]
    hook_handle = target_layer.register_forward_hook(patch_hook)

    past_key_values = None
    current_input_ids = input_ids
    step_logits_history = []
    generated_token_ids = []

    try:
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Prefill teacher-forced prefix vs Decode step
                if past_key_values is None:
                    outputs = model(input_ids=current_input_ids, use_cache=True)
                else:
                    outputs = model(
                        input_ids=current_input_ids[:, -1:], 
                        past_key_values=past_key_values, 
                        use_cache=True
                    )

                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[0, -1, :]
                
                step_logits_history.append(next_token_logits.cpu())

                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                generated_token_ids.append(next_token_id)
                current_input_ids = torch.cat(
                    [current_input_ids, torch.tensor([[next_token_id]], device=device)], dim=-1
                )

                if next_token_id == tokenizer.eos_token_id:
                    break

    finally:
        hook_handle.remove()

    # Backtrack through newly generated tokens to locate the final choice emission
    target_step_idx = None
    for idx in reversed(range(len(generated_token_ids))):
        if generated_token_ids[idx] in all_choice_token_ids:
            target_step_idx = idx
            break

    if target_step_idx is None:
        target_step_idx = len(step_logits_history) - 1

    # Extract log probabilities at the target decision step
    target_logits = step_logits_history[target_step_idx]
    target_log_probs = F.log_softmax(target_logits, dim=-1)

    choice_log_probs = {}
    for choice, token_ids in choice_map.items():
        choice_log_probs[choice] = max(target_log_probs[t_id].item() for t_id in token_ids)

    return {
        "full_tokens": current_input_ids[0].tolist(),
        "teacher_forced_prefix_tokens": tokens,
        "newly_generated_tokens": generated_token_ids,
        "decoded_prompt_and_completion": tokenizer.decode(current_input_ids[0]),
        "decoded_completion": tokenizer.decode(generated_token_ids),
        "decision_step_in_generation": target_step_idx,
        "choice_log_probs": choice_log_probs,
        "chosen_answer_token": tokenizer.decode([generated_token_ids[target_step_idx]])
    }

def generate_choice_logits(
    model,
    tokenizer,
    tokens: List[int],
    choices: List[str] = ["A", "B", "C", "D"],
    max_new_tokens: int = 512
) -> Dict[str, Any]:
    device = model.device
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    prompt_len = input_ids.shape[1]

    choice_map = {}
    all_choice_token_ids = []
    for c in choices:
        ids = list({
            tokenizer.encode(c, add_special_tokens=False)[-1],
            tokenizer.encode(f" {c}", add_special_tokens=False)[-1],
            tokenizer.encode(f"({c})", add_special_tokens=False)[-1],
        })
        choice_map[c] = ids
        all_choice_token_ids.extend(ids)
    
    all_choice_token_ids = list(set(all_choice_token_ids))

    past_key_values = None
    current_input_ids = input_ids
    step_logits_history = []
    generated_token_ids = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            if past_key_values is None:
                outputs = model(input_ids=current_input_ids, use_cache=True)
            else:
                outputs = model(
                    input_ids=current_input_ids[:, -1:], 
                    past_key_values=past_key_values, 
                    use_cache=True
                )

            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[0, -1, :]
            
            step_logits_history.append(next_token_logits.cpu())

            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            generated_token_ids.append(next_token_id)
            current_input_ids = torch.cat(
                [current_input_ids, torch.tensor([[next_token_id]], device=device)], dim=-1
            )

            if next_token_id == tokenizer.eos_token_id:
                break

    target_step_idx = None
    for idx in reversed(range(len(generated_token_ids))):
        if generated_token_ids[idx] in all_choice_token_ids:
            target_step_idx = idx
            break

    if target_step_idx is None:
        target_step_idx = len(step_logits_history) - 1

    target_logits = step_logits_history[target_step_idx]
    target_log_probs = F.log_softmax(target_logits, dim=-1)

    choice_log_probs = {}
    for choice, token_ids in choice_map.items():
        choice_log_probs[choice] = max(target_log_probs[t_id].item() for t_id in token_ids)

    return {
        "full_tokens": current_input_ids[0].tolist(),  # Prompt + generated tokens combined
        "prompt_tokens": tokens,                       # Original prefix tokens [t1, ..., tn]
        "generated_text": tokenizer.decode(generated_token_ids),       # Generated tokens only
        "decision_step": target_step_idx,
        "choice_log_probs": choice_log_probs,
        "chosen_answer_token": tokenizer.decode([generated_token_ids[target_step_idx]])
    }