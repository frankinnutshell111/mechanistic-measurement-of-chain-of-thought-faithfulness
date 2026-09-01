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

    # === FIXED: Robust Choice Map ===
    choice_map = {c: [] for c in choices}
    for choice in choices:
        for prefix in ["", " ", "\n", "("]:
            toks = tokenizer.encode(f"{prefix}{choice}", add_special_tokens=False)
            last_tok = toks[-1]
            if tokenizer.decode([last_tok]).strip(" \n\t().-:#*") == choice:
                choice_map[choice].append(last_tok)
        choice_map[choice] = list(set(choice_map[choice]))

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

    # === FIXED: Robust Reverse Scan ===
    target_step_idx = None
    for idx in reversed(range(len(generated_token_ids))):
        tok_id = generated_token_ids[idx]
        cleaned_tok = tokenizer.decode([tok_id]).strip(" \n\t().-:#*")
        if cleaned_tok in choices:
            target_step_idx = idx
            break

    if target_step_idx is None:
        target_step_idx = len(step_logits_history) - 1

    target_logits = step_logits_history[target_step_idx]
    target_log_probs = F.log_softmax(target_logits, dim=-1)

    choice_log_probs = {}
    for choice, token_ids in choice_map.items():
        if token_ids:
            choice_log_probs[choice] = max(target_log_probs[t_id].item() for t_id in token_ids)
        else:
            choice_log_probs[choice] = float("-inf")

    return {
        "decoded_prompt_and_completion": tokenizer.decode(current_input_ids[0]),
        "decision_step": target_step_idx,
        "choice_log_probs": choice_log_probs,
        "chosen_answer_token": tokenizer.decode([generated_token_ids[target_step_idx]]).strip()
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
    device = model.device
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    # === FIXED: Robust Choice Map ===
    choice_map = {c: [] for c in choices}
    for choice in choices:
        for prefix in ["", " ", "\n", "("]:
            toks = tokenizer.encode(f"{prefix}{choice}", add_special_tokens=False)
            last_tok = toks[-1]
            if tokenizer.decode([last_tok]).strip(" \n\t().-:#*") == choice:
                choice_map[choice].append(last_tok)
        choice_map[choice] = list(set(choice_map[choice]))

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

    # === FIXED: Robust Reverse Scan ===
    target_step_idx = None
    for idx in reversed(range(len(generated_token_ids))):
        tok_id = generated_token_ids[idx]
        cleaned_tok = tokenizer.decode([tok_id]).strip(" \n\t().-:#*")
        if cleaned_tok in choices:
            target_step_idx = idx
            break

    if target_step_idx is None:
        target_step_idx = len(step_logits_history) - 1

    target_logits = step_logits_history[target_step_idx]
    target_log_probs = F.log_softmax(target_logits, dim=-1)

    choice_log_probs = {}
    for choice, token_ids in choice_map.items():
        if token_ids:
            choice_log_probs[choice] = max(target_log_probs[t_id].item() for t_id in token_ids)
        else:
            choice_log_probs[choice] = float("-inf")

    return {
        "decoded_prompt_and_completion": tokenizer.decode(current_input_ids[0]),
        "decision_step_in_generation": target_step_idx,
        "choice_log_probs": choice_log_probs,
        "chosen_answer_token": tokenizer.decode([generated_token_ids[target_step_idx]]).strip()
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

    # Build token mapping strictly for valid choice tokens (avoiding punctuation traps)
    choice_map = {c: [] for c in choices}
    for choice in choices:
        for prefix in ["", " ", "\n", "("]:
            toks = tokenizer.encode(f"{prefix}{choice}", add_special_tokens=False)
            last_tok = toks[-1]
            if tokenizer.decode([last_tok]).strip(" \n\t().-:#*") == choice:
                choice_map[choice].append(last_tok)
        choice_map[choice] = list(set(choice_map[choice]))

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

    # Robust reverse scan: strip formatting to locate the exact choice token step
    target_step_idx = None
    for idx in reversed(range(len(generated_token_ids))):
        tok_id = generated_token_ids[idx]
        cleaned_tok = tokenizer.decode([tok_id]).strip(" \n\t().-:#*")
        if cleaned_tok in choices:
            target_step_idx = idx
            break

    if target_step_idx is None:
        target_step_idx = len(step_logits_history) - 1

    target_logits = step_logits_history[target_step_idx]
    target_log_probs = F.log_softmax(target_logits, dim=-1)

    choice_log_probs = {}
    for choice, token_ids in choice_map.items():
        if token_ids:
            choice_log_probs[choice] = max(target_log_probs[t_id].item() for t_id in token_ids)
        else:
            choice_log_probs[choice] = float("-inf")

    return {
        "full_tokens": current_input_ids[0].tolist(),
        "prompt_tokens": tokens,
        "decision_step": target_step_idx,
        "choice_log_probs": choice_log_probs,
        "chosen_answer_token": tokenizer.decode([generated_token_ids[target_step_idx]]).strip()
    }