import sys
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer

def find_cot_segment_indices(
    tokenizer, 
    tokens: List[int]
) -> Dict[str, Any]:
    think_start_idx = None
    think_end_idx = None
    raw_fullstop_indices = []

    # 1. First pass: locate <think>, </think>, and period candidates
    for idx, token_id in enumerate(tokens):
        decoded_tok = tokenizer.decode([token_id])
        
        if "<think>" in decoded_tok:
            think_start_idx = idx
        elif "</think>" in decoded_tok:
            think_end_idx = idx
        elif "." in decoded_tok:
            # Skip decimal points inside numbers (e.g., "3" "." "14")
            prev_tok = tokenizer.decode([tokens[idx - 1]]) if idx > 0 else ""
            next_tok = tokenizer.decode([tokens[idx + 1]]) if idx < len(tokens) - 1 else ""
            
            if prev_tok.strip().isdigit() and next_tok.strip().isdigit():
                continue
                
            raw_fullstop_indices.append(idx)

    # 2. Collapse contiguous dots (e.g. "..." -> keep index of final dot)
    deduped_fullstop_indices = []
    i = 0
    while i < len(raw_fullstop_indices):
        j = i
        while j + 1 < len(raw_fullstop_indices) and raw_fullstop_indices[j + 1] == raw_fullstop_indices[j] + 1:
            j += 1
        deduped_fullstop_indices.append(raw_fullstop_indices[j])
        i = j + 1

    # 3. Filter fullstops to those within the <think> ... </think> block
    cot_fullstop_indices = deduped_fullstop_indices
    if think_start_idx is not None and think_end_idx is not None:
        cot_fullstop_indices = [
            idx for idx in deduped_fullstop_indices 
            if think_start_idx < idx < think_end_idx
        ]

    # 4. Build ordered list of boundary indices
    boundary_indices = [0]
    if think_start_idx is not None:
        boundary_indices.append(think_start_idx)
    boundary_indices.extend(cot_fullstop_indices[:-1])
    if think_end_idx is not None:
        boundary_indices.append(think_end_idx)
    
    boundary_indices = sorted(boundary_indices)


    # 5. Decode token slices between boundaries into natural language strings
    decoded_segments = []
    prev_idx = -1
    for b_idx in boundary_indices:
        seg_tokens = tokens[prev_idx + 1 : b_idx + 1]
        if seg_tokens:
            decoded_segments.append(tokenizer.decode(seg_tokens))
        prev_idx = b_idx

    return {
        "think_start_idx": think_start_idx,
        "think_end_idx": think_end_idx,
        "cot_fullstop_indices": cot_fullstop_indices,
        "segment_boundary_indices": boundary_indices,
        "decoded_segments": decoded_segments
    }
