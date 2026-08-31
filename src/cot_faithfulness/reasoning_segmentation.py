import sys
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer

def find_cot_segment_indices(
    tokenizer, 
    tokens: List[int]
) -> Dict[str, Any]:
    """
    Finds token position indices for '<think>', all fullstops ('.'), and '</think>'.
    """
    think_start_idx = None
    think_end_idx = None
    fullstop_indices = []

    for idx, token_id in enumerate(tokens):
        decoded_tok = tokenizer.decode([token_id])
        
        # Check for start tag <think>
        if "<think>" in decoded_tok:
            think_start_idx = idx
            
        # Check for closing tag </think>
        elif "</think>" in decoded_tok:
            think_end_idx = idx
            
        # Check for fullstops / period tokens
        elif "." in decoded_tok:
            fullstop_indices.append(idx)

    # Filter fullstops to only those occurring inside the thinking block if both tags are present
    cot_fullstop_indices = fullstop_indices
    if think_start_idx is not None and think_end_idx is not None:
        cot_fullstop_indices = [
            idx for idx in fullstop_indices 
            if think_start_idx < idx < think_end_idx
        ]

    # Combine all segment boundary token positions in sorted order
    boundary_indices = [0]
    if think_start_idx is not None:
        boundary_indices.append(think_start_idx)
    boundary_indices.extend(cot_fullstop_indices[:-1])
    if think_end_idx is not None:
        boundary_indices.append(think_end_idx)

    return {
        "think_start_idx": think_start_idx,
        "think_end_idx": think_end_idx,
        "all_fullstop_indices": fullstop_indices,
        "cot_fullstop_indices": cot_fullstop_indices,
        "segment_boundary_indices": sorted(boundary_indices)
    }
