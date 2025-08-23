# weak_labels
from __future__ import annotations
from typing import Dict, List
import re
import torch


def heuristic_labels(seq: str, patterns: Dict[str, str], binary: bool = True) -> Dict[str, List[int]]:
    
    L = len(seq)
    flags  = [0]*L
    labels = [0]*L
    bounds = [0]*L

    if binary:
        for pat in patterns.values():
            for m in re.finditer(pat, seq):
                a, b = m.start(), m.end()
                for i in range(a, b):
                    flags[i]  = 1
                    labels[i] = 1
                if 0 <= a < L:     bounds[a]   = 1
                if 0 <= b-1 < L:   bounds[b-1] = 1
        return {"motif_flags": flags, "motif_labels": labels, "motif_boundaries": bounds}

    label_id = 1
    for pat in patterns.values():
        for m in re.finditer(pat, seq):
            a, b = m.start(), m.end()
            for i in range(a, b):
                flags[i]  = 1
                labels[i] = label_id
            if 0 <= a < L:     bounds[a]   = 1
            if 0 <= b-1 < L:   bounds[b-1] = 1
        label_id += 1

    return {"motif_flags": flags, "motif_labels": labels, "motif_boundaries": bounds}


@torch.no_grad()
def pseudo_labels_from_model_binary(
    model,
    head,           
    tokenizer,
    seq: str,
    prob_thresh: float = 0.9,
) -> Dict[str, List[int]]:
    ids = tokenizer.encode(seq)
    if not ids:
        return {"motif_labels": [], "motif_flags": [], "motif_boundaries": []}

    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    logits = head(model.encoder(x))  
    probs = torch.softmax(logits, dim=-1)[0]  
    conf, _ = probs.max(dim=-1)              

    labels: List[int] = []
    flags:  List[int] = []
    for c in conf.tolist():
        if c >= prob_thresh:
            labels.append(1)
            flags.append(1)
        else:
            labels.append(0)  
            flags.append(0)

    bounds = [0]*len(labels)
    return {"motif_labels": labels, "motif_flags": flags, "motif_boundaries": bounds}
