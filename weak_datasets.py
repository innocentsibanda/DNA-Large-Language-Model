# weak_datasets
from __future__ import annotations
from typing import List, Dict, Any, Callable
import torch

class WeakSupervisedMotifDataset(torch.utils.data.Dataset):
    
    def __init__(self, partial_sequences: List[str], labeller: Callable[[str], Dict[str, List[int]]]):
        self.items: List[Dict[str, Any]] = []
        for s in partial_sequences:
            ann = labeller(s)
            itm = {"seq": s}
            itm.update(ann)
            self.items.append(itm)

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]
