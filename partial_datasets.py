# partial_datasets 
from __future__ import annotations
import random
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset


class PartialViewDataset(Dataset):
    def __init__(
        self,
        base_sequences: List[str],
        window_size: int,
        windows_per_seq: int = 1,
        overlap: int = 0,
        strategy: str = "random",
        seed: int = 42,
    ):
        assert window_size > 0
        self.ws = int(window_size)
        self.wps = int(windows_per_seq)
        self.overlap = max(0, int(overlap))
        self.strategy = (strategy or "random").lower()
        self.rng = random.Random(seed)

        self.views: List[str] = []
        for s in base_sequences:
            if not s:
                continue
            L = len(s)
            if L <= self.ws:
                self.views.append(s) 
                continue

            if self.strategy == "random":
                for _ in range(self.wps):
                    start = self.rng.randint(0, L - self.ws)
                    self.views.append(s[start:start + self.ws])
            else:
                step = max(1, self.ws - self.overlap)
                for start in range(0, max(1, L - self.ws + 1), step):
                    self.views.append(s[start:start + self.ws])

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"seq": self.views[idx]}
