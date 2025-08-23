# tokenizer utils

import os
import random
import gzip
import bz2
from typing import List, Dict, Any, Optional

import torch


def _open_file(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_datasets(
    filepaths: List[str],
    fraction: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    
    datasets: Dict[str, List[str]] = {}
    for path in filepaths:
        try:
            with _open_file(path) as f:
                lines = [line.strip() for line in f if line.strip()]

            if 0 < fraction < 1.0:
                random.seed(seed)
                sample_size = int(len(lines) * fraction)
                if sample_size > 0:
                    lines = random.sample(lines, sample_size)
                elif verbose:
                    print(f"Warning: fraction={fraction} resulted in empty sample for {path}")

            datasets[os.path.basename(path)] = lines

            if verbose:
                print(f"Loaded {len(lines)} lines from {path}")

        except IOError as e:
            if verbose:
                print(f"Error loading file {path}: {e}")
            datasets[os.path.basename(path)] = []

    return datasets


def pad_sequences(
    sequences: List[List[int]],
    pad_token_id: int = 0,
    max_len: Optional[int] = None,
    padding: str = "right",
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    
    if not sequences:
        return torch.empty((0, 0), dtype=dtype)

    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        trunc = seq[:max_len]
        pad_len = max(0, max_len - len(trunc))
        if padding == "right":
            padded.append(trunc + [pad_token_id] * pad_len)
        elif padding == "left":
            padded.append([pad_token_id] * pad_len + trunc)
        else:
            raise ValueError("padding must be 'right' or 'left'")

    return torch.tensor(padded, dtype=dtype)


def batchify(
    sequences: List[List[int]],
    pad_id: int,
    max_len: Optional[int] = None,
    padding: str = "right",
    dtype: torch.dtype = torch.long,
    return_positions: bool = True,
) -> Dict[str, torch.Tensor]:
    
    ids = pad_sequences(sequences, pad_token_id=pad_id, max_len=max_len, padding=padding, dtype=dtype)
    attn_mask = (ids != pad_id).to(dtype=torch.long)

    out: Dict[str, torch.Tensor] = {
        "input_ids": ids,
        "attention_mask": attn_mask,
    }

    if return_positions:
        seq_len = ids.size(1)
        pos = torch.arange(seq_len, dtype=dtype).unsqueeze(0).expand_as(ids)
        out["position_ids"] = pos

    return out
