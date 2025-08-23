# pretraining_supervised_learning 

from __future__ import annotations

import random
from typing import Tuple, Optional, Callable
import torch

Tensor = torch.Tensor
AugmentFn = Callable[[Tensor, Optional[Tensor]], Tensor]

def _as_long_2d(x: Tensor | list) -> Tensor:
    if isinstance(x, list):
        x = torch.tensor(x, dtype=torch.long)
    elif not torch.is_tensor(x):
        raise TypeError(f"Expected list or Tensor, got {type(x)}")
    x = x.long()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() != 2:
        raise ValueError(f"Expected 1D or 2D tensor, got {tuple(x.shape)}")
    return x


def _restore_shape(x: Tensor, like: Tensor | list) -> Tensor:
    if torch.is_tensor(like) and like.dim() == 1:
        return x.squeeze(0)
    if isinstance(like, list):
        return x.squeeze(0)
    return x


def _set_seeds(seed: Optional[int]) -> None:
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

def apply_masked_motif_modeling(
    token_ids: Tensor | list,
    motif_flags: Tensor | list,
    mask_token_id: int,
    mask_prob: float = 0.15,
    pad_token_id: int = 0,
    seed: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    
    _set_seeds(seed)

    x = _as_long_2d(token_ids)
    m = _as_long_2d(motif_flags)
    if x.shape != m.shape:
        raise ValueError(f"token_ids and motif_flags must have same shape, got {x.shape} vs {m.shape}")

    B, L = x.shape
    device = x.device
    input_ids = x.clone()
    labels = torch.full_like(x, fill_value=-100)

    valid = (x != pad_token_id) & (m > 0)        
    choose = (torch.rand(B, L, device=device) < mask_prob) & valid

    labels[choose] = x[choose]

    r = torch.rand(B, L, device=device)
    to_mask   = (r < 0.8) & choose
    to_random = (r >= 0.8) & (r < 0.9) & choose

    input_ids[to_mask] = mask_token_id
    if to_random.any():
        max_id = max(mask_token_id, pad_token_id, int(x.max().item()))
        input_ids[to_random] = torch.randint(0, max_id + 1, size=(to_random.sum().item(),), device=device)

    return _restore_shape(input_ids, token_ids), _restore_shape(labels, token_ids)


def generate_cml_pairs(
    token_ids: Tensor | list,
    motif_flags: Tensor | list,
    augment_fn: AugmentFn,
    seed: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    _set_seeds(seed)
    x = _as_long_2d(token_ids)
    m = _as_long_2d(motif_flags)
    if x.shape != m.shape:
        raise ValueError(f"token_ids and motif_flags must have same shape, got {x.shape} vs {m.shape}")
    anchor = x.clone()
    positive = augment_fn(x, m)
    return _restore_shape(anchor, token_ids), _restore_shape(positive, token_ids)


def apply_map_task(
    token_ids: Tensor | list,
    motif_labels: Tensor | list,
) -> Tuple[Tensor, Tensor]:
    x = _as_long_2d(token_ids)
    y = _as_long_2d(motif_labels)
    if x.shape != y.shape:
        raise ValueError(f"token_ids and motif_labels must have same shape, got {x.shape} vs {y.shape}")
    return _restore_shape(x, token_ids), _restore_shape(y, token_ids)


def apply_motif_boundary_prediction(
    token_ids: Tensor | list,
    motif_boundaries: Tensor | list,
) -> Tuple[Tensor, Tensor]:
    x = _as_long_2d(token_ids)
    y = _as_long_2d(motif_boundaries)
    if x.shape != y.shape:
        raise ValueError(f"token_ids and motif_boundaries must have same shape, got {x.shape} vs {y.shape}")
    return _restore_shape(x, token_ids), _restore_shape(y, token_ids)

def cml_augment_random_mask(
    token_ids: Tensor | list,
    motif_flags: Optional[Tensor | list],
    mask_token_id: int = 0,
    mask_prob: float = 0.15,
    pad_token_id: int = 0,
    seed: Optional[int] = None,
) -> Tensor:
    _set_seeds(seed)
    x = _as_long_2d(token_ids).clone()
    device = x.device
    B, L = x.shape

    valid = (x != pad_token_id)
    choose = (torch.rand(B, L, device=device) < mask_prob) & valid
    r = torch.rand(B, L, device=device)

    to_mask   = (r < 0.8) & choose
    to_random = (r >= 0.8) & (r < 0.9) & choose

    x[to_mask] = mask_token_id
    if to_random.any():
        max_id = max(mask_token_id, pad_token_id, int(x.max().item()))
        x[to_random] = torch.randint(0, max_id + 1, size=(to_random.sum().item(),), device=device)
    return _restore_shape(x, token_ids)


def augment_mask_non_motif(
    token_ids: Tensor | list,
    motif_flags: Optional[Tensor | list],
    mask_token_id: int,
    mask_prob: float = 0.15,
    pad_token_id: int = 0,
    seed: Optional[int] = None,
) -> Tensor:
    _set_seeds(seed)
    x = _as_long_2d(token_ids).clone()
    m = None if motif_flags is None else _as_long_2d(motif_flags)
    device = x.device
    B, L = x.shape

    outside = torch.ones_like(x, dtype=torch.bool) if m is None else (m == 0)
    valid = (x != pad_token_id) & outside
    choose = (torch.rand(B, L, device=device) < mask_prob) & valid

    r = torch.rand(B, L, device=device)
    to_mask   = (r < 0.8) & choose
    to_random = (r >= 0.8) & (r < 0.9) & choose

    x[to_mask] = mask_token_id
    if to_random.any():
        max_id = max(mask_token_id, pad_token_id, int(x.max().item()))
        x[to_random] = torch.randint(0, max_id + 1, size=(to_random.sum().item(),), device=device)
    return _restore_shape(x, token_ids)


def augment_window_centered(
    token_ids: Tensor | list,
    motif_flags: Optional[Tensor | list],
    window_size: int = 64,
    pad_token_id: int = 0,
    seed: Optional[int] = None,
) -> Tensor:

    _set_seeds(seed)
    x = _as_long_2d(token_ids)
    m = None if motif_flags is None else _as_long_2d(motif_flags)
    B, L = x.shape
    out = torch.full_like(x, fill_value=pad_token_id)

    for b in range(B):
        if m is not None:
            pos_idx = (m[b] > 0).nonzero(as_tuple=False).flatten().tolist()
        else:
            pos_idx = []
        if len(pos_idx) == 0:
            start = random.randint(0, max(0, L - window_size))
        else:
            center = random.choice(pos_idx)
            start = max(0, min(center - window_size // 2, L - window_size))
        end = min(start + window_size, L)
        crop = x[b, start:end]
        out[b, : crop.size(0)] = crop
    return _restore_shape(out, token_ids)


def augment_kmer_shuffle_outside(
    token_ids: Tensor | list,
    motif_flags: Optional[Tensor | list],
    k: int = 3,
    pad_token_id: int = 0,
    seed: Optional[int] = None,
) -> Tensor:
    
    _set_seeds(seed)
    x = _as_long_2d(token_ids)
    m = None if motif_flags is None else _as_long_2d(motif_flags)
    B, L = x.shape
    out = x.clone()

    for b in range(B):
        blocks = []
        touches_motif = []
        for i in range(0, L, k):
            sl = slice(i, min(i + k, L))
            blocks.append(out[b, sl].clone())
            if m is None:
                touches_motif.append(False)
            else:
                touches_motif.append((m[b, sl] > 0).any().item())

        idx_outside = [i for i, t in enumerate(touches_motif) if not t]
        shuffled = idx_outside[:]
        random.shuffle(shuffled)

        new_blocks = list(blocks)
        for orig, sh in zip(idx_outside, shuffled):
            new_blocks[orig] = blocks[sh]

        ptr = 0
        for blk in new_blocks:
            n = blk.size(0)
            out[b, ptr:ptr + n] = blk
            ptr += n

    return _restore_shape(out, token_ids)


def choose_motif_aware_augment(augment_name: str) -> AugmentFn:
  
    name = (augment_name or "").lower()
    if name == "mask_non_motif":
        return augment_mask_non_motif
    if name == "window_centered":
        return augment_window_centered
    if name == "kmer_shuffle_outside":
        return augment_kmer_shuffle_outside
    return cml_augment_random_mask
