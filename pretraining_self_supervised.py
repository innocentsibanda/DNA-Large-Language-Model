# pretraining_self_supervised learning

from __future__ import annotations

import random
from typing import Tuple, Union, Optional

import torch

from config import MASK_TOKEN

TensorOrList = Union[list, torch.Tensor]


def _as_long_tensor(x: TensorOrList) -> torch.Tensor:
    if isinstance(x, list):
        return torch.tensor(x, dtype=torch.long)
    if not torch.is_tensor(x):
        raise TypeError(f"Expected list or Tensor, got {type(x)}")
    return x.long()


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2:
        return x
    raise ValueError(f"Expected 1D or 2D tensor, got shape {tuple(x.shape)}")


def _restore_shape(x: torch.Tensor, like: torch.Tensor | list) -> torch.Tensor:
    if torch.is_tensor(like) and like.dim() == 1:
        return x.squeeze(0)
    if isinstance(like, list):
        return x.squeeze(0)
    return x


def _set_seeds(seed: Optional[int]) -> None:
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

def _effective_vocab_size_for(tokenizer) -> int:
    if hasattr(tokenizer, "effective_vocab_size"):
        try:
            return int(tokenizer.effective_vocab_size())  
        except Exception:
            pass

    if hasattr(tokenizer, "kmer_tokenizer") and hasattr(tokenizer, "bpe_tokenizer"):
        try:
            kmer_vs = len(tokenizer.kmer_tokenizer.vocab)
            kmer_vs = int(getattr(tokenizer, "kmer_vocab_size", kmer_vs) or kmer_vs)
        except Exception:
            kmer_vs = 0
        try:
            bpe_vs = len(tokenizer.bpe_tokenizer.vocab)
        except Exception:
            bpe_vs = 0
        fused = int(kmer_vs) + int(bpe_vs)
        if fused > 0:
            return fused

    vocab = getattr(tokenizer, "vocab", None) or {}
    return int(len(vocab))


def _get_vocab_and_ids(tokenizer):
    
    vocab = getattr(tokenizer, "vocab", None) or getattr(tokenizer, "token2id", None)
    if vocab is None:
        raise ValueError("Tokenizer must have 'vocab' or 'token2id' attribute.")
    if MASK_TOKEN not in vocab:
        raise ValueError(f"Tokenizer vocab must include mask token '{MASK_TOKEN}'.")
    mask_id = int(vocab[MASK_TOKEN])
    if not hasattr(tokenizer, "pad_id"):
        raise ValueError("Tokenizer must implement pad_id() to indicate padding index.")
    pad_id = int(tokenizer.pad_id())
    vocab_size = _effective_vocab_size_for(tokenizer)
    return vocab, mask_id, pad_id, int(vocab_size)


def apply_mlm_masking(
    token_ids: TensorOrList,
    tokenizer,
    mask_prob: float = 0.15,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _set_seeds(seed)

    x = _as_long_tensor(token_ids)
    x2d = _ensure_2d(x).clone()
    B, L = x2d.shape

    _, mask_id, pad_id, vocab_size = _get_vocab_and_ids(tokenizer)

    labels = torch.full_like(x2d, fill_value=-100)
    input_ids = x2d.clone()

    valid = (x2d != pad_id)
    to_mask = (torch.rand(B, L, device=x2d.device) < mask_prob) & valid

    labels[to_mask] = x2d[to_mask]

    r = torch.rand(B, L, device=x2d.device)
    m_mask = (r < 0.8) & to_mask
    m_rand = (r >= 0.8) & (r < 0.9) & to_mask

    input_ids[m_mask] = mask_id
    if m_rand.any():
        input_ids[m_rand] = torch.randint(0, vocab_size, size=(m_rand.sum().item(),), device=x2d.device)

    input_ids.clamp_(0, vocab_size - 1)
    return _restore_shape(input_ids, x), _restore_shape(labels, x)


def apply_span_corruption(
    token_ids: TensorOrList,
    tokenizer,
    mask_prob: float = 0.15,
    max_span: int = 5,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _set_seeds(seed)

    x = _as_long_tensor(token_ids)
    x2d = _ensure_2d(x).clone()
    B, L = x2d.shape

    _, mask_id, pad_id, vocab_size = _get_vocab_and_ids(tokenizer)

    labels = torch.full_like(x2d, fill_value=-100)
    input_ids = x2d.clone()

    for b in range(B):
        i = 0
        while i < L:
            if input_ids[b, i].item() == pad_id:
                i += 1
                continue
            if random.random() < mask_prob:
                span = random.randint(1, max_span)
                end = i
                steps = 0
                while end < L and input_ids[b, end].item() != pad_id and steps < span:
                    end += 1
                    steps += 1
                labels[b, i:end] = x2d[b, i:end]
                input_ids[b, i:end] = mask_id
                i = end
            else:
                i += 1

    input_ids.clamp_(0, vocab_size - 1)
    return _restore_shape(input_ids, x), _restore_shape(labels, x)


def reorder_kmer_window(
    token_ids: TensorOrList,
    window_size: int = 5,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _set_seeds(seed)

    x = _as_long_tensor(token_ids)
    x2d = _ensure_2d(x)
    B, L = x2d.shape

    corrupted = x2d.clone()
    labels = x2d.clone()

    if L <= 1:
        return _restore_shape(corrupted, x), _restore_shape(labels, x)

    for b in range(B):
        w = max(2, min(window_size, L))
        start = random.randint(0, L - w)
        end = start + w
        window = corrupted[b, start:end].clone()
        perm = torch.randperm(w, device=corrupted.device)
        corrupted[b, start:end] = window[perm]

    return _restore_shape(corrupted, x), _restore_shape(labels, x)

def generate_dae_pair(
    token_ids: TensorOrList,
    tokenizer,
    mask_prob: float = 0.15,
    seed: Optional[int] = None,
    masked_only: bool = True, 
) -> Tuple[torch.Tensor, torch.Tensor]:
    _set_seeds(seed)
    x = _as_long_tensor(token_ids)
    x2d = _ensure_2d(x).clone()
    B, L = x2d.shape

    _, mask_id, pad_id, vocab_size = _get_vocab_and_ids(tokenizer)

    corrupted = x2d.clone()
    valid = (x2d != pad_id)
    to_corrupt = (torch.rand(B, L, device=x2d.device) < mask_prob) & valid

    r = torch.rand(B, L, device=x2d.device)
    to_mask   = (r < 0.8) & to_corrupt
    to_random = (r >= 0.8) & (r < 0.9) & to_corrupt

    corrupted[to_mask] = mask_id
    if to_random.any():
        corrupted[to_random] = torch.randint(0, vocab_size, (to_random.sum().item(),), device=x2d.device)

    corrupted.clamp_(0, vocab_size - 1)

    if masked_only:
        labels = torch.full_like(x2d, -100)
        labels[to_corrupt] = x2d[to_corrupt]
    else:
        labels = x2d

    return _restore_shape(corrupted, x), _restore_shape(labels, x)
