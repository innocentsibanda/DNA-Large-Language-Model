# Collate Functions 
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F

import config as cfg
from pretraining_self_supervised import (
    apply_mlm_masking,
    apply_span_corruption,
    reorder_kmer_window,
    generate_dae_pair,
)
from pretraining_supervised import (
    apply_masked_motif_modeling,
    apply_motif_boundary_prediction,
    cml_augment_random_mask,
    choose_motif_aware_augment,
)

def _pad_2d(list_of_1d: List[torch.Tensor], pad_value: int, max_len: Optional[int]) -> torch.Tensor:
    if len(list_of_1d) == 0:
        return torch.empty(0, 0, dtype=torch.long)

    L = max_len if max_len is not None else max(int(x.numel()) for x in list_of_1d)
    if L <= 0:
        return torch.empty(len(list_of_1d), 0, dtype=torch.long)

    out = torch.full((len(list_of_1d), L), fill_value=pad_value, dtype=torch.long)
    for i, x in enumerate(list_of_1d):
        n = min(L, int(x.numel()))
        if n > 0:
            out[i, :n] = x[:n]
    return out

def _to_long_list(ids_list: List[List[int]]) -> List[torch.Tensor]:
    return [torch.tensor(x, dtype=torch.long) for x in ids_list]

def _build_positions(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)


def make_selfsup_collate(tokenizer, task: str = "mlm"):
    tok_type = getattr(cfg, "TOKENIZER_TYPE", "kmer").lower()
    hybrid_mode = getattr(cfg, "HYBRID_MODE", "dual").lower()
    is_dual = (tok_type == "hybrid" and hybrid_mode == "dual")

    pad_id = tokenizer.pad_id()
    mask_id = tokenizer.vocab.get(cfg.MASK_TOKEN, pad_id)
    MAX = getattr(cfg, "MAX_LEN", None)

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        seqs = [b["seq"] for b in batch]

        if not is_dual:
            ids_list = [tokenizer.encode(s) for s in seqs]
            ids = _pad_2d(_to_long_list(ids_list), pad_value=pad_id, max_len=MAX)
            attn = (ids != pad_id).long()
            pos = _build_positions(ids.size(0), ids.size(1), device=ids.device)

            t = (task or "mlm").lower()
            if t == "mlm":
                inp, labels = apply_mlm_masking(ids, tokenizer, mask_prob=cfg.MASK_PROB)
            elif t == "span":
                inp, labels = apply_span_corruption(ids, tokenizer, mask_prob=cfg.MASK_PROB, max_span=cfg.SPAN_MAX_LEN)
            elif t == "kmer_reorder":
                inp, labels = reorder_kmer_window(ids, window_size=cfg.REORDER_WINDOW)
            elif t == "dae":
                inp, labels = generate_dae_pair(ids, tokenizer, mask_prob=cfg.MASK_PROB, masked_only=True)
            else:
                inp, labels = ids, torch.full_like(ids, -100)

            return {
                "input_ids": inp,
                "attention_mask": attn,
                "position_ids": pos,
                "labels": labels,
            }
        encs = [tokenizer.encode_for_embedding(s) for s in seqs]
        kmer_list = [torch.tensor(e["kmer_input_ids"], dtype=torch.long) for e in encs]
        bpe_list  = [torch.tensor(e["bpe_input_ids"],  dtype=torch.long) for e in encs]

        kmer_ids = _pad_2d(kmer_list, pad_value=pad_id, max_len=MAX)
        bpe_ids  = _pad_2d(bpe_list,  pad_value=pad_id, max_len=MAX)

        kmer_am = (kmer_ids != pad_id).long()
        bpe_am  = (bpe_ids  != pad_id).long()

        pos_k = _build_positions(kmer_ids.size(0), kmer_ids.size(1), device=kmer_ids.device)
        pos_b = _build_positions(bpe_ids.size(0),  bpe_ids.size(1),  device=bpe_ids.device)

        stream = getattr(cfg, "SELF_SUP_STREAM", "kmer").lower()
        if stream == "kmer":
            if task == "mlm":
                _, labels_k = apply_mlm_masking(kmer_ids, tokenizer, mask_prob=cfg.MASK_PROB)
            elif task == "span":
                _, labels_k = apply_span_corruption(kmer_ids, tokenizer, mask_prob=cfg.MASK_PROB, max_span=cfg.SPAN_MAX_LEN)
            elif task == "kmer_reorder":
                _, labels_k = reorder_kmer_window(kmer_ids, window_size=cfg.REORDER_WINDOW)
            elif task == "dae":
                _, labels_k = generate_dae_pair(kmer_ids, tokenizer, mask_prob=cfg.MASK_PROB, masked_only=True)
            else:
                labels_k = torch.full_like(kmer_ids, -100)
            labels_b = torch.full_like(bpe_ids, -100)
        else:
            if task == "mlm":
                _, labels_b = apply_mlm_masking(bpe_ids, tokenizer, mask_prob=cfg.MASK_PROB)
            elif task == "span":
                _, labels_b = apply_span_corruption(bpe_ids, tokenizer, mask_prob=cfg.MASK_PROB, max_span=cfg.SPAN_MAX_LEN)
            elif task == "kmer_reorder":
                _, labels_b = reorder_kmer_window(bpe_ids, window_size=cfg.REORDER_WINDOW)
            elif task == "dae":
                _, labels_b = generate_dae_pair(bpe_ids, tokenizer, mask_prob=cfg.MASK_PROB, masked_only=True)
            else:
                labels_b = torch.full_like(bpe_ids, -100)
            labels_k = torch.full_like(kmer_ids, -100)

        return {
            "kmer_input_ids": kmer_ids,
            "kmer_attention_mask": kmer_am,
            "kmer_position_ids": pos_k,
            "bpe_input_ids": bpe_ids,
            "bpe_attention_mask": bpe_am,
            "bpe_position_ids": pos_b,
            "labels_kmer": labels_k,
            "labels_bpe": labels_b,
            "motif_flags": torch.zeros_like(kmer_ids, dtype=torch.long),
        }

    return collate

def make_supervised_collate(tokenizer, task_name: str, augment: Optional[str] = None):
    
    pad_id = tokenizer.pad_id()
    mask_id = tokenizer.vocab.get(cfg.MASK_TOKEN, pad_id)
    MAX = getattr(cfg, "MAX_LEN", None)

    aug_fn = choose_motif_aware_augment(augment or "random_mask")

    def _extract_arrays(item: Dict[str, Any], key: str, L_ids: int) -> torch.Tensor:
        arr = item.get(key, None)
        if arr is None:
            return torch.zeros(L_ids, dtype=torch.long)
        t = torch.tensor(arr, dtype=torch.long)
        if t.numel() >= L_ids:
            return t[:L_ids]
        out = torch.zeros(L_ids, dtype=torch.long)
        out[: t.numel()] = t
        return out

    map_label_key = getattr(cfg, "MAP_LABEL_KEY", "global_label")

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        seqs = [b["seq"] for b in batch]

        if hasattr(tokenizer, "encode_for_embedding"):
            encs = [tokenizer.encode_for_embedding(s) for s in seqs]
            ids_list = [e["kmer_input_ids"] for e in encs]
        else:
            ids_list = [tokenizer.encode(s) for s in seqs]

        ids = _pad_2d(_to_long_list(ids_list), pad_value=pad_id, max_len=MAX)  # (B, L)
        attn = (ids != pad_id).long()
        pos = _build_positions(ids.size(0), ids.size(1), device=ids.device)

        motif_flags_rows: List[torch.Tensor] = []
        motif_labels_rows: List[torch.Tensor] = []
        motif_bounds_rows: List[torch.Tensor] = []

        for itm, enc in zip(batch, ids_list):
            L_ids = min(len(enc), MAX if MAX is not None else len(enc))
            motif_flags_rows.append(_extract_arrays(itm, "motif_flags", L_ids))
            motif_labels_rows.append(_extract_arrays(itm, "motif_labels", L_ids))
            motif_bounds_rows.append(_extract_arrays(itm, "motif_boundaries", L_ids))

        motif_flags = _pad_2d(motif_flags_rows, pad_value=0, max_len=ids.size(1))
        motif_labels = _pad_2d(motif_labels_rows, pad_value=0, max_len=ids.size(1))
        motif_bounds = _pad_2d(motif_bounds_rows, pad_value=0, max_len=ids.size(1))

        name = (task_name or "masked_motif").lower()
        if name == "masked_motif":
            inp, labels = apply_masked_motif_modeling(
                ids, motif_flags, mask_token_id=mask_id, mask_prob=cfg.MASK_PROB, pad_token_id=pad_id
            )
        elif name == "mbp":
            inp, labels = apply_motif_boundary_prediction(ids, motif_bounds)
        elif name == "map":
            have_all_seq_labels = all(
                (map_label_key in itm) and (itm[map_label_key] is not None)
                for itm in batch
            )
            if have_all_seq_labels:
                seq_labels: List[int] = []
                for itm in batch:
                    val = itm[map_label_key]
                    if not isinstance(val, int):
                        raise ValueError(f"MAP collate: '{map_label_key}' must be int id, got {type(val)}")
                    seq_labels.append(int(val))
                labels = torch.tensor(seq_labels, dtype=torch.long) 
                inp = ids  
            else:
                token_labels = motif_labels.clone()
                token_labels[attn == 0] = -100
                labels = token_labels                     
                inp = ids                                  
        else:
            inp, labels = ids, torch.full_like(ids, -100)

        view2 = aug_fn(ids, motif_flags, mask_token_id=mask_id, pad_token_id=pad_id)
        am2 = (view2 != pad_id).long()

        out: Dict[str, torch.Tensor] = {
            "input_ids": inp,
            "attention_mask": attn,
            "position_ids": pos,
            "labels": labels, 
            "input_ids_view2": view2,
            "attention_mask_view2": am2,
            "motif_flags": motif_flags,
            "motif_labels": motif_labels,
            "motif_boundaries": motif_bounds,
        }
        return out

    return collate
