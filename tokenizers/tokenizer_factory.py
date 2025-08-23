# tokenizer_factory

from __future__ import annotations 

from typing import List, Tuple, Dict
import torch

from .kmer_tokenizer import KmerTokenizer
from .bpe_tokenizer import BPETokenizer
from .hybrid_tokenizer_fusion import HybridTokenizerTokenFusion
from .hybrid_tokenizer_embedding_fusion import HybridTokenizerEmbeddingFusion
from .utils import batchify


DEFAULT_HYBRID_MODE = "dual"  # "dual" -> embedding fusion; "token"  --- > token level fusion


def build_special_tokens(cfg) -> Tuple[List[str], str, str]:
    
    pad = getattr(cfg, "PAD_TOKEN", "[PAD]")
    unk = getattr(cfg, "UNK_TOKEN", "[UNK]")

    extras: List[str] = []
    for name in ("MASK_TOKEN", "CLS_TOKEN", "SEP_TOKEN"):
        tok = getattr(cfg, name, None)
        if tok and tok not in extras and tok not in (pad, unk):
            extras.append(tok)

    return extras, pad, unk


def create_tokenizer(cfg):

    specials, pad, unk = build_special_tokens(cfg)
    tok_type = getattr(cfg, "TOKENIZER_TYPE", "kmer").lower()

    if tok_type == "kmer":
        return KmerTokenizer(
            k=int(getattr(cfg, "KMER_SIZE", 3)),
            special_tokens=specials,
            max_vocab_size=(None if getattr(cfg, "VOCAB_SIZE", None) is None else int(cfg.VOCAB_SIZE)),
        )

    if tok_type == "bpe":
        return BPETokenizer(
            vocab_size=int(getattr(cfg, "VOCAB_SIZE", 1000)),
            merge_num=getattr(cfg, "MERGE_NUM", None),
            special_tokens=specials,
        )

    if tok_type == "hybrid":
        mode = getattr(cfg, "HYBRID_MODE", DEFAULT_HYBRID_MODE).lower()
        if mode == "token":
            return HybridTokenizerTokenFusion(
                k=int(getattr(cfg, "KMER_SIZE", 3)),
                vocab_size=int(getattr(cfg, "VOCAB_SIZE", 1000)),
                merge_num=getattr(cfg, "MERGE_NUM", None),
                special_tokens=specials,
            )
        return HybridTokenizerEmbeddingFusion(
            k=int(getattr(cfg, "KMER_SIZE", 3)),
            vocab_size=int(getattr(cfg, "VOCAB_SIZE", 1000)),
            merge_num=getattr(cfg, "MERGE_NUM", None),
            special_tokens=specials,
        )

    raise ValueError(f"Unknown TOKENIZER_TYPE: {tok_type!r}")


def collate_single_stream(batch_ids: List[List[int]], pad_id: int, max_len=None) -> Dict[str, "torch.Tensor"]:
    return batchify(batch_ids, pad_id=pad_id, max_len=max_len, padding="right", return_positions=True)
