#Hybrid toknizer with embedding fusion

from __future__ import annotations

from typing import List, Tuple, Dict, Union, Optional

import torch

from .tokenizer_base import BaseTokenizer
from .kmer_tokenizer import KmerTokenizer
from .bpe_tokenizer import BPETokenizer
from .utils import pad_sequences


class HybridTokenizerEmbeddingFusion(BaseTokenizer):

    def __init__(
        self,
        k: int = 3,
        vocab_size: int = 1000,
        special_tokens: Optional[List[str]] = None,
        merge_num: Optional[int] = None,
    ):
        super().__init__(special_tokens)

        self.k = int(k)
        self.vocab_size = int(vocab_size)
        self.merge_num = int(merge_num) if merge_num is not None else None

        self.kmer_tokenizer = KmerTokenizer(k=self.k, special_tokens=self.special_tokens)
        self.bpe_tokenizer = BPETokenizer(
            vocab_size=self.vocab_size,
            merge_num=self.merge_num,
            special_tokens=self.special_tokens,
        )

        self.kmer_vocab_size = 0 

    def train(self, sequences: List[str]) -> None:
        self.kmer_tokenizer.train(sequences)
        self.bpe_tokenizer.train(sequences)
        self.kmer_vocab_size = len(self.kmer_tokenizer.vocab)

    def encode(self, sequence: str) -> Tuple[List[int], List[int]]:
        
        kmer_encoded = self.kmer_tokenizer.encode(sequence)
        bpe_encoded = self.bpe_tokenizer.encode(sequence)
        return kmer_encoded, bpe_encoded

    def decode(self, token_ids: List[int]) -> str:
        
        return self.kmer_tokenizer.decode(token_ids)

    def decode_streams(self, ids: Tuple[List[int], List[int]]) -> Tuple[str, str]:
        
        kmer_ids, bpe_ids = ids
        kmer_decoded = self.kmer_tokenizer.decode(kmer_ids)
        bpe_decoded = self.bpe_tokenizer.decode(bpe_ids)
        return kmer_decoded, bpe_decoded

    def decode_to_tokens(self, ids: Tuple[List[int], List[int]]) -> Tuple[List[str], List[str]]:
       
        kmer_ids, bpe_ids = ids

        if hasattr(self.kmer_tokenizer, "decode_to_kmers"):
            kmer_tokens = self.kmer_tokenizer.decode_to_kmers(kmer_ids)
        else:
            kmer_tokens = [self.kmer_tokenizer.inv_vocab.get(i, "<UNK>") for i in kmer_ids]

        end = getattr(self.bpe_tokenizer, "end_of_token", "</w>")
        bpe_tokens = [
            self.bpe_tokenizer.inv_vocab.get(i, "<UNK>").replace(end, "")
            for i in bpe_ids
        ]

        return kmer_tokens, bpe_tokens

    def encode_for_embedding(self, sequence: str) -> Dict[str, List[int]]:
        
        k_ids, b_ids = self.encode(sequence)
        return {"kmer_input_ids": k_ids, "bpe_input_ids": b_ids}

    def collate_batch_for_embedding(
        self,
        batch: List[Dict[str, List[int]]],
        max_len_k: Optional[int] = None,
        max_len_b: Optional[int] = None,
        padding: str = "right",
        dtype: torch.dtype = torch.long,
        return_positions: bool = False,
    ) -> Dict[str, torch.Tensor]:
       
        k_pad = self.kmer_tokenizer.pad_id()
        b_pad = self.bpe_tokenizer.pad_id()

        k_list = [ex["kmer_input_ids"] for ex in batch]
        b_list = [ex["bpe_input_ids"] for ex in batch]

        k_ids = pad_sequences(k_list, pad_token_id=k_pad, max_len=max_len_k, padding=padding, dtype=dtype)
        b_ids = pad_sequences(b_list, pad_token_id=b_pad, max_len=max_len_b, padding=padding, dtype=dtype)

        k_mask = (k_ids != k_pad).to(dtype=torch.long)
        b_mask = (b_ids != b_pad).to(dtype=torch.long)

        out: Dict[str, torch.Tensor] = {
            "kmer_input_ids": k_ids,
            "kmer_attention_mask": k_mask,
            "bpe_input_ids": b_ids,
            "bpe_attention_mask": b_mask,
        }

        if return_positions:
            Lk = k_ids.size(1)
            Lb = b_ids.size(1)
            out["kmer_position_ids"] = torch.arange(Lk, dtype=dtype).unsqueeze(0).expand_as(k_ids)
            out["bpe_position_ids"] = torch.arange(Lb, dtype=dtype).unsqueeze(0).expand_as(b_ids)

        return out
