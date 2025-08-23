# Hybrid tokenizer with fusion at tokenization level

from __future__ import annotations

from typing import List, Tuple, Union

from .tokenizer_base import BaseTokenizer
from .kmer_tokenizer import KmerTokenizer
from .bpe_tokenizer import BPETokenizer

try:
    import torch
except Exception: 
    torch = None


class HybridTokenizerTokenFusion(BaseTokenizer):

    def __init__(
        self,
        k: int = 3,
        vocab_size: int = 1000,
        special_tokens: List[str] | None = None,
        merge_num: int | None = None,
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

        self.kmer_vocab_size: int = 0

    def train(self, sequences: List[str]) -> None:
        self.kmer_tokenizer.train(sequences)
        self.bpe_tokenizer.train(sequences)
        self.kmer_vocab_size = len(self.kmer_tokenizer.vocab)

    def _ensure_offset_ready(self) -> None:
        if self.kmer_vocab_size == 0:
            self.kmer_vocab_size = len(self.kmer_tokenizer.vocab)
            if self.kmer_vocab_size == 0:
                raise RuntimeError(
                    "HybridTokenizerTokenFusion: call train() before encode() so kmer_vocab_size is set."
                )

    def encode(self, sequence: str) -> List[int]:
        
        self._ensure_offset_ready()

        kmer_ids = self.kmer_tokenizer.encode(sequence)
        bpe_ids = self.bpe_tokenizer.encode(sequence)

        combined: List[int] = []
        max_len = max(len(kmer_ids), len(bpe_ids))
        for i in range(max_len):
            if i < len(kmer_ids):
                combined.append(kmer_ids[i])
            if i < len(bpe_ids):
                combined.append(bpe_ids[i] + self.kmer_vocab_size)
        return combined

    def _split_streams(self, token_ids: List[int]) -> Tuple[List[int], List[int], List[int], List[int]]:
        kmer_ids: List[int] = []
        bpe_ids: List[int] = []
        kpos: List[int] = []
        bpos: List[int] = []

        for pos, tid in enumerate(token_ids):
            if tid < self.kmer_vocab_size:
                kmer_ids.append(tid)
                kpos.append(pos)
            else:
                bpe_ids.append(tid - self.kmer_vocab_size)
                bpos.append(pos)

        return kmer_ids, bpe_ids, kpos, bpos

    def decode_to_tokens(self, token_ids: Union[List[int], "torch.Tensor"]) -> List[str]:
       
        if torch is not None and hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        kmer_ids, bpe_ids, kpos, bpos = self._split_streams(token_ids)

        if hasattr(self.kmer_tokenizer, "decode_to_kmers"):
            kmer_tokens = self.kmer_tokenizer.decode_to_kmers(kmer_ids)
        else:
            kmer_tokens = [self.kmer_tokenizer.inv_vocab.get(i, "<UNK>") for i in kmer_ids]

        end = getattr(self.bpe_tokenizer, "end_of_token", "</w>")
        bpe_tokens = [self.bpe_tokenizer.inv_vocab.get(i, "<UNK>").replace(end, "") for i in bpe_ids]

        merged: List[str] = [None] * (len(kpos) + len(bpos)) 
        for pos, tok in zip(kpos, kmer_tokens):
            merged[pos] = tok
        for pos, tok in zip(bpos, bpe_tokens):
            merged[pos] = tok

        return [m if m is not None else "<UNK>" for m in merged]

    def decode(self, token_ids: Union[List[int], "torch.Tensor"]) -> str:
        
        if torch is not None and hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist() 
        kmer_ids, _, _, _ = self._split_streams(token_ids)
        return self.kmer_tokenizer.decode(kmer_ids)
