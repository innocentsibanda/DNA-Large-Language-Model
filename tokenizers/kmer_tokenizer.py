# kmer tokenizer 

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Dict, Any

from .tokenizer_base import BaseTokenizer


class KmerTokenizer(BaseTokenizer):

    def __init__(
        self,
        k: int = 3,
        special_tokens: Optional[List[str]] = None,
        max_vocab_size: Optional[int] = None,
    ):
        super().__init__(special_tokens)
        self.k = int(k)
        self.max_vocab_size = int(max_vocab_size) if max_vocab_size is not None else None

    def train(self, sequences: List[str]) -> None:
        
        if self.k <= 0:
            raise ValueError("k must be a positive integer")

        kmer_counts = Counter()
        for seq in sequences:
            seq = self.preprocess(seq)
            if len(seq) < self.k:
                continue
            kmers = (seq[i : i + self.k] for i in range(len(seq) - self.k + 1))
            kmer_counts.update(kmers)

        sorted_kmers = (k for k, _ in kmer_counts.most_common())

        remaining = None
        if self.max_vocab_size is not None:
            remaining = max(self.max_vocab_size - len(self.vocab), 0)

        for kmer in sorted_kmers:
            if kmer in self.vocab:
                continue
            if remaining is not None and remaining == 0:
                break
            idx = len(self.vocab)
            self.vocab[kmer] = idx
            self.inv_vocab[idx] = kmer
            if remaining is not None:
                remaining -= 1

    def encode(self, sequence: str) -> List[int]:
      
        sequence = self.preprocess(sequence)
        if len(sequence) < self.k:
            return []

        tokens = [sequence[i : i + self.k] for i in range(len(sequence) - self.k + 1)]
        unk_id = self.unk_id()
        return [self.vocab.get(tok, unk_id) for tok in tokens]

    def decode(self, token_ids: List[int]) -> str:
      
        return self.decode_to_sequence(token_ids)

    
    def decode_to_kmers(self, token_ids: List[int]) -> List[str]:
        return [self.inv_vocab.get(idx, "<UNK>") for idx in token_ids]

    def decode_to_sequence(self, token_ids: List[int]) -> str:
        
        toks = self.decode_to_kmers(token_ids)
        if not toks:
            return ""
        
        return toks[0] + "".join(t[-1] for t in toks[1:] if t)

    def preprocess(self, seq: str) -> str:
        
        return (seq or "").upper().replace(" ", "")

    def save_state(self, path: str) -> None:
        
        import json

        data: Dict[str, Any] = {
            "type": self.__class__.__name__,
            "k": self.k,
            "max_vocab_size": self.max_vocab_size,
        }


        vocab_payload = {
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
        }
        data.update({"_vocab_payload": vocab_payload})

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_state(self, path: str) -> None:
       
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.k = int(data.get("k", self.k))
        mvs = data.get("max_vocab_size", self.max_vocab_size)
        self.max_vocab_size = int(mvs) if mvs is not None else None

        vocab_payload = data.get("_vocab_payload", {})
        loaded_vocab = vocab_payload.get("vocab", {})
        loaded_specials = vocab_payload.get("special_tokens", self.special_tokens)

        specials = list(loaded_specials or [])
        if "<PAD>" not in specials:
            specials.insert(0, "<PAD>")
        if "<UNK>" not in specials:
            specials.insert(1, "<UNK>")

        self.special_tokens = specials
        self.vocab = {tok: int(idx) for tok, idx in loaded_vocab.items()}
        for tok in self.special_tokens:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}
