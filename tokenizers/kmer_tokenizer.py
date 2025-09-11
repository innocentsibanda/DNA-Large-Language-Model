# kmer_tokenizer

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
        allow_ambiguous: bool = True,
        alphabet: str = "ACGTN",
        enforce_alphabet: bool = False,
    ):
        super().__init__(special_tokens)
        self.k = int(k)
        self.max_vocab_size = int(max_vocab_size) if max_vocab_size is not None else None
        self.allow_ambiguous = bool(allow_ambiguous)
        self.alphabet = "".join(sorted(set(alphabet.upper())))
        self.enforce_alphabet = bool(enforce_alphabet)


    def train(self, sequences: List[str]) -> None:
        if self.k <= 0:
            raise ValueError("k must be a positive integer")

        kmer_counts = Counter()
        for seq in sequences:
            seq = self.preprocess(seq)
            if len(seq) < self.k:
                continue
            kmers = (seq[i : i + self.k] for i in range(len(seq) - self.k + 1))
            kmers = (k for k in kmers if self._valid_kmer(k))
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

        toks = [sequence[i : i + self.k] for i in range(len(sequence) - self.k + 1)]
        toks = [t for t in toks if self._valid_kmer(t)]
        unk = self.unk_id()
        return [self.vocab.get(t, unk) for t in toks]

    def decode(self, token_ids: List[int]) -> str:
        return self.decode_to_sequence(token_ids)


    def decode_to_kmers(self, token_ids: List[int]) -> List[str]:
        return [self.inv_vocab.get(idx, "<UNK>") for idx in token_ids]

    def decode_to_sequence(self, token_ids: List[int]) -> str:
        
        toks = self.decode_to_kmers(token_ids)
        if not toks:
            return ""

        toks = [t for t in toks if t not in self.special_tokens]

        if not toks:
            return ""

        k = self.k
        norm: List[str] = []
        for t in toks:
            if t == "<UNK>":
                norm.append("N" * k)
            elif isinstance(t, str) and len(t) == k:
                norm.append(t)

        if not norm:
            return ""

        seq = norm[0]
        if k == 1:

            for t in norm[1:]:
                seq += t
            return seq

        for t in norm[1:]:
            seq += t[-1]
        return seq

    def preprocess(self, seq: str) -> str:
        return (seq or "").upper().replace(" ", "")

    def _valid_kmer(self, kmer: str) -> bool:
        """Validate a k-mer according to ambiguity and alphabet rules."""
        if len(kmer) != self.k:
            return False
        if not self.allow_ambiguous and ("N" in kmer):
            return False
        if self.enforce_alphabet:
            for c in kmer:
                if c not in self.alphabet:
                    return False
        return True

    def save_state(self, path: str) -> None:
        import json

        data: Dict[str, Any] = {
            "type": self.__class__.__name__,
            "k": self.k,
            "max_vocab_size": self.max_vocab_size,
            "allow_ambiguous": self.allow_ambiguous,
            "alphabet": self.alphabet,
            "enforce_alphabet": self.enforce_alphabet,
            "_vocab_payload": {
                "special_tokens": self.special_tokens,
                "vocab": self.vocab,
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_state(self, path: str) -> None:
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.k = int(data.get("k", self.k))
        mvs = data.get("max_vocab_size", self.max_vocab_size)
        self.max_vocab_size = int(mvs) if mvs is not None else None
        self.allow_ambiguous = bool(data.get("allow_ambiguous", self.allow_ambiguous))
        self.alphabet = str(data.get("alphabet", self.alphabet)).upper()
        self.enforce_alphabet = bool(data.get("enforce_alphabet", self.enforce_alphabet))

        vocab_payload = data.get("_vocab_payload", {})
        loaded_vocab = vocab_payload.get("vocab", {})
        loaded_specials = vocab_payload.get("special_tokens", self.special_tokens)

        specials = list(loaded_specials or [])
        if "<PAD>" not in specials:
            specials.insert(0, "<PAD>")
        if "<UNK>" not in specials:
            insert_at = 1 if len(specials) > 0 else 0
            specials.insert(insert_at, "<UNK>")

        self.special_tokens = specials
        self.vocab = {tok: int(idx) for tok, idx in loaded_vocab.items()}

        for tok in self.special_tokens:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)

        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}
