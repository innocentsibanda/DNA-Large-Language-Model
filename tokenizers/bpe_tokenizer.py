# bpe_tokenizer
from __future__ import annotations

import json
import hashlib
from collections import Counter
from typing import List, Optional, Tuple, Dict, Any

from .tokenizer_base import BaseTokenizer

class BPETokenizer(BaseTokenizer):

    def __init__(
        self,
        vocab_size: int = 1000,
        merge_num: Optional[int] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        super().__init__(special_tokens)
        self.vocab_size = int(vocab_size)
        self.merge_num = int(merge_num) if merge_num is not None else None
        self.end_of_token = "</w>"
        self.bpe_ranks: Dict[Tuple[str, str], int] = {} 
        self._frozen: bool = False
        self.tokenizer_version: str = "bpe.v1"

    def preprocess(self, seq: str) -> str:
        s = (seq or "").upper().replace(" ", "")
        allowed = set("ACGTUN")  
        return "".join(ch if ch in allowed else "N" for ch in s)

    def _word_tokens_with_eow(self, seq: str) -> List[str]:
        s = self.preprocess(seq)
        if not s:
            return []
        if len(s) == 1:
            return [s + self.end_of_token]
        return list(s[:-1]) + [s[-1] + self.end_of_token]

    @staticmethod
    def _deterministic_best_pair(pairs: Counter) -> Optional[Tuple[str, str]]:
        if not pairs:
            return None
        max_count = max(pairs.values())
        candidates = [p for p, c in pairs.items() if c == max_count]
        return sorted(candidates)[0]

    @staticmethod
    def _merge_once(tokens: List[str], pair: Tuple[str, str]) -> List[str]:
        A, B = pair
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == A and tokens[i + 1] == B:
                merged.append(A + B)  # concat strings
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def train(self, sequences: List[str]) -> None:
        if self._frozen:
            raise RuntimeError("Tokenizer is frozen; refuse to retrain (avoid leakage).")

        corpus_tokens: List[List[str]] = [self._word_tokens_with_eow(seq) for seq in sequences if seq]

        while True:
            if self.merge_num is not None and len(self.bpe_ranks) >= self.merge_num:
                break
            if len(self.bpe_ranks) >= max(0, self.vocab_size - len(self.vocab)):
                break

            pair_counts = Counter()
            for toks in corpus_tokens:
                for i in range(len(toks) - 1):
                    pair_counts[(toks[i], toks[i + 1])] += 1

            best = self._deterministic_best_pair(pair_counts)
            if best is None or pair_counts[best] <= 0:
                break

            self.bpe_ranks[best] = len(self.bpe_ranks) 
            corpus_tokens = [self._merge_once(toks, best) for toks in corpus_tokens]

            if self.merge_num is None and len(self.bpe_ranks) >= 10 * self.vocab_size:
                break

        for toks in corpus_tokens:
            for tok in toks:
                if tok not in self.vocab:
                    if len(self.vocab) >= self.vocab_size:
                        break
                    idx = len(self.vocab)
                    self.vocab[tok] = idx
                    self.inv_vocab[idx] = tok


    def encode(self, sequence: str) -> List[int]:
        toks = self._word_tokens_with_eow(sequence)
        if not toks:
            return []
        pair2rank = self.bpe_ranks
        if not pair2rank:
            unk = self.unk_id()
            return [self.vocab.get(t, unk) for t in toks]

        while True:
            best_pair = None
            best_rank = None
            for i in range(len(toks) - 1):
                p = (toks[i], toks[i + 1])
                r = pair2rank.get(p)
                if r is None:
                    continue
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best_pair = p
            if best_pair is None:
                break
            toks = self._merge_once(toks, best_pair)

        unk_id = self.unk_id()
        return [self.vocab.get(t, unk_id) for t in toks]

    def decode(self, token_ids: List[int]) -> str:
        toks = [self.inv_vocab.get(int(idx), self.special_tokens_map.get("UNK", "<UNK>")) for idx in token_ids]
        return "".join(t.replace(self.end_of_token, "") for t in toks)

    def effective_vocab_size(self) -> int:
        return int(len(self.vocab))

    def freeze(self) -> None:
        self._frozen = True

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def _checksum(self) -> str:
        h = hashlib.sha256()
        for tok, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
            h.update(f"{idx}:{tok}".encode("utf-8"))
        for (a, b), r in sorted(self.bpe_ranks.items(), key=lambda x: x[1]):
            h.update(f"{r}:{a}|{b}".encode("utf-8"))
        return h.hexdigest()

    def save_state(self, path: str) -> None:
        data: Dict[str, Any] = {
            "type": self.__class__.__name__,
            "version": self.tokenizer_version,
            "vocab_size": self.vocab_size,
            "merge_num": self.merge_num,
            "special_tokens": self.special_tokens,
            "end_of_token": self.end_of_token,
            "vocab": self.vocab,
            "bpe_ranks": {f"{a}\t{b}": r for (a, b), r in self.bpe_ranks.items()},
            "frozen": self._frozen,
            "checksum": self._checksum(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_state(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.tokenizer_version = data.get("version", self.tokenizer_version)
        self.vocab_size = int(data.get("vocab_size", self.vocab_size))
        self.merge_num = data.get("merge_num", self.merge_num)
        self.special_tokens = data.get("special_tokens", self.special_tokens)
        self.end_of_token = data.get("end_of_token", self.end_of_token)

        self.vocab = {tok: int(idx) for tok, idx in data["vocab"].items()}
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}

        self.bpe_ranks = {}
        for k, r in data.get("bpe_ranks", {}).items():
            a, b = k.split("\t")
            self.bpe_ranks[(a, b)] = int(r)

        self._frozen = bool(data.get("frozen", False))
    
        _saved = data.get("checksum")
       
