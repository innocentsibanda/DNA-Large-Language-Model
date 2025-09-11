# bpe tokenizer
from __future__ import annotations

import json
import os
from pathlib import Path
from collections import Counter
from typing import List, Optional, Tuple, Dict, Any, Iterable, Union

from .tokenizer_base import BaseTokenizer

Pair = Tuple[str, str]
PathLike = Union[str, os.PathLike]


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
        self.bpe_ranks: Dict[Pair, int] = {}
        self.end_of_token = "</w>"

    def preprocess(self, seq: str) -> str:
        return (seq or "").upper().replace(" ", "")

    @staticmethod
    def _pairs(tokens: List[str]) -> Iterable[Pair]:

        return zip(tokens, tokens[1:])

    def get_stats(self, tokens_list: List[List[str]]) -> Counter:
        pairs = Counter()
        for tokens in tokens_list:
            pairs.update(self._pairs(tokens))
        return pairs

    def merge_vocab(self, tokens_list: List[List[str]], pair: Pair) -> List[List[str]]:
        bigram = "".join(pair)
        new_tokens_list: List[List[str]] = []
        for tokens in tokens_list:
            merged_tokens: List[str] = []
            i = 0
            n = len(tokens)
            while i < n:
                if i < n - 1 and (tokens[i], tokens[i + 1]) == pair:
                    merged_tokens.append(bigram)
                    i += 2
                else:
                    merged_tokens.append(tokens[i])
                    i += 1
            new_tokens_list.append(merged_tokens)
        return new_tokens_list

    def train(self, sequences: List[str]) -> None:
        eot = self.end_of_token
        tokens_list: List[List[str]] = [
            [ch + eot for ch in self.preprocess(seq)]
            for seq in sequences
        ]

        prev_pairs: Optional[Counter] = None
        while (
            (self.merge_num is None or len(self.bpe_ranks) < self.merge_num)
            and len(self.vocab) < self.vocab_size
        ):
            pairs = self.get_stats(tokens_list)
            if not pairs or pairs == prev_pairs:
                break
            prev_pairs = pairs

            best_pair = max(pairs, key=pairs.get)
            self.bpe_ranks[best_pair] = len(self.bpe_ranks)
            tokens_list = self.merge_vocab(tokens_list, best_pair)

        for tokens in tokens_list:
            for token in tokens:
                if token not in self.vocab:
                    if len(self.vocab) >= self.vocab_size:
                        return
                    idx = len(self.vocab)
                    self.vocab[token] = idx
                    self.inv_vocab[idx] = token

    def encode(self, sequence: str) -> List[int]:
       
        eot = self.end_of_token
        tokens = [ch + eot for ch in self.preprocess(sequence)]
        i = 0
        n = len(tokens)
        while i < n - 1:
            pair = (tokens[i], tokens[i + 1])
            if pair in self.bpe_ranks:
                merged_token = "".join(pair)
                tokens[i : i + 2] = [merged_token]
                n -= 1  
                i = max(i - 1, 0)
            else:
                i += 1

        unk_id = self.unk_id()
        return [self.vocab.get(token, unk_id) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.inv_vocab.get(int(idx), "<UNK>") for idx in token_ids]
        eot = self.end_of_token
        return "".join(tok.replace(eot, "") for tok in tokens)

    def save_state(self, path: PathLike) -> None:
        data: Dict[str, Any] = {
            "type": self.__class__.__name__,
            "vocab_size": self.vocab_size,
            "merge_num": self.merge_num,
            "special_tokens": self.special_tokens,
            "end_of_token": self.end_of_token,
            "vocab": self.vocab,
            "bpe_ranks": {f"{a}\t{b}": r for (a, b), r in self.bpe_ranks.items()},
        }
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path) 

    def load_state(self, path: PathLike) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

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

    @classmethod
    def from_config_path(cls, cfg_path: PathLike) -> "BPETokenizer":
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        tok = cls(
            vocab_size=int(cfg.get("vocab_size", 1000)),
            merge_num=cfg.get("merge_num"),
            special_tokens=cfg.get("special_tokens"),
        )
        eot = cfg.get("end_of_token")
        if eot is not None:
            tok.end_of_token = str(eot)
        return tok

    def update_from_config(self, cfg: Dict[str, Any]) -> None:
        if "vocab_size" in cfg:
            self.vocab_size = int(cfg["vocab_size"])
        if "merge_num" in cfg:
            self.merge_num = cfg["merge_num"]
        if "special_tokens" in cfg:
            self.special_tokens = cfg["special_tokens"]
        if "end_of_token" in cfg:
            self.end_of_token = str(cfg["end_of_token"])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(vocab_size={self.vocab_size}, "
            f"merge_num={self.merge_num}, end_of_token={self.end_of_token!r}, "
            f"vocab={len(self.vocab)}, merges={len(self.bpe_ranks)})"
        )
