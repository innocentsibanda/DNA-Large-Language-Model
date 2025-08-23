#tokenizer base 

import json
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Type, TypeVar, Any

T = TypeVar("T", bound="BaseTokenizer")

class BaseTokenizer(ABC):

    def __init__(self, special_tokens: List[str] | None = None):
        
        special_tokens = list(special_tokens or [])
        if "<PAD>" not in special_tokens:
            special_tokens.insert(0, "<PAD>")
        if "<UNK>" not in special_tokens:
            insert_at = 1 if len(special_tokens) > 0 else 0
            special_tokens.insert(insert_at, "<UNK>")

        self.special_tokens: List[str] = special_tokens

        self.vocab: Dict[str, int] = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.inv_vocab: Dict[int, str] = {i: tok for tok, i in self.vocab.items()}

    @abstractmethod
    def train(self, sequences: List[str]) -> None:
        
        pass

    @abstractmethod
    def encode(self, sequence: str) -> List[int]:
        
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        pass

    def pad_id(self) -> int:
        return self.vocab["<PAD>"]

    def unk_id(self) -> int:
        return self.vocab["<UNK>"]

    def save_vocab(self, filepath: str) -> None:
    
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "special_tokens": self.special_tokens,
                        "vocab": self.vocab,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except IOError as e:
            print(f"Error saving vocab to {filepath}: {e}")

    def load_vocab(self, filepath: str) -> None:
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            loaded_vocab = data.get("vocab", {})
            loaded_specials = data.get("special_tokens", self.special_tokens)

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

        except IOError as e:
            print(f"Error loading vocab from {filepath}: {e}")

    def get_config(self) -> Dict[str, Union[List[str], int, str]]:
        return {
            "type": self.__class__.__name__,
            "special_tokens": self.special_tokens,
            "vocab_size": len(self.vocab),
        }

    @classmethod
    def from_config(cls: Type[T], config: Dict[str, Any]) -> T:
        
        special_tokens = config.get("special_tokens", [])
        return cls(special_tokens=special_tokens) 
