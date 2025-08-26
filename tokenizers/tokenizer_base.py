# tokenizer_base
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Type, TypeVar, Any, Optional

T = TypeVar("T", bound="BaseTokenizer")

class BaseTokenizer(ABC):

    ROLE_SYNONYMS: Dict[str, List[str]] = {
        "PAD":  ["[PAD]", "<PAD>", "PAD"],
        "UNK":  ["[UNK]", "<UNK>", "UNK"],
        "MASK": ["[MASK]", "<MASK>", "MASK"],
        "CLS":  ["[CLS]", "<CLS>", "CLS"],
        "SEP":  ["[SEP]", "<SEP>", "SEP"],
    }

    def __init__(
        self,
        special_tokens: Optional[List[str]] = None,
        require_mask_cls_sep: bool = True,
    ):
    
        provided = list(special_tokens or [])
        seen: set[str] = set()
        specials: List[str] = []
        for tok in provided:
            if tok not in seen:
                specials.append(tok)
                seen.add(tok)

        for role in ("PAD", "UNK"):
            if not self._any_synonym_present(specials, role):
                specials.insert(len(specials), self.ROLE_SYNONYMS[role][0])

        if require_mask_cls_sep:
            for role in ("MASK", "CLS", "SEP"):
                if not self._any_synonym_present(specials, role):
                    specials.insert(len(specials), self.ROLE_SYNONYMS[role][0])

        self.special_tokens: List[str] = specials
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        for i, tok in enumerate(self.special_tokens):
            self.vocab[tok] = i
            self.inv_vocab[i] = tok

        self.special_tokens_map: Dict[str, str] = {}
        for role in self.ROLE_SYNONYMS.keys():
            tok = self._resolve_present_token(role)
            if tok is not None:
                self.special_tokens_map[role] = tok

        self._frozen: bool = False

    @abstractmethod
    def train(self, sequences: List[str]) -> None:
        ...

    @abstractmethod
    def encode(self, sequence: str) -> List[int]:
        ...

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        ...

    def _any_synonym_present(self, tokens: List[str], role: str) -> bool:
        syns = self.ROLE_SYNONYMS.get(role, [])
        return any(s in tokens for s in syns)

    def _resolve_present_token(self, role: str) -> Optional[str]:
        syns = self.ROLE_SYNONYMS.get(role, [])
        for s in syns:
            if s in self.vocab:
                return s
        return None

    def _id_for_any(self, names: List[str], *, default_to_unk: bool = False) -> int:
        for n in names:
            if n in self.vocab:
                return self.vocab[n]
        if default_to_unk:
            return self.unk_id()
        raise KeyError(f"None of the names {names} found in vocab specials={self.special_tokens}")

    def pad_id(self) -> int:
        return self._id_for_any(self.ROLE_SYNONYMS["PAD"])

    def unk_id(self) -> int:
        return self._id_for_any(self.ROLE_SYNONYMS["UNK"])

    def mask_id(self) -> int:
        try:
            return self._id_for_any(self.ROLE_SYNONYMS["MASK"])
        except KeyError:
            return self.unk_id()

    def cls_id(self) -> int:
        try:
            return self._id_for_any(self.ROLE_SYNONYMS["CLS"])
        except KeyError:
            return self.unk_id()

    def sep_id(self) -> int:
        try:
            return self._id_for_any(self.ROLE_SYNONYMS["SEP"])
        except KeyError:
            return self.unk_id()

    def special_id(self, role_or_token: str) -> int:
        role = role_or_token.upper()
        if role in self.ROLE_SYNONYMS:
            return self._id_for_any(self.ROLE_SYNONYMS[role], default_to_unk=False)

        return self.vocab.get(role_or_token, self.unk_id())

    def freeze(self) -> None:
        self._frozen = True

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def effective_vocab_size(self) -> int:
        return int(len(self.vocab))

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

            loaded_vocab = {tok: int(idx) for tok, idx in data.get("vocab", {}).items()}
            loaded_specials = list(data.get("special_tokens", []))

            def _ensure_role(specials: List[str], role: str) -> None:
                if not any(t in specials for t in self.ROLE_SYNONYMS[role]):
                    specials.append(self.ROLE_SYNONYMS[role][0])

            if not loaded_specials:
                specials = []
                for role in ("PAD", "UNK", "MASK", "CLS", "SEP"):
                    for s in self.ROLE_SYNONYMS[role]:
                        if s in loaded_vocab:
                            specials.append(s)
                            break
    
                _ensure_role(specials, "PAD")
                _ensure_role(specials, "UNK")
            else:
                specials = []
                seen = set()
                for t in loaded_specials:
                    if t not in seen:
                        specials.append(t); seen.add(t)
                _ensure_role(specials, "PAD")
                _ensure_role(specials, "UNK")

            self.special_tokens = specials
            self.vocab = dict(loaded_vocab)  

            for tok in self.special_tokens:
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)

            self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}

            self.special_tokens_map = {}
            for role in self.ROLE_SYNONYMS.keys():
                tok = self._resolve_present_token(role)
                if tok is not None:
                    self.special_tokens_map[role] = tok

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
        specials = config.get("special_tokens", [])
        return cls(special_tokens=specials)
