# dataset_loader
from __future__ import annotations

import json
from typing import List, Dict, Any, Optional, Iterable, Tuple
import gzip
import bz2
from torch.utils.data import Dataset


def _open_any(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    if "," in s and " " not in s:
        parts = s.split(",")
    else:
        parts = s.split()
    out: List[int] = []
    for x in parts:
        if x == "":
            continue
        out.append(int(x))
    return out


class SelfSupDataset(Dataset):
    def __init__(self, data: Iterable[str] | List[str] | None = None, files: Optional[List[str]] = None):
        self.samples: List[str] = []
        if data is not None:
            self.samples.extend([s.strip() for s in data if s and str(s).strip()])
        if files:
            for p in files:
                with _open_any(p) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.samples.append(line)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"seq": self.samples[idx]}


class SupervisedMotifDataset(Dataset):
    def __init__(
        self,
        jsonl_files: Optional[List[str]] = None,
        records: Optional[List[Dict[str, Any]]] = None,
        txt_files: Optional[List[str]] = None,
        txt_schema: str = "seq",
        txt_sep: Optional[str] = None,   
        map_label_key: str = "global_label",
        label_vocab: Optional[Dict[str, int]] = None,   
        auto_build_label_vocab: bool = True,          
    ):
        self.items: List[Dict[str, Any]] = []
        self.map_label_key: str = map_label_key

        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}

        if label_vocab is not None:
            for k, v in label_vocab.items():
                if not isinstance(v, int):
                    raise ValueError(f"label_vocab values must be ints, got {type(v)} for key {k!r}")
            self.label2id = dict(label_vocab)
            self.id2label = {v: k for k, v in self.label2id.items()}
            auto = False
        else:
            auto = auto_build_label_vocab

        def _maybe_int(s: Any) -> Tuple[bool, int]:
            """Try to parse strict int (not labels like 'ENV')."""
            if isinstance(s, int):
                return True, int(s)
            if isinstance(s, str):
                try:
                    return True, int(s)
                except ValueError:
                    return False, 0
            return False, 0

        def _label_to_id(lbl: Any) -> int:
            """Return an int id for lbl (int passthrough, string mapped/auto-built)."""
            ok, val = _maybe_int(lbl)
            if ok:
                return val

            if not isinstance(lbl, str):
                raise ValueError(f"Unsupported label type: {type(lbl)} (value={lbl!r})")

            if lbl in self.label2id:
                return self.label2id[lbl]

            if auto:
                new_id = 0 if len(self.label2id) == 0 else (max(self.label2id.values()) + 1)
                self.label2id[lbl] = new_id
                self.id2label[new_id] = lbl
                return new_id

            raise ValueError(f"Label {lbl!r} not found in label_vocab and auto_build_label_vocab=False")

        def _maybe_split_cols(line: str) -> List[str]:
            if txt_sep is None:
                return line.split("\t") if ("\t" in line) else line.split()
            return line.split(txt_sep)

        if records:
            for r in records:
                if "seq" in r and isinstance(r["seq"], str) and r["seq"]:
                    item = dict(r) 
                    if self.map_label_key in item and item[self.map_label_key] is not None:
                        item[self.map_label_key] = _label_to_id(item[self.map_label_key])
                    self.items.append(item)

        if jsonl_files:
            for p in jsonl_files:
                with _open_any(p) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        if "seq" not in obj or not isinstance(obj["seq"], str) or not obj["seq"]:
                            continue
                        item = dict(obj)
                        if self.map_label_key in item and item[self.map_label_key] is not None:
                            item[self.map_label_key] = _label_to_id(item[self.map_label_key])
                        self.items.append(item)

        
        if txt_files:
            parse = txt_schema.lower().strip()
            for p in txt_files:
                with _open_any(p) as f:
                    for raw in f:
                        line = raw.strip()
                        if not line:
                            continue
                        cols = _maybe_split_cols(line)
                        if not cols:
                            continue

                        if parse == "seq":
                            seq = cols[0]
                            if seq:
                                self.items.append({"seq": seq})

                        elif parse == "seq_label":
                            seq = cols[0]
                            if not seq:
                                continue
                            if len(cols) < 2 or cols[1] == "":
                                self.items.append({"seq": seq})
                            else:
                                raw_label = cols[1]
                                try:
                                    lbl_id = _label_to_id(raw_label)
                                except ValueError:
                                    raise
                                self.items.append({"seq": seq, self.map_label_key: lbl_id})

                        elif parse == "seq_token_arrays":
                            seq = cols[0]
                            if not seq:
                                continue
                            entry: Dict[str, Any] = {"seq": seq}
                            if len(cols) >= 2 and cols[1].strip():
                                entry["motif_flags"] = _parse_int_list(cols[1])
                            if len(cols) >= 3 and cols[2].strip():
                                entry["motif_labels"] = _parse_int_list(cols[2])
                            if len(cols) >= 4 and cols[3].strip():
                                entry["motif_boundaries"] = _parse_int_list(cols[3])
                            self.items.append(entry)

                        else:
                            raise ValueError(f"Unsupported txt_schema: {txt_schema!r}")

        if auto and self.label2id and not self.id2label:
            self.id2label = {v: k for k, v in self.label2id.items()}

        for it in self.items:
            if self.map_label_key in it and it[self.map_label_key] is not None:
                if not isinstance(it[self.map_label_key], int):
                    ok, val = _maybe_int(it[self.map_label_key])
                    if not ok:
                        raise ValueError(
                            f"Sample has non-integer label after mapping: {it[self.map_label_key]!r}"
                        )
                    it[self.map_label_key] = val

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]
