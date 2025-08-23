# Main.py  
from __future__ import annotations

import math
import inspect
import random
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import config as cfg

from tokenizers.tokenizer_factory import create_tokenizer
from dataset_loader import SelfSupDataset, SupervisedMotifDataset
from collate_functions import make_selfsup_collate, make_supervised_collate

from model import (
    MultiTaskPretrainingModel,
    SimpleEncoder,
    DualStreamEncoder,
    MLMHead,
    MaskedMotifModelingHead,
    MotifAnnotatedPretrainingHead,
    ContrastiveMotifLearningHead,
)
from train_eval import train, evaluate, supervised_contrastive_loss

from partial_datasets import PartialViewDataset
from pretraining_supervised import cml_augment_random_mask

def _seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _sample_indices(n: int, fraction: float, seed: int, strategy: str = "first_k") -> list[int]:
    if fraction >= 1.0 or n == 0:
        return list(range(n))
    k = max(1, int(math.ceil(n * max(fraction, 1e-9))))
    if strategy == "first_k":
        return list(range(k))
    rnd = random.Random(seed)
    return sorted(rnd.sample(range(n), k))

def _subsample_dataset(ds, fraction: float, seed: int, strategy: str = "first_k"):
    n = len(ds)
    if fraction >= 1.0 or n == 0:
        return ds
    k = max(1, int(math.ceil(n * max(fraction, 1e-9))))
    if strategy == "first_k":
        idx = list(range(k))
    else:
        rnd = random.Random(seed)
        idx = sorted(rnd.sample(range(n), k))
    from torch.utils.data import Subset
    return Subset(ds, idx)

def _probe_first_batch(loader):
    if not cfg.PROBE_FIRST_BATCH:
        return None
    print("Sanity-check")
    batch = next(iter(loader))
    if "input_ids" in batch:
        masked = int((batch.get("labels", torch.zeros(1)) != -100).sum()) if "labels" in batch else 0
        print("single-stream batch", batch["input_ids"].shape, "masked:", masked)
    elif "kmer_input_ids" in batch:
        print("dual-stream batch")
        for k in ("kmer_input_ids", "bpe_input_ids", "labels_kmer", "labels_bpe"):
            if k in batch and isinstance(batch[k], torch.Tensor):
                print(" ", k, batch[k].shape)
        if "labels_kmer" in batch:
            print(" masked kmer:", int((batch["labels_kmer"] != -100).sum()))
        if "labels_bpe" in batch:
            print(" masked bpe:", int((batch["labels_bpe"] != -100).sum()))
    else:
        print("sequence-classification batch keys:", list(batch.keys()))
    return batch

def _save_tokenizer(tokenizer):
    try:
        tok_type = getattr(cfg, "TOKENIZER_TYPE", "kmer").lower()
        if tok_type == "bpe":
            path = cfg.TOKENIZER_BPE_PATH
        elif tok_type == "hybrid":
            path = cfg.TOKENIZER_HYBRID_PATH
        else:
            path = cfg.TOKENIZER_KMER_PATH
        tokenizer.save_vocab(path)
        print(f"Saved tokenizer vocab to {path}")
    except Exception as e:
        print("Warning: failed to save tokenizer vocab:", e)

def _effective_vocab_size_for_token_fusion(tokenizer) -> int:
    if hasattr(tokenizer, "effective_vocab_size"):
        try:
            return int(tokenizer.effective_vocab_size()) 
        except Exception:
            pass

    kmer_vs = len(getattr(tokenizer, "kmer_tokenizer", type("x", (), {"vocab": {}})).vocab)
    try:
        kmer_vs = int(getattr(tokenizer, "kmer_vocab_size", kmer_vs) or kmer_vs)
    except Exception:
        pass
    bpe_vs = len(getattr(tokenizer, "bpe_tokenizer", type("x", (), {"vocab": {}})).vocab)
    return max(int(kmer_vs) + int(bpe_vs), 0)

def _dedupe_params(params_iterable):
    seen = set()
    out = []
    for p in params_iterable:
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out

def _build_model_and_head(tokenizer, device: torch.device):
    tok_type = getattr(cfg, "TOKENIZER_TYPE", "kmer").lower()
    hybrid_mode = getattr(cfg, "HYBRID_MODE", "dual").lower()
    is_dual = (tok_type == "hybrid" and hybrid_mode == "dual")

    if is_dual:
        print("Using DualStreamEncoder (hybrid dual-stream).")
        kmer_vocab = len(tokenizer.kmer_tokenizer.vocab)
        bpe_vocab = len(tokenizer.bpe_tokenizer.vocab)

        dual_encoder = DualStreamEncoder(
            kmer_vocab_size=kmer_vocab,
            bpe_vocab_size=bpe_vocab,
            fusion=getattr(cfg, "DUAL_FUSION", "sum"),
            embed_dim=cfg.EMBED_DIM,
        ).to(device)

        stream = getattr(cfg, "SELF_SUP_STREAM", "kmer").lower()
        if stream not in ("kmer", "bpe"):
            raise ValueError("cfg.SELF_SUP_STREAM must be 'kmer' or 'bpe' for dual-stream training.")
        vocab_for_head = kmer_vocab if stream == "kmer" else bpe_vocab

        head = MLMHead(embedding_dim=cfg.EMBED_DIM, vocab_size=vocab_for_head).to(device)

        class ModelWrap(torch.nn.Module):
            def __init__(self, enc):
                super().__init__()
                self.encoder = enc

        model = ModelWrap(dual_encoder).to(device)
        params = _dedupe_params([p for p in model.parameters() if p.requires_grad] +
                                [p for p in head.parameters() if p.requires_grad])
        metrics_vocab = vocab_for_head
        return model, head, params, True, metrics_vocab

    if tok_type == "hybrid" and hybrid_mode == "token":
        vocab_for_head = _effective_vocab_size_for_token_fusion(tokenizer)
    else:
        vocab_for_head = len(tokenizer.vocab)

    try:
        model = MultiTaskPretrainingModel(
            vocab_size=vocab_for_head,
            motif_vocab_size=cfg.MOTIF_VOCAB_SIZE,
            dual_stream=False,
        ).to(device)
        head = model.mlm_head
        print("MultiTaskPretrainingModel.")
        params = _dedupe_params([p for p in model.parameters() if p.requires_grad])
    except Exception:
        enc = SimpleEncoder(vocab_size=vocab_for_head, motif_vocab_size=cfg.MOTIF_VOCAB_SIZE).to(device)
        head = MLMHead(enc.embedding.embedding_dim, vocab_for_head).to(device)

        class ModelWrap(torch.nn.Module):
            def __init__(self, enc):
                super().__init__()
                self.encoder = enc

        model = ModelWrap(enc)
        print("Fallback: SimpleEncoder + MLMHead.")
        params = _dedupe_params([p for p in model.parameters() if p.requires_grad] +
                                [p for p in head.parameters() if p.requires_grad])

    metrics_vocab = vocab_for_head
    return model, head, params, False, metrics_vocab

def _load_texts(files: list[str]) -> list[str]:
    ds = SelfSupDataset(files=files)
    return [ds[i]["seq"] for i in range(len(ds))]

def _pretty_print_metrics(tag: str, metrics: dict):
    top5 = metrics.get("task_top5", {})
    def _r(v):
        try:
            return round(float(v), 4)
        except Exception:
            return v
    compact = {k: _r(v) for k, v in top5.items()}
    print(f"{tag} {compact}")

def _train_with_signature_guard(*args, **kwargs):
    sig = inspect.signature(train)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    if "print_batch_metrics" in allowed:
        filtered["print_batch_metrics"] = False
    return train(*args, **filtered)

def _coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def _encode_any(
    model,
    x,
    attention_mask=None,
    motif_flags=None,
    position_ids=None,
    *,
    bpe_input_ids=None,
    bpe_attention_mask=None,
    bpe_position_ids=None,
):
 
    enc = getattr(model, "encoder", model)

    if isinstance(x, dict):
        k_ids = x.get("kmer_input_ids", x.get("input_ids"))
        b_ids = _coalesce(x.get("bpe_input_ids"), bpe_input_ids)
        k_am  = x.get("kmer_attention_mask", x.get("attention_mask"))
        b_am  = _coalesce(x.get("bpe_attention_mask"), bpe_attention_mask)
        k_pos = x.get("kmer_position_ids", x.get("position_ids"))
        b_pos = _coalesce(x.get("bpe_position_ids"), bpe_position_ids)
    else:
        k_ids, k_am, k_pos = x, attention_mask, position_ids
        b_ids, b_am, b_pos = bpe_input_ids, bpe_attention_mask, bpe_position_ids

    if isinstance(enc, DualStreamEncoder):
        if b_ids is None:
            if k_ids is None:
                raise ValueError("DualStreamEncoder requires at least k-mer input ids.")
            b_ids = torch.full_like(k_ids, fill_value=cfg.PAD_TOKEN_ID)
            b_am = torch.zeros_like(k_ids)
            b_pos = None
        return enc(
            kmer_input_ids=k_ids,
            bpe_input_ids=b_ids,
            kmer_attention_mask=k_am,
            bpe_attention_mask=b_am,
            kmer_position_ids=k_pos,
            bpe_position_ids=b_pos,
            motif_flags=motif_flags,
        )

    return enc(k_ids, attention_mask=k_am, motif_flags=motif_flags, position_ids=k_pos)

def _make_seqcls_collate(tokenizer, label_key: str = "global_label"):
    pad_id = tokenizer.pad_id()
    is_dual = hasattr(tokenizer, "encode_for_embedding")
    mask_id = tokenizer.vocab.get(cfg.MASK_TOKEN, pad_id)

    def collate(batch):
        import torch
        seqs = [b["seq"] for b in batch]
        labels = torch.tensor([int(b[label_key]) for b in batch], dtype=torch.long)

        if not is_dual:
            ids = [tokenizer.encode(s) for s in seqs]
            from tokenizers.utils import batchify
            out = batchify(ids, pad_id=pad_id, max_len=getattr(cfg, "MAX_LEN", None), return_positions=True)
            base_ids = out["input_ids"]
        else:
            enc = [tokenizer.encode_for_embedding(s) for s in seqs]
            out = tokenizer.collate_batch_for_embedding(
                enc, max_len_k=getattr(cfg, "MAX_LEN", None), max_len_b=getattr(cfg, "MAX_LEN", None), return_positions=True
            )
            base_ids = out["kmer_input_ids"]

        view1 = cml_augment_random_mask(base_ids, None, mask_token_id=mask_id, pad_token_id=pad_id)
        view2 = cml_augment_random_mask(base_ids, None, mask_token_id=mask_id, pad_token_id=pad_id)

        out["labels"] = labels
        out["input_ids_view1"] = view1
        out["input_ids_view2"] = view2
        out["attention_mask_view1"] = (view1 != pad_id).long()
        out["attention_mask_view2"] = (view2 != pad_id).long()
        return out

    return collate

def main():
    _seed_all(cfg.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = create_tokenizer(cfg)

    ds_all_1 = SelfSupDataset(files=cfg.DATA_FILES)
    tok_idx = _sample_indices(len(ds_all_1), getattr(cfg, "DATASET_FRACTION", 1.0), cfg.RANDOM_SEED, "first_k")
    raw_corpus = [ds_all_1[i]["seq"][: cfg.MAX_LEN] for i in tok_idx]
    print(f"Tokenizer corpus: {len(tok_idx)} seqs (fraction={getattr(cfg, 'DATASET_FRACTION', 1.0)})")
    tokenizer.train(raw_corpus)
    print(f"Tokenizer trained. Vocab size: {len(tokenizer.vocab)}")

    tok_type = getattr(cfg, "TOKENIZER_TYPE", "kmer").lower()
    hybrid_mode = getattr(cfg, "HYBRID_MODE", "dual").lower()
    if tok_type == "hybrid" and hybrid_mode == "token":
        fused = _effective_vocab_size_for_token_fusion(tokenizer)
        km = len(tokenizer.kmer_tokenizer.vocab)
        bp = len(tokenizer.bpe_tokenizer.vocab)
        print(f"[hybrid-token] kmer_vocab={km}  bpe_vocab={bp}  fused_vocab={fused}")
        if fused <= 5:
            raise RuntimeError(
                "Hybrid token-fusion fused vocab did not grow (>5). "
                "Increase DATASET_FRACTION / adjust KMER_SIZE / increase BPE VOCAB_SIZE."
            )

    selected_task = getattr(cfg, "PRETRAIN_TASK", "mlm").lower()
    if selected_task in ("mlm", "span", "dae"):
        assert cfg.MASK_TOKEN in tokenizer.vocab, f"{cfg.MASK_TOKEN} not in tokenizer.vocab; add to special tokens."

    model, head, params, is_dual, metrics_vocab = _build_model_and_head(tokenizer, device)
    optimizer = torch.optim.AdamW(params, lr=cfg.LEARNING_RATE, weight_decay=1e-2)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    train_fraction = getattr(cfg, "TRAINSET_FRACTION", getattr(cfg, "DATASET_FRACTION", 1.0))

    pin = torch.cuda.is_available()

    print("\n====== STAGE 1: SELF-SUPERVISED LEARNING ON COMPLETE SEQUENCES OF RNA VIRUSES ======")
    full_ds = SelfSupDataset(files=cfg.DATA_FILES)
    if len(full_ds) == 0:
        print(f"No self-supervised data found in {cfg.DATA_FILES}. Exiting.")
        return

    full_ds_sub = _subsample_dataset(full_ds, train_fraction, cfg.RANDOM_SEED, "first_k")
    if len(full_ds_sub) > 1:
        indices = list(range(len(full_ds_sub)))
        tr_idx, va_idx = train_test_split(indices, test_size=0.3, random_state=cfg.RANDOM_SEED)
        ss_train = torch.utils.data.Subset(full_ds_sub, tr_idx)
        ss_val = torch.utils.data.Subset(full_ds_sub, va_idx)
    else:
        ss_train = full_ds_sub
        ss_val = full_ds_sub

    collate = make_selfsup_collate(tokenizer, task=selected_task)
    ss_train_loader = DataLoader(ss_train, batch_size=cfg.BATCH_SIZE, shuffle=True,  collate_fn=collate, num_workers=0, pin_memory=pin)
    ss_val_loader   = DataLoader(ss_val,   batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=pin)

    _probe_first_batch(ss_train_loader)
    for epoch in range(cfg.EPOCHS):
        trm = _train_with_signature_guard(
            model, head, ss_train_loader, optimizer, criterion, device,
            vocab_size=metrics_vocab, max_seq_len=cfg.MAX_LEN,
            ambiguous_token_id=tokenizer.vocab.get("N", None),
            kmer_size=cfg.KMER_SIZE, dual_stream=is_dual,
            collect_selfsup_extras=True, task=selected_task,
        )
        vam = evaluate(
            model, head, ss_val_loader, criterion, device,
            vocab_size=metrics_vocab, max_seq_len=cfg.MAX_LEN,
            ambiguous_token_id=tokenizer.vocab.get("N", None),
            kmer_size=cfg.KMER_SIZE, dual_stream=is_dual,
            collect_selfsup_extras=True, task=selected_task,
        )
        print(f"[Stage1/{selected_task}] Epoch {epoch + 1}/{cfg.EPOCHS}")
        _pretty_print_metrics("  Train:", trm)
        _pretty_print_metrics("  Valid:", vam)
        torch.nn.utils.clip_grad_norm_([p for p in params if p.requires_grad], max_norm=cfg.GRAD_CLIP)

    try:
        torch.save(model.state_dict(), cfg.CKPT_STAGE1A)
        print(f"Saved {cfg.CKPT_STAGE1A}")
    except Exception as e:
        print("Warning: could not save stage 1 checkpoint:", e)

    _save_tokenizer(tokenizer)


    if cfg.PARTIAL_ENABLE:
        print("\n======= STAGE 2: SELF-SUPERVISED LEARNING ON PARTIAL SEQUENCES OF RNA VIRUSES =======")
        if cfg.PARTIAL_DATA_FILES and len(cfg.PARTIAL_DATA_FILES) > 0:
            partial_texts = _load_texts(cfg.PARTIAL_DATA_FILES)
            pv_items = [{"seq": s} for s in partial_texts]
            print(f"Loaded {len(pv_items)} partial sequences from files: {cfg.PARTIAL_DATA_FILES}")
        else:
            full_texts = [full_ds[i]["seq"] for i in range(len(full_ds))]
            pv_ds = PartialViewDataset(
                base_sequences=full_texts,
                window_size=cfg.PARTIAL_WINDOW_SIZE,
                windows_per_seq=cfg.PARTIAL_WINDOWS_PER_SEQ,
                overlap=cfg.PARTIAL_OVERLAP,
                strategy=cfg.PARTIAL_STRATEGY,
                seed=cfg.RANDOM_SEED,
            )
            pv_items = [pv_ds[i] for i in range(len(pv_ds))]
            print(f"Cropped {len(pv_items)} partial windows from Stage 1 data.")

        if 0 < train_fraction < 1.0:
            k = max(1, int(len(pv_items) * train_fraction))
            pv_items = pv_items[:k]

        class _PVShim(torch.utils.data.Dataset):
            def __init__(self, items): self.items = items
            def __len__(self): return len(self.items)
            def __getitem__(self, i): return self.items[i]

        coll = make_selfsup_collate(tokenizer, task=selected_task)
        pv_loader = DataLoader(_PVShim(pv_items), batch_size=cfg.BATCH_SIZE, shuffle=True,
                               collate_fn=coll, num_workers=0, pin_memory=pin)
        for ep in range(cfg.EPOCHS_PARTIAL):
            trm = _train_with_signature_guard(
                model, head, pv_loader, optimizer, criterion, device,
                vocab_size=metrics_vocab, max_seq_len=cfg.PARTIAL_WINDOW_SIZE,
                ambiguous_token_id=tokenizer.vocab.get("N", None),
                kmer_size=cfg.KMER_SIZE, dual_stream=is_dual,
                collect_selfsup_extras=True, task=selected_task,
            )
            _pretty_print_metrics(f"[Stage2/{selected_task}] Epoch {ep+1}/{cfg.EPOCHS_PARTIAL} Train:", trm)

        try:
            torch.save(model.state_dict(), cfg.CKPT_STAGE1B)
            print(f"Saved {cfg.CKPT_STAGE1B}")
        except Exception as e:
            print("Warning: could not save stage 2 checkpoint:", e)

   
    print("\n====== STAGE 3: SELF-SUPERVISED LEARNING ON COMPLETE RIBOZYME SEQUENCES ======")
    ss3_full = SelfSupDataset(files=cfg.STAGE2_RIBOZYME_FULL_FILES)
    if len(ss3_full) > 0:
        ss3_sub = _subsample_dataset(ss3_full, train_fraction, cfg.RANDOM_SEED, "first_k")
        if len(ss3_sub) > 1:
            idx = list(range(len(ss3_sub)))
            tr_idx, va_idx = train_test_split(idx, test_size=0.3, random_state=cfg.RANDOM_SEED)
            ds_tr, ds_va = torch.utils.data.Subset(ss3_sub, tr_idx), torch.utils.data.Subset(ss3_sub, va_idx)
        else:
            ds_tr, ds_va = ss3_sub, ss3_sub

        coll = make_selfsup_collate(tokenizer, task=selected_task)
        tr_loader = DataLoader(ds_tr, batch_size=cfg.BATCH_SIZE, shuffle=True,  collate_fn=coll, num_workers=0, pin_memory=pin)
        va_loader = DataLoader(ds_va, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=coll, num_workers=0, pin_memory=pin)

        for ep in range(cfg.EPOCHS_STAGE2):
            trm = _train_with_signature_guard(
                model, head, tr_loader, optimizer, criterion, device,
                vocab_size=metrics_vocab, max_seq_len=cfg.MAX_LEN,
                ambiguous_token_id=tokenizer.vocab.get("N", None),
                kmer_size=cfg.KMER_SIZE, dual_stream=is_dual,
                collect_selfsup_extras=True, task=selected_task,
            )
            vam = evaluate(
                model, head, va_loader, criterion, device,
                vocab_size=metrics_vocab, max_seq_len=cfg.MAX_LEN,
                ambiguous_token_id=tokenizer.vocab.get("N", None),
                kmer_size=cfg.KMER_SIZE, dual_stream=is_dual,
                collect_selfsup_extras=True, task=selected_task,
            )
            _pretty_print_metrics(f"[Stage3/{selected_task}] Epoch {ep+1}/{cfg.EPOCHS_STAGE2} Train:", trm)
            _pretty_print_metrics("  Valid:", vam)

        try:
            torch.save(model.state_dict(), cfg.CKPT_STAGE2)
            print(f"Saved {cfg.CKPT_STAGE2}")
        except Exception as e:
            print("Warning: could not save stage3 checkpoint:", e)
    else:
        print("Stage 3 dataset empty  skipped.")

    
    print("\n====== STAGE 4: SELF-SUPERVISED LEARNING ON PARTIAL RIBOZYME SEQUENCES ======")
    if cfg.STAGE3_RIBOZYME_PARTIAL_FILES:
        ss4 = SelfSupDataset(files=cfg.STAGE3_RIBOZYME_PARTIAL_FILES)
        ss4_sub = _subsample_dataset(ss4, train_fraction, cfg.RANDOM_SEED, "first_k")
        pv_items = [{"seq": ss4_sub[i]["seq"]} for i in range(len(ss4_sub))]
    else:
        full_texts = _load_texts(cfg.STAGE2_RIBOZYME_FULL_FILES)
        pv_ds = PartialViewDataset(
            base_sequences=full_texts,
            window_size=cfg.PARTIAL_WINDOW_SIZE,
            windows_per_seq=cfg.PARTIAL_WINDOWS_PER_SEQ,
            overlap=cfg.PARTIAL_OVERLAP,
            strategy=cfg.PARTIAL_STRATEGY,
            seed=cfg.RANDOM_SEED,
        )
        pv_items = [pv_ds[i] for i in range(len(pv_ds))]

    if 0 < train_fraction < 1.0:
        k = max(1, int(len(pv_items) * train_fraction))
        pv_items = pv_items[:k]

    class _PVShim2(torch.utils.data.Dataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]

    coll = make_selfsup_collate(tokenizer, task=selected_task)
    pv_loader = DataLoader(_PVShim2(pv_items), batch_size=cfg.BATCH_SIZE, shuffle=True,
                           collate_fn=coll, num_workers=0, pin_memory=pin)
    for ep in range(cfg.EPOCHS_STAGE3):
        trm = _train_with_signature_guard(
            model, head, pv_loader, optimizer, criterion, device,
            vocab_size=metrics_vocab, max_seq_len=cfg.PARTIAL_WINDOW_SIZE,
            ambiguous_token_id=tokenizer.vocab.get("N", None),
            kmer_size=cfg.KMER_SIZE, dual_stream=is_dual,
            collect_selfsup_extras=True, task=selected_task,
        )
        _pretty_print_metrics(f"[Stage4/{selected_task}] Epoch {ep+1}/{cfg.EPOCHS_STAGE3} Train:", trm)

    try:
        torch.save(model.state_dict(), cfg.CKPT_STAGE3)
        print(f"Saved {cfg.CKPT_STAGE3}")
    except Exception as e:
        print("Warning: could not save stage 4 checkpoint:", e)

    if cfg.STAGE4_WEAK_SEQCLS and len(cfg.STAGE4_WEAK_SEQCLS) > 0:
        print("\n====== STAGE 5:SUPERVISED LEARNING ON LABELED PARTIAL RNA VIRUS SEQUENCES  =====")
        ds5 = SupervisedMotifDataset(txt_files=cfg.STAGE4_WEAK_SEQCLS, txt_schema="seq_label")
        ds5 = _subsample_dataset(ds5, train_fraction, cfg.RANDOM_SEED, "first_k")
        coll5 = _make_seqcls_collate(tokenizer, label_key=cfg.MAP_LABEL_KEY)
        loader5 = DataLoader(ds5, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=coll5,
                             num_workers=0, pin_memory=pin)

        head5 = MotifAnnotatedPretrainingHead(embedding_dim=cfg.EMBED_DIM, output_dim=cfg.MAP_NUM_CLASSES_COARSE).to(device)
        proj5 = getattr(model, "cml_head", ContrastiveMotifLearningHead().to(device))

        opt5 = torch.optim.AdamW(
            _dedupe_params(list(model.parameters()) + list(head5.parameters()) + list(proj5.parameters())),
            lr=cfg.LEARNING_RATE * cfg.LR_MULT_STAGE4, weight_decay=1e-2
        )

        for ep in range(cfg.EPOCHS_STAGE4):
            model.train(); head5.train(); proj5.train()
            for batch in loader5:
                xk = _coalesce(batch.get("kmer_input_ids"), batch.get("input_ids")).to(device)
                xb = batch.get("bpe_input_ids")
                xb = xb.to(device) if xb is not None else None
                amk = _coalesce(batch.get("kmer_attention_mask"), batch.get("attention_mask"))
                amb = batch.get("bpe_attention_mask", None)
                if amk is not None: amk = amk.to(device)
                if amb is not None: amb = amb.to(device)
                y = batch["labels"].to(device)

                enc = _encode_any(
                    model, xk, attention_mask=amk,
                    bpe_input_ids=xb, bpe_attention_mask=amb,
                )
                logits = head5(enc)
                loss_ce = torch.nn.functional.cross_entropy(logits, y)

                v1 = batch["input_ids_view1"].to(device)
                v2 = batch["input_ids_view2"].to(device)
                am1 = batch["attention_mask_view1"].to(device)
                am2 = batch["attention_mask_view2"].to(device)

                z1 = proj5(_encode_any(
                    model, v1, attention_mask=am1,
                    bpe_input_ids=xb, bpe_attention_mask=amb,
                ))
                z2 = proj5(_encode_any(
                    model, v2, attention_mask=am2,
                    bpe_input_ids=xb, bpe_attention_mask=amb,
                ))

                z = torch.cat([z1, z2], dim=0)
                y2 = torch.cat([y, y], dim=0)
                loss_supcon = supervised_contrastive_loss(z, y2, temperature=cfg.SUPCON_TEMPERATURE)

                loss = cfg.CE_WEIGHT * loss_ce + cfg.SUPCON_WEIGHT * loss_supcon

                opt5.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt5.step()

            print(f"[Stage5] epoch {ep+1}: CE={loss_ce.item():.4f}  SupCon={loss_supcon.item():.4f}  Total={loss.item():.4f}")

        try:
            torch.save(model.state_dict(), cfg.CKPT_STAGE4)
            print(f"Saved {cfg.CKPT_STAGE4}")
        except Exception as e:
            print("Warning: could not save stage 5 checkpoint:", e)

    if cfg.STAGE5_WEAK_RIBOZYME_SEQCLS and len(cfg.STAGE5_WEAK_RIBOZYME_SEQCLS) > 0:
        print("\n====== STAGE 6: SUPERVISED LEARNING ON LABELED PARTIAL RIBOZYME SEQUENCES ======")
        ds6w = SupervisedMotifDataset(txt_files=cfg.STAGE5_WEAK_RIBOZYME_SEQCLS, txt_schema="seq_label")
        ds6w = _subsample_dataset(ds6w, train_fraction, cfg.RANDOM_SEED, "first_k")
        coll6w = _make_seqcls_collate(tokenizer, label_key=cfg.MAP_LABEL_KEY)
        loader6w = DataLoader(ds6w, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=coll6w,
                              num_workers=0, pin_memory=pin)

        head6w = MotifAnnotatedPretrainingHead(embedding_dim=cfg.EMBED_DIM, output_dim=cfg.MAP_NUM_CLASSES_RIBOZYME).to(device)
        proj6w = getattr(model, "cml_head", ContrastiveMotifLearningHead().to(device))

        opt6w = torch.optim.AdamW(
            _dedupe_params(list(model.parameters()) + list(head6w.parameters()) + list(proj6w.parameters())),
            lr=cfg.LEARNING_RATE * cfg.LR_MULT_STAGE5, weight_decay=1e-2
        )

        for ep in range(cfg.EPOCHS_STAGE5):
            model.train(); head6w.train(); proj6w.train()
            for batch in loader6w:
                xk = _coalesce(batch.get("kmer_input_ids"), batch.get("input_ids")).to(device)
                xb = batch.get("bpe_input_ids")
                xb = xb.to(device) if xb is not None else None
                amk = _coalesce(batch.get("kmer_attention_mask"), batch.get("attention_mask"))
                amb = batch.get("bpe_attention_mask", None)
                if amk is not None: amk = amk.to(device)
                if amb is not None: amb = amb.to(device)
                y = batch["labels"].to(device)

                enc = _encode_any(
                    model, xk, attention_mask=amk,
                    bpe_input_ids=xb, bpe_attention_mask=amb,
                )
                logits = head6w(enc)
                loss_ce = torch.nn.functional.cross_entropy(logits, y)

                v1 = batch["input_ids_view1"].to(device)
                v2 = batch["input_ids_view2"].to(device)
                am1 = batch["attention_mask_view1"].to(device)
                am2 = batch["attention_mask_view2"].to(device)

                z1 = proj6w(_encode_any(
                    model, v1, attention_mask=am1,
                    bpe_input_ids=xb, bpe_attention_mask=amb,
                ))
                z2 = proj6w(_encode_any(
                    model, v2, attention_mask=am2,
                    bpe_input_ids=xb, bpe_attention_mask=amb,
                ))

                z = torch.cat([z1, z2], dim=0)
                y2 = torch.cat([y, y], dim=0)
                loss_supcon = supervised_contrastive_loss(z, y2, temperature=cfg.SUPCON_TEMPERATURE)

                loss = cfg.CE_WEIGHT * loss_ce + cfg.SUPCON_WEIGHT * loss_supcon

                opt6w.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt6w.step()

            print(f"[Stage6] epoch {ep+1}: CE={loss_ce.item():.4f}  SupCon={loss_supcon.item():.4f}  Total={loss.item():.4f}")

        try:
            torch.save(model.state_dict(), cfg.CKPT_STAGE5)
            print(f"Saved {cfg.CKPT_STAGE5}")
        except Exception as e:
            print("Warning: could not save stage 6 checkpoint:", e)


    if cfg.STAGE6_GOLD_MIXED and len(cfg.STAGE6_GOLD_MIXED) > 0:
        print("\n====== STAGE 7: MOTIF-AWARE SUPERVISED LEARNING ON RNA VIRUSES AND RIBOZYME SEQUENCES ======")
        ds7 = SupervisedMotifDataset(jsonl_files=cfg.STAGE6_GOLD_MIXED)
        ds7 = _subsample_dataset(ds7, train_fraction, cfg.RANDOM_SEED, "first_k")

        for task in cfg.STAGE6_TASKS:
            print(f"[7] task = {task}")
            if task in ("masked_motif", "mbp"):
                coll7 = make_supervised_collate(tokenizer, task_name=task, augment="mask_non_motif")
                loader7 = DataLoader(ds7, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=coll7,
                                     num_workers=0, pin_memory=pin)
                if task == "masked_motif":
                    sup_head = model.masked_motif_head if hasattr(model, "masked_motif_head") \
                               else MaskedMotifModelingHead(cfg.EMBED_DIM, cfg.MOTIF_VOCAB_SIZE).to(device)
                    sup_crit = torch.nn.CrossEntropyLoss(ignore_index=-100)
                    vsz = cfg.MOTIF_VOCAB_SIZE
                else: 
                    from model import MotifBoundaryPredictionHead
                    sup_head = model.mbp_head if hasattr(model, "mbp_head") \
                               else MotifBoundaryPredictionHead(cfg.EMBED_DIM).to(device)
                    sup_crit = torch.nn.CrossEntropyLoss()
                    vsz = 2

                sup_opt = torch.optim.AdamW(
                    _dedupe_params(list(model.parameters()) + list(sup_head.parameters())),
                    lr=cfg.LEARNING_RATE * cfg.LR_MULT_STAGE6, weight_decay=1e-2,
                )
                for ep in range(cfg.EPOCHS_STAGE6):
                    tr_metrics = train(
                        model, sup_head, loader7, sup_opt, sup_crit, device,
                        vocab_size=vsz, max_seq_len=cfg.MAX_LEN,
                        ambiguous_token_id=tokenizer.vocab.get("N", None),
                        kmer_size=cfg.KMER_SIZE, dual_stream=False,
                        collect_selfsup_extras=True, task=task,
                    )
                    _pretty_print_metrics(f"[Stage7/{task}] epoch {ep+1}:", tr_metrics)

            elif task == "map":
                coll7 = make_supervised_collate(tokenizer, task_name="map", augment="mask_non_motif")
                loader7 = DataLoader(ds7, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=coll7,
                                     num_workers=0, pin_memory=pin)

                sup_head = None
                sup_crit = None
                vsz = None
                sup_opt = None

                for ep in range(cfg.EPOCHS_STAGE6):
                    model.train()
                    epoch_metrics = None

                    for batch in loader7:
                        labels = batch["labels"]
                        is_token_level = (labels.dim() == 2)

                        if sup_head is None:
                            if is_token_level:
                                vsz = int(getattr(cfg, "MOTIF_VOCAB_SIZE", 2))
                                sup_head = (getattr(model, "masked_motif_head", None)
                                            or MaskedMotifModelingHead(cfg.EMBED_DIM, vsz).to(device))
                                sup_crit = torch.nn.CrossEntropyLoss(ignore_index=-100)
                                print("[Stage7/map] Detected token-level labels -> using token-level MAP head.")
                            else:
                                vsz = int(getattr(cfg, "MAP_NUM_CLASSES_COARSE", 2))
                                sup_head = (getattr(model, "map_head", None)
                                            or MotifAnnotatedPretrainingHead(cfg.EMBED_DIM, output_dim=vsz).to(device))
                                sup_crit = torch.nn.CrossEntropyLoss()
                                print("[Stage7/map] Detected sequence-level labels -> using sequence-level MAP head.")

                            sup_opt = torch.optim.AdamW(
                                _dedupe_params(list(model.parameters()) + list(sup_head.parameters())),
                                lr=cfg.LEARNING_RATE * cfg.LR_MULT_STAGE6, weight_decay=1e-2,
                            )

                        metrics = train(
                            model, sup_head, [(batch)], sup_opt, sup_crit, device,
                            vocab_size=vsz, max_seq_len=cfg.MAX_LEN,
                            ambiguous_token_id=tokenizer.vocab.get("N", None),
                            kmer_size=cfg.KMER_SIZE, dual_stream=False,
                            collect_selfsup_extras=True, task=("masked_motif" if is_token_level else "map"),
                        )
                        epoch_metrics = metrics

                    if epoch_metrics is not None:
                        _pretty_print_metrics(f"[Stage7/map] epoch {ep+1}:", epoch_metrics)

            elif task == "cml":
                coll_cml = make_supervised_collate(tokenizer, task_name="masked_motif", augment="mask_non_motif")
                loader_cml = DataLoader(ds7, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=coll_cml,
                                        num_workers=0, pin_memory=pin)
                cml_head = getattr(model, "cml_head", ContrastiveMotifLearningHead().to(device))
                cml_opt = torch.optim.AdamW(
                    _dedupe_params(list(model.parameters()) + list(cml_head.parameters())),
                    lr=cfg.LEARNING_RATE * cfg.LR_MULT_STAGE6, weight_decay=1e-2,
                )
                for ep in range(cfg.EPOCHS_STAGE6):
                    model.train(); cml_head.train()
                    epoch_loss = 0.0
                    epoch_nce = 0.0
                    tp = tn = fp = fn = 0
                    n_batches = 0

                    for batch in loader_cml:
                        x1 = _coalesce(batch.get("input_ids"), batch.get("kmer_input_ids")).to(device)
                        x2 = batch.get("input_ids_view2")
                        x2 = x2.to(device) if x2 is not None else x1.clone()

                        am1 = _coalesce(batch.get("attention_mask"), batch.get("kmer_attention_mask"))
                        am2 = batch.get("attention_mask_view2")
                        if am1 is not None: am1 = am1.to(device)
                        if am2 is not None: am2 = am2.to(device)

                        z1 = cml_head(_encode_any(model, x1, attention_mask=am1))
                        z2 = cml_head(_encode_any(model, x2, attention_mask=am2))

                        z1 = torch.nn.functional.normalize(z1, dim=-1)
                        z2 = torch.nn.functional.normalize(z2, dim=-1)
                        z = torch.cat([z1, z2], dim=0)
                        y = torch.arange(z1.size(0), device=z.device).repeat(2)

                        sim = torch.matmul(z, z.t()) / cfg.SUPCON_TEMPERATURE
                        N = z.size(0)
                        self_mask = torch.eye(N, dtype=torch.bool, device=z.device)
                        sim = sim.masked_fill(self_mask, -1e9)
                        pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & (~self_mask)
                        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
                        loss = -(log_prob[pos_mask]).mean()

                        nce_loss = loss

                        with torch.no_grad():
                            preds = sim.argmax(dim=1)
                            B = z1.size(0)
                            truth = torch.cat([torch.arange(B, 2*B, device=z.device), torch.arange(0, B, device=z.device)])
                            correct = (preds == truth)
                            tp += int(correct.sum().item())
                            fp += int((preds != truth).sum().item())

                        cml_opt.zero_grad(set_to_none=True); loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        cml_opt.step()

                        epoch_loss += float(loss.item())
                        epoch_nce  += float(nce_loss.item())
                        n_batches  += 1

                    contrastive_accuracy  = tp / max(tp + fp + tn + fn, 1)
                    contrastive_precision = tp / max(tp + fp, 1)
                    contrastive_recall    = tp / max(tp + fn, 1)

                    print(
                        f"[Stage7/cml] epoch {ep+1}: "
                        f"loss={(epoch_loss/max(n_batches,1)):.4f}  "
                        f"nce_loss={(epoch_nce/max(n_batches,1)):.4f}  "
                        f"contrastive_accuracy={contrastive_accuracy:.4f}  "
                        f"contrastive_precision={contrastive_precision:.4f}  "
                        f"contrastive_recall={contrastive_recall:.4f}"
                    )

        try:
            torch.save(model.state_dict(), cfg.CKPT_STAGE6)
            print(f"Saved {cfg.CKPT_STAGE6}")
        except Exception as e:
            print("Warning: could not save stage 7 checkpoint:", e)

    print("\nAll stages complete.")


if __name__ == "__main__":
    main()
