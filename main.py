# main
from __future__ import annotations

import math
import json
import time
import inspect
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split, StratifiedKFold

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


from train_eval import (
    train, evaluate,
    train_cls_ce_supcon_epoch, eval_cls_ce_supcon_epoch,
    train_cml_epoch, eval_cml_epoch,
    promotion_gate_satisfied,                    
    EMAWeights,                                   
    compute_class_weights,                       
)


AMP_ENABLE = bool(getattr(cfg, "MIXED_PRECISION", True))


AUX_TASK = str(getattr(cfg, "AUX_TASK", "mlm"))              
AUX_WEIGHT = float(getattr(cfg, "AUX_LOSS_WEIGHT", 0.1))     
REPLAY_FRACTION = float(getattr(cfg, "REPLAY_FRACTION", 0.01))  
UNFREEZE_TOP_N = int(getattr(cfg, "UNFREEZE_TOP_N", 0))   
USE_EMA = bool(getattr(cfg, "USE_EMA", True))
EMA_DECAY = float(getattr(cfg, "EMA_DECAY", 0.999))
WARMUP_STEPS = int(getattr(cfg, "WARMUP_STEPS", 100))
COSINE_TOTAL_STEPS = int(getattr(cfg, "COSINE_TOTAL_STEPS", 1000))
EARLY_STOP_PATIENCE = int(getattr(cfg, "EARLY_STOP_PATIENCE", 5))
SAVE_DIR = Path(getattr(cfg, "SAVE_DIR", "."))


STAGE_LOG_PATH = SAVE_DIR / str(getattr(cfg, "STAGE_LOG_PATH", "stage_gates.jsonl"))
RUN_CONFIG_SNAPSHOT = SAVE_DIR / str(getattr(cfg, "RUN_CONFIG_SNAPSHOT", "config_snapshot.json"))
TOKENIZER_META_PATH = SAVE_DIR / str(getattr(cfg, "TOKENIZER_META_PATH", "tokenizer_meta.json"))

def _seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _subsample(ds, fraction: float, seed: int, mode: str = "first_k"):
    n = len(ds)
    if fraction >= 1.0 or n <= 1:
        return ds
    k = max(1, int(math.ceil(n * max(fraction, 1e-9))))
    if mode == "first_k":
        idx = list(range(k))
    else:
        rnd = random.Random(seed)
        idx = sorted(rnd.sample(range(n), k))
    return Subset(ds, idx)

def _save_jsonl(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _save_json(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _snapshot_config():
    cfg_dict = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
    _save_json(RUN_CONFIG_SNAPSHOT, cfg_dict)

def _sha1_texts(texts: List[str]) -> str:
    h = hashlib.sha1()
    for s in texts[:1000]:  
        h.update(s.encode("utf-8"))
    return h.hexdigest()

def _build_optimizer(params, lr: float, weight_decay: float):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return opt

def _build_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def _freeze_encoder_layers(model, unfreeze_top_n: int = 0):

    enc = getattr(model, "encoder", model)
    for p in enc.parameters():
        p.requires_grad = False

    stack = getattr(enc, "encoder", None)
    if stack is None:
        return  
    layers = getattr(stack, "layers", None)
    if layers is None:
        return

    if unfreeze_top_n > 0:
        for p in enc.embedding.parameters():
            p.requires_grad = True  
        for layer in layers[-unfreeze_top_n:]:
            for p in layer.parameters():
                p.requires_grad = True

def _collect_trainable_params(*modules):
    seen, params = set(), []
    for m in modules:
        if m is None: 
            continue
        for p in m.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p)); params.append(p)
    return params

def _maybe_lock_tokenizer(tokenizer):
    if hasattr(tokenizer, "train"):
        tokenizer.train = lambda *args, **kwargs: None 
    return tokenizer

def _stage_gate_and_log(stage_name: str, prev_metrics: Dict[str, Any], cur_metrics: Dict[str, Any]) -> bool:
    ok, reasons = promotion_gate_satisfied(prev_metrics, cur_metrics)
    row = {
        "ts": time.time(),
        "stage": stage_name,
        "promote": bool(ok),
        "reasons": reasons,
        "prev_top5": prev_metrics.get("task_top5", {}),
        "cur_top5": cur_metrics.get("task_top5", {}),
    }
    _save_jsonl(STAGE_LOG_PATH, row)
    return ok

def _build_model_and_head_for_selfsup(tokenizer, device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module, int, bool, int]:
    tok_type = getattr(cfg, "TOKENIZER_TYPE", "kmer").lower()
    hybrid_mode = getattr(cfg, "HYBRID_MODE", "dual").lower()
    is_dual = (tok_type == "hybrid" and hybrid_mode == "dual")

    if is_dual:
        kmer_vocab = len(tokenizer.kmer_tokenizer.vocab)
        bpe_vocab = len(tokenizer.bpe_tokenizer.vocab)
        enc = DualStreamEncoder(
            kmer_vocab_size=kmer_vocab, bpe_vocab_size=bpe_vocab,
            fusion=getattr(cfg, "DUAL_FUSION", "sum"), embed_dim=cfg.EMBED_DIM
        ).to(device)
        
        stream = getattr(cfg, "SELF_SUP_STREAM", "kmer").lower()
        head_vocab = kmer_vocab if stream == "kmer" else bpe_vocab
        head = MLMHead(cfg.EMBED_DIM, head_vocab).to(device)
        class Wrap(torch.nn.Module):
            def __init__(self, e): super().__init__(); self.encoder = e
        model = Wrap(enc).to(device)
        metrics_vocab = head_vocab
        return model, head, metrics_vocab, True, head_vocab

    vocab_for_head = len(getattr(tokenizer, "vocab", {}))
    try:
        model = MultiTaskPretrainingModel(vocab_size=vocab_for_head, motif_vocab_size=cfg.MOTIF_VOCAB_SIZE).to(device)
        head = model.mlm_head
    except Exception:
        enc = SimpleEncoder(vocab_size=vocab_for_head, motif_vocab_size=cfg.MOTIF_VOCAB_SIZE).to(device)
        head = MLMHead(cfg.EMBED_DIM, vocab_for_head).to(device)
        class Wrap(torch.nn.Module):
            def __init__(self, e): super().__init__(); self.encoder = e
        model = Wrap(enc).to(device)
    return model, head, vocab_for_head, False, vocab_for_head

def _build_aux_head(model, aux_task: str, vocab_size: int) -> torch.nn.Module:
    aux = aux_task.lower()
    if aux == "mlm" or aux == "dae" or aux == "span" or aux == "kmer_reorder":
        return MLMHead(cfg.EMBED_DIM, vocab_size).to(next(model.parameters()).device)
    if aux == "masked_motif":
        return MaskedMotifModelingHead(cfg.EMBED_DIM, cfg.MOTIF_VOCAB_SIZE).to(next(model.parameters()).device)
    return MLMHead(cfg.EMBED_DIM, vocab_size).to(next(model.parameters()).device)

def _aux_step_selfsup(
    model, aux_head, aux_loader, criterion, device, vocab_size, kmer_size, ambiguous_token_id, task
) -> Dict[str, Any]:
    return train(
        model, aux_head, aux_loader,
        optimizer=torch.optim.SGD(_collect_trainable_params(model, aux_head), lr=1e-4),  
        criterion=criterion, device=device,
        vocab_size=vocab_size, max_seq_len=cfg.MAX_LEN,
        ambiguous_token_id=ambiguous_token_id, kmer_size=kmer_size,
        dual_stream=False, collect_selfsup_extras=True, task=task,
    )

def _build_replay_loader(
    tokenizer, stage1_files: List[str], batch_size: int, aux_task: str, fraction: float, pin: bool
) -> Optional[DataLoader]:
    if fraction <= 0.0:
        return None
    ds = SelfSupDataset(files=stage1_files)
    ds = _subsample(ds, fraction, cfg.RANDOM_SEED, "first_k")
    if len(ds) == 0:
        return None
    collate = make_selfsup_collate(tokenizer, task=aux_task)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0, pin_memory=pin)

def _linear_probe_eval(
    model, tokenizer, files: List[str], label_key: str, n_splits: int = 3
) -> Dict[str, Any]:
    try:
        ds = SupervisedMotifDataset(txt_files=files, txt_schema="seq_label", map_label_key=label_key)
        if len(ds) < 8:
            return {"linear_probe_top1": float("nan"), "linear_probe_macro_f1": float("nan")}
        device = next(model.parameters()).device
        encs, ys = [], []
        loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=make_supervised_collate(tokenizer, "map"))
        with torch.no_grad():
            for batch in loader:
                x = (batch.get("input_ids") or batch.get("kmer_input_ids")).to(device)
                am = (batch.get("attention_mask") or batch.get("kmer_attention_mask"))
                if am is not None: am = am.to(device)
                h = getattr(model, "encoder", model)(x, attention_mask=am) 
                cls = h[:, 0, :].detach().cpu().numpy()
                encs.append(cls); ys.append(batch["labels"].numpy())
        X = np.concatenate(encs); y = np.concatenate(ys)
        skf = StratifiedKFold(n_splits=min(n_splits, len(np.unique(y))))
        accs, f1s = [], []
        for tr, te in skf.split(X, y):
            means = {}
            for c in np.unique(y[tr]):
                means[c] = X[tr][y[tr] == c].mean(axis=0)
            def pred(vec):
                best, bestc = -1e9, None
                for c, mu in means.items():
                    s = np.dot(vec, mu) / (1e-8 + np.linalg.norm(vec) * np.linalg.norm(mu))
                    if s > best: best, bestc = s, c
                return bestc
            yp = np.array([pred(v) for v in X[te]])
            accs.append((yp == y[te]).mean())
            from sklearn.metrics import f1_score
            f1s.append(f1_score(y[te], yp, average="macro", zero_division=0))
        return {"linear_probe_top1": float(np.mean(accs)), "linear_probe_macro_f1": float(np.mean(f1s))}
    except Exception:
        return {"linear_probe_top1": float("nan"), "linear_probe_macro_f1": float("nan")}

def main():
    _seed_all(getattr(cfg, "RANDOM_SEED", 42))
    device = _device()
    pin = torch.cuda.is_available()

    _snapshot_config()

    stage1_all = SelfSupDataset(files=cfg.DATA_FILES)
    if len(stage1_all) == 0:
        print(f"[FATAL] No Stage-1 data found at {cfg.DATA_FILES}")
        return

    tok_fraction = float(getattr(cfg, "DATASET_FRACTION", 1.0))
    stage1_tok = _subsample(stage1_all, tok_fraction, cfg.RANDOM_SEED, "first_k")
    tok_corpus = [stage1_tok[i]["seq"][: cfg.MAX_LEN] for i in range(len(stage1_tok))]

    tokenizer = create_tokenizer(cfg)
    tokenizer.train(tok_corpus)
    vocab_sha = _sha1_texts(tok_corpus)
    _save_json(TOKENIZER_META_PATH, {
        "kmer_size": int(getattr(cfg, "KMER_SIZE", 3)),
        "vocab_size": int(getattr(cfg, "VOCAB_SIZE", 1000)),
        "merge_num": getattr(cfg, "MERGE_NUM", None),
        "hybrid_mode": str(getattr(cfg, "HYBRID_MODE", "dual")),
        "tokenizer_type": str(getattr(cfg, "TOKENIZER_TYPE", "kmer")),
        "corpus_hash": vocab_sha,
    })
    tokenizer = _maybe_lock_tokenizer(tokenizer)
    print(f"[Tokenizer] Trained & locked. Vocab={len(getattr(tokenizer,'vocab', {}))}")

    model, ss_head, metrics_vocab, is_dual, head_vocab = _build_model_and_head_for_selfsup(tokenizer, device)

    params = _collect_trainable_params(model, ss_head)
    optimizer = _build_optimizer(params, lr=cfg.LEARNING_RATE, weight_decay=1e-2)
    scheduler = _build_warmup_cosine_scheduler(optimizer, WARMUP_STEPS, COSINE_TOTAL_STEPS)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLE)
    ema = EMAWeights(model, decay=EMA_DECAY) if USE_EMA else None

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    print("\n====== STAGE 1: SELF-SUP (RNA viruses, complete) ======")
    stage1_frac = float(getattr(cfg, "TRAINSET_FRACTION", tok_fraction))
    stage1_sub = _subsample(stage1_all, stage1_frac, cfg.RANDOM_SEED, "first_k")
    s1_idx = list(range(len(stage1_sub)))
    if len(s1_idx) > 1:
        tr_idx, va_idx = train_test_split(s1_idx, test_size=0.3, random_state=cfg.RANDOM_SEED)
        s1_tr, s1_va = Subset(stage1_sub, tr_idx), Subset(stage1_sub, va_idx)
    else:
        s1_tr = stage1_sub; s1_va = stage1_sub

    collate_s1 = make_selfsup_collate(tokenizer, task=getattr(cfg, "PRETRAIN_TASK", "mlm"))
    s1_tr_loader = DataLoader(s1_tr, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_s1, num_workers=0, pin_memory=pin)
    s1_va_loader = DataLoader(s1_va, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_s1, num_workers=0, pin_memory=pin)

    prev_metrics = {}
    for ep in range(getattr(cfg, "EPOCHS", 1)):
        m_tr = train(
            model, ss_head, s1_tr_loader, optimizer, criterion, device,
            vocab_size=metrics_vocab, max_seq_len=cfg.MAX_LEN,
            ambiguous_token_id=tokenizer.vocab.get("N", None), kmer_size=cfg.KMER_SIZE,
            dual_stream=is_dual, collect_selfsup_extras=True, task=getattr(cfg, "PRETRAIN_TASK", "mlm"),
        )
        m_va = evaluate(
            model, ss_head, s1_va_loader, criterion, device,
            vocab_size=metrics_vocab, max_seq_len=cfg.MAX_LEN,
            ambiguous_token_id=tokenizer.vocab.get("N", None), kmer_size=cfg.KMER_SIZE,
            dual_stream=is_dual, collect_selfsup_extras=True, task=getattr(cfg, "PRETRAIN_TASK", "mlm"),
        )
        scheduler.step()
        if ema: ema.update(model)
        print(f"[Stage1] ep {ep+1}: train={m_tr.get('task_top5',{})}  valid={m_va.get('task_top5',{})}")
        prev_metrics = m_va

    torch.save(model.state_dict(), cfg.CKPT_STAGE1A)

    replay_loader = _build_replay_loader(
        tokenizer, cfg.DATA_FILES, batch_size=max(1, cfg.BATCH_SIZE // 2),
        aux_task=AUX_TASK, fraction=REPLAY_FRACTION, pin=pin
    )

    print("\n====== STAGE 2: SELF-SUP (RNA viruses, partial) ======")
    if cfg.PARTIAL_ENABLE:
        if cfg.PARTIAL_DATA_FILES and len(cfg.PARTIAL_DATA_FILES) > 0:
            s2 = SelfSupDataset(files=cfg.PARTIAL_DATA_FILES)
        else:
            from partial_datasets import PartialViewDataset
            full_texts = [stage1_all[i]["seq"] for i in range(len(stage1_all))]
            pv = PartialViewDataset(
                base_sequences=full_texts,
                window_size=cfg.PARTIAL_WINDOW_SIZE,
                windows_per_seq=cfg.PARTIAL_WINDOWS_PER_SEQ,
                overlap=cfg.PARTIAL_OVERLAP,
                strategy=cfg.PARTIAL_STRATEGY,
                seed=cfg.RANDOM_SEED,
            )
            class _Shim(torch.utils.data.Dataset):
                def __init__(self, views): self.views=views
                def __len__(self): return len(self.views)
                def __getitem__(self, i): return {"seq": self.views[i]}
            s2 = _Shim([pv[i]["seq"] for i in range(len(pv))])

        s2 = _subsample(s2, stage1_frac, cfg.RANDOM_SEED, "first_k")
        collate_s2 = make_selfsup_collate(tokenizer, task=getattr(cfg, "PRETRAIN_TASK", "mlm"))
        s2_loader = DataLoader(s2, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_s2, num_workers=0, pin_memory=pin)
        best = prev_metrics
        for ep in range(getattr(cfg, "EPOCHS_PARTIAL", 1)):
            m_tr = train(
                model, ss_head, s2_loader, optimizer, criterion, device,
                vocab_size=metrics_vocab, max_seq_len=cfg.PARTIAL_WINDOW_SIZE,
                ambiguous_token_id=tokenizer.vocab.get("N", None), kmer_size=cfg.KMER_SIZE,
                dual_stream=is_dual, collect_selfsup_extras=True, task=getattr(cfg, "PRETRAIN_TASK", "mlm"),
            )
            scheduler.step()
            if ema: ema.update(model)
            print(f"[Stage2] ep {ep+1}: train={m_tr.get('task_top5',{})}")
            cur = m_tr
            if _stage_gate_and_log("stage2", best, cur):
                best = cur
        torch.save(model.state_dict(), cfg.CKPT_STAGE1B)

    print("\n====== STAGE 3: SELF-SUP (Ribozymes, full) ======")
    s3_full = SelfSupDataset(files=cfg.STAGE2_RIBOZYME_FULL_FILES)
    if len(s3_full) > 0:
        s3_sub = _subsample(s3_full, stage1_frac, cfg.RANDOM_SEED, "first_k")
        idx = list(range(len(s3_sub)))
        if len(idx) > 1:
            tr, va = train_test_split(idx, test_size=0.3, random_state=cfg.RANDOM_SEED)
            s3_tr, s3_va = Subset(s3_sub, tr), Subset(s3_sub, va)
        else:
            s3_tr = s3_sub; s3_va = s3_sub
        collate_s3 = make_selfsup_collate(tokenizer, task=getattr(cfg, "PRETRAIN_TASK", "mlm"))
        s3_tr_loader = DataLoader(s3_tr, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_s3, num_workers=0, pin_memory=pin)
        s3_va_loader = DataLoader(s3_va, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_s3, num_workers=0, pin_memory=pin)
        best = prev_metrics
        for ep in range(getattr(cfg, "EPOCHS_STAGE2", 3)):
            m_tr = train(
                model, ss_head, s3_tr_loader, optimizer, criterion, device,
                vocab_size=metrics_vocab, max_seq_len=cfg.MAX_LEN,
                ambiguous_token_id=tokenizer.vocab.get("N", None), kmer_size=cfg.KMER_SIZE,
                dual_stream=is_dual, collect_selfsup_extras=True, task=getattr(cfg, "PRETRAIN_TASK", "mlm"),
            )
            m_va = evaluate(
                model, ss_head, s3_va_loader, criterion, device,
                vocab_size=metrics_vocab, max_seq_len=cfg.MAX_LEN,
                ambiguous_token_id=tokenizer.vocab.get("N", None), kmer_size=cfg.KMER_SIZE,
                dual_stream=is_dual, collect_selfsup_extras=True, task=getattr(cfg, "PRETRAIN_TASK", "mlm"),
            )
            scheduler.step()
            if ema: ema.update(model)
            print(f"[Stage3] ep {ep+1}: train={m_tr.get('task_top5',{})} valid={m_va.get('task_top5',{})}")
            if _stage_gate_and_log("stage3", m_tr, m_va):
                best = m_va
        torch.save(model.state_dict(), cfg.CKPT_STAGE2)
    else:
        print("[Stage3] No ribozyme full data. Skipping.")

    print("\n====== STAGE 4: SELF-SUP (Ribozymes, partial) ======")
    if cfg.STAGE3_RIBOZYME_PARTIAL_FILES:
        s4 = SelfSupDataset(files=cfg.STAGE3_RIBOZYME_PARTIAL_FILES)
        s4 = _subsample(s4, stage1_frac, cfg.RANDOM_SEED, "first_k")
    else:
        full_texts = [SelfSupDataset(files=cfg.STAGE2_RIBOZYME_FULL_FILES)[i]["seq"]
                      for i in range(len(SelfSupDataset(files=cfg.STAGE2_RIBOZYME_FULL_FILES)))]
        if len(full_texts) == 0:
            s4 = SelfSupDataset(files=[])  # empty
        else:
            from partial_datasets import PartialViewDataset
            pv = PartialViewDataset(
                base_sequences=full_texts,
                window_size=cfg.PARTIAL_WINDOW_SIZE,
                windows_per_seq=cfg.PARTIAL_WINDOWS_PER_SEQ,
                overlap=cfg.PARTIAL_OVERLAP,
                strategy=cfg.PARTIAL_STRATEGY,
                seed=cfg.RANDOM_SEED,
            )
            class _Shim2(torch.utils.data.Dataset):
                def __init__(self, views): self.views=views
                def __len__(self): return len(self.views)
                def __getitem__(self, i): return {"seq": self.views[i]}
            s4 = _Shim2([pv[i]["seq"] for i in range(len(pv))])

    if len(s4) > 0:
        collate_s4 = make_selfsup_collate(tokenizer, task=getattr(cfg, "PRETRAIN_TASK", "mlm"))
        s4_loader = DataLoader(s4, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_s4, num_workers=0, pin_memory=pin)
        best = prev_metrics
        for ep in range(getattr(cfg, "EPOCHS_STAGE3", 3)):
            m_tr = train(
                model, ss_head, s4_loader, optimizer, criterion, device,
                vocab_size=metrics_vocab, max_seq_len=cfg.PARTIAL_WINDOW_SIZE,
                ambiguous_token_id=tokenizer.vocab.get("N", None), kmer_size=cfg.KMER_SIZE,
                dual_stream=is_dual, collect_selfsup_extras=True, task=getattr(cfg, "PRETRAIN_TASK", "mlm"),
            )
            scheduler.step()
            if ema: ema.update(model)
            print(f"[Stage4] ep {ep+1}: train={m_tr.get('task_top5',{})}")
            if _stage_gate_and_log("stage4", best, m_tr):
                best = m_tr
        torch.save(model.state_dict(), cfg.CKPT_STAGE3)
    else:
        print("[Stage4] No ribozyme partial data. Skipping.")

    print("\n====== INDUCTIVE TRANSFER BEGINS (Stages 5–7) ======")
    _freeze_encoder_layers(model, unfreeze_top_n=UNFREEZE_TOP_N)

    if cfg.STAGE4_WEAK_SEQCLS and len(cfg.STAGE4_WEAK_SEQCLS) > 0:
        print("\n====== STAGE 5: SUP (RNA viruses, partial, seq-level) ======")
        ds5 = SupervisedMotifDataset(txt_files=cfg.STAGE4_WEAK_SEQCLS, txt_schema="seq_label", map_label_key=cfg.MAP_LABEL_KEY)
        ds5 = _subsample(ds5, stage1_frac, cfg.RANDOM_SEED, "first_k")
        coll5 = lambda b: make_supervised_collate(tokenizer, task_name="map", augment="mask_non_motif")(b)
        loader5 = DataLoader(ds5, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=coll5, num_workers=0, pin_memory=pin)

        head5 = MotifAnnotatedPretrainingHead(cfg.EMBED_DIM, output_dim=cfg.MAP_NUM_CLASSES_COARSE).to(device)
        proj5 = getattr(model, "cml_head", ContrastiveMotifLearningHead().to(device))

        cw = compute_class_weights(ds5, label_key=cfg.MAP_LABEL_KEY)
        ce_weight_vec = torch.tensor(cw, dtype=torch.float32, device=device) if cw is not None else None

        optimizer5 = _build_optimizer(_collect_trainable_params(model, head5, proj5), lr=cfg.LEARNING_RATE * cfg.LR_MULT_STAGE4, weight_decay=1e-2)
        scheduler5 = _build_warmup_cosine_scheduler(optimizer5, WARMUP_STEPS, COSINE_TOTAL_STEPS)

        aux_head5 = _build_aux_head(model, AUX_TASK, metrics_vocab)
        aux_loader5 = _build_replay_loader(tokenizer, cfg.DATA_FILES, batch_size=max(1, cfg.BATCH_SIZE // 2), aux_task=AUX_TASK, fraction=REPLAY_FRACTION, pin=pin)

        best_score = -1e9
        patience = EARLY_STOP_PATIENCE
        for ep in range(getattr(cfg, "EPOCHS_STAGE4", 2)):
            mt = train_cls_ce_supcon_epoch(
                model, head5, proj5, loader5, optimizer5, device,
                ce_weight=float(getattr(cfg, "CE_WEIGHT", 1.0)),
                supcon_weight=float(getattr(cfg, "SUPCON_WEIGHT", 0.5)),
                temperature=float(getattr(cfg, "SUPCON_TEMPERATURE", 0.07)),
            )
            if aux_loader5 is not None and AUX_WEIGHT > 0:
                aux_metrics = _aux_step_selfsup(
                    model, aux_head5, aux_loader5, torch.nn.CrossEntropyLoss(ignore_index=-100), device,
                    vocab_size=metrics_vocab, kmer_size=cfg.KMER_SIZE,
                    ambiguous_token_id=tokenizer.vocab.get("N", None), task=AUX_TASK
                )
                scheduler5.step()
                print(f"[Stage5][AUX {AUX_TASK}] top5={aux_metrics.get('task_top5',{})}")
            else:
                scheduler5.step()

            if ema: ema.update(model)
            mv = eval_cls_ce_supcon_epoch(model, head5, proj5, loader5, device)
            print(f"[Stage5] ep {ep+1}: train={mt.get('task_top5',{})} valid={mv.get('task_top5',{})}")

            score = float(mt.get("macro_f1_cls", float("nan")))
            if not math.isnan(score) and score > best_score:
                best_score = score
                patience = EARLY_STOP_PATIENCE
                torch.save(model.state_dict(), cfg.CKPT_STAGE4)
            else:
                patience -= 1
                if patience <= 0:
                    print("[Stage5] Early stopping triggered.")
                    break

    if cfg.STAGE5_WEAK_RIBOZYME_SEQCLS and len(cfg.STAGE5_WEAK_RIBOZYME_SEQCLS) > 0:
        print("\n====== STAGE 6: SUP (Ribozymes, partial, seq-level) ======")
        ds6 = SupervisedMotifDataset(txt_files=cfg.STAGE5_WEAK_RIBOZYME_SEQCLS, txt_schema="seq_label", map_label_key=cfg.MAP_LABEL_KEY)
        ds6 = _subsample(ds6, stage1_frac, cfg.RANDOM_SEED, "first_k")
        coll6 = lambda b: make_supervised_collate(tokenizer, task_name="map", augment="mask_non_motif")(b)
        loader6 = DataLoader(ds6, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=coll6, num_workers=0, pin_memory=pin)

        head6 = MotifAnnotatedPretrainingHead(cfg.EMBED_DIM, output_dim=cfg.MAP_NUM_CLASSES_RIBOZYME).to(device)
        proj6 = getattr(model, "cml_head", ContrastiveMotifLearningHead().to(device))
        optimizer6 = _build_optimizer(_collect_trainable_params(model, head6, proj6), lr=cfg.LEARNING_RATE * cfg.LR_MULT_STAGE5, weight_decay=1e-2)
        scheduler6 = _build_warmup_cosine_scheduler(optimizer6, WARMUP_STEPS, COSINE_TOTAL_STEPS)

        aux_head6 = _build_aux_head(model, AUX_TASK, metrics_vocab)
        aux_loader6 = _build_replay_loader(tokenizer, cfg.DATA_FILES, batch_size=max(1, cfg.BATCH_SIZE // 2), aux_task=AUX_TASK, fraction=REPLAY_FRACTION, pin=pin)

        best_score = -1e9
        patience = EARLY_STOP_PATIENCE
        for ep in range(getattr(cfg, "EPOCHS_STAGE5", 3)):
            mt = train_cls_ce_supcon_epoch(
                model, head6, proj6, loader6, optimizer6, device,
                ce_weight=float(getattr(cfg, "CE_WEIGHT", 1.0)),
                supcon_weight=float(getattr(cfg, "SUPCON_WEIGHT", 0.5)),
                temperature=float(getattr(cfg, "SUPCON_TEMPERATURE", 0.07)),
            )
            if aux_loader6 is not None and AUX_WEIGHT > 0:
                aux_metrics = _aux_step_selfsup(
                    model, aux_head6, aux_loader6, torch.nn.CrossEntropyLoss(ignore_index=-100), device,
                    vocab_size=metrics_vocab, kmer_size=cfg.KMER_SIZE,
                    ambiguous_token_id=tokenizer.vocab.get("N", None), task=AUX_TASK
                )
                scheduler6.step()
                print(f"[Stage6][AUX {AUX_TASK}] top5={aux_metrics.get('task_top5',{})}")
            else:
                scheduler6.step()

            if ema: ema.update(model)
            mv = eval_cls_ce_supcon_epoch(model, head6, proj6, loader6, device)
            print(f"[Stage6] ep {ep+1}: train={mt.get('task_top5',{})} valid={mv.get('task_top5',{})}")

            score = float(mt.get("macro_f1_cls", float("nan")))
            if not math.isnan(score) and score > best_score:
                best_score = score
                patience = EARLY_STOP_PATIENCE
                torch.save(model.state_dict(), cfg.CKPT_STAGE5)
            else:
                patience -= 1
                if patience <= 0:
                    print("[Stage6] Early stopping triggered.")
                    break

    if cfg.STAGE6_GOLD_MIXED and len(cfg.STAGE6_GOLD_MIXED) > 0:
        print("\n====== STAGE 7: SUP (Token-level motifs on mixed RNA/Ribozymes) ======")
        ds7 = SupervisedMotifDataset(jsonl_files=cfg.STAGE6_GOLD_MIXED)
        ds7 = _subsample(ds7, stage1_frac, cfg.RANDOM_SEED, "first_k")

    
        coll7_mm = make_supervised_collate(tokenizer, task_name="masked_motif", augment="mask_non_motif")
        loader7_mm = DataLoader(ds7, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=coll7_mm, num_workers=0, pin_memory=pin)
        head_mm = getattr(model, "masked_motif_head", MaskedMotifModelingHead(cfg.EMBED_DIM, cfg.MOTIF_VOCAB_SIZE).to(device))
        opt_mm = _build_optimizer(_collect_trainable_params(model, head_mm), lr=cfg.LEARNING_RATE * cfg.LR_MULT_STAGE6, weight_decay=1e-2)
        sch_mm = _build_warmup_cosine_scheduler(opt_mm, WARMUP_STEPS, COSINE_TOTAL_STEPS)

        from model import MotifBoundaryPredictionHead
        coll7_mbp = make_supervised_collate(tokenizer, task_name="mbp", augment="mask_non_motif")
        loader7_mbp = DataLoader(ds7, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=coll7_mbp, num_workers=0, pin_memory=pin)
        head_mbp = getattr(model, "mbp_head", MotifBoundaryPredictionHead(cfg.EMBED_DIM).to(device))
        opt_mbp = _build_optimizer(_collect_trainable_params(model, head_mbp), lr=cfg.LEARNING_RATE * cfg.LR_MULT_STAGE6, weight_decay=1e-2)
        sch_mbp = _build_warmup_cosine_scheduler(opt_mbp, WARMUP_STEPS, COSINE_TOTAL_STEPS)

        aux_head7 = _build_aux_head(model, AUX_TASK, metrics_vocab)
        aux_loader7 = _build_replay_loader(tokenizer, cfg.DATA_FILES, batch_size=max(1, cfg.BATCH_SIZE // 2), aux_task=AUX_TASK, fraction=REPLAY_FRACTION, pin=pin)

        for ep in range(getattr(cfg, "EPOCHS_STAGE6", 3)):
            mt_mm = train(
                model, head_mm, loader7_mm, opt_mm, torch.nn.CrossEntropyLoss(ignore_index=-100), device,
                vocab_size=cfg.MOTIF_VOCAB_SIZE, max_seq_len=cfg.MAX_LEN,
                ambiguous_token_id=tokenizer.vocab.get("N", None), kmer_size=cfg.KMER_SIZE,
                dual_stream=False, collect_selfsup_extras=True, task="masked_motif",
            )
            sch_mm.step()
            print(f"[Stage7/MM] ep {ep+1}: {mt_mm.get('task_top5',{})}")

            mt_mbp = train(
                model, head_mbp, loader7_mbp, opt_mbp, torch.nn.CrossEntropyLoss(), device,
                vocab_size=2, max_seq_len=cfg.MAX_LEN,
                ambiguous_token_id=tokenizer.vocab.get("N", None), kmer_size=cfg.KMER_SIZE,
                dual_stream=False, collect_selfsup_extras=True, task="mbp",
            )
            sch_mbp.step()
            print(f"[Stage7/MBP] ep {ep+1}: {mt_mbp.get('task_top5',{})}")

            if aux_loader7 is not None and AUX_WEIGHT > 0:
                aux_metrics = _aux_step_selfsup(
                    model, aux_head7, aux_loader7, torch.nn.CrossEntropyLoss(ignore_index=-100), device,
                    vocab_size=metrics_vocab, kmer_size=cfg.KMER_SIZE,
                    ambiguous_token_id=tokenizer.vocab.get("N", None), task=AUX_TASK
                )
                print(f"[Stage7][AUX {AUX_TASK}] top5={aux_metrics.get('task_top5',{})}")

            if ema: ema.update(model)

        torch.save(model.state_dict(), cfg.CKPT_STAGE6)

    probe = _linear_probe_eval(
        model, tokenizer,
        files=(cfg.STAGE4_WEAK_SEQCLS if (cfg.STAGE4_WEAK_SEQCLS and len(cfg.STAGE4_WEAK_SEQCLS)>0) else cfg.STAGE5_WEAK_RIBOZYME_SEQCLS),
        label_key=getattr(cfg, "MAP_LABEL_KEY", "global_label")
    )
    print(f"\n[Extrinsic probe] {probe}")

    print("\nAll stages complete.")

if __name__ == "__main__":
    main()
