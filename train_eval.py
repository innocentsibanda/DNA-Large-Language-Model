
# train and evaluation

from __future__ import annotations

import math
from typing import Optional, Dict, Tuple, Any, List

import numpy as np
import torch
from torch import Tensor
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

try:
    from config import PRETRAIN_TASK, SELF_SUP_STREAM
except Exception:
    PRETRAIN_TASK = "mlm"
    SELF_SUP_STREAM = "kmer"

def _to_numpy(x: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()

def _safe_mean(x: List[float]) -> float:
    return float(np.mean(x)) if len(x) > 0 else float("nan")

def compute_per_class_accuracy(preds: Tensor, labels: Tensor, num_classes: int) -> list:
    per_class_acc = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum().item() == 0:
            per_class_acc.append(float("nan"))
        else:
            correct = (preds[mask] == labels[mask]).sum().item()
            per_class_acc.append(correct / mask.sum().item())
    return per_class_acc

def token_entropy(probs: np.ndarray) -> float:
    if probs.size == 0:
        return float("nan")
    epsilon = 1e-12
    entropy_per_token = -np.sum(probs * np.log(probs + epsilon), axis=1)
    return float(np.mean(entropy_per_token))

def ambiguous_base_error_rate(labels: np.ndarray, preds: np.ndarray, ambiguous_token_id: int) -> float:
    mask = labels == ambiguous_token_id
    if np.sum(mask) == 0:
        return float("nan")
    errors = np.sum(preds[mask] != labels[mask])
    return errors / np.sum(mask)

def position_wise_accuracy(preds: np.ndarray, labels: np.ndarray, max_seq_len: Optional[int]) -> list:
    if max_seq_len is None or max_seq_len <= 0:
        return []
    if len(preds) == 0 or len(labels) == 0 or (len(preds) % max_seq_len != 0):
        return []
    preds_2d = preds.reshape(-1, max_seq_len)
    labels_2d = labels.reshape(-1, max_seq_len)
    pos_acc = []
    for pos in range(max_seq_len):
        mask = labels_2d[:, pos] != -100
        if np.sum(mask) == 0:
            pos_acc.append(float("nan"))
        else:
            correct = np.sum(preds_2d[mask, pos] == labels_2d[mask, pos])
            pos_acc.append(correct / np.sum(mask))
    return pos_acc

def transition_transversion_error_rate(labels: np.ndarray, preds: np.ndarray) -> dict:
    transitions = {(0, 2), (2, 0), (1, 3), (3, 1)}
    transversions = set()
    bases = {0, 1, 2, 3}
    for b1 in bases:
        for b2 in bases:
            if b1 != b2 and (b1, b2) not in transitions:
                transversions.add((b1, b2))

    mask = (labels >= 0) & (labels <= 3)
    filtered_labels = labels[mask]
    filtered_preds = preds[mask]
    if filtered_labels.size == 0:
        return {"transition_rate": float("nan"), "transversion_rate": float("nan")}

    errors = filtered_preds != filtered_labels
    if np.sum(errors) == 0:
        return {"transition_rate": 0.0, "transversion_rate": 0.0}

    error_pairs = list(zip(filtered_labels[errors], filtered_preds[errors]))
    transition_errors = sum(1 for pair in error_pairs if pair in transitions)
    transversion_errors = sum(1 for pair in error_pairs if pair in transversions)
    total_errors = transition_errors + transversion_errors
    return {
        "transition_rate": transition_errors / total_errors if total_errors > 0 else 0.0,
        "transversion_rate": transversion_errors / total_errors if total_errors > 0 else 0.0,
    }

def kmer_accuracy(preds: np.ndarray, labels: np.ndarray, k: int, vocab_size: Optional[int]) -> float:
    if vocab_size is None or k <= 0 or len(preds) < k or len(labels) < k:
        return float("nan")

    def encode_kmer(arr):
        kmer_ids = []
        base = vocab_size
        for i in range(len(arr) - k + 1):
            kmer_id = 0
            for j in range(k):
                kmer_id = kmer_id * base + int(arr[i + j])
            kmer_ids.append(kmer_id)
        return np.array(kmer_ids, dtype=np.int64)

    pred_kmers = encode_kmer(preds)
    label_kmers = encode_kmer(labels)
    if len(label_kmers) == 0:
        return float("nan")
    correct = np.sum(pred_kmers == label_kmers)
    total = len(label_kmers)
    return correct / total if total > 0 else float("nan")

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    if probs.size == 0 or labels.size == 0:
        return float("nan")
    valid = labels >= 0
    if not np.any(valid):
        return float("nan")
    probs = probs[valid]
    labels = labels[valid].astype(np.int64)

    bins = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    ece = 0.0
    N = float(len(labels))
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences >= lo) & (confidences <= hi if i == n_bins - 1 else confidences < hi)
        if np.sum(mask) > 0:
            avg_conf = float(np.mean(confidences[mask]))
            avg_acc = float(np.mean(accuracies[mask]))
            ece += (np.sum(mask) / N) * abs(avg_conf - avg_acc)
    return float(ece)


def prediction_entropy_distribution_stats(probs: np.ndarray) -> dict:
    if probs.size == 0:
        return {"entropy_mean": float("nan"), "entropy_std": float("nan"),
                "entropy_max": float("nan"), "entropy_min": float("nan")}
    epsilon = 1e-12
    h = -np.sum(probs * np.log(probs + epsilon), axis=1)
    return {
        "entropy_mean": float(np.mean(h)),
        "entropy_std": float(np.std(h)),
        "entropy_max": float(np.max(h)),
        "entropy_min": float(np.min(h)),
    }

def cls_top1_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    if preds.size == 0 or labels.size == 0:
        return float("nan")
    return float((preds == labels).mean())


def cls_macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    if preds.size == 0 or labels.size == 0:
        return float("nan")
    return float(f1_score(labels, preds, average="macro", zero_division=0))

def cls_ece_from_logits(logits: Tensor, labels: Tensor) -> float:
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        valid = labels >= 0
        if valid.sum().item() == 0:
            return float("nan")
        p = probs[valid].detach().cpu().numpy()
        y = labels[valid].detach().cpu().numpy()
        return expected_calibration_error(p, y)

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        logpt = torch.nn.functional.log_softmax(logits, dim=-1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        pt = pt.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            w = self.weight.gather(0, target)
            loss = loss * w
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_mult: float = 0.0,
) -> LambdaLR:

    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_mult + (1.0 - min_lr_mult) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

class EMAWeights:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, Tensor] = {}
        self.backup: Dict[str, Tensor] = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self.backup:
                p.data.copy_(self.backup[name].data)
        self.backup = {}

def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int | None = None,
    *,
    method: str = "effective_num",  
    beta: float = 0.9999,
    eps: float = 1e-8,
) -> torch.Tensor:
    labels = labels.view(-1).to(torch.long)
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1 if labels.numel() > 0 else 0
    counts = torch.bincount(labels, minlength=num_classes).float()

    if num_classes == 0:
        return torch.tensor([], dtype=torch.float32)

    if method == "freq_inv":
        w = 1.0 / (counts + eps)
        w = w * (num_classes / (w.sum() + eps))
        return w

    if method == "effective_num":
        beta_t = torch.tensor(beta, dtype=torch.float32)
        effective_num = 1.0 - torch.pow(beta_t, counts)
        w = (1.0 - beta_t) / (effective_num + eps)
        w = w * (num_classes / (w.sum() + eps))
        return w

    raise ValueError(f"Unknown method={method!r}. Use 'effective_num' or 'freq_inv'.")

def compute_class_weights_from_loader(
    dataloader,
    *,
    label_key: str = "labels",
    num_classes: int | None = None,
    method: str = "effective_num",
    beta: float = 0.9999,
    eps: float = 1e-8,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    all_labels = []
    for batch in dataloader:
        if label_key not in batch:
            continue
        y = batch[label_key]
        if torch.is_tensor(y):
            all_labels.append(y.detach().view(-1).to(torch.long).cpu())
    if not all_labels:
        return torch.tensor([], dtype=torch.float32)
    labels = torch.cat(all_labels, dim=0)
    w = compute_class_weights(labels, num_classes=num_classes, method=method, beta=beta, eps=eps)
    if device is not None:
        w = w.to(device)
    return w

get_class_weights = compute_class_weights

def _unpack_single_stream(batch: Dict[str, Tensor], device: torch.device) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    x = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    attention_mask = attention_mask.to(device) if attention_mask is not None else None
    position_ids = batch.get("position_ids")
    position_ids = position_ids.to(device) if position_ids is not None else None
    labels = batch.get("labels")
    labels = labels.to(device) if labels is not None else None
    return x, attention_mask, position_ids, labels

def _unpack_dual_stream(batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    out: Dict[str, Tensor] = {}
    for k in (
        "kmer_input_ids", "kmer_attention_mask", "kmer_position_ids",
        "bpe_input_ids", "bpe_attention_mask", "bpe_position_ids",
        "labels_kmer", "labels_bpe",
    ):
        if k in batch and batch[k] is not None:
            out[k] = batch[k].to(device)
    if "motif_flags" in batch and batch["motif_flags"] is not None:
        out["motif_flags"] = batch["motif_flags"].to(device)
    return out

def _encode_any(
    model: torch.nn.Module,
    x: Tensor,
    attention_mask: Optional[Tensor] = None,
    motif_flags: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
) -> Tensor:
    enc = getattr(model, "encoder", model)
    try:
        return enc(x, attention_mask=attention_mask, motif_flags=motif_flags, position_ids=position_ids)
    except TypeError:
        return enc(
            kmer_input_ids=x,
            bpe_input_ids=x,
            kmer_attention_mask=attention_mask,
            bpe_attention_mask=attention_mask,
            kmer_position_ids=position_ids,
            bpe_position_ids=position_ids,
            motif_flags=motif_flags,
        )

def _forward_single(
    model: torch.nn.Module,
    head: Optional[torch.nn.Module],
    x: Tensor,
    attention_mask: Optional[Tensor],
    position_ids: Optional[Tensor],
    motif_flags: Optional[Tensor],
) -> Tensor:
    encoded = _encode_any(model, x, attention_mask=attention_mask, motif_flags=motif_flags, position_ids=position_ids)
    return head(encoded) if head is not None else encoded

def _forward_dual(
    dual_encoder: torch.nn.Module,
    head: Optional[torch.nn.Module],
    batch: Dict[str, Tensor],
) -> Tensor:
    encoded = dual_encoder(
        kmer_input_ids=batch["kmer_input_ids"],
        bpe_input_ids=batch["bpe_input_ids"],
        kmer_attention_mask=batch.get("kmer_attention_mask"),
        bpe_attention_mask=batch.get("bpe_attention_mask"),
        kmer_position_ids=batch.get("kmer_position_ids"),
        bpe_position_ids=batch.get("bpe_position_ids"),
        motif_flags=batch.get("motif_flags"),
    )
    return head(encoded) if head is not None else encoded

def _pick_dual_labels(b: dict, stream_name: str) -> Tuple[torch.Tensor, str]:
    stream = (stream_name or "kmer").lower()
    for key in (f"labels_{stream}", "labels_kmer", "labels_bpe"):
        t = b.get(key, None)
        if t is not None:
            return t, key
    raise ValueError("Dual-stream batch is missing 'labels_kmer'/'labels_bpe'.")

def _compute_loss_and_basic_counts(
    logits: Tensor,
    labels: Tensor,
    criterion: torch.nn.Module,
    attention_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, int, int, int, Tensor, Tensor]:
    if logits.dim() == 3 and labels.dim() == 2:
        V = logits.size(-1)
        mask = (labels != -100)
        loss = criterion(logits.view(-1, V), labels.view(-1))
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        batch_masked = int(mask.sum().item())
        if attention_mask is not None:
            batch_valid = int(attention_mask.long().sum().item())
        else:
            batch_valid = int((labels != -100).numel())
        batch_correct = int(((preds == labels) & mask).sum().item())
        return loss, batch_masked, batch_valid, batch_correct, preds, probs

    if logits.dim() == 2 and (labels.dim() == 1 or (labels.dim() == 2 and labels.size(1) == 1)):
        if labels.dim() == 2:
            labels = labels.squeeze(1)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        batch_masked = int(labels.numel())
        batch_valid = batch_masked
        batch_correct = int((preds == labels).sum().item())
        return loss, batch_masked, batch_valid, batch_correct, preds, probs

    raise ValueError(
        f"Head/labels shape mismatch: logits shape {tuple(logits.shape)} vs labels shape {tuple(labels.shape)}. "
        "Use a token-level head (B,L,V) with token labels (B,L), "
        "or a sequence-level head (B,C) with sequence labels (B)."
    )


def _aggregate_selfsup_metrics(
    avg_loss: float,
    total_tokens: int,
    total_masked_tokens: int,
    all_preds: List[Tensor],
    all_labels: List[Tensor],
    all_probs: List[np.ndarray],
    vocab_size: Optional[int],
    max_seq_len: Optional[int],
    ambiguous_token_id: Optional[int],
    kmer_size: int,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    preds_concat = _to_numpy(torch.cat(all_preds)) if all_preds else np.array([], dtype=np.int64)
    labels_concat = _to_numpy(torch.cat(all_labels)) if all_labels else np.array([], dtype=np.int64)
    probs_concat = np.concatenate(all_probs) if all_probs else np.array([], dtype=np.float32)

    if labels_concat.size > 0:
        try:
            precision, recall, f1_w, _ = precision_recall_fscore_support(
                labels_concat, preds_concat, average="weighted", zero_division=0
            )
        except Exception:
            precision = recall = f1_w = float("nan")

        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        per_class_acc = (
            compute_per_class_accuracy(torch.tensor(preds_concat), torch.tensor(labels_concat), int(vocab_size))
            if vocab_size else []
        )
        try:
            cm_mask = labels_concat >= 0
            conf_mat = confusion_matrix(labels_concat[cm_mask], preds_concat[cm_mask]) if np.any(cm_mask) else None
        except Exception:
            conf_mat = None

        entropy_avg = token_entropy(probs_concat)
        ambiguous_error = (
            ambiguous_base_error_rate(labels_concat, preds_concat, ambiguous_token_id)
            if ambiguous_token_id is not None else float("nan")
        )
        pos_acc = position_wise_accuracy(preds_concat, labels_concat, max_seq_len)
        transv_rates = transition_transversion_error_rate(labels_concat, preds_concat)
        kmer_acc = kmer_accuracy(preds_concat, labels_concat, kmer_size, vocab_size)
        ece = expected_calibration_error(probs_concat, labels_concat)

        metrics.update({
            "precision": precision,
            "recall": recall,
            "f1": f1_w,
            "perplexity": perplexity,
            "per_class_accuracy": per_class_acc,
            "confusion_matrix": conf_mat,
            "avg_token_entropy": entropy_avg,
            "ambiguous_base_error_rate": ambiguous_error,
            "position_wise_accuracy": pos_acc,
            "transition_rate": transv_rates.get("transition_rate", float("nan")),
            "transversion_rate": transv_rates.get("transversion_rate", float("nan")),
            "kmer_accuracy": kmer_acc,
            "expected_calibration_error": ece,
            **prediction_entropy_distribution_stats(probs_concat),
        })
        try:
            metrics["macro_f1"] = float(f1_score(labels_concat, preds_concat, average="macro", zero_division=0))
        except Exception:
            metrics["macro_f1"] = float("nan")

        try:
            if probs_concat.size and probs_concat.shape[1] == 2:
                mask = labels_concat >= 0
                if np.any(mask):
                    auroc = roc_auc_score(labels_concat[mask], probs_concat[mask, 1])
                else:
                    auroc = float("nan")
                metrics["auroc"] = float(auroc)
            else:
                metrics["auroc"] = float("nan")
        except Exception:
            metrics["auroc"] = float("nan")
    else:
        metrics.update({
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "perplexity": float("inf"),
            "per_class_accuracy": [],
            "confusion_matrix": None,
            "avg_token_entropy": float("nan"),
            "ambiguous_base_error_rate": float("nan"),
            "position_wise_accuracy": [],
            "transition_rate": float("nan"),
            "transversion_rate": float("nan"),
            "kmer_accuracy": float("nan"),
            "expected_calibration_error": float("nan"),
            "entropy_mean": float("nan"),
            "entropy_std": float("nan"),
            "entropy_max": float("nan"),
            "entropy_min": float("nan"),
            "macro_f1": float("nan"),
            "auroc": float("nan"),
        })
    return metrics

def _select_top5_for_task(task: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    t = (task or PRETRAIN_TASK).lower()

    loss = metrics.get("loss", float("nan"))
    acc = metrics.get("accuracy", float("nan"))
    ppl = metrics.get("perplexity", float("inf"))
    ent = metrics.get("avg_token_entropy", float("nan"))
    ece = metrics.get("expected_calibration_error", float("nan"))
    masking_ratio = metrics.get("masking_ratio", float("nan"))

    if t == "mlm":
        return {"loss": loss, "accuracy_masked": acc, "perplexity": ppl,
                "masking_ratio": masking_ratio, "expected_calibration_error": ece}
    if t in ("span", "dae", "kmer_reorder"):
        return {"loss": loss, "accuracy_masked": acc, "perplexity": ppl,
                "masking_ratio": masking_ratio, "seq_reconstruction_rate": metrics.get("seq_reconstruction_rate", float("nan"))}
    if t in ("masked_motif", "map"):
        return {"loss": loss, "token_accuracy": acc, "masking_ratio": masking_ratio,
                "avg_token_entropy": ent, "macro_f1": metrics.get("macro_f1", float("nan"))}
    if t == "mbp":
        return {"loss": loss, "token_accuracy": acc, "avg_token_entropy": ent,
                "macro_f1": metrics.get("macro_f1", float("nan")), "auroc": metrics.get("auroc", float("nan"))}
    if t == "ce_supcon":
        return {"loss_ce": metrics.get("loss_ce", float("nan")), "loss_supcon": metrics.get("loss_supcon", float("nan")),
                "top1_accuracy": metrics.get("top1_accuracy", float("nan")),
                "macro_f1": metrics.get("macro_f1_cls", float("nan")), "ece": metrics.get("ece_cls", float("nan"))}
    if t == "cml":
        return {"contrastive_loss": metrics.get("contrastive_loss", float("nan")),
                "pos_cosine_mean": metrics.get("pos_cosine_mean", float("nan")),
                "neg_cosine_mean": metrics.get("neg_cosine_mean", float("nan")),
                "embedding_norm_mean": metrics.get("embedding_norm_mean", float("nan")),
                "alignment_uniformity": metrics.get("alignment_uniformity", float("nan"))}
    return {"loss": loss, "accuracy": acc, "perplexity": ppl, "entropy_mean": ent, "expected_calibration_error": ece}

def train(
    model: torch.nn.Module,
    head: Optional[torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Optional[torch.nn.Module],
    device: torch.device,
    vocab_size: Optional[int] = None,
    print_batch_metrics: bool = False,
    max_seq_len: Optional[int] = None,
    ambiguous_token_id: Optional[int] = None,
    kmer_size: int = 3,
    dual_stream: bool = False,
    collect_selfsup_extras: Optional[bool] = None,
    task: Optional[str] = None,
    *,
    mixed_precision: bool = False,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    task = (task or PRETRAIN_TASK).lower()
    model.train()
    if head is not None:
        head.train()

    total_loss = 0.0
    total_correct = 0
    total_valid_tokens = 0
    total_masked_tokens = 0

    dae_total_corrupted = 0
    dae_total_corrupted_correct = 0
    seq_with_corr = 0
    seq_full_correct = 0

    all_preds: List[Tensor] = []
    all_labels: List[Tensor] = []
    all_probs: List[np.ndarray] = []

    use_amp = bool(mixed_precision and torch.cuda.is_available())
    scaler = scaler if scaler is not None else GradScaler(enabled=use_amp)

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)

        if dual_stream:
            b = _unpack_dual_stream(batch, device)
            with autocast(enabled=use_amp):
                logits = _forward_dual(model.encoder if hasattr(model, "encoder") else model, head, b)
                labels, label_key = _pick_dual_labels(b, SELF_SUP_STREAM)
                attention_mask = b.get("kmer_attention_mask") if label_key == "labels_kmer" else b.get("bpe_attention_mask")
                if criterion is None:
                    raise ValueError("criterion is required for training.")
                loss, batch_masked, batch_valid, batch_correct, preds, probs = _compute_loss_and_basic_counts(
                    logits, labels, criterion, attention_mask
                )
        else:
            x, attention_mask, position_ids, labels = _unpack_single_stream(batch, device)
            motif_flags = batch.get("motif_flags")
            motif_flags = motif_flags.to(device) if motif_flags is not None else None
            with autocast(enabled=use_amp):
                logits = _forward_single(model, head, x, attention_mask, position_ids, motif_flags)
                if criterion is None or labels is None:
                    raise ValueError("criterion and labels are required for training.")
                loss, batch_masked, batch_valid, batch_correct, preds, probs = _compute_loss_and_basic_counts(
                    logits, labels, criterion, attention_mask
                )

        if use_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                list(p for p in model.parameters() if p.requires_grad) +
                ([] if head is None else [p for p in head.parameters() if p.requires_grad]),
                max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(p for p in model.parameters() if p.requires_grad) +
                ([] if head is None else [p for p in head.parameters() if p.requires_grad]),
                max_norm=1.0
            )
            optimizer.step()

        total_correct       += batch_correct
        total_masked_tokens += batch_masked
        total_valid_tokens  += batch_valid
        total_loss          += float(loss.item()) * max(1, batch_masked)

        if print_batch_metrics and batch_masked > 0:
            print(f"Batch {batch_idx + 1} - loss: {loss.item():.4f}  acc: {batch_correct / batch_masked:.4f}  count: {batch_masked}")

        if logits.dim() == 3 and labels.dim() == 2:
            mask = (labels != -100)
            if mask.any():
                all_preds.append(preds[mask].detach().cpu())
                all_labels.append(labels[mask].detach().cpu())
                all_probs.append(probs[mask].detach().cpu().numpy())

        if task in ("dae", "span", "kmer_reorder") and logits.dim() == 3 and labels.dim() == 2:
            valid_bool = (attention_mask.long() > 0) if attention_mask is not None else torch.ones_like(labels, dtype=torch.bool)
            corrupted = (x != labels) & valid_bool if not dual_stream else torch.zeros_like(labels, dtype=torch.bool)
            if corrupted.numel() > 0:
                dae_total_corrupted += int(corrupted.sum().item())
                dae_total_corrupted_correct += int(((preds == labels) & corrupted).sum().item())
                if corrupted.dim() == 2:
                    row_has_corr = corrupted.any(dim=1)
                    row_all_corr_correct = ((preds == labels) | (~corrupted)).all(dim=1)
                    seq_with_corr += int(row_has_corr.sum().item())
                    seq_full_correct += int((row_all_corr_correct & row_has_corr).sum().item())

    avg_loss = total_loss / max(1, total_masked_tokens)
    accuracy = total_correct / max(1, total_masked_tokens)

    metrics: Dict[str, Any] = {"loss": avg_loss, "accuracy": accuracy}

    if collect_selfsup_extras is None:
        collect_selfsup_extras = PRETRAIN_TASK in ["mlm", "span", "kmer_reorder", "dae"]

    if collect_selfsup_extras and total_masked_tokens > 0 and len(all_labels) > 0:
        rich = _aggregate_selfsup_metrics(
            avg_loss=avg_loss,
            total_tokens=total_valid_tokens,
            total_masked_tokens=total_masked_tokens,
            all_preds=all_preds,
            all_labels=all_labels,
            all_probs=all_probs,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            ambiguous_token_id=ambiguous_token_id,
            kmer_size=kmer_size,
        )
        metrics.update(rich)

    metrics["masking_ratio"] = total_masked_tokens / max(1, total_valid_tokens) if total_valid_tokens > 0 else float("nan")

    if task in ("dae", "span", "kmer_reorder"):
        metrics["corruption_rate"] = dae_total_corrupted / max(1, total_valid_tokens) if total_valid_tokens > 0 else float("nan")
        metrics["masked_accuracy"] = dae_total_corrupted_correct / max(1, dae_total_corrupted) if dae_total_corrupted > 0 else float("nan")
        metrics["seq_reconstruction_rate"] = (seq_full_correct / max(1, seq_with_corr)) if seq_with_corr > 0 else float("nan")

    metrics["task_top5"] = _select_top5_for_task(task, metrics)
    return metrics

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    head: Optional[torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    criterion: Optional[torch.nn.Module],
    device: torch.device,
    vocab_size: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    ambiguous_token_id: Optional[int] = None,
    kmer_size: int = 3,
    dual_stream: bool = False,
    collect_selfsup_extras: Optional[bool] = None,
    task: Optional[str] = None,
    *,
    EMAWeights: Optional[EMAWeights] = None,
    use_EMAWeights_for_eval: bool = False,
    mixed_precision: bool = False,
) -> Dict[str, float]:
    task = (task or PRETRAIN_TASK).lower()

    if EMAWeights is not None and use_EMAWeights_for_eval:
        EMAWeights.apply_shadow(model)

    model.eval()
    if head is not None:
        head.eval()

    total_loss = 0.0
    total_correct = 0
    total_valid_tokens = 0
    total_masked_tokens = 0

    dae_total_corrupted = 0
    dae_total_corrupted_correct = 0
    seq_with_corr = 0
    seq_full_correct = 0

    all_preds: List[Tensor] = []
    all_labels: List[Tensor] = []
    all_probs: List[np.ndarray] = []

    use_amp = bool(mixed_precision and torch.cuda.is_available())

    for batch in dataloader:
        if dual_stream:
            b = _unpack_dual_stream(batch, device)
            with autocast(enabled=use_amp):
                logits = _forward_dual(model.encoder if hasattr(model, "encoder") else model, head, b)
                labels, label_key = _pick_dual_labels(b, SELF_SUP_STREAM)
                attention_mask = b.get("kmer_attention_mask") if label_key == "labels_kmer" else b.get("bpe_attention_mask")
                if criterion is None:
                    raise ValueError("criterion is required for evaluation.")
                loss, batch_masked, batch_valid, batch_correct, preds, probs = _compute_loss_and_basic_counts(
                    logits, labels, criterion, attention_mask
                )
        else:
            x, attention_mask, position_ids, labels = _unpack_single_stream(batch, device)
            motif_flags = batch.get("motif_flags")
            motif_flags = motif_flags.to(device) if motif_flags is not None else None
            with autocast(enabled=use_amp):
                logits = _forward_single(model, head, x, attention_mask, position_ids, motif_flags)
                if criterion is None or labels is None:
                    raise ValueError("criterion and labels are required for evaluation.")
                loss, batch_masked, batch_valid, batch_correct, preds, probs = _compute_loss_and_basic_counts(
                    logits, labels, criterion, attention_mask
                )

        total_correct       += batch_correct
        total_masked_tokens += batch_masked
        total_valid_tokens  += batch_valid
        total_loss          += float(loss.item()) * max(1, batch_masked)

        if logits.dim() == 3 and labels.dim() == 2:
            mask = (labels != -100)
            if mask.any():
                all_preds.append(preds[mask].cpu())
                all_labels.append(labels[mask].cpu())
                all_probs.append(probs[mask].cpu().numpy())

        if task in ("dae", "span", "kmer_reorder") and logits.dim() == 3 and labels.dim() == 2:
            valid_bool = (attention_mask.long() > 0) if attention_mask is not None else torch.ones_like(labels, dtype=torch.bool)
            corrupted = (x != labels) & valid_bool if not dual_stream else torch.zeros_like(labels, dtype=torch.bool)
            if corrupted.numel() > 0:
                dae_total_corrupted += int(corrupted.sum().item())
                dae_total_corrupted_correct += int(((preds == labels) & corrupted).sum().item())
                if corrupted.dim() == 2:
                    row_has_corr = corrupted.any(dim=1)
                    row_all_corr_correct = ((preds == labels) | (~corrupted)).all(dim=1)
                    seq_with_corr += int(row_has_corr.sum().item())
                    seq_full_correct += int((row_all_corr_correct & row_has_corr).sum().item())

    avg_loss = total_loss / max(1, total_masked_tokens)
    accuracy = total_correct / max(1, total_masked_tokens)

    metrics: Dict[str, Any] = {"loss": avg_loss, "accuracy": accuracy}

    if collect_selfsup_extras is None:
        collect_selfsup_extras = PRETRAIN_TASK in ["mlm", "span", "kmer_reorder", "dae"]

    if collect_selfsup_extras and total_masked_tokens > 0 and len(all_labels) > 0:
        rich = _aggregate_selfsup_metrics(
            avg_loss=avg_loss,
            total_tokens=total_valid_tokens,
            total_masked_tokens=total_masked_tokens,
            all_preds=all_preds,
            all_labels=all_labels,
            all_probs=all_probs,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            ambiguous_token_id=ambiguous_token_id,
            kmer_size=kmer_size,
        )
        metrics.update(rich)

    metrics["masking_ratio"] = total_masked_tokens / max(1, total_valid_tokens) if total_valid_tokens > 0 else float("nan")

    if task in ("dae", "span", "kmer_reorder"):
        metrics["corruption_rate"] = dae_total_corrupted / max(1, total_valid_tokens) if total_valid_tokens > 0 else float("nan")
        metrics["masked_accuracy"] = dae_total_corrupted_correct / max(1, dae_total_corrupted) if dae_total_corrupted > 0 else float("nan")
        metrics["seq_reconstruction_rate"] = (seq_full_correct / max(1, seq_with_corr)) if seq_with_corr > 0 else float("nan")

    metrics["task_top5"] = _select_top5_for_task(task, metrics)

    if EMAWeights is not None and use_EMAWeights_for_eval:
        EMAWeights.restore(model)

    return metrics

def supervised_contrastive_loss(z: Tensor, y: Tensor, temperature: float = 0.07) -> Tensor:
    z = torch.nn.functional.normalize(z, dim=-1)
    y = y.long().view(-1)
    sim = torch.matmul(z, z.t()) / max(temperature, 1e-8)
    N = z.size(0)
    device = z.device
    mask_self = torch.eye(N, dtype=torch.bool, device=device)
    labels = y.view(-1, 1)
    pos_mask = (labels == labels.t()) & (~mask_self)
    sim = sim.masked_fill(mask_self, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_log_prob = log_prob[pos_mask]
    if pos_log_prob.numel() == 0:
        return torch.zeros((), device=device)
    return -pos_log_prob.mean()


def train_cls_ce_supcon_epoch(
    model: torch.nn.Module,
    head_cls: torch.nn.Module,
    cml_head: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ce_weight: float = 1.0,
    supcon_weight: float = 1.0,
    temperature: float = 0.07,
    *,
    mixed_precision: bool = False,
    scaler: Optional[GradScaler] = None,
    focal_loss: bool = False,
    class_weights: Optional[Tensor] = None,
) -> Dict[str, Any]:
    model.train(); head_cls.train(); cml_head.train()

    use_amp = bool(mixed_precision and torch.cuda.is_available())
    scaler = scaler if scaler is not None else GradScaler(enabled=use_amp)

    all_logits: List[Tensor] = []
    all_labels: List[Tensor] = []

    ce_losses: List[float] = []
    supcon_losses: List[float] = []

    if focal_loss:
        ce_crit = FocalLoss(gamma=2.0, weight=class_weights.to(device) if class_weights is not None else None)
    else:
        ce_crit = torch.nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    for batch in dataloader:
        x = (batch.get("input_ids") or batch.get("kmer_input_ids")).to(device)
        am = batch.get("attention_mask") or batch.get("kmer_attention_mask")
        if am is not None: am = am.to(device)
        y = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            enc = _encode_any(model, x, attention_mask=am)
            logits = head_cls(enc)
            loss_ce = ce_crit(logits, y)

            v1 = batch["input_ids_view1"].to(device)
            v2 = batch["input_ids_view2"].to(device)
            am1 = batch["attention_mask_view1"].to(device)
            am2 = batch["attention_mask_view2"].to(device)

            z1 = cml_head(_encode_any(model, v1, attention_mask=am1))
            z2 = cml_head(_encode_any(model, v2, attention_mask=am2))
            z = torch.cat([z1, z2], dim=0)
            y2 = torch.cat([y, y], dim=0)

            loss_supcon = supervised_contrastive_loss(z, y2, temperature=temperature)
            loss = ce_weight * loss_ce + supcon_weight * loss_supcon

        if use_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head_cls.parameters()) + list(cml_head.parameters()), 1.0
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head_cls.parameters()) + list(cml_head.parameters()), 1.0
            )
            optimizer.step()

        all_logits.append(logits.detach())
        all_labels.append(y.detach())
        ce_losses.append(float(loss_ce.item()))
        supcon_losses.append(float(loss_supcon.item()))

    logits_cat = torch.cat(all_logits) if all_logits else torch.empty(0, device=device)
    labels_cat = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long, device=device)

    with torch.no_grad():
        top1 = float("nan")
        macro_f1 = float("nan")
        ece_cls = float("nan")
        if logits_cat.numel() > 0:
            preds = logits_cat.argmax(dim=-1)
            top1 = cls_top1_accuracy(_to_numpy(preds), _to_numpy(labels_cat))
            macro_f1 = cls_macro_f1(_to_numpy(preds), _to_numpy(labels_cat))
            ece_cls = cls_ece_from_logits(logits_cat, labels_cat)

    out = {
        "loss_ce": _safe_mean(ce_losses),
        "loss_supcon": _safe_mean(supcon_losses),
        "top1_accuracy": top1,
        "macro_f1_cls": macro_f1,
        "ece_cls": ece_cls,
    }
    out["task_top5"] = _select_top5_for_task("ce_supcon", out)
    return out

@torch.no_grad()
def eval_cls_ce_supcon_epoch(
    model: torch.nn.Module,
    head_cls: torch.nn.Module,
    cml_head: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval(); head_cls.eval(); cml_head.eval()

    all_logits: List[Tensor] = []
    all_labels: List[Tensor] = []

    for batch in dataloader:
        x = (batch.get("input_ids") or batch.get("kmer_input_ids")).to(device)
        am = batch.get("attention_mask") or batch.get("kmer_attention_mask")
        if am is not None: am = am.to(device)
        y = batch["labels"].to(device)

        enc = _encode_any(model, x, attention_mask=am)
        logits = head_cls(enc)

        all_logits.append(logits.detach())
        all_labels.append(y.detach())

    logits_cat = torch.cat(all_logits) if all_logits else torch.empty(0, device=device)
    labels_cat = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long, device=device)

    top1 = float("nan")
    macro_f1 = float("nan")
    ece_cls = float("nan")
    if logits_cat.numel() > 0:
        preds = logits_cat.argmax(dim=-1)
        top1 = cls_top1_accuracy(_to_numpy(preds), _to_numpy(labels_cat))
        macro_f1 = cls_macro_f1(_to_numpy(preds), _to_numpy(labels_cat))
        ece_cls = cls_ece_from_logits(logits_cat, labels_cat)

    out = {
        "loss_ce": float("nan"),
        "loss_supcon": float("nan"),
        "top1_accuracy": top1,
        "macro_f1_cls": macro_f1,
        "ece_cls": ece_cls,
    }
    out["task_top5"] = _select_top5_for_task("ce_supcon", out)
    return out


@torch.no_grad()
def _cosine_stats(z1: Tensor, z2: Tensor) -> Tuple[float, float]:
    pos_cos = torch.nn.functional.cosine_similarity(z1, z2, dim=-1)
    pos_mean = float(pos_cos.mean().item())
    if z2.size(0) > 1:
        neg_cos = torch.nn.functional.cosine_similarity(z1, z2.roll(1, dims=0), dim=-1)
        neg_mean = float(neg_cos.mean().item())
    else:
        neg_mean = float("nan")
    return pos_mean, neg_mean

def _alignment_uniformity(z1: Tensor, z2: Tensor) -> float:
    with torch.no_grad():
        return float(torch.norm(z1 - z2, dim=-1).mean().item())

def train_cml_epoch(
    model: torch.nn.Module,
    cml_head: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 0.07,
    *,
    mixed_precision: bool = False,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, Any]:
    model.train(); cml_head.train()

    use_amp = bool(mixed_precision and torch.cuda.is_available())
    scaler = scaler if scaler is not None else GradScaler(enabled=use_amp)

    all_pos = []; all_neg = []; all_norms = []; losses = []

    for batch in dataloader:
        x1 = (batch.get("input_ids") or batch.get("kmer_input_ids")).to(device)
        x2 = batch.get("input_ids_view2").to(device) if "input_ids_view2" in batch else x1.clone()
        am1 = batch.get("attention_mask") or batch.get("kmer_attention_mask")
        am2 = batch.get("attention_mask_view2")
        if am1 is not None: am1 = am1.to(device)
        if am2 is not None: am2 = am2.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            z1 = cml_head(_encode_any(model, x1, attention_mask=am1))
            z2 = cml_head(_encode_any(model, x2, attention_mask=am2))

            z1 = torch.nn.functional.normalize(z1, dim=-1)
            z2 = torch.nn.functional.normalize(z2, dim=-1)
            z = torch.cat([z1, z2], dim=0)
            y = torch.arange(z1.size(0), device=z.device).repeat(2)

            sim = torch.matmul(z, z.t()) / max(temperature, 1e-8)
            N = z.size(0)
            self_mask = torch.eye(N, dtype=torch.bool, device=z.device)
            sim = sim.masked_fill(self_mask, -1e9)
            pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & (~self_mask)
            log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
            loss = -(log_prob[pos_mask]).mean()

        if use_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(cml_head.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(cml_head.parameters()), 1.0)
            optimizer.step()

        losses.append(float(loss.item()))
        pos_mean, neg_mean = _cosine_stats(z1, z2)
        all_pos.append(pos_mean)
        all_neg.append(neg_mean)
        all_norms.append(float(z.norm(dim=-1).mean().item()))

    metrics = {
        "contrastive_loss": _safe_mean(losses),
        "pos_cosine_mean": _safe_mean(all_pos),
        "neg_cosine_mean": _safe_mean(all_neg),
        "embedding_norm_mean": _safe_mean(all_norms),
        "alignment_uniformity": float("nan"),
    }
    metrics["task_top5"] = _select_top5_for_task("cml", metrics)
    return metrics

@torch.no_grad()
def eval_cml_epoch(
    model: torch.nn.Module,
    cml_head: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    temperature: float = 0.07,
) -> Dict[str, Any]:
    model.eval(); cml_head.eval()
    all_pos = []; all_neg = []; all_norms = []

    for batch in dataloader:
        x1 = (batch.get("input_ids") or batch.get("kmer_input_ids")).to(device)
        x2 = batch.get("input_ids_view2").to(device) if "input_ids_view2" in batch else x1.clone()
        am1 = batch.get("attention_mask") or batch.get("kmer_attention_mask")
        am2 = batch.get("attention_mask_view2")
        if am1 is not None: am1 = am1.to(device)
        if am2 is not None: am2 = am2.to(device)

        z1 = torch.nn.functional.normalize(cml_head(_encode_any(model, x1, attention_mask=am1)), dim=-1)
        z2 = torch.nn.functional.normalize(cml_head(_encode_any(model, x2, attention_mask=am2)), dim=-1)

        pos_mean, neg_mean = _cosine_stats(z1, z2)
        all_pos.append(pos_mean)
        all_neg.append(neg_mean)
        all_norms.append(float(torch.cat([z1, z2], dim=0).norm(dim=-1).mean().item()))

    metrics = {
        "contrastive_loss": float("nan"),
        "pos_cosine_mean": _safe_mean(all_pos),
        "neg_cosine_mean": _safe_mean(all_neg),
        "embedding_norm_mean": _safe_mean(all_norms),
        "alignment_uniformity": float("nan"),
    }
    metrics["task_top5"] = _select_top5_for_task("cml", metrics)
    return metrics

def promotion_gate_satisfied(
    prev_metrics: Dict[str, Any],
    curr_metrics: Dict[str, Any],
    gates: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    
    def _get(m, k, default=float("nan")):
        v = m.get(k, default)
        try:
            return float(v)
        except Exception:
            return default

    acc_min = float(gates.get("masked_accuracy_min_delta", 0.02))
    ece_max = float(gates.get("ece_max_delta", -0.01))             
    rec_min = float(gates.get("reconstruction_min_delta", 0.02))
    ent_eps = float(gates.get("entropy_stability_eps", 0.01))
    need_two = bool(gates.get("require_any_two", True))

    prev_acc = _get(prev_metrics, "accuracy", float("nan"))
    curr_acc = _get(curr_metrics, "accuracy", float("nan"))

    prev_ece = _get(prev_metrics, "expected_calibration_error", float("nan"))
    curr_ece = _get(curr_metrics, "expected_calibration_error", float("nan"))

    prev_recon = _get(prev_metrics, "seq_reconstruction_rate", float("nan"))
    curr_recon = _get(curr_metrics, "seq_reconstruction_rate", float("nan"))

    prev_ent = _get(prev_metrics, "avg_token_entropy", _get(prev_metrics, "entropy_mean", float("nan")))
    curr_ent = _get(curr_metrics, "avg_token_entropy", _get(curr_metrics, "entropy_mean", float("nan")))

    d_acc  = curr_acc  - prev_acc  if (not math.isnan(curr_acc)  and not math.isnan(prev_acc))  else float("nan")
    d_ece  = curr_ece  - prev_ece  if (not math.isnan(curr_ece)  and not math.isnan(prev_ece))  else float("nan")
    d_rec  = curr_recon - prev_recon if (not math.isnan(curr_recon) and not math.isnan(prev_recon)) else float("nan")
    d_ent  = curr_ent  - prev_ent  if (not math.isnan(curr_ent)  and not math.isnan(prev_ent))  else float("nan")

    c_acc  = (not math.isnan(d_acc)) and (d_acc >= acc_min)
    c_ece  = (not math.isnan(d_ece)) and (d_ece <= ece_max)        
    c_rec  = (not math.isnan(d_rec)) and (d_rec >= rec_min)
    c_ent  = (not math.isnan(d_ent)) and (abs(d_ent) <= ent_eps)  

    satisfied = [c_acc, c_ece, c_rec, c_ent]
    num_ok = sum(1 for c in satisfied if c)

    passed = (num_ok >= 2) if need_two else any(satisfied)

    details = {
        "deltas": {
            "accuracy": d_acc,
            "ece": d_ece,
            "seq_reconstruction_rate": d_rec,
            "avg_token_entropy": d_ent,
        },
        "criteria": {
            "accuracy_up": c_acc,
            "ece_down": c_ece,
            "reconstruction_up": c_rec,
            "entropy_stable": c_ent,
        },
        "num_satisfied": num_ok,
        "require_any_two": need_two,
        "passed": passed,
    }
    return passed, details


def gate_summary(details: Dict[str, Any]) -> str:
    """
    Pretty string for logs/dashboards.
    """
    d = details.get("deltas", {})
    c = details.get("criteria", {})
    return (
        f"[gate] satisfied={details.get('num_satisfied', 0)} "
        f"(need_two={details.get('require_any_two', True)} => passed={details.get('passed', False)}) | "
        f"acc={d.get('accuracy')} (ok={c.get('accuracy_up')}), "
        f"ECE={d.get('ece')} (ok={c.get('ece_down')}), "
        f"recon={d.get('seq_reconstruction_rate')} (ok={c.get('reconstruction_up')}), "
        f"entropy={d.get('avg_token_entropy')} (ok={c.get('entropy_stable')})"
    )
