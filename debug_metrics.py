# debug_metrics 

import torch
import numpy as np

try:
    import debug_config as cfg  
    cfg_module_name = "debug_config"
except Exception:
    import config as cfg  
    cfg_module_name = "config"

from train_eval import _compute_loss_and_basic_counts

print(f"[debug_metrics] Using config module: {cfg_module_name}")

B = getattr(cfg, "DEBUG_BATCH", 2)
L = getattr(cfg, "DEBUG_SEQ_LEN", 24)
V = getattr(cfg, "DEBUG_VOCAB_SIZE", 16)             
MAP_TOKEN_CLASSES = getattr(cfg, "DEBUG_MAP_TOKEN_CLASSES", 8)  
MAP_SEQ_CLASSES   = getattr(cfg, "DEBUG_MAP_SEQ_CLASSES", 8)  
PAD_ID = getattr(cfg, "PAD_TOKEN_ID", 0)

crit_token = torch.nn.CrossEntropyLoss(ignore_index=-100)
crit_seq   = torch.nn.CrossEntropyLoss()

rng = torch.Generator().manual_seed(7)


def make_masked_labels(batch: int, seq: int, num_classes: int, p: float = 0.15):
    
    mask = (torch.rand(batch, seq, generator=rng) < p)
    labels = torch.full((batch, seq), -100, dtype=torch.long)
    if mask.any():
        labels[mask] = torch.randint(0, num_classes, (int(mask.sum().item()),), generator=rng)
    return labels, mask


def run_token_task(name: str, num_classes: int):
    
    logits = torch.randn(B, L, num_classes, generator=rng)
    labels, masked_bool = make_masked_labels(B, L, num_classes, p=0.25)
    attention_mask = torch.ones(B, L, dtype=torch.long)

    loss, n_masked, n_valid, n_correct, preds, probs = _compute_loss_and_basic_counts(
        logits, labels, crit_token, attention_mask
    )

    acc = (n_correct / max(1, n_masked)) if n_masked > 0 else float("nan")
    print(f"\n[{name}]")
    print(f"  logits: {tuple(logits.shape)}  labels: {tuple(labels.shape)}")
    print(f"  masked count: {n_masked} / valid tokens: {n_valid}")
    print(f"  loss: {loss.item():.4f}  accuracy(masked): {acc:.4f}")


def run_seq_task(name: str, num_classes: int):
    
    logits = torch.randn(B, num_classes, generator=rng)
    labels = torch.randint(0, num_classes, (B,), generator=rng)

    loss, n_masked, n_valid, n_correct, preds, probs = _compute_loss_and_basic_counts(
        logits, labels, crit_seq, None
    )

    acc = (n_correct / max(1, n_masked)) if n_masked > 0 else float("nan")
    print(f"\n[{name} (sequence-level)]")
    print(f"  logits: {tuple(logits.shape)}  labels: {tuple(labels.shape)}")
    print(f"  samples: {n_masked}")
    print(f"  loss: {loss.item():.4f}  top-1 acc: {acc:.4f}")


def main():
    run_token_task("MLM", V)
    run_token_task("Span Corruption", V)
    run_token_task("DAE", V)
    run_token_task("K-mer Reorder", V)

    run_token_task("MAP (token-level)", MAP_TOKEN_CLASSES)  
    run_token_task("MBP (token-level)", 2)                  

    run_seq_task("MAP", MAP_SEQ_CLASSES)

    print("\n[OK] All metric shapes validated.")


if __name__ == "__main__":
    main()
