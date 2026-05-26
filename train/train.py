"""Behaviour-cloning trainer: fit PolicyValueNet to ISMCTS targets.

Usage:
    python train/train.py --data train/data/selfplay.npz \
                          --epochs 20 --batch-size 1024 --lr 1e-3 \
                          --out train/checkpoints/bc.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import PolicyValueNet, masked_log_softmax, ENC_DIM, ACTION_DIM, N_PLAYERS


def load_dataset(path: Path):
    z = np.load(path)
    feats  = torch.from_numpy(z["features"]).float()
    policy = torch.from_numpy(z["policy"]).float()
    legal  = torch.from_numpy(z["legal"]).bool()
    value  = torch.from_numpy(z["value"]).float()
    assert feats.shape[1]  == ENC_DIM,    f"features shape {feats.shape}"
    assert policy.shape[1] == ACTION_DIM, f"policy shape {policy.shape}"
    assert legal.shape  == policy.shape
    assert value.shape[1] == N_PLAYERS,   f"value shape {value.shape}  (re-run data_gen with new code)"
    return feats, policy, legal, value


def train_one_epoch(model, loader, optim, device, value_weight: float):
    model.train()
    total_pol, total_val, n = 0.0, 0.0, 0
    for feats, policy_t, legal, value_t in loader:
        feats    = feats.to(device, non_blocking=True)
        policy_t = policy_t.to(device, non_blocking=True)
        legal    = legal.to(device, non_blocking=True)
        value_t  = value_t.to(device, non_blocking=True)

        policy_logits, value_pred = model(feats)
        log_probs = masked_log_softmax(policy_logits, legal)

        # Masked cross-entropy: -sum_a target(a) * log_p(a), where target=0 on illegal.
        # Replace any -inf where target is 0 with 0 contribution.
        ce = -(policy_t * log_probs.clamp_min(-100.0)).sum(dim=-1).mean()
        mse = F.mse_loss(value_pred, value_t)

        loss = ce + value_weight * mse
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optim.step()

        bs = feats.size(0)
        total_pol += ce.item() * bs
        total_val += mse.item() * bs
        n += bs
    return total_pol / n, total_val / n


@torch.no_grad()
def eval_split(model, loader, device, value_weight: float):
    model.eval()
    total_pol, total_val, top1_correct, n = 0.0, 0.0, 0, 0
    for feats, policy_t, legal, value_t in loader:
        feats    = feats.to(device); policy_t = policy_t.to(device)
        legal    = legal.to(device); value_t  = value_t.to(device)

        policy_logits, value_pred = model(feats)
        log_probs = masked_log_softmax(policy_logits, legal)

        ce  = -(policy_t * log_probs.clamp_min(-100.0)).sum(dim=-1).mean()
        mse = F.mse_loss(value_pred, value_t)

        # Top-1 agreement with ISMCTS argmax on legal moves
        teacher_argmax = policy_t.argmax(dim=-1)
        student_argmax = log_probs.argmax(dim=-1)
        top1_correct += (teacher_argmax == student_argmax).sum().item()

        bs = feats.size(0)
        total_pol += ce.item() * bs
        total_val += mse.item() * bs
        n += bs
    return total_pol / n, total_val / n, top1_correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",        type=str,   default="train/data/selfplay.npz")
    ap.add_argument("--out",         type=str,   default="train/checkpoints/bc.pt")
    ap.add_argument("--epochs",      type=int,   default=20)
    ap.add_argument("--batch-size",  type=int,   default=1024)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--val-frac",    type=float, default=0.1)
    ap.add_argument("--hidden",      type=int,   default=512)
    ap.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed",        type=int,   default=0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / args.data
    out_path  = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {args.device}")
    print(f"Loading {data_path}")
    feats, policy, legal, value = load_dataset(data_path)
    N = feats.shape[0]
    print(f"Samples: {N:,}")

    # Train/val split
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed))
    n_val = int(N * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = TensorDataset(feats[train_idx], policy[train_idx], legal[train_idx], value[train_idx])
    val_ds   = TensorDataset(feats[val_idx],   policy[val_idx],   legal[val_idx],   value[val_idx])

    pin = (args.device != "cpu")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, pin_memory=pin)

    model = PolicyValueNet(hidden=args.hidden).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params (hidden={args.hidden})")
    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}\n")

    best_val_pol = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_pol, tr_val = train_one_epoch(model, train_loader, optim, args.device, args.value_weight)
        va_pol, va_val, va_top1 = eval_split(model, val_loader, args.device, args.value_weight)
        sched.step()

        marker = ""
        if va_pol < best_val_pol:
            best_val_pol = va_pol
            torch.save({"model": model.state_dict(),
                        "hidden": args.hidden,
                        "enc_dim": ENC_DIM,
                        "action_dim": ACTION_DIM}, out_path)
            marker = " *"

        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train pol={tr_pol:.4f} val={tr_val:.4f}   "
              f"val pol={va_pol:.4f} val={va_val:.4f}  top1={va_top1*100:5.2f}%  "
              f"lr={sched.get_last_lr()[0]:.2e}  {time.perf_counter()-t0:.1f}s{marker}")

    print(f"\nBest val policy loss: {best_val_pol:.4f}  ->  {out_path}")


if __name__ == "__main__":
    main()
