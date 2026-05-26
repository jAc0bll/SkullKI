"""Train the BeliefNet from a self-play dataset.

Loss: binary cross-entropy on non-self rows (the perspective player's own
hand is trivially derivable from the observation so we mask it out).

Usage:
    python train/train_belief.py --data train/data/belief_v1.npz \
                                 --epochs 30 --batch-size 2048 \
                                 --out train/checkpoints/belief.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from belief_model import BeliefNet, ENC_DIM, N_PLAYERS, N_CARDS


def load(path: Path):
    z = np.load(path)
    feats       = torch.from_numpy(z["features"]).float()
    hands       = torch.from_numpy(z["hands"]).float()        # (N, 4, 70)
    perspective = torch.from_numpy(z["perspective"]).long()   # (N,)
    assert feats.shape[1]    == ENC_DIM
    assert hands.shape[1:]   == (N_PLAYERS, N_CARDS)
    return feats, hands, perspective


def perspective_mask(perspective: torch.Tensor) -> torch.Tensor:
    """Returns (B, N_PLAYERS, N_CARDS) bool mask with True wherever the
    cell should contribute to the loss (= all non-self entries)."""
    B = perspective.shape[0]
    players = torch.arange(N_PLAYERS, device=perspective.device).view(1, N_PLAYERS)
    not_self = (players != perspective.view(B, 1))           # (B, N_PLAYERS)
    return not_self.unsqueeze(-1).expand(-1, -1, N_CARDS)


def run_epoch(model, loader, optim, device, train: bool):
    if train: model.train()
    else:     model.eval()

    total_loss = 0.0
    total_correct = 0
    total_cells = 0
    total_positive = 0
    total_positive_correct = 0
    n_total = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for feats, hands, perspective in loader:
            feats       = feats.to(device,       non_blocking=True)
            hands       = hands.to(device,       non_blocking=True)
            perspective = perspective.to(device, non_blocking=True)

            logits = model(feats)  # (B, 4, 70)
            mask   = perspective_mask(perspective)
            loss_per_cell = F.binary_cross_entropy_with_logits(logits, hands, reduction='none')
            denom = mask.sum().clamp_min(1).float()
            loss  = (loss_per_cell * mask).sum() / denom

            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()

            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5)
                truth = hands > 0.5
                correct = (pred == truth) & mask
                total_correct += correct.sum().item()
                total_cells   += mask.sum().item()
                pos_mask = truth & mask
                total_positive += pos_mask.sum().item()
                total_positive_correct += (pred & pos_mask).sum().item()

            B = feats.size(0)
            total_loss += loss.item() * B
            n_total    += B

    avg_loss = total_loss / max(1, n_total)
    acc      = total_correct / max(1, total_cells)
    recall   = total_positive_correct / max(1, total_positive)
    return avg_loss, acc, recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",         type=str,   default="train/data/belief_v1.npz")
    ap.add_argument("--out",          type=str,   default="train/checkpoints/belief.pt")
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch-size",   type=int,   default=2048)
    ap.add_argument("--lr",           type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac",     type=float, default=0.1)
    ap.add_argument("--hidden",       type=int,   default=512)
    ap.add_argument("--device",       type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed",         type=int,   default=0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / args.data
    out_path  = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {args.device}")
    print(f"Loading {data_path}")
    feats, hands, perspective = load(data_path)
    N = feats.shape[0]
    print(f"Samples: {N:,}")

    perm = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed))
    n_val = int(N * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    pin = (args.device != "cpu")
    train_ds = TensorDataset(feats[train_idx], hands[train_idx], perspective[train_idx])
    val_ds   = TensorDataset(feats[val_idx],   hands[val_idx],   perspective[val_idx])
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  pin_memory=pin)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, pin_memory=pin)

    model = BeliefNet(hidden=args.hidden).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params (hidden={args.hidden})")
    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}\n")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc, tr_rec = run_epoch(model, train_ld, optim, args.device, train=True)
        va_loss, va_acc, va_rec = run_epoch(model, val_ld,   optim, args.device, train=False)
        sched.step()
        marker = ""
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(),
                        "hidden": args.hidden,
                        "enc_dim": ENC_DIM,
                        "n_players": N_PLAYERS,
                        "n_cards":   N_CARDS}, out_path)
            marker = " *"
        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train loss={tr_loss:.4f} acc={tr_acc*100:5.2f}% rec={tr_rec*100:5.2f}%   "
              f"val loss={va_loss:.4f} acc={va_acc*100:5.2f}% rec={va_rec*100:5.2f}%  "
              f"lr={sched.get_last_lr()[0]:.2e}  {time.perf_counter()-t0:.1f}s{marker}")

    print(f"\nBest val loss: {best_val:.4f}  ->  {out_path}")


if __name__ == "__main__":
    main()
