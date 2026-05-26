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

from model import build_model, masked_log_softmax, ENC_DIM, ACTION_DIM, N_PLAYERS


def load_dataset(paths):
    """Load one or more npz shards and concatenate. Each shard must share the
    same column layout (features/policy/legal/value)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    feats_l, policy_l, legal_l, value_l = [], [], [], []
    for p in paths:
        z = np.load(p)
        assert z["features"].shape[1]  == ENC_DIM,    f"features shape {z['features'].shape} in {p}"
        assert z["policy"].shape[1]    == ACTION_DIM, f"policy shape {z['policy'].shape} in {p}"
        assert z["legal"].shape == z["policy"].shape
        assert z["value"].shape[1] == N_PLAYERS, f"value shape {z['value'].shape} in {p} (re-run data_gen)"
        feats_l.append(z["features"]); policy_l.append(z["policy"])
        legal_l.append(z["legal"]);    value_l.append(z["value"])
    feats  = torch.from_numpy(np.concatenate(feats_l,  axis=0)).float()
    policy = torch.from_numpy(np.concatenate(policy_l, axis=0)).float()
    legal  = torch.from_numpy(np.concatenate(legal_l,  axis=0)).bool()
    value  = torch.from_numpy(np.concatenate(value_l,  axis=0)).float()
    return feats, policy, legal, value


def train_one_epoch(model, loader, optim, device, value_weight: float,
                    scaler=None, epoch: int | None = None,
                    total_epochs: int | None = None):
    """If `scaler` is a GradScaler, runs forward+backward under autocast
    (mixed precision). Otherwise plain FP32."""
    model.train()
    total_pol, total_val, n = 0.0, 0.0, 0
    n_batches = len(loader)
    log_every = max(1, n_batches // 20)
    tag = f"epoch {epoch:>3}/{total_epochs}" if epoch is not None else "train"
    t_start = time.perf_counter()
    use_amp = scaler is not None
    for i, (feats, policy_t, legal, value_t) in enumerate(loader, 1):
        feats    = feats.to(device, non_blocking=True)
        policy_t = policy_t.to(device, non_blocking=True)
        legal    = legal.to(device, non_blocking=True)
        value_t  = value_t.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            policy_logits, value_pred = model(feats)
            # log-softmax in fp32 for numerical stability under fp16 logits.
            log_probs = masked_log_softmax(policy_logits.float(), legal)
            ce  = -(policy_t * log_probs.clamp_min(-100.0)).sum(dim=-1).mean()
            mse = F.mse_loss(value_pred.float(), value_t)
            loss = ce + value_weight * mse

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()

        bs = feats.size(0)
        total_pol += ce.item() * bs
        total_val += mse.item() * bs
        n += bs
        if i % log_every == 0 or i == n_batches:
            pct  = i / n_batches * 100
            secs = time.perf_counter() - t_start
            eta  = secs / i * (n_batches - i)
            print(f"  {tag}  {i:>5}/{n_batches}  ({pct:5.1f}%)  "
                  f"pol={total_pol/n:.4f}  val={total_val/n:.4f}  "
                  f"{secs:5.1f}s  eta {eta:5.1f}s",
                  flush=True)
    return total_pol / n, total_val / n


@torch.no_grad()
def eval_split(model, loader, device, value_weight: float, use_amp: bool = False):
    model.eval()
    total_pol, total_val, top1_correct, n = 0.0, 0.0, 0, 0
    for feats, policy_t, legal, value_t in loader:
        feats    = feats.to(device, non_blocking=True)
        policy_t = policy_t.to(device, non_blocking=True)
        legal    = legal.to(device, non_blocking=True)
        value_t  = value_t.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            policy_logits, value_pred = model(feats)
        log_probs = masked_log_softmax(policy_logits.float(), legal)
        ce  = -(policy_t * log_probs.clamp_min(-100.0)).sum(dim=-1).mean()
        mse = F.mse_loss(value_pred.float(), value_t)

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
    ap.add_argument("--data",        type=str,   nargs="+", default=["train/data/selfplay.npz"],
                    help="One or more npz shards (concatenated). Used by the AlphaZero "
                         "loop to pass a replay window of the last N iterations.")
    ap.add_argument("--out",         type=str,   default="train/checkpoints/bc.pt")
    ap.add_argument("--init",        type=str,   default="",
                    help="Optional PyTorch checkpoint to warm-start from. "
                         "Must match --hidden. Used by AlphaZero loop to inherit "
                         "the previous best model instead of re-training from scratch.")
    ap.add_argument("--epochs",      type=int,   default=20)
    ap.add_argument("--batch-size",  type=int,   default=1024)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--val-frac",    type=float, default=0.1)
    ap.add_argument("--arch",        type=str,   default="v1", choices=["v1", "v2"],
                    help="v1 = original 2-layer MLP (~1.9M params at hidden=1024). "
                         "v2 = residual MLP (~25M params at hidden=2048, num_blocks=3).")
    ap.add_argument("--hidden",      type=int,   default=512)
    ap.add_argument("--num-blocks",  type=int,   default=3,
                    help="Only used for --arch v2: number of residual blocks.")
    ap.add_argument("--dropout",     type=float, default=0.0)
    ap.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--num-workers", type=int,   default=4,
                    help="DataLoader worker processes. Overlaps host-side shuffling "
                         "and pinned-memory transfer with GPU compute.")
    ap.add_argument("--amp",         action=argparse.BooleanOptionalAction, default=None,
                    help="Mixed precision (fp16 autocast + GradScaler). "
                         "Default: on for cuda, off otherwise. Use --no-amp to force off.")
    args = ap.parse_args()
    if args.amp is None:
        args.amp = args.device.startswith("cuda")

    repo_root = Path(__file__).resolve().parent.parent
    data_paths = [repo_root / d for d in args.data]
    out_path   = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {args.device}")
    if len(data_paths) == 1:
        print(f"Loading {data_paths[0]}")
    else:
        print(f"Loading {len(data_paths)} shards (replay buffer):")
        for p in data_paths:
            print(f"  {p}")
    feats, policy, legal, value = load_dataset(data_paths)
    N = feats.shape[0]
    print(f"Samples: {N:,}")

    # Train/val split
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed))
    n_val = int(N * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = TensorDataset(feats[train_idx], policy[train_idx], legal[train_idx], value[train_idx])
    val_ds   = TensorDataset(feats[val_idx],   policy[val_idx],   legal[val_idx],   value[val_idx])

    pin = (args.device != "cpu")
    nw  = args.num_workers
    persistent = nw > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=pin, num_workers=nw,
                              persistent_workers=persistent)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              pin_memory=pin, num_workers=nw,
                              persistent_workers=persistent)

    model = build_model(args.arch, hidden=args.hidden,
                        num_blocks=args.num_blocks, dropout=args.dropout).to(args.device)
    if args.init:
        init_path = repo_root / args.init
        ck = torch.load(init_path, map_location=args.device, weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        ck_arch   = ck.get("arch",   "v1") if isinstance(ck, dict) else "v1"
        ck_hidden = ck.get("hidden")        if isinstance(ck, dict) else None
        if ck_arch != args.arch:
            raise ValueError(f"--init arch={ck_arch} mismatches --arch {args.arch}.")
        if ck_hidden is not None and ck_hidden != args.hidden:
            raise ValueError(
                f"--init hidden={ck_hidden} mismatches --hidden {args.hidden}. "
                f"Either pass the matching --hidden or retrain from scratch.")
        model.load_state_dict(sd)
        print(f"Warm-start from {init_path}")
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    n_params = sum(p.numel() for p in model.parameters())
    arch_desc = f"arch={args.arch} hidden={args.hidden}"
    if args.arch == "v2":
        arch_desc += f" num_blocks={args.num_blocks} dropout={args.dropout}"
    print(f"Model: {n_params/1e6:.2f}M params ({arch_desc})")
    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  "
          f"DataLoader workers: {nw}  AMP: {'on' if args.amp else 'off'}\n")

    # Establish baseline (so a warm-started model that gets worse during training
    # still leaves the original at --out, and the loop doesn't crash on missing file).
    init_val_pol, init_val_v, init_top1 = eval_split(model, val_loader, args.device, args.value_weight,
                                                     use_amp=args.amp)
    print(f"epoch   0/{args.epochs}  init                            "
          f"val pol={init_val_pol:.4f} val={init_val_v:.4f}  top1={init_top1*100:5.2f}%")
    torch.save({"model": model.state_dict(),
                "arch": args.arch,
                "hidden": args.hidden,
                "num_blocks": args.num_blocks,
                "dropout": args.dropout,
                "enc_dim": ENC_DIM,
                "action_dim": ACTION_DIM}, out_path)
    best_val_pol = init_val_pol
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_pol, tr_val = train_one_epoch(model, train_loader, optim, args.device, args.value_weight,
                                         scaler=scaler, epoch=epoch, total_epochs=args.epochs)
        va_pol, va_val, va_top1 = eval_split(model, val_loader, args.device, args.value_weight,
                                             use_amp=args.amp)
        sched.step()

        marker = ""
        if va_pol < best_val_pol:
            best_val_pol = va_pol
            torch.save({"model": model.state_dict(),
                        "arch": args.arch,
                        "hidden": args.hidden,
                        "num_blocks": args.num_blocks,
                        "dropout": args.dropout,
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
