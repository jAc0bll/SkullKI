"""Timing diagnostic — finds the bottleneck in Deep CFR training."""
from __future__ import annotations
import multiprocessing
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from skull_king.training.cfr.buffers import AdvantageBuffer, StrategyBuffer
from skull_king.training.cfr.networks import AdvantageNet, StrategyNet
from skull_king.training.cfr.train import load_config
from skull_king.training.cfr.traversal import worker_init, worker_task

CONFIG = "cfr_config_v4_server.yaml"
cfg = load_config(CONFIG)
ctx = multiprocessing.get_context("fork")

device = torch.device("cpu")
hidden = tuple(cfg.net_hidden)
adv_net = AdvantageNet(hidden=hidden).to(device)
strat_net = StrategyNet(hidden=hidden).to(device)
adv_weights = {k: v.cpu() for k, v in adv_net.state_dict().items()}
strat_weights = {k: v.cpu() for k, v in strat_net.state_dict().items()}

tasks = [
    (i % cfg.n_players, 42 + i, cfg.n_players)
    for i in range(cfg.traversals_per_player * cfg.n_players)
]
print(f"\nConfig: {cfg.num_workers} workers | {len(tasks)} traversals/iter | "
      f"net={cfg.net_hidden} | steps={cfg.adv_train_steps}")
print("=" * 60)

# ── 1. Pool creation with EMPTY buffers ──────────────────────────
t0 = time.time()
pool = ctx.Pool(cfg.num_workers, initializer=worker_init,
                initargs=(adv_weights, cfg.n_players))
pool.close(); pool.join()
t_pool_empty = time.time() - t0
print(f"[1] Pool creation (empty buffers):   {t_pool_empty:.2f}s")

# ── 2. Traversal collection ───────────────────────────────────────
t0 = time.time()
pool = ctx.Pool(cfg.num_workers, initializer=worker_init,
                initargs=(adv_weights, cfg.n_players))
n_adv_total = n_strat_total = 0
all_adv_obs, all_adv_masks, all_adv_targets, all_adv_actions = [], [], [], []
all_strat_obs, all_strat_masks, all_strat_strats = [], [], []
for result in pool.imap_unordered(worker_task, tasks, chunksize=4):
    a_obs, a_masks, a_targets, a_actions, s_obs, s_masks, s_strats = result
    n_adv_total += a_obs.shape[0]
    n_strat_total += s_obs.shape[0]
    if a_obs.shape[0]:
        all_adv_obs.append(a_obs); all_adv_masks.append(a_masks)
        all_adv_targets.append(a_targets); all_adv_actions.append(a_actions)
    if s_obs.shape[0]:
        all_strat_obs.append(s_obs); all_strat_masks.append(s_masks)
        all_strat_strats.append(s_strats)
pool.close(); pool.join()
t_collect = time.time() - t0
print(f"[2] Traversal collection:            {t_collect:.2f}s  "
      f"({n_adv_total:,} adv samples, {n_strat_total:,} strat samples)")

# ── 3. Pool creation with FULL pre-allocated buffers ─────────────
adv_buf = AdvantageBuffer(capacity=cfg.adv_buffer_capacity)
strat_buf = StrategyBuffer(capacity=cfg.strat_buffer_capacity)
print(f"\n    Pre-allocated buffer RAM:")
print(f"      adv_buf : {adv_buf._obs.nbytes / 1e9:.2f} GB  (obs only)")
print(f"      strat_buf: {strat_buf._obs.nbytes / 1e9:.2f} GB  (obs only)")
total_gb = (adv_buf._obs.nbytes + adv_buf._masks.nbytes + adv_buf._targets.nbytes
            + strat_buf._obs.nbytes + strat_buf._masks.nbytes
            + strat_buf._strategies.nbytes) / 1e9
print(f"      total   : {total_gb:.2f} GB")

t0 = time.time()
pool = ctx.Pool(cfg.num_workers, initializer=worker_init,
                initargs=(adv_weights, cfg.n_players))
pool.close(); pool.join()
t_pool_full = time.time() - t0
print(f"\n[3] Pool creation (full buffers pre-alloc): {t_pool_full:.2f}s  "
      f"({'OK' if t_pool_full < 2 else 'BOTTLENECK'})")

# ── 4. Advantage net training ─────────────────────────────────────
if all_adv_obs:
    adv_buf.add_batch_vec(
        np.concatenate(all_adv_obs),
        np.concatenate(all_adv_masks),
        np.concatenate(all_adv_targets),
        np.concatenate(all_adv_actions),
    )

adv_opt = torch.optim.Adam(adv_net.parameters(), lr=cfg.adv_lr)
adv_net.train()
t0 = time.time()
for _ in range(cfg.adv_train_steps):
    obs, _, targets, actions = adv_buf.sample(cfg.adv_batch_size)
    obs_t = torch.FloatTensor(obs)
    tgt_t = torch.FloatTensor(targets)
    act_t = torch.LongTensor(actions)
    pred = adv_net(obs_t)
    loss = torch.nn.functional.mse_loss(
        pred.gather(1, act_t.unsqueeze(1)).squeeze(1),
        tgt_t.gather(1, act_t.unsqueeze(1)).squeeze(1),
    )
    adv_opt.zero_grad(); loss.backward()
    adv_opt.step()
t_adv = time.time() - t0
print(f"[4] Adv net training ({cfg.adv_train_steps} steps):   {t_adv:.2f}s")

# ── 5. Strategy net training ──────────────────────────────────────
if all_strat_obs:
    strat_buf.add_batch_vec(
        np.concatenate(all_strat_obs),
        np.concatenate(all_strat_masks),
        np.concatenate(all_strat_strats),
    )

strat_opt = torch.optim.Adam(strat_net.parameters(), lr=cfg.strat_lr)
strat_net.train()
t0 = time.time()
for _ in range(cfg.strat_train_steps):
    obs, masks, strategies = strat_buf.sample(cfg.strat_batch_size)
    obs_t = torch.FloatTensor(obs)
    mask_t = torch.BoolTensor(masks)
    strat_t = torch.FloatTensor(strategies)
    logits = strat_net(obs_t).masked_fill(~mask_t, float("-inf"))
    loss = -(strat_t * torch.log_softmax(logits, dim=-1)).nan_to_num(0).sum(-1).mean()
    strat_opt.zero_grad(); loss.backward()
    strat_opt.step()
t_strat = time.time() - t0
print(f"[5] Strat net training ({cfg.strat_train_steps} steps): {t_strat:.2f}s")

# ── Summary ───────────────────────────────────────────────────────
total = t_pool_full + t_collect + t_adv + t_strat
print(f"\n{'='*60}")
print(f"Estimated time/iter:  {total:.1f}s")
print(f"  pool_create: {t_pool_full:.1f}s ({100*t_pool_full/total:.0f}%)")
print(f"  collection:  {t_collect:.1f}s ({100*t_collect/total:.0f}%)")
print(f"  adv_train:   {t_adv:.1f}s ({100*t_adv/total:.0f}%)")
print(f"  strat_train: {t_strat:.1f}s ({100*t_strat/total:.0f}%)")
print(f"\nTorch threads (main): {torch.get_num_threads()}")
print(f"CPU count:            {multiprocessing.cpu_count()}")
