"""Batched MCTS for AlphaZero-Skull-King.

Design:
  - One tree per parallel game; all games run simulations in lockstep.
  - At each simulation step the runner:
      1. Each tree traverses (PUCT-select) down to a leaf
      2. The leaf engine clones are stepped through opponent moves until
         it's seat-0's turn again (using policy-net sampled actions)
      3. All leaves are batch-evaluated by the network in ONE GPU call
      4. Visit/value backups are done per-tree

  This keeps every network forward at batch=N_GAMES instead of batch=1,
  which is critical for GPU utilization with deep MCTS.

  Opponents play "implicitly" — we don't run MCTS for them. Their actions
  are sampled from the same shared policy network. This is the standard
  AlphaZero adaptation for hidden-information games where opponents share
  the agent's strategy distribution.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from skull_king.env.skull_king_env import ACTION_SPACE_SIZE
from skull_king.game_state import GamePhase
from skull_king.training.rebel.public_belief_state import encode_pbs_batch
from skull_king.training.rebel.subgame import (
    _action_to_card,
    _build_action_mask,
    _fast_clone_engine,
)

if TYPE_CHECKING:
    from skull_king.engine import GameEngine
    from skull_king.training.alphazero.networks import AlphaZeroNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bid_mask_for_engine(eng) -> np.ndarray:
    m = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for b in range(eng._round + 1):
        m[b] = True
    return m


def _legal_mask(eng) -> np.ndarray:
    """Legal-action mask for the current player + phase."""
    return _bid_mask_for_engine(eng) if eng._phase == GamePhase.BIDDING else _build_action_mask(eng)


def _apply_action(eng, player: int, action: int) -> None:
    if eng._phase == GamePhase.BIDDING:
        eng.place_bid_no_state(player, action)
    else:
        card, tm = _action_to_card(action, eng)
        eng.play_card_no_state(player, card, tm)


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class MCTSNode:
    """Single node in an AlphaZero MCTS tree."""
    __slots__ = ("prior", "visits", "value_sum", "children", "expanded")

    def __init__(self, prior: float = 0.0) -> None:
        self.prior: float = prior
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.children: Optional[dict[int, "MCTSNode"]] = None
        self.expanded: bool = False

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


def _puct_select(node: MCTSNode, c_puct: float) -> int:
    """Pick the child action maximising the PUCT formula."""
    parent_visits = node.visits
    sqrt_parent = math.sqrt(parent_visits)
    best_action = -1
    best_score = -math.inf
    assert node.children is not None
    for a, child in node.children.items():
        u = c_puct * child.prior * sqrt_parent / (1.0 + child.visits)
        score = child.q + u
        if score > best_score:
            best_score = score
            best_action = a
    return best_action


def _add_dirichlet_noise(node: MCTSNode, alpha: float, eps: float, rng: np.random.Generator) -> None:
    """Mix Dirichlet noise into root priors to encourage exploration."""
    if node.children is None:
        return
    actions = list(node.children.keys())
    noise = rng.dirichlet([alpha] * len(actions))
    for a, n in zip(actions, noise):
        node.children[a].prior = (1.0 - eps) * node.children[a].prior + eps * float(n)


# ---------------------------------------------------------------------------
# Batched evaluator
# ---------------------------------------------------------------------------

def _batched_forward(
    network: "AlphaZeroNet",
    engines: list,
    players: list[int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Run a single batched forward pass over (engine, player) pairs.

    Returns (priors[B, A], values[B], masks[B, A]).
    """
    B = len(engines)
    encs = encode_pbs_batch(engines, players)              # [B, pbs_size]
    masks = np.stack([_legal_mask(eng) for eng in engines])  # [B, A]

    enc_t  = torch.from_numpy(encs).to(device)
    mask_t = torch.from_numpy(masks).to(device)
    with torch.no_grad():
        log_probs, value = network(enc_t, mask_t)
        priors = log_probs.exp().cpu().numpy()
        values = value.cpu().numpy()
    return priors, values, masks


def _advance_opponents(
    engines: list,
    network: "AlphaZeroNet",
    agent_seat: int,
    device: torch.device,
    rng: np.random.Generator,
) -> np.ndarray:
    """For each engine, sample opponent actions from the network until it
    is the agent's turn (or terminal). Operates in batched lock-step.

    Returns a boolean array marking which engines are now terminal.
    """
    terminal = np.zeros(len(engines), dtype=bool)
    # Mark already-terminal engines
    for i, eng in enumerate(engines):
        if eng._phase == GamePhase.GAME_OVER:
            terminal[i] = True

    while True:
        # Identify games where it's an OPPONENT's turn
        pending: list[int] = []
        for i, eng in enumerate(engines):
            if terminal[i]:
                continue
            cp = eng._current_player_index()
            if cp != agent_seat:
                pending.append(i)
        if not pending:
            return terminal

        sub_engines = [engines[i] for i in pending]
        sub_players = [engines[i]._current_player_index() for i in pending]

        priors, _, masks = _batched_forward(network, sub_engines, sub_players, device)

        # Sample legal actions from policy
        for k, i in enumerate(pending):
            legal = np.where(masks[k])[0]
            if len(legal) == 0:
                terminal[i] = True
                continue
            p_legal = priors[k, legal]
            s = p_legal.sum()
            p_legal = p_legal / s if s > 1e-9 else np.full(len(legal), 1.0 / len(legal))
            action = int(rng.choice(legal, p=p_legal))
            try:
                _apply_action(engines[i], sub_players[k], action)
            except Exception:
                terminal[i] = True
                continue
            if engines[i]._phase == GamePhase.GAME_OVER:
                terminal[i] = True

    # unreachable
    return terminal


# ---------------------------------------------------------------------------
# Public MCTS driver
# ---------------------------------------------------------------------------

def run_batched_mcts(
    engines_root: list,
    network: "AlphaZeroNet",
    n_simulations: int,
    device: torch.device,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    rng: np.random.Generator,
    add_root_noise: bool = True,
    agent_seat: int = 0,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Run MCTS for a batch of games.

    Assumes the engines are all at a state where ``agent_seat`` is to act
    and the game is not over.

    Returns:
        policies : list of length B, each entry an [A]-shaped float array
                   of visit-count probabilities at the root.
        root_v   : [B] float array of estimated state values (root q).
    """
    B = len(engines_root)
    roots = [MCTSNode() for _ in range(B)]

    # Initial root expansion: query network once at the actual root state
    priors, root_values, root_masks = _batched_forward(
        network, engines_root, [agent_seat] * B, device
    )
    for i in range(B):
        legal = np.where(root_masks[i])[0]
        roots[i].children = {int(a): MCTSNode(prior=float(priors[i, a])) for a in legal}
        roots[i].expanded = True
        if add_root_noise and len(legal) > 0:
            _add_dirichlet_noise(roots[i], dirichlet_alpha, dirichlet_eps, rng)

    from skull_king.training.cfr.traversal import _utility_from_scores

    # Run simulations
    for _ in range(n_simulations):
        # Tree traversal in LOCKSTEP — all games descend together one tree-level
        # at a time so opponent advancement can be batched across games.
        paths: list[list[MCTSNode]] = [[r] for r in roots]
        clones = [_fast_clone_engine(eng) for eng in engines_root]
        leaf_values = np.zeros(B, dtype=np.float32)
        done = np.zeros(B, dtype=bool)        # finished selection (hit leaf/terminal)

        # Helper to mark a game as terminal and record its value
        def _finalize_terminal(i: int) -> None:
            scores = [p.total_score for p in clones[i]._players]
            leaf_values[i] = _utility_from_scores(scores, agent_seat)
            done[i] = True

        # Catch any games that are already terminal at the root
        for i in range(B):
            if clones[i]._phase == GamePhase.GAME_OVER:
                _finalize_terminal(i)

        # Level-by-level descent until all games hit a leaf or terminal
        while not done.all():
            # Phase 1: PUCT-select for each still-active game (CPU only)
            stepping = [i for i in range(B) if not done[i]]
            actions: dict[int, int] = {}
            for i in stepping:
                node = paths[i][-1]
                if not node.expanded:
                    done[i] = True                # found leaf to expand
                    continue
                if node.children is None or not node.children:
                    done[i] = True
                    continue
                a = _puct_select(node, c_puct)
                actions[i] = a

            if not actions:
                break

            # Phase 2: apply agent actions in CPU
            advancing: list[int] = []
            advancing_engs: list = []
            for i, a in actions.items():
                try:
                    _apply_action(clones[i], agent_seat, a)
                except Exception:
                    clones[i]._phase = GamePhase.GAME_OVER
                paths[i].append(paths[i][-1].children[a])
                if clones[i]._phase == GamePhase.GAME_OVER:
                    _finalize_terminal(i)
                else:
                    advancing.append(i)
                    advancing_engs.append(clones[i])

            # Phase 3: batched opponent advancement across all advancing games
            if advancing_engs:
                terminal_after = _advance_opponents(
                    advancing_engs, network, agent_seat, device, rng,
                )
                for k, i in enumerate(advancing):
                    if terminal_after[k]:
                        _finalize_terminal(i)
            # Loop continues: any game still not done has reached a new node;
            # next iteration's PUCT-select decides if it descends further.

        # Batch-expand all non-terminal leaves
        needs_expand: list[int] = []
        for i in range(B):
            # leaf_values is 0 for non-terminal leaves; expand them.
            if clones[i]._phase != GamePhase.GAME_OVER:
                needs_expand.append(i)

        if needs_expand:
            sub_engines = [clones[i] for i in needs_expand]
            priors, values, masks = _batched_forward(
                network, sub_engines, [agent_seat] * len(sub_engines), device
            )
            for k, i in enumerate(needs_expand):
                leaf = paths[i][-1]
                legal = np.where(masks[k])[0]
                leaf.children = {int(a): MCTSNode(prior=float(priors[k, a])) for a in legal}
                leaf.expanded = True
                leaf_values[i] = float(values[k])

        # Backup
        for i in range(B):
            v = leaf_values[i]
            for n in paths[i]:
                n.visits += 1
                n.value_sum += v

    # Extract policies + root values
    policies: list[np.ndarray] = []
    root_v = np.zeros(B, dtype=np.float32)
    for i, root in enumerate(roots):
        visits = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        if root.children:
            for a, c in root.children.items():
                visits[a] = c.visits
        total = visits.sum()
        if total > 0:
            visits /= total
        policies.append(visits)
        root_v[i] = root.q
    return policies, root_v
