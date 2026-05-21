"""FastAPI web interface for Skull King AI.

Run:
    uvicorn skull_king.web.app:app --reload
    # then open http://localhost:8000
"""
from __future__ import annotations

import glob
import os
import re
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from skull_king.agents import HeuristicAgent, MCTSAgent, RandomAgent
from skull_king.agents.base_agent import BaseAgent
from skull_king.cards import Card, CardType, Suit, TigressMode, NUM_ROUNDS, TRUMP_SUIT
from skull_king.engine import GameEngine
from skull_king.game_state import GamePhase
from skull_king.resolver import TrickResolver
from skull_king.tournament.runner import TournamentRunner
from skull_king.trick import PlayedCard
from skull_king.training.self_play_env import SB3AgentWrapper

# ---------------------------------------------------------------------------
# Lazy PPO model loader
# ---------------------------------------------------------------------------

_MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "skull_king")
)
_DEFAULT_MODEL_PATH = os.path.normpath(os.path.join(_MODELS_DIR, "final.zip"))
_ppo_model: Any = None
_active_model_path: str = _DEFAULT_MODEL_PATH


def _load_ppo_model() -> Any:
    global _ppo_model
    if _ppo_model is None:
        if not os.path.exists(_active_model_path):
            raise HTTPException(
                404,
                f"No trained model found at {_active_model_path}. "
                "Run: python -m skull_king.training.train",
            )
        from sb3_contrib import MaskablePPO
        _ppo_model = MaskablePPO.load(_active_model_path)
    return _ppo_model

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Skull King AI", version="1.0")

_STATIC = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC), name="static")

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

_SESSION_TTL = 3600  # 1 hour


@dataclass
class GameSession:
    session_id: str
    engine: GameEngine
    human_seat: int
    agents: list[Optional[BaseAgent]]   # None at human_seat
    log: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


_sessions: dict[str, GameSession] = {}


def _cleanup() -> None:
    cutoff = time.time() - _SESSION_TTL
    stale = [k for k, v in _sessions.items() if v.created_at < cutoff]
    for k in stale:
        del _sessions[k]


def _get_session(sid: str) -> GameSession:
    if sid not in _sessions:
        raise HTTPException(404, "Session not found or expired")
    return _sessions[sid]


# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------

_SUIT_EMOJI = {"BLACK": "♠", "YELLOW": "⭐", "GREEN": "♣", "PURPLE": "♦"}
_TYPE_EMOJI = {
    CardType.SKULL_KING: "💀",
    CardType.PIRATE: "🏴‍☠️",
    CardType.MERMAID: "🧜",
    CardType.ESCAPE: "🏳️",
    CardType.TIGRESS: "🐯",
}
_TYPE_WEIGHT = {
    CardType.SKULL_KING: 0.95,
    CardType.PIRATE: 0.75,
    CardType.TIGRESS: 0.45,
    CardType.MERMAID: 0.25,
    CardType.ESCAPE: 0.00,
}


def _card_to_dict(card: Card) -> dict:
    if card.card_type == CardType.NUMBERED:
        suit_name = card.suit.value.capitalize()
        emoji = _SUIT_EMOJI.get(card.suit.value, "")
        display = f"{suit_name} {card.value}"
    else:
        emoji = _TYPE_EMOJI.get(card.card_type, "")
        display = card.card_type.value.replace("_", " ").title()
    return {
        "type": card.card_type.value,
        "suit": card.suit.value if card.suit else None,
        "value": card.value,
        "display": display,
        "emoji": emoji,
    }


def _dict_to_card(data: dict) -> Card:
    ct = CardType(data["type"])
    suit = Suit(data["suit"]) if data.get("suit") else None
    value = data.get("value")
    return Card(card_type=ct, suit=suit, value=value)


# ---------------------------------------------------------------------------
# State serialisation
# ---------------------------------------------------------------------------

def _build_state(session: GameSession) -> dict:
    engine = session.engine
    state = engine.get_state()
    human = session.human_seat

    is_human_turn = (
        state.current_player_index == human
        and state.phase not in (GamePhase.GAME_OVER,)
    )

    legal_bids: list[int] = []
    legal_cards: list[dict] = []

    if is_human_turn and state.phase == GamePhase.BIDDING:
        legal_bids = list(range(state.round_number + 1))
    elif is_human_turn and state.phase == GamePhase.PLAYING:
        hand = list(state.player_states[human].hand)
        raw = TrickResolver.legal_plays(list(state.current_trick_cards), hand)
        seen: set = set()
        for c in raw:
            key = (c.card_type, c.suit, c.value)
            if key not in seen:
                seen.add(key)
                legal_cards.append(_card_to_dict(c))

    players_out = []
    for i, ps in enumerate(state.player_states):
        is_human = i == human
        agent = session.agents[i]
        name = "You" if is_human else (agent.name if agent else "AI")
        hand = [_card_to_dict(c) for c in ps.hand] if is_human else []
        players_out.append({
            "seat": i,
            "name": name,
            "is_human": is_human,
            "hand": hand,
            "hand_size": len(ps.hand),
            "bid": ps.bid,
            "tricks_won": ps.tricks_won_this_round,
            "bonus": ps.accumulated_bonus,
            "total_score": ps.total_score,
            "is_current": state.current_player_index == i,
        })

    trick_out = [
        {
            "card": _card_to_dict(pc.card),
            "seat": pc.player_index,
            "player_name": players_out[pc.player_index]["name"],
            "tigress_mode": pc.tigress_mode.value if pc.tigress_mode else None,
        }
        for pc in state.current_trick_cards
    ]

    winner = None
    if state.phase == GamePhase.GAME_OVER:
        scores = [(ps["total_score"], ps["name"]) for ps in players_out]
        best = max(s for s, _ in scores)
        winners = [n for s, n in scores if s == best]
        winner = winners[0] if len(winners) == 1 else f"Tie: {', '.join(winners)}"

    return {
        "session_id": session.session_id,
        "phase": state.phase.value,
        "round_number": state.round_number,
        "trick_number": state.trick_number,
        "human_seat": human,
        "current_player_seat": state.current_player_index,
        "is_human_turn": is_human_turn,
        "player_states": players_out,
        "current_trick": trick_out,
        "legal_bids": legal_bids,
        "legal_cards": legal_cards,
        "game_over": state.phase == GamePhase.GAME_OVER,
        "winner": winner,
        "log": session.log[-30:],
    }


# ---------------------------------------------------------------------------
# AI auto-advance
# ---------------------------------------------------------------------------

def _trick_winner(prev_tricks_won: list[int], new_state, fallback: int) -> int:
    """Return seat index of whoever just won a trick (same-round case)."""
    for i, (old, new) in enumerate(
        zip(prev_tricks_won, [ps.tricks_won_this_round for ps in new_state.player_states])
    ):
        if new > old:
            return i
    return fallback


def _advance_ai(session: GameSession) -> list[dict]:
    """Run all AI turns until it's the human's turn; return animation events."""
    engine = session.engine
    human = session.human_seat
    events: list[dict] = []

    for _ in range(2000):   # safety bound
        state = engine.get_state()
        if state.phase == GamePhase.GAME_OVER:
            scores = [ps.total_score for ps in state.player_states]
            best = max(scores)
            who = [session.agents[i].name if i != human else "You"
                   for i, s in enumerate(scores) if s == best]
            session.log.append(f"Game over! Winner: {', '.join(who)} ({best} pts)")
            break
        seat = state.current_player_index
        if seat == human:
            break
        agent = session.agents[seat]
        if agent is None:
            break

        agent.before_move(engine)
        prev_round = state.round_number
        prev_trick = state.trick_number
        prev_tricks_won = [ps.tricks_won_this_round for ps in state.player_states]

        if state.phase == GamePhase.BIDDING:
            bid = agent.bid(state, seat)
            engine.place_bid(seat, bid)
            session.log.append(f"{agent.name} bids {bid}")
            events.append({"type": "bid", "seat": seat, "player_name": agent.name, "bid": bid})
        else:
            card, mode = agent.play(state, seat)
            engine.play_card(seat, card, mode)
            ms = f" ({mode.value})" if mode else ""
            session.log.append(f"{agent.name} plays {_card_to_dict(card)['display']}{ms}")
            events.append({
                "type": "play",
                "seat": seat,
                "player_name": agent.name,
                "card": _card_to_dict(card),
                "tigress_mode": mode.value if mode else None,
            })

            new_state = engine.get_state()
            trick_done = (
                new_state.trick_number != prev_trick
                or new_state.round_number != prev_round
                or new_state.phase == GamePhase.GAME_OVER
            )
            if trick_done:
                if new_state.round_number == prev_round and new_state.phase != GamePhase.GAME_OVER:
                    winner_seat = _trick_winner(prev_tricks_won, new_state, seat)
                    agent_w = session.agents[winner_seat]
                    winner_name = "You" if winner_seat == human else (agent_w.name if agent_w else "AI")
                    events.append({"type": "trick_won", "winner_seat": winner_seat, "winner_name": winner_name})
                else:
                    events.append({"type": "round_end", "round_number": prev_round})

    return events


# ---------------------------------------------------------------------------
# Hint helpers (self-contained, no private imports)
# ---------------------------------------------------------------------------

def _h_strength(card: Card, mode: Optional[TigressMode] = None) -> int:
    if card.card_type == CardType.SKULL_KING:
        return 10_000
    if card.card_type == CardType.PIRATE:
        return 9_000
    if card.card_type == CardType.TIGRESS:
        return 9_000 if mode == TigressMode.PIRATE else 0
    if card.card_type == CardType.MERMAID:
        return 5_000
    if card.card_type == CardType.ESCAPE:
        return 0
    trump_bonus = 1_000 if card.suit == TRUMP_SUIT else 0
    return trump_bonus + (card.value or 0) * 10


def _h_would_beat(
    card: Card,
    played: tuple[PlayedCard, ...],
    player_index: int,
    mode: Optional[TigressMode],
) -> bool:
    candidate = PlayedCard(
        card=card,
        player_index=player_index,
        play_order=len(played) + 1,
        tigress_mode=mode,
    )
    result = TrickResolver.resolve(list(played) + [candidate])
    return result.winner_player_index == player_index


def _h_candidates(legal: list[Card]) -> list[tuple[Card, Optional[TigressMode]]]:
    seen_tigress = False
    result = []
    for card in dict.fromkeys(legal):
        if card.card_type == CardType.TIGRESS:
            if not seen_tigress:
                result.append((card, TigressMode.PIRATE))
                result.append((card, TigressMode.ESCAPE))
                seen_tigress = True
        else:
            result.append((card, None))
    return result


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class NewGameReq(BaseModel):
    n_players: int = 4
    human_seat: int = 0
    opponent_types: list[str] = ["heuristic", "heuristic", "heuristic"]
    seed: int = 42


class BidReq(BaseModel):
    bid: int


class PlayReq(BaseModel):
    card: dict
    tigress_mode: Optional[str] = None


class TournamentReq(BaseModel):
    n_players: int = 4
    agents: list[str] = ["heuristic", "heuristic", "random", "random"]
    n_games: int = 20
    seed: int = 42


class SettingsReq(BaseModel):
    active_model: str


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _make_agent(kind: str, seat: int, n_players: int = 4) -> BaseAgent:
    kind = kind.lower()
    if kind == "random":
        a = RandomAgent(seat)
        a.name = "Random"
        return a
    if kind == "mcts":
        a = MCTSAgent(n_simulations=5, seed=seat)
        a.name = "MCTS"
        return a
    if kind == "ppo":
        model = _load_ppo_model()
        a = SB3AgentWrapper(model, n_players=n_players, name="PPO")
        return a
    a = HeuristicAgent()
    a.name = "Heuristic"
    return a


# ---------------------------------------------------------------------------
# Routes – static
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return FileResponse(os.path.join(_STATIC, "index.html"))


@app.get("/api/health")
def health():
    return {"ok": True}


# ---------------------------------------------------------------------------
# Routes – game
# ---------------------------------------------------------------------------

@app.post("/api/game")
def new_game(req: NewGameReq):
    _cleanup()
    n = req.n_players
    if not (2 <= n <= 6):
        raise HTTPException(400, "n_players must be 2–6")
    if not (0 <= req.human_seat < n):
        raise HTTPException(400, "human_seat out of range")
    expected_opponents = n - 1
    if len(req.opponent_types) != expected_opponents:
        raise HTTPException(400, f"Need {expected_opponents} opponent_types for {n} players")

    engine = GameEngine(n_players=n, seed=req.seed)
    engine.start()

    agents: list[Optional[BaseAgent]] = [None] * n
    ai_idx = 0
    for seat in range(n):
        if seat == req.human_seat:
            continue
        agents[seat] = _make_agent(req.opponent_types[ai_idx], seat, n)
        # Deduplicate names across seats
        ai_idx += 1

    # Give distinct names when multiple of same type
    name_counts: dict[str, int] = {}
    for seat in range(n):
        if agents[seat] is None:
            continue
        base = agents[seat].name
        name_counts[base] = name_counts.get(base, 0) + 1

    name_seen: dict[str, int] = {}
    for seat in range(n):
        if agents[seat] is None:
            continue
        base = agents[seat].name
        if name_counts[base] > 1:
            name_seen[base] = name_seen.get(base, 0) + 1
            agents[seat].name = f"{base}-{name_seen[base]}"

    sid = str(uuid.uuid4())
    session = GameSession(
        session_id=sid,
        engine=engine,
        human_seat=req.human_seat,
        agents=agents,
    )
    session.log.append("Game started — good luck!")
    _sessions[sid] = session

    _advance_ai(session)
    return _build_state(session)


@app.get("/api/game/{sid}")
def get_game(sid: str):
    return _build_state(_get_session(sid))


@app.post("/api/game/{sid}/bid")
def place_bid(sid: str, req: BidReq):
    session = _get_session(sid)
    state = session.engine.get_state()

    if state.phase != GamePhase.BIDDING:
        raise HTTPException(400, f"Not in BIDDING phase")
    if state.current_player_index != session.human_seat:
        raise HTTPException(400, "Not your turn to bid")
    if not (0 <= req.bid <= state.round_number):
        raise HTTPException(400, f"Bid must be 0–{state.round_number}")

    session.engine.place_bid(session.human_seat, req.bid)
    session.log.append(f"You bid {req.bid}")

    human_event = {"type": "bid", "seat": session.human_seat, "player_name": "You", "bid": req.bid}
    ai_events = _advance_ai(session)
    state_dict = _build_state(session)
    state_dict["events"] = [human_event] + ai_events
    return state_dict


@app.post("/api/game/{sid}/play")
def play_card(sid: str, req: PlayReq):
    session = _get_session(sid)
    state = session.engine.get_state()

    if state.phase != GamePhase.PLAYING:
        raise HTTPException(400, "Not in PLAYING phase")
    if state.current_player_index != session.human_seat:
        raise HTTPException(400, "Not your turn to play")

    try:
        card = _dict_to_card(req.card)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, f"Invalid card: {exc}")

    tigress_mode: Optional[TigressMode] = None
    if req.tigress_mode:
        try:
            tigress_mode = TigressMode(req.tigress_mode)
        except ValueError:
            raise HTTPException(400, f"Invalid tigress_mode: {req.tigress_mode}")

    human = session.human_seat
    prev_round = state.round_number
    prev_trick = state.trick_number
    prev_tricks_won = [ps.tricks_won_this_round for ps in state.player_states]

    try:
        session.engine.play_card(human, card, tigress_mode)
    except Exception as exc:
        raise HTTPException(400, str(exc))

    ms = f" ({tigress_mode.value})" if tigress_mode else ""
    session.log.append(f"You play {_card_to_dict(card)['display']}{ms}")

    human_events: list[dict] = [{
        "type": "play",
        "seat": human,
        "player_name": "You",
        "card": _card_to_dict(card),
        "tigress_mode": tigress_mode.value if tigress_mode else None,
    }]

    new_state = session.engine.get_state()
    trick_done = (
        new_state.trick_number != prev_trick
        or new_state.round_number != prev_round
        or new_state.phase == GamePhase.GAME_OVER
    )
    if trick_done:
        if new_state.round_number == prev_round and new_state.phase != GamePhase.GAME_OVER:
            winner_seat = _trick_winner(prev_tricks_won, new_state, human)
            agent_w = session.agents[winner_seat]
            winner_name = "You" if winner_seat == human else (agent_w.name if agent_w else "AI")
            human_events.append({"type": "trick_won", "winner_seat": winner_seat, "winner_name": winner_name})
        else:
            human_events.append({"type": "round_end", "round_number": prev_round})

    ai_events = _advance_ai(session)
    state_dict = _build_state(session)
    state_dict["events"] = human_events + ai_events
    return state_dict


@app.delete("/api/game/{sid}")
def delete_game(sid: str):
    _sessions.pop(sid, None)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Routes – hint
# ---------------------------------------------------------------------------

@app.get("/api/game/{sid}/hint")
def get_hint(sid: str):
    session = _get_session(sid)
    state = session.engine.get_state()
    human = session.human_seat

    if state.phase == GamePhase.GAME_OVER:
        return {"phase": "GAME_OVER", "reasoning": "The game is over."}
    if state.current_player_index != human:
        return {"phase": state.phase.value, "reasoning": "Not your turn yet — wait for AI players."}

    agent = HeuristicAgent()

    # ── BIDDING ──────────────────────────────────────────────────────────────
    if state.phase == GamePhase.BIDDING:
        hand = list(state.player_states[human].hand)
        recommended = agent.bid(state, human)

        card_weights = []
        total_weight = 0.0
        for card in hand:
            if card.card_type == CardType.NUMBERED:
                if card.suit == TRUMP_SUIT:
                    w = 0.15 + (card.value / 14) * 0.60
                else:
                    w = (card.value / 14) * 0.15
            else:
                w = _TYPE_WEIGHT.get(card.card_type, 0.0)
            total_weight += w
            card_weights.append({"card": _card_to_dict(card), "weight": round(w, 2)})

        card_weights.sort(key=lambda x: -x["weight"])
        reasoning = (
            f"Sum of expected-win weights = {total_weight:.2f} → rounds to bid {recommended}. "
            "High black numbered cards, pirates, and Skull King score highest."
        )
        return {
            "phase": "BIDDING",
            "recommended_bid": recommended,
            "weight_total": round(total_weight, 2),
            "card_weights": card_weights,
            "reasoning": reasoning,
        }

    # ── PLAYING ──────────────────────────────────────────────────────────────
    ps = state.player_states[human]
    bid = ps.bid if ps.bid is not None else 0
    tricks_won = ps.tricks_won_this_round
    want_to_win = tricks_won < bid
    tricks_needed = max(0, bid - tricks_won)

    rec_card, rec_mode = agent.play(state, human)
    played = state.current_trick_cards

    hand = list(ps.hand)
    legal = TrickResolver.legal_plays(list(played), hand)
    candidates = _h_candidates(legal)

    card_analysis = []
    for c, m in candidates:
        would_win = _h_would_beat(c, played, human, m) if played else None
        card_analysis.append({
            "card": _card_to_dict(c),
            "tigress_mode": m.value if m else None,
            "would_win": would_win,
            "strength": _h_strength(c, m),
            "is_recommended": (c == rec_card and m == rec_mode),
        })
    card_analysis.sort(key=lambda x: -x["strength"])

    if not played:
        if want_to_win:
            reasoning = (
                f"Leading the trick. Bid {bid}, won {tricks_won} so far — "
                "need more tricks. Playing strongest card."
            )
        else:
            reasoning = (
                f"Leading the trick. Bid {bid}, won {tricks_won} — "
                "bid already met or set to 0. Playing weakest/escape card."
            )
    else:
        winners_exist = any(x["would_win"] for x in card_analysis)
        if want_to_win:
            if winners_exist:
                reasoning = (
                    f"Need {tricks_needed} more trick(s). "
                    "Playing weakest card that beats the current trick leader — conserve strong cards."
                )
            else:
                reasoning = (
                    f"Need {tricks_needed} more trick(s) but nothing wins. "
                    "Minimising loss by playing weakest card."
                )
        else:
            reasoning = (
                f"Bid {bid} already met ({tricks_won}/{bid}). "
                "Playing weakest card that won't win the trick."
            )

    return {
        "phase": "PLAYING",
        "bid": bid,
        "tricks_won": tricks_won,
        "want_to_win": want_to_win,
        "tricks_needed": tricks_needed,
        "recommended": {
            "card": _card_to_dict(rec_card),
            "tigress_mode": rec_mode.value if rec_mode else None,
        },
        "reasoning": reasoning,
        "card_analysis": card_analysis,
    }


# ---------------------------------------------------------------------------
# Routes – tournament
# ---------------------------------------------------------------------------

@app.post("/api/tournament")
def run_tournament(req: TournamentReq):
    n = req.n_players
    if not (2 <= n <= 6):
        raise HTTPException(400, "n_players must be 2–6")
    if len(req.agents) != n:
        raise HTTPException(400, f"Need exactly {n} agent types")
    if not (1 <= req.n_games <= 100):
        raise HTTPException(400, "n_games must be 1–100")

    agents = [_make_agent(kind, i, n) for i, kind in enumerate(req.agents)]

    # Give distinct names
    name_counts: dict[str, int] = {}
    for a in agents:
        name_counts[a.name] = name_counts.get(a.name, 0) + 1
    name_seen: dict[str, int] = {}
    for a in agents:
        base = a.name
        if name_counts[base] > 1:
            name_seen[base] = name_seen.get(base, 0) + 1
            a.name = f"{base}-{name_seen[base]}"

    runner = TournamentRunner(seed=req.seed)
    result = runner.run(agents, n_games=req.n_games)

    wr = result.win_rates()
    avg = result.avg_scores()
    std = result.score_std()

    return {
        "agent_names": result.agent_names,
        "n_games": result.n_games,
        "win_rates": {k: float(round(v, 4)) for k, v in wr.items()},
        "avg_scores": {k: float(round(v, 1)) for k, v in avg.items()},
        "score_std": {k: float(round(v, 1)) for k, v in std.items()},
        "per_round_avg": {
            k: [float(round(x, 1)) for x in v]
            for k, v in result.per_round_avg().items()
        },
        "summary": result.summary(),
    }


# ---------------------------------------------------------------------------
# Routes – model management & settings
# ---------------------------------------------------------------------------

@app.get("/api/models")
def list_models():
    active = os.path.normpath(_active_model_path)
    models = []
    if os.path.isdir(_MODELS_DIR):
        for path in sorted(glob.glob(os.path.join(_MODELS_DIR, "*.zip"))):
            norm = os.path.normpath(path)
            stat = os.stat(norm)
            models.append({
                "name": os.path.basename(norm),
                "path": norm,
                "size_mb": round(stat.st_size / 1_048_576, 1),
                "is_active": norm == active,
                "created_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
    return {"models": models}


@app.get("/api/settings")
def get_settings():
    return {"active_model": _active_model_path}


@app.post("/api/settings")
def update_settings(req: SettingsReq):
    global _ppo_model, _active_model_path
    path = os.path.normpath(req.active_model)
    if not os.path.exists(path):
        raise HTTPException(404, f"Model not found: {path}")
    _active_model_path = path
    _ppo_model = None
    return {"ok": True, "active_model": path}


@app.post("/api/settings/reload")
def reload_model():
    global _ppo_model
    _ppo_model = None
    return {"ok": True, "message": "Model cache cleared; will reload on next use"}


# ---------------------------------------------------------------------------
# CFR Strategy Analysis (GTO) — lazy loader
# ---------------------------------------------------------------------------

_cfr_explorer: Any = None
_cfr_model_path: str = ""


def _find_default_cfr_model() -> str:
    """Return best available CFR strategy model path, or empty string."""
    candidates = [
        os.path.join(_MODELS_DIR, "cfr_final_strat.pt"),
    ]
    # Also look for any checkpoint *_strat.pt, prefer later iterations
    if os.path.isdir(_MODELS_DIR):
        extras = sorted(
            glob.glob(os.path.join(_MODELS_DIR, "*_strat.pt")),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        candidates += extras
    for p in candidates:
        if os.path.exists(p):
            return os.path.normpath(p)
    return ""


def _load_cfr_explorer():
    global _cfr_explorer, _cfr_model_path
    if _cfr_explorer is None:
        if not _cfr_model_path:
            _cfr_model_path = _find_default_cfr_model()
        if not _cfr_model_path:
            raise HTTPException(
                404,
                "No CFR strategy model found in models/skull_king/. "
                "Train with: python -m skull_king.training.cfr.train",
            )
        from skull_king.analysis.explorer import StrategyExplorer
        _cfr_explorer = StrategyExplorer(_cfr_model_path)
    return _cfr_explorer


class BidQueryRequest(BaseModel):
    hand: list[str]          # e.g. ["BLACK_14", "PIRATE", "ESCAPE", ...]
    round_num: int
    player_score: float = 0.0
    opponent_scores: list[float] = []


class PlayQueryRequest(BaseModel):
    hand: list[str]
    round_num: int
    trick_num: int
    my_bid: int
    tricks_won: int
    current_trick: list[str] = []
    seen_cards: list[str] = []
    player_score: float = 0.0
    opponent_scores: list[float] = []
    is_trick_leader: bool = False


def _parse_card(s: str) -> Card:
    """Parse card string like 'BLACK_14', 'PIRATE', 'SKULL_KING', 'MERMAID', 'ESCAPE', 'TIGRESS'.

    Numbered cards are identified by having a numeric suffix (e.g. 'BLACK_14').
    Special cards may contain underscores too (e.g. 'SKULL_KING'), so we check
    whether the last component is numeric before treating it as a numbered card.
    """
    s = s.upper().strip()
    if "_" in s:
        parts = s.rsplit("_", 1)
        if parts[1].isdigit():
            suit = Suit[parts[0]]
            value = int(parts[1])
            return Card(card_type=CardType.NUMBERED, suit=suit, value=value)
    return Card(card_type=CardType[s])


@app.post("/gto/bid")
async def gto_bid(req: BidQueryRequest):
    explorer = _load_cfr_explorer()
    try:
        hand = [_parse_card(c) for c in req.hand]
    except Exception as e:
        raise HTTPException(400, f"Invalid card: {e}")
    result = explorer.query_bid(
        hand, req.round_num, req.player_score, req.opponent_scores or None
    )
    return {
        "recommended_bid": result.recommended_bid,
        "hand_strength": result.hand_strength,
        "probabilities": result.probabilities,
        "round_num": result.round_num,
    }


@app.post("/gto/play")
async def gto_play(req: PlayQueryRequest):
    explorer = _load_cfr_explorer()
    try:
        hand = [_parse_card(c) for c in req.hand]
        current_trick = [_parse_card(c) for c in req.current_trick]
        seen_cards = [_parse_card(c) for c in req.seen_cards]
    except Exception as e:
        raise HTTPException(400, f"Invalid card: {e}")
    result = explorer.query_play(
        hand, req.round_num, req.trick_num, req.my_bid, req.tricks_won,
        current_trick, seen_cards, req.player_score,
        req.opponent_scores, req.is_trick_leader,
    )
    return {
        "recommended": result.recommended,
        "position_in_trick": result.position_in_trick,
        "probabilities": result.probabilities,
    }


@app.get("/gto/models")
async def gto_available_models():
    """List available CFR strategy models."""
    models = []
    if os.path.isdir(_MODELS_DIR):
        for f in sorted(os.listdir(_MODELS_DIR)):
            if f.endswith("_strat.pt"):
                full = os.path.join(_MODELS_DIR, f)
                stat = os.stat(full)
                models.append({
                    "name": f,
                    "path": os.path.normpath(full),
                    "size_mb": round(stat.st_size / 1_048_576, 1),
                    "active": os.path.normpath(full) == os.path.normpath(_cfr_model_path) if _cfr_model_path else False,
                })
    return {"models": models, "active": _cfr_model_path}


class GTOLoadRequest(BaseModel):
    path: str


@app.post("/gto/load")
async def gto_load_model(req: GTOLoadRequest):
    """Switch the active CFR strategy model."""
    global _cfr_explorer, _cfr_model_path
    path = os.path.normpath(req.path)
    if not os.path.exists(path):
        raise HTTPException(404, f"Model not found: {path}")
    _cfr_explorer = None
    _cfr_model_path = path
    return {"ok": True, "active": path}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("skull_king.web.app:app", host="0.0.0.0", port=8000, reload=True)
