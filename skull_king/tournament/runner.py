"""Tournament runner for benchmarking Skull King agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from skull_king.agents.base_agent import BaseAgent
from skull_king.cards import NUM_ROUNDS
from skull_king.engine import GameEngine
from skull_king.game_state import GamePhase


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TournamentResult:
    """Aggregated statistics from a completed tournament.

    Attributes
    ----------
    agent_names:
        Display names for each agent, indexed 0..n_agents-1.
    n_games:
        Number of games played.
    final_scores:
        ``ndarray`` of shape ``(n_games, n_agents)`` — total score at game end.
    round_cumulative:
        ``ndarray`` of shape ``(n_games, NUM_ROUNDS, n_agents)`` — cumulative
        score at the end of each round.  ``round_cumulative[g, r, a]`` is
        agent *a*'s total score after round *r+1* in game *g*.
    """

    agent_names: list[str]
    n_games: int
    final_scores: np.ndarray        # (n_games, n_agents)
    round_cumulative: np.ndarray    # (n_games, NUM_ROUNDS, n_agents)

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------

    @property
    def n_agents(self) -> int:
        return len(self.agent_names)

    def win_rates(self) -> dict[str, float]:
        """Fraction of games each agent won (strict majority; ties give 0)."""
        wins = np.zeros(self.n_agents)
        for g in range(self.n_games):
            row = self.final_scores[g]
            best = row.max()
            winners = np.where(row == best)[0]
            if len(winners) == 1:
                wins[winners[0]] += 1
        return {self.agent_names[i]: wins[i] / self.n_games for i in range(self.n_agents)}

    def avg_scores(self) -> dict[str, float]:
        """Mean final score per agent across all games."""
        means = self.final_scores.mean(axis=0)
        return {self.agent_names[i]: float(means[i]) for i in range(self.n_agents)}

    def score_std(self) -> dict[str, float]:
        """Standard deviation of final scores per agent."""
        stds = self.final_scores.std(axis=0)
        return {self.agent_names[i]: float(stds[i]) for i in range(self.n_agents)}

    def avg_round_delta(self) -> np.ndarray:
        """Average score gained each round; shape ``(NUM_ROUNDS, n_agents)``."""
        # round_cumulative[:,0,:] = delta round 1; diff gives subsequent deltas
        deltas = np.diff(self.round_cumulative, prepend=0, axis=1)
        return deltas.mean(axis=0)  # (NUM_ROUNDS, n_agents)

    def per_round_avg(self) -> dict[str, list[float]]:
        """Per-agent list of mean score delta per round."""
        deltas = self.avg_round_delta()
        return {
            self.agent_names[i]: [float(deltas[r, i]) for r in range(NUM_ROUNDS)]
            for i in range(self.n_agents)
        }

    def summary(self) -> str:
        """Formatted text table."""
        lines = [
            f"Tournament: {self.n_games} games, {self.n_agents} agents",
            f"{'Agent':<16} {'Win%':>6} {'AvgScore':>10} {'StdDev':>8}",
            "-" * 44,
        ]
        win_r = self.win_rates()
        avg_s = self.avg_scores()
        std_s = self.score_std()
        for name in sorted(self.agent_names, key=lambda n: avg_s[n], reverse=True):
            lines.append(
                f"{name:<16} {win_r[name]*100:>5.1f}% "
                f"{avg_s[name]:>10.1f} {std_s[name]:>8.1f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TournamentRunner:
    """Run multi-agent Skull King tournaments with seat rotation.

    Seat rotation ensures each agent spends equal time in every seat position,
    removing positional bias from the statistics.

    Parameters
    ----------
    seed:
        Base seed; game *g* uses ``seed + g`` so results are reproducible.
    """

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def run(
        self,
        agents: Sequence[BaseAgent],
        n_games: int,
        rotate_seats: bool = True,
    ) -> TournamentResult:
        """Play ``n_games`` games and return aggregated statistics.

        Parameters
        ----------
        agents:
            Agent list; ``len(agents)`` is the player count.
        n_games:
            Total games to play.
        rotate_seats:
            If True, shift seat assignments each game so each agent occupies
            each seat an equal number of times (approximately).
        """
        n = len(agents)
        if not (2 <= n <= 6):
            raise ValueError(f"Need 2–6 agents, got {n}")

        names = [a.name for a in agents]
        all_finals: list[np.ndarray] = []
        all_rounds: list[np.ndarray] = []

        for g in range(n_games):
            # Seat mapping: agent i → seat (i + g) % n  when rotating
            if rotate_seats:
                agent_to_seat = [(i + g) % n for i in range(n)]
            else:
                agent_to_seat = list(range(n))
            seat_to_agent = [0] * n
            for agent_idx, seat in enumerate(agent_to_seat):
                seat_to_agent[seat] = agent_idx

            finals, round_scores = self._run_game(
                agents, seat_to_agent, n, seed=self._seed + g
            )
            all_finals.append(finals)
            all_rounds.append(round_scores)

        final_arr = np.array(all_finals, dtype=np.float32)   # (n_games, n_agents)
        round_arr = np.array(all_rounds, dtype=np.float32)   # (n_games, NUM_ROUNDS, n_agents)

        return TournamentResult(
            agent_names=names,
            n_games=n_games,
            final_scores=final_arr,
            round_cumulative=round_arr,
        )

    # ------------------------------------------------------------------
    # Internal game runner
    # ------------------------------------------------------------------

    def _run_game(
        self,
        agents: Sequence[BaseAgent],
        seat_to_agent: list[int],
        n: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(final_scores, round_cumulative)`` both indexed by agent."""
        engine = GameEngine(n_players=n, seed=seed)
        state = engine.start()

        prev_round = 1
        round_snapshots: list[list[float]] = []  # up to NUM_ROUNDS entries

        while state.phase != GamePhase.GAME_OVER:
            # Detect round transition: new round started → record previous round's scores
            if state.phase == GamePhase.BIDDING and state.round_number > prev_round:
                round_snapshots.append(self._seat_scores_by_agent(state, seat_to_agent, n))
                prev_round = state.round_number

            seat = state.current_player_index
            agent_idx = seat_to_agent[seat]
            agent = agents[agent_idx]
            agent.before_move(engine)

            if state.phase == GamePhase.BIDDING:
                bid = agent.bid(state, seat)
                state = engine.place_bid(seat, bid)
            else:
                card, mode = agent.play(state, seat)
                state = engine.play_card(seat, card, mode)

        # Final round scores at GAME_OVER
        round_snapshots.append(self._seat_scores_by_agent(state, seat_to_agent, n))

        # Pad to NUM_ROUNDS if game ended early (shouldn't happen, but defensive)
        while len(round_snapshots) < NUM_ROUNDS:
            round_snapshots.append(round_snapshots[-1])

        final = np.array(round_snapshots[-1], dtype=np.float32)
        rounds = np.array(round_snapshots, dtype=np.float32)   # (NUM_ROUNDS, n_agents)
        return final, rounds

    @staticmethod
    def _seat_scores_by_agent(
        state,
        seat_to_agent: list[int],
        n: int,
    ) -> list[float]:
        """Return scores indexed by agent (not by seat)."""
        scores_by_agent = [0.0] * n
        for seat in range(n):
            agent_idx = seat_to_agent[seat]
            scores_by_agent[agent_idx] = float(state.player_states[seat].total_score)
        return scores_by_agent
