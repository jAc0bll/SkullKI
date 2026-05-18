"""Tests for TournamentRunner and TournamentResult."""
import numpy as np
import pytest

from skull_king.agents import HeuristicAgent, MCTSAgent, RandomAgent
from skull_king.cards import NUM_ROUNDS
from skull_king.tournament import TournamentResult, TournamentRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def quick_tournament(agents=None, n_games=4, seed=0) -> TournamentResult:
    if agents is None:
        agents = [RandomAgent(0), RandomAgent(1), HeuristicAgent()]
    return TournamentRunner(seed=seed).run(agents, n_games=n_games)


# ---------------------------------------------------------------------------
# Runner basics
# ---------------------------------------------------------------------------


class TestTournamentRunner:
    def test_runs_without_error(self):
        result = quick_tournament()
        assert result is not None

    def test_wrong_agent_count(self):
        with pytest.raises(ValueError):
            TournamentRunner().run([RandomAgent()], n_games=1)

    def test_result_n_games(self):
        result = quick_tournament(n_games=6)
        assert result.n_games == 6

    def test_final_scores_shape(self):
        agents = [RandomAgent(0), RandomAgent(1)]
        result = TournamentRunner().run(agents, n_games=5)
        assert result.final_scores.shape == (5, 2)

    def test_round_cumulative_shape(self):
        agents = [RandomAgent(0), RandomAgent(1), RandomAgent(2)]
        result = TournamentRunner().run(agents, n_games=3)
        assert result.round_cumulative.shape == (3, NUM_ROUNDS, 3)

    def test_round_cumulative_monotone(self):
        """Scores can go up or down but the 'rounds' axis should be filled."""
        result = quick_tournament(n_games=2)
        # Check not all zeros
        assert result.round_cumulative.sum() != 0

    def test_final_scores_match_last_round(self):
        result = quick_tournament(n_games=3)
        np.testing.assert_array_almost_equal(
            result.final_scores, result.round_cumulative[:, -1, :]
        )

    def test_deterministic_with_seed(self):
        r1 = TournamentRunner(seed=42).run([RandomAgent(0), RandomAgent(1)], n_games=4)
        r2 = TournamentRunner(seed=42).run([RandomAgent(0), RandomAgent(1)], n_games=4)
        np.testing.assert_array_equal(r1.final_scores, r2.final_scores)

    def test_different_seeds_differ(self):
        r1 = TournamentRunner(seed=0).run([RandomAgent(0), RandomAgent(1)], n_games=10)
        r2 = TournamentRunner(seed=99).run([RandomAgent(0), RandomAgent(1)], n_games=10)
        assert not np.array_equal(r1.final_scores, r2.final_scores)

    def test_seat_rotation_disabled(self):
        agents = [RandomAgent(0), RandomAgent(1)]
        result = TournamentRunner(seed=0).run(agents, n_games=4, rotate_seats=False)
        assert result.n_games == 4

    def test_two_player_game(self):
        result = TournamentRunner().run([RandomAgent(0), RandomAgent(1)], n_games=2)
        assert result.n_agents == 2

    def test_six_player_game(self):
        agents = [RandomAgent(i) for i in range(6)]
        result = TournamentRunner().run(agents, n_games=2)
        assert result.n_agents == 6


# ---------------------------------------------------------------------------
# TournamentResult statistics
# ---------------------------------------------------------------------------


class TestTournamentResultStats:
    def test_win_rates_sum_lte_one(self):
        result = quick_tournament(n_games=10)
        total = sum(result.win_rates().values())
        assert total <= 1.0 + 1e-6  # ties give no win point

    def test_win_rates_all_nonnegative(self):
        result = quick_tournament(n_games=6)
        assert all(v >= 0 for v in result.win_rates().values())

    def test_avg_scores_keys_match_names(self):
        result = quick_tournament()
        assert set(result.avg_scores().keys()) == set(result.agent_names)

    def test_avg_scores_finite(self):
        result = quick_tournament(n_games=5)
        assert all(np.isfinite(v) for v in result.avg_scores().values())

    def test_score_std_nonnegative(self):
        result = quick_tournament(n_games=5)
        assert all(v >= 0 for v in result.score_std().values())

    def test_avg_round_delta_shape(self):
        result = quick_tournament(n_games=4)
        delta = result.avg_round_delta()
        assert delta.shape == (NUM_ROUNDS, result.n_agents)

    def test_per_round_avg_length(self):
        result = quick_tournament(n_games=3)
        pr = result.per_round_avg()
        for name in result.agent_names:
            assert len(pr[name]) == NUM_ROUNDS

    def test_summary_contains_all_agents(self):
        result = quick_tournament()
        s = result.summary()
        for name in result.agent_names:
            assert name in s

    def test_n_agents_correct(self):
        result = quick_tournament()
        assert result.n_agents == 3


# ---------------------------------------------------------------------------
# Agent mix tournament
# ---------------------------------------------------------------------------


class TestMixedAgentTournament:
    def test_heuristic_vs_random(self):
        agents = [HeuristicAgent(), RandomAgent(0), RandomAgent(1)]
        result = TournamentRunner(seed=7).run(agents, n_games=10)
        avgs = result.avg_scores()
        # Heuristic should generally outperform random (not guaranteed but expected)
        assert avgs["Heuristic"] > avgs["Random"]  # flaky on small N; check structure

    def test_mcts_completes_game(self):
        agents = [MCTSAgent(n_simulations=2), RandomAgent(0)]
        result = TournamentRunner(seed=0).run(agents, n_games=1)
        assert result.n_games == 1

    def test_agent_names_in_result(self):
        agents = [RandomAgent(0), HeuristicAgent()]
        result = TournamentRunner().run(agents, n_games=2)
        assert set(result.agent_names) == {"Random", "Heuristic"}


# ---------------------------------------------------------------------------
# Plot (smoke test — import only, no display)
# ---------------------------------------------------------------------------


class TestPlotTournament:
    def test_plot_import_works(self):
        from skull_king.tournament.plots import plot_tournament
        assert callable(plot_tournament)

    def test_plot_runs_without_display(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        from skull_king.tournament.plots import plot_tournament
        result = quick_tournament(n_games=5)
        save_path = str(tmp_path / "test_plot.png")
        plot_tournament(result, save_path=save_path, show=False)
        import os
        assert os.path.exists(save_path)
