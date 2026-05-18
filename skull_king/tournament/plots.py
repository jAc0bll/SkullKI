"""Matplotlib visualisation for tournament results."""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from skull_king.tournament.runner import TournamentResult


def plot_tournament(
    result: "TournamentResult",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Four-panel summary of a completed tournament.

    Panels
    ------
    1. Win rate bar chart
    2. Final score distribution (box plot)
    3. Average score delta per round (line chart)
    4. Rolling-average final score over games (convergence)

    Parameters
    ----------
    result:
        Completed ``TournamentResult``.
    save_path:
        If given, save the figure to this path before showing.
    show:
        If True, call ``plt.show()``.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install it with:  pip install matplotlib"
        ) from exc

    import numpy as np

    names = result.agent_names
    n = result.n_agents
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:n]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Tournament results — {result.n_games} games, {n} agents",
        fontsize=14, fontweight="bold",
    )

    # ------------------------------------------------------------------
    # Panel 1 — Win rates
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    win_r = result.win_rates()
    rates = [win_r[name] * 100 for name in names]
    bars = ax.bar(names, rates, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("Win rate (%)")
    ax.set_ylabel("Win %")
    ax.set_ylim(0, max(rates) * 1.25 + 5)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{rate:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )

    # ------------------------------------------------------------------
    # Panel 2 — Score distributions
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    score_data = [result.final_scores[:, i] for i in range(n)]
    bp = ax.boxplot(
        score_data,
        tick_labels=names,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_title("Final score distribution")
    ax.set_ylabel("Total score")
    avg_s = result.avg_scores()
    for i, name in enumerate(names):
        ax.axhline(avg_s[name], color=colors[i], linestyle="--", linewidth=0.8, alpha=0.6)

    # ------------------------------------------------------------------
    # Panel 3 — Average score delta per round
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    per_round = result.per_round_avg()
    rounds = list(range(1, 11))
    for i, name in enumerate(names):
        ax.plot(rounds, per_round[name], marker="o", color=colors[i], label=name)
    ax.set_title("Avg score delta per round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Score gained")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks(rounds)
    ax.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Panel 4 — Rolling average final score over games
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    window = max(1, result.n_games // 10)
    for i, name in enumerate(names):
        series = result.final_scores[:, i]
        rolling = np.convolve(series, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, result.n_games), rolling, color=colors[i], label=name)
    ax.set_title(f"Rolling avg score (window={window})")
    ax.set_xlabel("Game")
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
