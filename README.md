# Skull King AI

A complete implementation of the [Skull King](https://www.schmidtspiele.de/files/Produkte/5/49353%20-%20Skull%20King/49353_Skull_King_GB_D_F_NL.pdf) card game with a reinforcement learning agent trained via self-play.

## Features

- **Full game engine** — all rules including bonuses, Black 14, Mermaid/Skull King interaction
- **Three agents** — Random, Heuristic (rule-based), MCTS (Monte Carlo Tree Search)
- **PPO self-play training** — Maskable PPO with curriculum learning and rank-based rewards
- **Browser UI** — play against any combination of agents, get AI hints, run tournaments
- **TensorBoard monitoring** — live training metrics

---

## Quick Start

### 1. Install

```bash
git clone <your-repo-url>
cd skull-king-projekt

pip install -r requirements.txt
```

Requires Python ≥ 3.11. GPU optional (CPU training works fine for this model size).

### 2. Play in the browser

```bash
uvicorn skull_king.web.app:app --reload
# open http://localhost:8000
```

### 3. Train a model

```bash
# Local machine (~8–12h for 10M steps)
python -m skull_king.training.train

# High-core-count server (~8–12h for 20M steps on 34+ cores)
python -m skull_king.training.train --config server_config.yaml
```

Monitor progress:
```bash
tensorboard --logdir logs/skull_king
# open http://localhost:6006
```

The trained model is saved to `models/skull_king/final.zip` and auto-detected by the web UI.

---

## Project Structure

```
skull_king/
  cards.py          Card types, suits, deck builder
  trick.py          Played card representation
  resolver.py       TrickResolver — legal plays, trick winner
  calculator.py     Score calculator (round scoring, bonuses)
  engine.py         GameEngine — full game orchestration
  game_state.py     GameState, PlayerState, GamePhase

  env/
    skull_king_env.py     Gymnasium single-agent environment

  agents/
    base_agent.py         BaseAgent ABC
    random_agent.py       Uniform-random legal-move agent
    heuristic_agent.py    Rule-based: weight-sum bidding + want-to-win/lose play
    mcts_agent.py         Determinization MCTS (PIMC)

  training/
    self_play_env.py      SelfPlaySkullKingEnv — opponents driven by frozen model
    callbacks.py          SelfPlayCallback, CurriculumCallback,
                          TournamentEvalCallback, StepCheckpointCallback
    train.py              Training entry point (reads YAML config)
    analyze.py            Post-training model analysis and strategy report

  tournament/
    runner.py             TournamentRunner
    plots.py              Matplotlib score charts

  web/
    app.py                FastAPI backend
    static/index.html     Browser UI (single-page)

tests/                    pytest suite (~350 tests)
docs/rules_spec.md        Authoritative rules + resolved ambiguities
training_config.yaml      Local training hyperparameters
server_config.yaml        Server training hyperparameters (34+ cores)
COMMANDS.md               Quick command reference
```

---

## Training Guide

### Key design decisions

**Reward: `shaped` + rank bonus**
Every game ends with a placement bonus (+0.50 first, -0.30 last). This teaches the agent that *beating opponents* matters, not just maximising its own score.

**Curriculum (`heuristic_mix`)**
Opponent quality is annealed over training:
- Step 0 → 80% of opponent turns use the Heuristic agent (near-optimal play)
- End → 8% heuristic, 92% self-play

This prevents the agent from learning to exploit random-policy mistakes and forces it to develop real strategy from the start.

**Self-play (`SelfPlayCallback`)**
Every 25 000 steps the current policy is frozen and used as the opponent in subsequent rollouts.

### Config options

| Field | Local default | Server default | Notes |
|---|---|---|---|
| `total_timesteps` | 10 000 000 | 20 000 000 | Scale with compute |
| `n_envs` | 12 | 48 | One subprocess per env |
| `reward_mode` | shaped | shaped | shaped > round > sparse |
| `curriculum_start_mix` | 0.80 | 0.80 | Start vs heuristic |
| `curriculum_end_mix` | 0.08 | 0.08 | End vs self-play |
| `n_steps` | 2048 | 2048 | Rollout steps per env |
| `batch_size` | 512 | 1024 | Gradient batch size |
| `learning_rate` | 3e-4 | 2e-4 | Adam LR |
| `ent_coef` | 0.03 | 0.03 | Exploration entropy |

### TensorBoard metrics

| Metric | What improving looks like |
|---|---|
| `eval/win_rate_vs_random` | Rising above 0.25 (random baseline in 4P) |
| `eval/win_rate_vs_heuristic` | Rising above 0.25 — real learning |
| `eval/bid_accuracy` | Fraction of rounds where bid == tricks won |
| `eval/mean_episode_reward` | Steady upward trend |
| `train/entropy_loss` | Slowly decreasing — less random over time |

### On a high-core server (34 cores / 54 threads)

`SubprocVecEnv` spawns one OS process per env — no Python GIL, true parallelism.
`n_envs=48` saturates ~48 cores with game simulation while 1 core runs the model.

```bash
python -m skull_king.training.train --config server_config.yaml
```

Expected throughput: ~7 000–8 000 steps/s → 20M steps in ~8–10h.

---

## Model Management

The web UI Settings tab lets you browse all `.zip` checkpoints in `models/skull_king/`
and switch the active model at runtime without restarting the server.

To copy a trained model from server to local:

```bash
scp user@server:/path/to/project/models/skull_king/final.zip models/skull_king/final.zip
```

---

## Rules Reference

`docs/rules_spec.md` contains the complete formal rules including all resolved ambiguities:

- **AMBIGUOUS-03** — Black is strict-follow (cannot play Black over a colored led suit if you hold that suit)
- **AMBIGUOUS-04** — Bonuses only on bid success (bid > 0 and tricks_won == bid)
- **AMBIGUOUS-05** — Black 14 bonus (+20) only when trick won by a special card

---

## Development

```bash
# Run tests
python -m pytest

# Lint + type check
ruff check .
mypy skull_king/

# Analyze a trained model
python -m skull_king.training.analyze
python -m skull_king.training.analyze --model models/skull_king/ppo_v3_server_10000000 --games 500
```
