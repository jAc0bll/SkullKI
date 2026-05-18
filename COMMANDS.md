# Skull King AI — Commands Reference

## Web UI

```bash
# Start (run from project root, then open http://localhost:8000)
uvicorn skull_king.web.app:app --reload

# Stop
Ctrl+C
```

The `--reload` flag restarts the server automatically when you edit Python files.
Remove it for a stable session (e.g. during a long game).

---

## Training

```bash
# Train with default config (training_config.yaml)
python -m skull_king.training.train

# Train with a custom config file
python -m skull_king.training.train --config my_config.yaml
```

Output goes to:
- `models/skull_king/final.zip` — the finished model
- `models/skull_king/ppo_selfplay_v1_25000.zip` ... `_100000.zip` — checkpoints every 25k steps
- `logs/skull_king/ppo_selfplay_v1/` — TensorBoard event files

Key settings in `training_config.yaml`:

| Setting | Default | Notes |
|---|---|---|
| `total_timesteps` | 100 000 | Scale to 2 000 000+ for real training |
| `n_envs` | 4 | Parallel envs; 8 is a good bump if you have CPU cores |
| `reward_mode` | round | `sparse` / `round` / `shaped` |
| `eval_freq` | 10 000 | How often to run the tournament eval and log metrics |
| `checkpoint_freq` | 25 000 | How often to save an intermediate model |

---

## Monitoring training progress (TensorBoard)

```bash
tensorboard --logdir logs/skull_king
# then open http://localhost:6006
```

Metrics to watch (all under the `eval/` prefix):

| Metric | What improving looks like |
|---|---|
| `eval/win_rate_vs_random` | Rising above 0.25 (random baseline in 4-player) |
| `eval/win_rate_vs_heuristic` | Rising above 0.25 — harder, means real learning |
| `eval/bid_accuracy` | Climbing — model learning to bid what it can win |
| `eval/mean_episode_reward` | Steady upward trend |
| `train/entropy_loss` | Decreasing slowly — less random over time |

---

## Tests

```bash
# Run the full test suite (350 tests)
python -m pytest

# Run a specific file
python -m pytest tests/test_engine.py

# Run with coverage report
python -m pytest --cov=skull_king --cov-report=term-missing
```

---

## Linting & type checking

```bash
ruff check .          # lint
ruff check . --fix    # lint and auto-fix safe issues
mypy skull_king/      # type check
```

---

## Project structure

```
skull_king/
  cards.py          Card types, suits, deck builder
  trick.py          PlayedCard, trick representation
  resolver.py       TrickResolver — legal plays, trick winner
  calculator.py     Score calculator (round scoring, bonuses)
  engine.py         GameEngine — orchestrates a full game
  game_state.py     GameState, PlayerState, GamePhase

  env/
    skull_king_env.py   Gymnasium single-agent env (for RL training)

  agents/
    base_agent.py       BaseAgent ABC
    random_agent.py     Random legal-move agent
    heuristic_agent.py  Weight-based bidding + want-to-win play logic
    mcts_agent.py       Determinization-MCTS (PIMC) agent

  training/
    self_play_env.py    SelfPlaySkullKingEnv + SB3AgentWrapper
    callbacks.py        SelfPlayCallback, TournamentEvalCallback, StepCheckpointCallback
    train.py            Training entry point (reads training_config.yaml)

  tournament/
    runner.py           TournamentRunner — runs N games, returns TournamentResult
    plots.py            Matplotlib helpers for score charts

  web/
    app.py              FastAPI app — game sessions, hint, tournament endpoints
    static/index.html   Single-page browser UI

tests/               pytest test suite (mirrors skull_king/ structure)
docs/rules_spec.md   Authoritative Skull King rules + resolved ambiguities
training_config.yaml RL training hyperparameters
requirements.txt     All Python dependencies
```

---

## Model analysis (what did it learn?)

```bash
# Analyse the default model (200 games, prints strategy report)
python -m skull_king.training.analyze

# Analyse a specific checkpoint
python -m skull_king.training.analyze --model models/skull_king/ppo_selfplay_v1_50000

# More games = more accurate statistics
python -m skull_king.training.analyze --games 500
```

The report covers:
- Tournament win rate vs random and heuristic
- Bid calibration (hand strength → how much the model bids)
- Bid accuracy per round
- Play style (aggressive vs passive depending on bid status)
- When it plays special cards (Skull King, Pirates, Mermaids, Escapes)
- Human-readable takeaways you can apply yourself

---

## Quick-start cheat sheet

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Play in browser
uvicorn skull_king.web.app:app --reload

# 3. Train a better model
python -m skull_king.training.train

# 4. Watch it improve
tensorboard --logdir logs/skull_king

# 5. Run tests
python -m pytest
```
