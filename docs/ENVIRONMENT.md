# Environment Notes

This document records known setup issues and expected environment behavior for Diffusive Othello.

## Current Environment Findings

The repository uses a local Python virtual environment named `venv`.

Verified observations:

- `python -m py_compile` succeeds for the main Python files.
- `requirements.txt` contains the minimal PVP desktop dependencies.
- `requirements-ai.txt` contains AI/training dependencies:
  - `torch`;
  - `numpy`;
  - `tqdm`.
- `requirements-dev.txt` contains developer/test dependencies.

Recommended Windows setup:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Optional AI/training setup for CUDA 13.0:
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-ai.txt

# Optional developer tools:
python -m pip install -r requirements-dev.txt
```

## Runtime Profiles

The project should support separate runtime profiles instead of requiring every dependency for every task.

### PVP Desktop Game

Required dependencies:

- Python;
- pygame;
- pyyaml.

Expected behavior:

- `python gui.py` opens a start screen where `Play PvP` starts PVP.
- `python gui.py --mode PVP` remains available as a quick start and should not
  require torch.
- PVP should show the win-rate prediction bar during gameplay.
- When a configured model is available, PVP and PVE should merge the current
  human player's top-three legal policy probabilities into the legal-move
  breathing highlights. PVE hides all legal-position and probability highlights
  during the AI's turn.
- PVP should attempt to load the configured model for prediction when available,
  and show `Invalid` at 50:50 when model loading or inference fails.

### PVE Desktop Game

Required dependencies:

- Python;
- pygame;
- pyyaml;
- torch;
- a valid model checkpoint.

Expected behavior:

- `python gui.py` opens a start screen where `Play PvE` starts PVE.
- `python gui.py --mode PVE` remains available as a quick start and should
  enable the first-player vs. second-player win-rate prediction bar during
  gameplay.
- Missing model dependencies should put the bar in the `Invalid` 50:50 state,
  produce a clear error message, and use fallback legal moves for AI turns.
- PvE labels the assigned sides as `YOU` and `COMPUTER` throughout the desktop
  UI. The gameplay sidebar can hide either model-analysis view and can end a
  match early, using the current canonical board score for the result.

### Training and Self-Play

Required dependencies:

- Python;
- torch;
- tqdm;
- pytest for tests;
- optional visualization/debug dependencies as needed.

Expected behavior:

- Training should run through command-line scripts, not notebooks only.
- Root training entry points are `.\train.ps1` on Windows and `bash ./train.sh`
  on Linux. Both start the multi-stage trainer in the background and write logs
  under `logs/`; the Linux script prefers `venv/bin/python` and falls back to
  `python3`.
- Python training implementation scripts live under `src/train`; keep the root
  launcher as the documented training entry point.
- All training dependencies should be documented and installable.
- MCTS should not depend on an undeclared external package.
- Self-play can batch multiple active games in one process with
  `--self-play-batch-size`, which is the preferred CUDA path because it batches
  neural-network leaf evaluations.
- Self-play can shard games across processes with `--self-play-workers`, which
  is most useful for CPU generation. On a single GPU, prefer one worker and a
  larger self-play batch to avoid loading one model per process.
- `python -m src.train.train_multistage --schedule continue` is the guarded
  long-running training path for an existing model. It starts from
  `models/latest.pth` by default, repeats the stage-5 workload, and promotes a
  candidate only after it scores strictly above 50% in a color-balanced,
  seeded-random MCTS arena against the checkpoint that started that stage.
  Final moves are sampled from MCTS visit distributions by default; set the
  arena temperature to zero for deterministic best-visit evaluation.
  Each successful candidate becomes the product model used by the next stage;
  a rejected candidate ends the continuation job early.
- `python -m src.train.arena --candidate <path> --incumbent <path> --promote`
  runs the same checkpoint comparison independently. Arena game counts must be
  even; its result is saved next to the candidate as `<candidate>.arena.json`.

### Web Game

The web game is a future target and should have its own documented setup once the stack is chosen.

Likely profiles:

- Static frontend only for browser PVP.
- Python backend plus frontend for PVE and AI inference.

## Recommended Dependency Layout

Use separate dependency files or optional extras:

- `requirements.txt`: minimal runtime for PVP desktop play.
- `requirements-ai.txt`: torch, tqdm, training, evaluation, and model tooling.
- `requirements-dev.txt`: pytest and developer tooling.
- Web dependencies should live under the web client directory if a Node-based frontend is added.

## Configuration Notes

Current config loading reads `config.yaml` relative to the process working directory. This is fragile when scripts are launched from other directories.

Current behavior:

- `config.yaml` is resolved relative to the repository root.
- `game.board_size` controls the default desktop board size.
- `ai.runtime.device` controls AI device selection: `auto`, `cpu`, or `cuda`.
- `ai.runtime.model_path` controls the pygame checkpoint path used by the
  desktop win-rate bar and PVE AI moves.
- `ai.self_play` and `ai.train` define generated data and checkpoint defaults.
- `ai.arena` defines default game count, MCTS visit-sampling temperature,
  reproducibility seeds, and promotion threshold. Arena games must be even;
  the candidate receives exactly half as first player and half as second
  player, in a seed-shuffled order.
- The continuation schedule uses `ai.runtime.model_path` as its default
  incumbent and promotion target. The stage-5-equivalent settings are defined
  as `CONTINUATION_STAGE` in `src/train/train_multistage.py`; CLI arena options
  override only evaluation settings, not the saved model architecture. An
  explicit `--initial-checkpoint` seeds only the first continuation stage;
  later stages load the latest accepted product checkpoint.

## Dataset Format Notes

Self-play datasets use `az-do-dataset-v2`.

Stored training tensors:

- `states`: current-player encoded board tensors;
- `legal_masks`: flattened legal action masks;
- `policies`: MCTS visit-count distributions;
- `values`: final outcome targets from the encoded player's perspective.

Stored analysis metadata:

- `sample_metadata`: exact game id, ply, absolute player, chosen action,
  temperature, MCTS root/chosen-child statistics, policy concentration,
  legal counts, piece counts, current margin, and flipped piece count;
- `game_metadata`: exact game boundaries, first player, winner, move count,
  pass count, terminal counts, final margin, and sample ranges.

The data visualizer expects v2 datasets and uses these metadata fields for
exact per-game statistics instead of estimating them from tensor occupancy.

Recommended improvement:

- Allow CLI overrides for model path and more runtime options.
- Store large generated data and checkpoints outside source-controlled paths by default.

## Artifact Notes

Large files under `data/` and `models/` should be treated as generated artifacts unless intentionally versioned.

Recommended improvement:

- Keep large generated datasets out of normal commits.
- Save model checkpoints with metadata.
- Document how to regenerate datasets and checkpoints.
- Keep rejected continuation candidates and their arena JSON records for later
  analysis; only the configured incumbent path is replaced after a win.
