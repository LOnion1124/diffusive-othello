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

- `python gui.py --mode PVP` should not require torch.
- PVP should not load model files.

### PVE Desktop Game

Required dependencies:

- Python;
- pygame;
- pyyaml;
- torch;
- a valid model checkpoint.

Expected behavior:

- `python gui.py --mode PVE` should load AI lazily.
- Missing model dependencies should produce a clear error message.

### Training and Self-Play

Required dependencies:

- Python;
- torch;
- tqdm;
- pytest for tests;
- optional visualization/debug dependencies as needed.

Expected behavior:

- Training should run through command-line scripts, not notebooks only.
- All training dependencies should be documented and installable.
- MCTS should not depend on an undeclared external package.
- Self-play can batch multiple active games in one process with
  `--self-play-batch-size`, which is the preferred CUDA path because it batches
  neural-network leaf evaluations.
- Self-play can shard games across processes with `--self-play-workers`, which
  is most useful for CPU generation. On a single GPU, prefer one worker and a
  larger self-play batch to avoid loading one model per process.

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
- `ai.runtime.model_path` controls the pygame PVE checkpoint path.
- `ai.self_play` and `ai.train` define generated data and checkpoint defaults.

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
