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
  - `tqdm`;
  - `mcts-simple`.
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
- tqdm;
- mcts-simple for the current experimental MCTS path;
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

### Web Game

The web game is a future target and should have its own documented setup once the stack is chosen.

Likely profiles:

- Static frontend only for browser PVP.
- Python backend plus frontend for PVE and AI inference.

## Recommended Dependency Layout

Use separate dependency files or optional extras:

- `requirements.txt`: minimal runtime for PVP desktop play.
- `requirements-ai.txt`: torch, tqdm, MCTS, training, evaluation, and model tooling.
- `requirements-dev.txt`: pytest and developer tooling.
- Web dependencies should live under the web client directory if a Node-based frontend is added.

## Configuration Notes

Current config loading reads `config.yaml` relative to the process working directory. This is fragile when scripts are launched from other directories.

Recommended improvement:

- Resolve `config.yaml` relative to the repository root or package path.
- Allow CLI overrides for board size, model path, device, and runtime mode.
- Store generated data and checkpoints under configurable output directories.

## Artifact Notes

Large files under `data/` and `model/` should be treated as generated artifacts unless intentionally versioned.

Recommended improvement:

- Keep large generated datasets out of normal commits.
- Save model checkpoints with metadata.
- Document how to regenerate datasets and checkpoints.
