# Diffusive Othello

A variant of traditional board game [Reversi](https://en.wikipedia.org/wiki/Reversi), supporting both player vs. player and player vs. AI.

## Get Started

Build python environment:

```sh
# create virtual environment
python -m venv venv
# then activate it base on your system
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# install minimal packages for PVP desktop play
pip install -r requirements.txt

# optional: install AI/training packages
# CUDA 13.0 GPU build:
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-ai.txt

# CPU-only fallback:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install -r requirements-ai.txt
# then set `ai.runtime.device` to `cpu` in config.yaml
```

## How to Play

Play with GUI made by pygame:

```sh
# play PVE (use model in config.yaml)
python gui.py --mode PVE
# play PVP
python gui.py --mode PVP
```

PVP only needs the minimal dependencies in `requirements.txt`. PVE loads the AI stack only when the AI first moves; if the model or optional AI dependencies are unavailable, the game reports the error in the terminal and uses a simple legal-move fallback.

A command-line interface is also provided, but only supports 2-player mode:

```sh
python cli.py
```

### Game Rules

This game is a variant of the classic board game *Reversi*. It is played on a 9×9 board. At the start of the game, each player has two pieces placed in diagonal corners of the board. The players then take turns placing pieces.

A move is valid only if:

1. The target cell is not already occupied; and
2. At least one adjacent orthogonally neighboring cell (up, down, left, or right) contains one of the player’s own pieces.

Whenever a player makes a valid move, all of the opponent’s pieces located in the eight surrounding cells (orthogonal and diagonal) are flipped to the player’s color. If a player has no valid moves, the turn automatically passes to the opponent. The game ends when neither player can make a move (usually when the board is full). The player with more pieces on the board at the end wins.

This game is still under development, and its rules may be adjusted in future updates.

## Train Your Own AI

AI defaults live under the `ai` section of `config.yaml`:

- `ai.runtime`: inference/training device and pygame model path.
- `ai.model`: AlphaNet architecture, board size, residual trunk size, and value-head hidden size.
- `ai.mcts`: search parameters for self-play.
- `ai.self_play`: generated dataset defaults, including self-play batch size and worker count.
- `ai.train`: dataset, checkpoint, and optimizer defaults.

The default AlphaNet is `alphanet-v2` with `num_filters: 96`,
`num_res_blocks: 6`, and `value_hidden_dim: 128`. This is intended to better
match later-stage MCTS targets with 192/256 simulations than the earlier
64-filter, 3-block model. Checkpoints from the smaller `alphanet-v1` shape
should be loaded with matching model settings or retrained under the new config.

Generate a versioned AlphaZero-style self-play dataset:

```sh
python -m src.train.selfplay --output data/selfplay.pt --games 10 --simulations 64 --device cpu
```

Self-play supports two speed knobs:

- `--self-play-batch-size N` advances up to `N` games together in one process, batching neural-network leaf evaluations. This is the preferred CUDA path.
- `--self-play-workers N` shards games across worker processes. This is most useful for CPU self-play; on a single GPU, prefer one worker and a larger self-play batch.

Example CUDA generation with batched inference:

```sh
python -m src.train.selfplay --checkpoint models/latest.pth --output data/selfplay.pt --games 100 --simulations 128 --device cuda --self-play-batch-size 8
```

Train a model that the current pygame PVE mode can load through `config.yaml`:

```sh
python -m src.train.train --dataset data/selfplay.pt --output models/latest.pth --epochs 5 --device cpu
```

For a tiny end-to-end smoke run, generate data during training:

```sh
python -m src.train.train --generate-games 1 --simulations 4 --epochs 1 --device cpu
```

The dataset file uses `az-do-dataset-v2`. It stores the training tensors
`state`, `legal_mask`, MCTS `policy` visit distributions, and final `value`
targets, plus per-sample and per-game metadata. The metadata records exact game
boundaries, true ply, absolute player, chosen action, MCTS root statistics,
flip counts, pass counts, terminal scores, final margins, and generator
settings. The training command saves a raw `AlphaNet` state dict to
`models/latest.pth`, keeping it compatible with the pygame `GameAI` loader.

Inspect generated v2 datasets with the local data visualizer:

```sh
python webtools/data_visualizer/server.py --data-dir data
```

Use the root training entry point to run the default multi-stage schedule in
the background. It writes logs under `logs/` and runs the package module from
`src/train`:

```powershell
.\train.ps1
```

The default launcher uses CUDA, batched self-play, `--resume`, and
`--promote-latest`. Common overrides:

```powershell
.\train.ps1 -Device cpu
.\train.ps1 -Device cuda -SelfPlayBatchSize 16
.\train.ps1 -StartStage 3 -InitialCheckpoint models\stage2_iter01.pth
```

Watch the background logs:

```powershell
Get-Content .\logs\train_multistage.out -Tail 50 -Wait
Get-Content .\logs\train_multistage.err -Tail 50 -Wait
```

Stop a background run by PID, using the value printed by `train.ps1`:

```powershell
Stop-Process -Id <PID>
```

The default multi-stage schedule writes independent datasets and checkpoints:

- `data/stage1_bootstrap.pt` -> `models/stage1_bootstrap.pth`
- `data/stage2_iter01.pt` -> `models/stage2_iter01.pth`
- `data/stage3_iter02.pt` -> `models/stage3_iter02.pth`
- `data/stage4_iter03.pt` -> `models/stage4_iter03.pth`
- `data/stage5_final.pt` -> `models/stage5_final.pth`

The implementation scripts live under `src/train/train_multistage.py` and
`src/train/train_pipeline.py`; keep `train.ps1` as the root training entry
point.
