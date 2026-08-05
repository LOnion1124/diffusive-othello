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

This game is a variant of the classic board game *Reversi*. It is played on a 9×9 board. At the start of the game, each player has two pieces placed in the corners on their respective opposite sides of the board. The players then take turns placing pieces.

A move is valid only if:

1. The target cell is not already occupied; and
2. At least one adjacent orthogonally neighboring cell (up, down, left, or right) contains one of the player’s own pieces.

Whenever a player makes a valid move, all of the opponent’s pieces located in the eight surrounding cells (orthogonal and diagonal) are flipped to the player’s color. If a player has no valid moves, the turn automatically passes to the opponent. The game ends when neither player can make a move (usually when the board is full). The player with more pieces on the board at the end wins.

This game is still under development, and its rules may be adjusted in future updates.

## Train Your Own AI

AI defaults live under the `ai` section of `config.yaml`:

- `ai.runtime`: inference/training device and pygame model path.
- `ai.model`: AlphaNet architecture and board size.
- `ai.mcts`: search parameters for self-play.
- `ai.self_play`: generated dataset defaults.
- `ai.train`: dataset, checkpoint, and optimizer defaults.

Generate a versioned AlphaZero-style self-play dataset:

```sh
python -m src.selfplay.selfplay --output data/selfplay.pt --games 10 --simulations 64 --device cpu
```

Train a model that the current pygame PVE mode can load through `config.yaml`:

```sh
python -m src.train.train --dataset data/selfplay.pt --output model/latest.pth --epochs 5 --device cpu
```

For a tiny end-to-end smoke run, generate data during training:

```sh
python -m src.train.train --generate-games 1 --simulations 4 --epochs 1 --device cpu
```

The dataset file stores `state`, `legal_mask`, MCTS `policy` visit distributions, final `value` targets, and metadata for dataset format, rule version, board size, and model version. The training command saves a raw `AlphaNet` state dict to `model/latest.pth`, keeping it compatible with the pygame `GameAI` loader.
