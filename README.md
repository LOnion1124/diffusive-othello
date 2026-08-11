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
python gui.py
```

The start screen lets you choose `Play PvP` or `Play PvE`. For scripted quick
starts, `python gui.py --mode PVP` and `python gui.py --mode PVE` are still
available.

Both GUI modes show a first-player vs. second-player win-rate prediction bar during gameplay. The bar attempts to load the configured model in either PVP or PVE; if the model or optional AI dependencies are unavailable, it locks to 50:50 and displays `Invalid`. PVE also reports the error in the terminal and uses a simple legal-move fallback for AI moves.

In PvE, the sidebar, turn indicator, win projection, and result dialog identify the
two sides as `YOU` and `COMPUTER`; a `YOU` victory is displayed as `YOU WIN`.
The sidebar also provides checkboxes to hide or restore the win projection and
move suggestions, plus `END MATCH` to end the current game and settle its
current board score immediately.

When model inference is available, the board uses the existing legal-move
breathing highlight for the current human player's three highest-probability
policy moves. Their center dots become probability values; the expanded ring
keeps the normal legal-move color and breathes up to a cell-safe maximum size,
while the text uses bright gray for the first choice and light gray for the
other two.
PVE hides all legal-position and probability highlights during the AI's turn.

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

On Linux, use the equivalent Bash entry point. It automatically prefers
`venv/bin/python`, falling back to `python3`, and also runs in the background:

```sh
bash ./train.sh
```

The default launcher uses CUDA, batched self-play, `--resume`, and
`--promote-latest`. Common overrides:

```powershell
.\train.ps1 -Device cpu
.\train.ps1 -Device cuda -SelfPlayBatchSize 16
.\train.ps1 -StartStage 3 -InitialCheckpoint models\stage2_iter01.pth
```

Linux accepts equivalent kebab-case options:

```sh
bash ./train.sh --device cuda --self-play-batch-size 16
bash ./train.sh --start-stage 3 --initial-checkpoint models/stage2_iter01.pth
```

### Continue from a strong checkpoint

Use the continuation schedule to repeat the stage-5 workload (2,000 self-play
games, 256 MCTS simulations, and five training epochs) from an existing strong
model. Its default baseline and promotion target are `models/latest.pth`:

```powershell
.\train.ps1 -Schedule continue
```

```sh
bash ./train.sh --schedule continue
```

Each round writes an independent candidate such as
`data/continue1_stage5.pt` and `models/continue1_stage5.pth`. Before replacing
the incumbent, it runs an arena with an equal number of first-player and
second-player games. With its default positive temperature, each final move is
sampled from the MCTS visit distribution; root noise remains disabled. Arena
games are therefore randomized but reproducible for a given seed. Set the
arena temperature to `0` to use deterministic best-visit moves. A candidate
replaces the incumbent only when its match score is strictly above 50% (`win +
0.5 * draw`); ties keep the existing checkpoint. Run several sequential
continuation rounds in one background job with:

```powershell
.\train.ps1 -Schedule continue -EndStage 3
```

```sh
bash ./train.sh --schedule continue --end-stage 3
```

The continuation schedule keeps the current incumbent after a rejected round,
and stops the job before generating any later rounds. After a successful arena,
the candidate is promoted to `models/latest.pth`; the next stage uses that
accepted product model as both its self-play and training checkpoint. The
arena always compares a candidate with the checkpoint that started its current
stage. `-ArenaGames`, `-ArenaSimulations`, `-ArenaTemperature`, and
`-ArenaMinimumScore` expose the corresponding arena settings. Arena game
counts must be even so that each checkpoint receives the same number of games
as each color. Defaults live under `ai.arena` in `config.yaml`.

You can also compare and optionally promote checkpoints independently:

```sh
python -m src.train.arena --candidate models/continue1_stage5.pth --incumbent models/latest.pth --games 40 --simulations 256 --move-temperature 1.0 --seed 0 --device cuda --promote
```

The command writes a full per-game record, including per-game random seeds and
candidate first-player/second-player summaries, to
`models/continue1_stage5.pth.arena.json`. Promotion atomically replaces the
incumbent checkpoint and its `.pth.json` metadata sidecar only after the
candidate clears `--minimum-score` (default `0.5`). Checkpoints must use the
model architecture supplied by `config.yaml` or matching command-line model
overrides.

Watch the background logs:

```powershell
Get-Content .\logs\train_multistage.out -Tail 50 -Wait
Get-Content .\logs\train_multistage.err -Tail 50 -Wait
```

Stop a background run by PID, using the value printed by `train.ps1`:

```powershell
Stop-Process -Id <PID>
```

On Linux, follow the logs and stop a process with:

```sh
tail -n 50 -f logs/train_multistage.out
tail -n 50 -f logs/train_multistage.err
kill <PID>
```

Pass `--foreground` to `train.sh` to keep the trainer attached to the current
terminal instead of starting it with `nohup`.

The default multi-stage schedule writes independent datasets and checkpoints:

- `data/stage1_bootstrap.pt` -> `models/stage1_bootstrap.pth`
- `data/stage2_iter01.pt` -> `models/stage2_iter01.pth`
- `data/stage3_iter02.pt` -> `models/stage3_iter02.pth`
- `data/stage4_iter03.pt` -> `models/stage4_iter03.pth`
- `data/stage5_final.pt` -> `models/stage5_final.pth`

Continuation rounds use separate artifacts and do not overwrite the curriculum
stage files:

- `data/continue1_stage5.pt` -> `models/continue1_stage5.pth`
- `data/continue2_stage5.pt` -> `models/continue2_stage5.pth`

The implementation scripts live under `src/train/train_multistage.py` and
`src/train/train_pipeline.py`; keep `train.ps1` (Windows) and `train.sh`
(Linux) as the root training entry points.
