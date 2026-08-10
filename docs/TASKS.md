# Diffusive Othello Improvement Tasks

This document turns the current project assessment into an implementation roadmap. The two primary goals are:

1. Build an excellent, reliable game implementation.
2. Build an AI training method that can learn effective play.

## Current Quality Assessment

### 1. Game Model Design

Current status: prototype quality.

The core game rules are readable, but the implementation is not reliable enough to serve as the source of truth for training.

Key issues:

- The game rules are implemented more than once: `src/game/logic.py` and `src/model/mcts/mymcts.py` each contain their own board and move logic.
- Player and score representation relies on fragile Python indexing behavior for `-1`.
- Move application assumes valid input and does not return a structured result.
- There is no single public API for legal moves, pass turns, terminal detection, winner calculation, state copying, or serialization.
- There are no rule-level tests for initial position, legal move masks, diffusion flips, pass turns, scoring, or terminal states.

Target design:

- Create one authoritative game engine.
- Make the game state easy to copy, serialize, test, and consume from UI, CLI, MCTS, and neural-network code.
- Treat all rule behavior as testable pure logic where possible.

### 2. Game UI Design

Current status: playable pygame prototype.

The pygame UI works as a first interface, but rendering, event handling, game control, and AI loading are tightly coupled.

Key issues:

- `gui.py` initializes pygame, loads AI, creates globals, and starts the main loop at import time.
- PVP mode still initializes `GameAI`, which can break local play when model or torch dependencies are missing.
- UI code reads internal board objects directly instead of using a stable game-state API.
- Rendering, input handling, state transitions, and AI turns are mixed in one file.

Target design:

- Keep pygame as a maintained desktop client.
- Split the pygame client into app, renderer, input controller, and player adapters.
- Load AI only when needed.
- Add a web game client as a separate development target, backed by the same authoritative game engine.

### 3. Game AI Training Method

Current status: barely working.

The main blocker is not only model architecture. The training loop does not yet form a consistent reinforcement-learning pipeline.

Key issues:

- The self-play path and MCTS path use different game logic and different data formats.
- The MCTS wrapper depends on `mcts_simple`, which is not listed in `requirements.txt` and is not available in the current environment.
- Current policy targets are mostly one-hot selected moves rather than MCTS visit-count distributions.
- Current value targets are inconsistent: one path uses a positive heuristic, while the network outputs values in `[-1, 1]`.
- Some simulated moves do not apply the full diffusion rule.
- The game termination condition differs between game logic and MCTS code.

Target design:

- Use AlphaZero-style self-play:
  - neural network predicts policy prior and value;
  - MCTS searches from the canonical game state;
  - training data stores `(state, legal_mask, visit_distribution, outcome_value)`;
  - the model trains with policy cross entropy plus value MSE;
  - new checkpoints are accepted through arena evaluation.

### 4. Overall Code Structure

Current status: small prototype with weak module boundaries.

Key issues:

- Game rules, dataset generation, inference, MCTS, UI, and scripts are coupled through ad hoc data formats.
- Configuration is loaded from `config.yaml` relative to the process working directory.
- Dependencies are incomplete.
- Large generated artifacts exist under `data/` and `models/`.
- There is no test suite, reproducible training command, evaluation command, or checkpoint metadata.

Target design:

- Establish clear module ownership.
- Keep generated artifacts separate from source.
- Add tests and reproducible commands before expanding AI experiments.

## Refactoring Roadmap

### Phase 1: Authoritative Game Engine

Goal: make the game implementation reliable enough for UI and AI.

Status: implemented. The canonical rule API lives in `src/game/state.py`, with legacy UI/CLI compatibility through `src/game/logic.py` and CPU-only rule tests in `tests/test_game_state.py`.

Tasks:

- Add a canonical game state module, for example `src/game/state.py`.
- Represent the board with a simple data structure suitable for copying and tensor conversion.
- Add public APIs:
  - `new_game(size)`;
  - `legal_moves(state, player)`;
  - `legal_mask(state, player)`;
  - `apply_move(state, player, move)`;
  - `pass_turn(state)`;
  - `is_terminal(state)`;
  - `winner(state)`;
  - `score(state)`;
  - `encode_state(state, player)`.
- Remove duplicated rule logic from MCTS and self-play modules.
- Add rule tests for all core game behavior.
- Fix player-score handling so `1`, `-1`, and empty cells are counted explicitly.

Acceptance criteria:

- CLI, pygame UI, self-play, and inference can all consume the same game engine.
- Rule tests pass on CPU without torch.
- There is only one implementation of legal moves and diffusion flips.

### Phase 2: Pygame Client Cleanup

Goal: keep the desktop game usable while decoupling it from AI and training code.

Status: implemented. The desktop client now enters through thin `gui.py`, with game flow in `src/ui/game_controller.py`, input mapping in `src/ui/input_controller.py`, rendering in `src/ui/pygame_renderer.py`, and the pygame loop in `src/ui/pygame_app.py`. PVP and PVE both attempt model-backed first-player vs. second-player win-rate prediction during gameplay; when model loading or inference fails, the bar locks to 50:50 and displays `Invalid`. PVE also uses a first-legal-move fallback when AI move inference cannot run.

Tasks:

- Move pygame app code into a package such as `src/ui/pygame_app.py`.
- Keep top-level `gui.py` as a thin entry point.
- Split renderer, input handling, and game-controller logic.
- Delay-load AI prediction in gameplay.
- Add a fallback legal-move policy if AI inference fails.
- Ensure PVP works without torch or model files installed by showing an invalid
  win-rate prediction state.
- Show first-player vs. second-player win-rate prediction in PVE and PVP.

Acceptance criteria:

- `python gui.py --mode PVP` runs in a minimal pygame environment with the
  win-rate bar marked invalid when optional AI dependencies are missing.
- `python gui.py --mode PVE` reports clear errors when model dependencies are missing.
- UI code does not access private board internals.

### Phase 3: Web Game Client

Goal: add a browser-playable version outside the pygame game.

Tasks:

- Choose a web architecture:
  - simple option: local static app using TypeScript and a port of the game engine;
  - Python-backed option: FastAPI service exposing legal moves, apply move, and AI inference endpoints;
  - full app option: React or Vite frontend plus Python AI backend.
- Define a shared game-state JSON format.
- Implement PVP browser play first.
- Add legal move highlighting, scoreboard, pass-turn handling, restart, and game-over state.
- Add optional PVE mode through a backend AI endpoint after the training/inference API stabilizes.

Acceptance criteria:

- The first screen is the playable board, not a landing page.
- PVP web play matches the canonical game-engine tests.
- Web and pygame clients produce identical results for the same move sequence.

### Phase 4: AI Data Pipeline

Goal: replace ad hoc self-play data with a consistent training dataset.

Status: implemented. The canonical AlphaZero-style v2 dataset lives in `src/train/dataset.py`, local PUCT MCTS lives in `src/model/mcts/mymcts.py`, self-play generation lives in `src/train/selfplay.py`, and training utilities consume soft MCTS visit distributions. The old heuristic-value self-play path and external `mcts_simple` dependency have been replaced. Dataset v2 also records exact per-sample and per-game analysis metadata for the local data visualizer.

Tasks:

- Define one training sample schema:
  - `state`: current-player perspective tensor;
  - `legal_mask`: flattened legal action mask;
  - `policy`: MCTS visit-count distribution;
  - `value`: final outcome from the current player's perspective.
- Remove or quarantine the old heuristic-value self-play path.
- Store datasets in a versioned format with rule version, board size, and model version metadata.
- Store exact analysis metadata:
  - per-sample game id, ply, absolute player, selected action, MCTS root stats,
    policy concentration, piece counts, current margin, and flip count;
  - per-game boundaries, winner, move count, pass count, terminal score, and
    final margin.
- Add dataset validation:
  - policy sums to 1 over legal moves;
  - illegal moves have zero probability;
  - value is in `[-1, 1]`;
  - tensor shapes match board size.

Acceptance criteria:

- A generated dataset can be loaded and validated without notebooks.
- The dataset format is shared by training, evaluation, and debugging tools.

### Phase 5: MCTS and Self-Play

Goal: build a useful search-driven training loop.

Status: implemented and optimized. Project-local MCTS now supports cached
legal move/mask/terminal data, batched neural-network leaf evaluation, batched
multi-game self-play, and process-level self-play sharding.

Tasks:

- Implement or vendor a small project-local MCTS instead of depending on missing `mcts_simple`.
- Use the canonical game engine for all transitions.
- Add neural-network priors and value estimates.
- Add Dirichlet noise at the root during training games.
- Use visit-count distributions as policy targets.
- Add temperature scheduling for early-game exploration and late-game exploitation.
- Batch neural-network leaf evaluations across active self-play games.
- Shard self-play games across worker processes for CPU generation.
- Support random, heuristic, MCTS, and neural-network players for benchmarking.

Acceptance criteria:

- Self-play can generate valid samples from scratch.
- MCTS never produces illegal moves.
- Random-vs-MCTS and old-model-vs-new-model matches can be run from scripts.

### Phase 6: Training and Evaluation

Goal: make model improvement measurable.

Status: partially implemented. Training checkpoints include dataset and
optimizer metadata, and `src/train/arena.py` now runs balanced checkpoint
seeded-random arenas. The `continue` multi-stage schedule repeats stage-5 strength from
`models/latest.pth` by default and promotes a candidate only when its arena
score strictly exceeds the configured threshold. Each accepted candidate is
the next stage's product checkpoint; a failed arena stops the run before later
stages train. Baseline win-rate tracking against random and heuristic players
remains future work.

Tasks:

- Add a proper training script, for example `python -m src.train.train`.
- Save checkpoint metadata:
  - board size;
  - rule version;
  - training steps;
  - optimizer settings;
  - dataset version;
  - evaluation results.
- Add arena evaluation before replacing `models/latest.pth`.
- Track baseline win rates against random, heuristic, and previous model players.
- Add CPU smoke tests for model forward pass and one tiny training step.

Acceptance criteria:

- Training is reproducible from documented commands.
- New checkpoints are promoted only when evaluation improves.
- The model can be used by both pygame and web clients through the same inference adapter.

### Phase 7: Repository Hygiene

Goal: make the project maintainable for future agents and humans.

Tasks:

- Complete dependency documentation.
- Add `pytest` tests.
- Add linting or formatting if the project grows.
- Keep generated data and checkpoints out of normal source commits unless intentionally versioned.
- Update docs whenever a phase changes behavior, commands, architecture, or file ownership.

Acceptance criteria:

- A fresh contributor can install, play PVP, run tests, and understand the next milestone from docs.
- Agent handoffs start from `AGENT.md`.
