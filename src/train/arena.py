"""Seeded-random checkpoint arena evaluation and guarded promotion.

The arena gives each checkpoint the same number of games as first and second
player. Moves are sampled from MCTS visit counts without self-play noise, so
the result is randomized while remaining reproducible for a given seed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config import (
    add_alphanet_model_args,
    apply_alphanet_arg_overrides,
    get_ai_config,
    get_alphanet_kwargs,
    resolve_torch_device,
)
from src.game.state import (
    PLAYER_ONE,
    apply_move,
    is_terminal,
    legal_moves,
    new_game,
    pass_turn,
    score,
    winner,
)
from src.model.mcts.mymcts import (
    AlphaZeroMCTS,
    Evaluator,
    MCTSConfig,
    NeuralEvaluator,
    choose_action_from_distribution,
)
from src.train.selfplay import load_model_for_self_play


@dataclass(frozen=True)
class ArenaConfig:
    """Settings for a balanced, seeded-random checkpoint comparison."""

    board_size: int = 9
    games: int = 40
    num_simulations: int = 256
    c_puct: float = 1.5
    move_temperature: float = 1.0
    seed: int | None = 0


@dataclass(frozen=True)
class ArenaGameResult:
    game_index: int
    candidate_player: int
    winner: int
    candidate_margin: int
    move_count: int
    pass_count: int
    game_seed: int = 0


@dataclass(frozen=True)
class ArenaResult:
    candidate_wins: int
    incumbent_wins: int
    draws: int
    games: list[ArenaGameResult]

    @property
    def candidate_score(self) -> float:
        """Candidate match points, treating a draw as half a win."""

        total_games = len(self.games)
        if total_games == 0:
            return 0.0
        return (self.candidate_wins + 0.5 * self.draws) / total_games

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_score"] = self.candidate_score
        payload["candidate_as_first"] = self.color_summary(PLAYER_ONE)
        payload["candidate_as_second"] = self.color_summary(-PLAYER_ONE)
        return payload

    def color_summary(self, candidate_player: int) -> dict[str, float | int]:
        """Summarize candidate results for one assigned color."""

        games = [game for game in self.games if game.candidate_player == candidate_player]
        candidate_wins = sum(game.winner == candidate_player for game in games)
        incumbent_wins = sum(game.winner == -candidate_player for game in games)
        draws = len(games) - candidate_wins - incumbent_wins
        margin_sum = sum(game.candidate_margin for game in games)
        game_count = len(games)
        return {
            "games": game_count,
            "candidate_wins": candidate_wins,
            "incumbent_wins": incumbent_wins,
            "draws": draws,
            "candidate_score": (
                (candidate_wins + 0.5 * draws) / game_count if game_count else 0.0
            ),
            "candidate_margin_sum": margin_sum,
            "candidate_average_margin": margin_sum / game_count if game_count else 0.0,
        }


def run_checkpoint_arena(
    *,
    candidate_checkpoint: str | Path,
    incumbent_checkpoint: str | Path,
    device: str,
    config: ArenaConfig,
    model_kwargs: dict[str, int] | None = None,
) -> ArenaResult:
    """Load two AlphaNet checkpoints and compare them in a balanced arena."""

    _validate_arena_config(config)
    candidate_model = load_model_for_self_play(
        candidate_checkpoint,
        board_size=config.board_size,
        device=device,
        model_kwargs=model_kwargs,
    )
    incumbent_model = load_model_for_self_play(
        incumbent_checkpoint,
        board_size=config.board_size,
        device=device,
        model_kwargs=model_kwargs,
    )
    if candidate_model is None or incumbent_model is None:
        raise ValueError("Arena evaluation requires both candidate and incumbent checkpoints.")

    return run_arena(
        candidate_evaluator=NeuralEvaluator(candidate_model, device=device),
        incumbent_evaluator=NeuralEvaluator(incumbent_model, device=device),
        config=config,
    )


def run_arena(
    *,
    candidate_evaluator: Evaluator,
    incumbent_evaluator: Evaluator,
    config: ArenaConfig,
) -> ArenaResult:
    """Compare two evaluators with randomized games and balanced colors.

    The public evaluator-based form keeps the game loop independently testable
    and allows future non-neural baselines to use the same arena.
    """

    _validate_arena_config(config)
    seed_rng = random.Random(config.seed)
    candidate_players = [PLAYER_ONE] * (config.games // 2) + [-PLAYER_ONE] * (
        config.games // 2
    )
    seed_rng.shuffle(candidate_players)
    game_results: list[ArenaGameResult] = []
    candidate_wins = 0
    incumbent_wins = 0
    draws = 0

    for game_index, candidate_player in enumerate(candidate_players):
        game_seed = seed_rng.randrange(2**63)
        result = _play_game(
            candidate_evaluator=candidate_evaluator,
            incumbent_evaluator=incumbent_evaluator,
            candidate_player=candidate_player,
            config=config,
            rng=random.Random(game_seed),
            game_index=game_index,
            game_seed=game_seed,
        )
        game_results.append(result)
        if result.winner == 0:
            draws += 1
        elif result.winner == candidate_player:
            candidate_wins += 1
        else:
            incumbent_wins += 1

    return ArenaResult(
        candidate_wins=candidate_wins,
        incumbent_wins=incumbent_wins,
        draws=draws,
        games=game_results,
    )


def should_promote(result: ArenaResult, *, minimum_score: float = 0.5) -> bool:
    """Return whether a candidate strictly clears the required match score."""

    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be between 0 and 1.")
    return result.candidate_score > minimum_score


def promote_if_stronger(
    *,
    candidate_checkpoint: str | Path,
    incumbent_checkpoint: str | Path,
    result: ArenaResult,
    minimum_score: float = 0.5,
) -> bool:
    """Atomically replace the incumbent and its metadata only after a win."""

    if not should_promote(result, minimum_score=minimum_score):
        return False

    candidate_path = Path(candidate_checkpoint)
    incumbent_path = Path(incumbent_checkpoint)
    if candidate_path.resolve() == incumbent_path.resolve():
        raise ValueError("Candidate and incumbent checkpoints must be different files.")
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate checkpoint does not exist: {candidate_path}")

    _copy_file_atomically(candidate_path, incumbent_path)
    candidate_metadata = _metadata_path(candidate_path)
    incumbent_metadata = _metadata_path(incumbent_path)
    if candidate_metadata.exists():
        _copy_file_atomically(candidate_metadata, incumbent_metadata)
    elif incumbent_metadata.exists():
        incumbent_metadata.unlink()
    return True


def save_arena_result(
    result: ArenaResult,
    path: str | Path,
    *,
    candidate_checkpoint: str | Path | None = None,
    incumbent_checkpoint: str | Path | None = None,
    minimum_score: float | None = None,
    promoted: bool | None = None,
) -> None:
    """Write the complete arena record as JSON for later inspection."""

    payload = result.to_dict()
    if candidate_checkpoint is not None:
        payload["candidate_checkpoint"] = str(candidate_checkpoint)
    if incumbent_checkpoint is not None:
        payload["incumbent_checkpoint"] = str(incumbent_checkpoint)
    if minimum_score is not None:
        payload["minimum_score"] = minimum_score
    if promoted is not None:
        payload["promoted"] = promoted
    _write_json_atomically(Path(path), payload)


def update_checkpoint_metadata(
    checkpoint: str | Path,
    result: ArenaResult,
    *,
    incumbent_checkpoint: str | Path,
    minimum_score: float,
    promoted: bool,
) -> None:
    """Attach the most recent arena outcome to a training checkpoint sidecar."""

    path = _metadata_path(Path(checkpoint))
    metadata: dict[str, Any] = {}
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            metadata = loaded
    metadata["arena"] = {
        **result.to_dict(),
        "incumbent_checkpoint": str(incumbent_checkpoint),
        "minimum_score": minimum_score,
        "promoted": promoted,
    }
    _write_json_atomically(path, metadata)


def _play_game(
    *,
    candidate_evaluator: Evaluator,
    incumbent_evaluator: Evaluator,
    candidate_player: int,
    config: ArenaConfig,
    rng: random.Random,
    game_index: int,
    game_seed: int,
) -> ArenaGameResult:
    state = new_game(config.board_size)
    evaluators = {
        candidate_player: candidate_evaluator,
        -candidate_player: incumbent_evaluator,
    }
    searches = {
        player: AlphaZeroMCTS(
            evaluator=evaluator,
            config=MCTSConfig(num_simulations=config.num_simulations, c_puct=config.c_puct),
            rng=random.Random(rng.randrange(2**63)),
        )
        for player, evaluator in evaluators.items()
    }
    move_count = 0
    pass_count = 0

    while not is_terminal(state):
        player = state.current_player
        if not legal_moves(state, player):
            state = pass_turn(state).state
            pass_count += 1
            continue

        root = searches[player].search(state, add_root_noise=False)
        action = _select_arena_action(
            searches[player],
            root,
            move_temperature=config.move_temperature,
            rng=rng,
        )
        state = apply_move(
            state,
            player,
            (action // state.size, action % state.size),
            validate=False,
        ).state
        move_count += 1

    counts = score(state)
    return ArenaGameResult(
        game_index=game_index,
        candidate_player=candidate_player,
        winner=winner(state),
        candidate_margin=counts[candidate_player] - counts[-candidate_player],
        move_count=move_count,
        pass_count=pass_count,
        game_seed=game_seed,
    )


def _best_visit_action(root: Any) -> int:
    legal_children = [
        (action, child)
        for action, child in root.children.items()
        if action >= 0
    ]
    if not legal_children:
        raise RuntimeError("Arena MCTS did not produce a legal move.")
    return max(legal_children, key=lambda item: (item[1].visit_count, -item[0]))[0]


def _select_arena_action(
    search: AlphaZeroMCTS,
    root: Any,
    *,
    move_temperature: float,
    rng: random.Random,
) -> int:
    """Sample an action from MCTS visits, or choose the best action at zero temperature."""

    if move_temperature == 0:
        return _best_visit_action(root)
    distribution = search.visit_distribution(root, temperature=move_temperature)
    return choose_action_from_distribution(distribution, rng=rng)


def _validate_arena_config(config: ArenaConfig) -> None:
    if config.games < 2 or config.games % 2 != 0:
        raise ValueError("Arena games must be an even number of at least 2.")
    if config.num_simulations < 1:
        raise ValueError("Arena simulations must be at least 1.")
    if config.c_puct <= 0:
        raise ValueError("Arena c_puct must be positive.")
    if not math.isfinite(config.move_temperature) or config.move_temperature < 0:
        raise ValueError("Arena move_temperature must be finite and non-negative.")


def _metadata_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(checkpoint.suffix + ".json")


def _copy_file_atomically(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    ai_config = get_ai_config()
    model_config = get_alphanet_kwargs()
    runtime_config = ai_config["runtime"]
    mcts_config = ai_config["mcts"]
    arena_config = ai_config["arena"]

    parser = argparse.ArgumentParser(
        description="Compare AlphaNet checkpoints with balanced seeded-random MCTS games."
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--incumbent", default=runtime_config["model_path"])
    parser.add_argument("--device", default=runtime_config["device"])
    parser.add_argument("--board-size", type=int, default=model_config["board_size"])
    parser.add_argument("--games", type=int, default=arena_config["games"])
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--c-puct", type=float, default=mcts_config["c_puct"])
    parser.add_argument(
        "--move-temperature",
        type=float,
        default=arena_config["move_temperature"],
        help="Sample final moves from MCTS visits; use 0 for deterministic best-visit moves.",
    )
    parser.add_argument("--seed", type=int, default=arena_config["seed"])
    parser.add_argument("--minimum-score", type=float, default=arena_config["minimum_score"])
    parser.add_argument("--result", default=None)
    parser.add_argument("--promote", action="store_true")
    add_alphanet_model_args(parser, model_config)
    args = parser.parse_args()

    device = resolve_torch_device(args.device)
    model_config = apply_alphanet_arg_overrides(model_config, args)
    result = run_checkpoint_arena(
        candidate_checkpoint=args.candidate,
        incumbent_checkpoint=args.incumbent,
        device=device,
        config=ArenaConfig(
            board_size=args.board_size,
            games=args.games,
            num_simulations=args.simulations,
            c_puct=args.c_puct,
            move_temperature=args.move_temperature,
            seed=args.seed,
        ),
        model_kwargs=model_config,
    )
    promoted = False
    if args.promote:
        promoted = promote_if_stronger(
            candidate_checkpoint=args.candidate,
            incumbent_checkpoint=args.incumbent,
            result=result,
            minimum_score=args.minimum_score,
        )

    result_path = Path(args.result) if args.result else Path(args.candidate).with_suffix(
        Path(args.candidate).suffix + ".arena.json"
    )
    save_arena_result(
        result,
        result_path,
        candidate_checkpoint=args.candidate,
        incumbent_checkpoint=args.incumbent,
        minimum_score=args.minimum_score,
        promoted=promoted,
    )
    first = result.color_summary(PLAYER_ONE)
    second = result.color_summary(-PLAYER_ONE)
    print(
        "Arena: candidate {candidate_wins}-{incumbent_wins}-{draws} incumbent "
        "(score={score:.3f}; first={first_wins}-{first_losses}-{first_draws}; "
        "second={second_wins}-{second_losses}-{second_draws}; promoted={promoted})".format(
            candidate_wins=result.candidate_wins,
            incumbent_wins=result.incumbent_wins,
            draws=result.draws,
            score=result.candidate_score,
            first_wins=first["candidate_wins"],
            first_losses=first["incumbent_wins"],
            first_draws=first["draws"],
            second_wins=second["candidate_wins"],
            second_losses=second["incumbent_wins"],
            second_draws=second["draws"],
            promoted=promoted,
        )
    )
    print(f"Saved arena result to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
