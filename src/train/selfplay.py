"""Self-play data generation for the AlphaZero-style training pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures
import random
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from src.train.dataset import DEFAULT_MODEL_VERSION, DODataset, make_dataset, save_dataset, validate_dataset
from src.config import (
    add_alphanet_model_args,
    apply_alphanet_arg_overrides,
    get_ai_config,
    get_alphanet_kwargs,
    resolve_torch_device,
)
from src.game.state import (
    GameState,
    apply_move,
    encode_state,
    is_terminal,
    legal_mask,
    legal_moves,
    new_game,
    pass_turn,
    winner,
)
from src.model.alphanet.network import AlphaNet
from src.model.mcts.mymcts import (
    AlphaZeroMCTS,
    MCTSConfig,
    NeuralEvaluator,
    UniformEvaluator,
    choose_action_from_distribution,
)


@dataclass(frozen=True)
class SelfPlayConfig:
    board_size: int = 9
    games: int = 1
    num_simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_drop_move: int = 20
    seed: int | None = None
    add_root_noise: bool = True
    model_version: str = DEFAULT_MODEL_VERSION
    batch_size: int = 1
    workers: int = 1


@dataclass
class RecordedSample:
    state: list[list[list[int]]]
    legal_mask: list[bool]
    policy: list[float]
    player: int


@dataclass
class _ActiveSelfPlayGame:
    state: GameState
    samples: list[RecordedSample]
    move_index: int
    rng: random.Random


def generate_self_play_dataset(
    *,
    model: torch.nn.Module | None = None,
    device: str = "cpu",
    config: SelfPlayConfig | None = None,
    model_kwargs: dict[str, int] | None = None,
    save_path: str | Path | None = None,
    show_progress: bool = False,
    progress_desc: str = "Self-play",
) -> DODataset:
    config = config or SelfPlayConfig()
    _validate_self_play_config(config)
    if config.workers > 1:
        dataset = _generate_self_play_dataset_parallel(
            model=model,
            device=device,
            config=config,
            model_kwargs=model_kwargs,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
    else:
        dataset = _generate_self_play_dataset_single(
            model=model,
            device=device,
            config=config,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )

    if save_path is not None:
        save_dataset(dataset, save_path)
    return dataset


def _generate_self_play_dataset_single(
    *,
    model: torch.nn.Module | None,
    device: str,
    config: SelfPlayConfig,
    show_progress: bool,
    progress_desc: str,
) -> DODataset:
    rng = random.Random(config.seed)
    evaluator = NeuralEvaluator(model, device=device) if model is not None else UniformEvaluator()
    progress_bar = None
    if show_progress:
        from tqdm import tqdm

        progress_bar = tqdm(total=config.games, desc=progress_desc, unit="game")

    try:
        results = play_self_play_games(
            evaluator=evaluator,
            config=config,
            rng=rng,
            on_game_done=(
                lambda completed, sample_count: _update_progress_bar(
                    progress_bar,
                    completed,
                    sample_count,
                )
            )
            if progress_bar is not None
            else None,
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return _make_dataset_from_game_results(
        results,
        board_size=config.board_size,
        model_version=config.model_version,
    )


def _generate_self_play_dataset_parallel(
    *,
    model: torch.nn.Module | None,
    device: str,
    config: SelfPlayConfig,
    model_kwargs: dict[str, int] | None,
    show_progress: bool,
    progress_desc: str,
) -> DODataset:
    workers = min(config.workers, config.games)
    if device.startswith("cuda"):
        warnings.warn(
            "Self-play workers with CUDA load one model per process. "
            "For one GPU, prefer workers=1 and a larger self-play batch size.",
            RuntimeWarning,
            stacklevel=2,
        )

    model_state_dict = None
    if model is not None:
        if model_kwargs is None:
            raise ValueError("model_kwargs is required when using workers with a model.")
        model_state_dict = {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        }

    shard_sizes = _split_games(config.games, workers)
    shard_configs = [
        replace(
            config,
            games=games,
            seed=_shard_seed(config.seed, index),
            workers=1,
        )
        for index, games in enumerate(shard_sizes)
        if games > 0
    ]

    progress_bar = None
    if show_progress:
        from tqdm import tqdm

        progress_bar = tqdm(total=config.games, desc=progress_desc, unit="game")

    payloads: list[dict[str, Any]] = []
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _self_play_worker,
                    shard_config,
                    device,
                    model_state_dict,
                    model_kwargs,
                )
                for shard_config in shard_configs
            ]
            for future in concurrent.futures.as_completed(futures):
                payload = future.result()
                payloads.append(payload)
                if progress_bar is not None:
                    progress_bar.update(int(payload["games"]))
                    progress_bar.set_postfix(samples=int(payload["states"].shape[0]))
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return _merge_dataset_payloads(
        payloads,
        board_size=config.board_size,
        model_version=config.model_version,
    )


def play_self_play_games(
    *,
    evaluator: UniformEvaluator | NeuralEvaluator,
    config: SelfPlayConfig,
    rng: random.Random,
    on_game_done: Callable[[int, int], None] | None = None,
) -> list[tuple[list[RecordedSample], int]]:
    results: list[tuple[list[RecordedSample], int]] = []
    active_games: list[_ActiveSelfPlayGame] = []
    games_started = 0
    total_samples = 0
    batch_size = max(1, config.batch_size)

    mcts = AlphaZeroMCTS(
        evaluator=evaluator,
        config=MCTSConfig(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
        ),
        rng=rng,
    )

    def start_games() -> None:
        nonlocal games_started
        while games_started < config.games and len(active_games) < batch_size:
            active_games.append(
                _ActiveSelfPlayGame(
                    state=new_game(config.board_size),
                    samples=[],
                    move_index=0,
                    rng=random.Random(rng.randrange(2**63)),
                )
            )
            games_started += 1

    start_games()
    while active_games:
        playable: list[_ActiveSelfPlayGame] = []
        completed: list[_ActiveSelfPlayGame] = []

        for game in active_games:
            while not is_terminal(game.state) and not legal_moves(
                game.state,
                game.state.current_player,
            ):
                game.state = pass_turn(game.state).state

            if is_terminal(game.state):
                completed.append(game)
            else:
                playable.append(game)

        if completed:
            for game in completed:
                game_winner = winner(game.state)
                results.append((game.samples, game_winner))
                total_samples += len(game.samples)
                if on_game_done is not None:
                    on_game_done(1, total_samples)
            completed_ids = {id(game) for game in completed}
            active_games = [game for game in active_games if id(game) not in completed_ids]
            start_games()
            if not playable:
                continue

        roots = mcts.search_batch(
            [game.state for game in playable],
            add_root_noise=config.add_root_noise,
        )

        for game, root in zip(playable, roots):
            player = game.state.current_player
            search_temperature = (
                config.temperature
                if game.move_index < config.temperature_drop_move
                else 0.0
            )
            policy = mcts.visit_distribution(root, temperature=1.0)
            action_distribution = mcts.visit_distribution(
                root,
                temperature=search_temperature,
            )
            action = choose_action_from_distribution(
                action_distribution,
                temperature=1.0,
                rng=game.rng,
            )

            mask = root.legal_mask_cache or legal_mask(game.state, player)
            if not mask[action]:
                raise RuntimeError("MCTS selected an illegal move.")

            game.samples.append(
                RecordedSample(
                    state=encode_state(game.state, player),
                    legal_mask=mask,
                    policy=policy,
                    player=player,
                )
            )

            size = game.state.size
            game.state = apply_move(
                game.state,
                player,
                (action // size, action % size),
                validate=False,
            ).state
            game.move_index += 1

    return results


def _make_dataset_from_game_results(
    game_results: list[tuple[list[RecordedSample], int]],
    *,
    board_size: int,
    model_version: str,
) -> DODataset:
    states: list[torch.Tensor] = []
    legal_masks: list[torch.Tensor] = []
    policies: list[torch.Tensor] = []
    values: list[float] = []

    for samples, game_winner in game_results:
        for sample in samples:
            states.append(torch.tensor(sample.state, dtype=torch.float32))
            legal_masks.append(torch.tensor(sample.legal_mask, dtype=torch.bool))
            policies.append(torch.tensor(sample.policy, dtype=torch.float32))
            if game_winner == 0:
                values.append(0.0)
            else:
                values.append(1.0 if game_winner == sample.player else -1.0)

    if not states:
        raise ValueError("Self-play produced no trainable samples.")

    dataset = make_dataset(
        torch.stack(states),
        torch.stack(legal_masks),
        torch.stack(policies),
        torch.tensor(values, dtype=torch.float32),
        board_size=board_size,
        model_version=model_version,
    )
    validate_dataset(dataset)
    return dataset


def _self_play_worker(
    config: SelfPlayConfig,
    device: str,
    model_state_dict: dict[str, torch.Tensor] | None,
    model_kwargs: dict[str, int] | None,
) -> dict[str, Any]:
    model = None
    if model_state_dict is not None:
        kwargs = dict(model_kwargs or {})
        kwargs["board_size"] = config.board_size
        model = AlphaNet(**kwargs).to(device)
        model.load_state_dict(model_state_dict)

    dataset = _generate_self_play_dataset_single(
        model=model,
        device=device,
        config=config,
        show_progress=False,
        progress_desc="Self-play",
    )
    payload = dataset.to_payload()
    payload["games"] = config.games
    return payload


def _merge_dataset_payloads(
    payloads: list[dict[str, Any]],
    *,
    board_size: int,
    model_version: str,
) -> DODataset:
    if not payloads:
        raise ValueError("Self-play produced no shards.")
    dataset = make_dataset(
        torch.cat([payload["states"] for payload in payloads], dim=0),
        torch.cat([payload["legal_masks"] for payload in payloads], dim=0),
        torch.cat([payload["policies"] for payload in payloads], dim=0),
        torch.cat([payload["values"] for payload in payloads], dim=0),
        board_size=board_size,
        model_version=model_version,
    )
    validate_dataset(dataset)
    return dataset


def _split_games(games: int, workers: int) -> list[int]:
    base = games // workers
    remainder = games % workers
    return [base + (1 if index < remainder else 0) for index in range(workers)]


def _shard_seed(seed: int | None, index: int) -> int | None:
    if seed is None:
        return None
    return seed + index * 1_000_003


def _validate_self_play_config(config: SelfPlayConfig) -> None:
    if config.games < 1:
        raise ValueError("Self-play games must be at least 1.")
    if config.num_simulations < 1:
        raise ValueError("MCTS simulations must be at least 1.")
    if config.batch_size < 1:
        raise ValueError("Self-play batch size must be at least 1.")
    if config.workers < 1:
        raise ValueError("Self-play workers must be at least 1.")


def _update_progress_bar(progress_bar: Any, completed: int, sample_count: int) -> None:
    progress_bar.update(completed)
    progress_bar.set_postfix(samples=sample_count)


def play_self_play_game(
    *,
    evaluator: UniformEvaluator | NeuralEvaluator,
    config: SelfPlayConfig,
    rng: random.Random,
) -> tuple[list[RecordedSample], int]:
    return play_self_play_games(
        evaluator=evaluator,
        config=replace(config, games=1, batch_size=1, workers=1),
        rng=rng,
    )[0]


def load_model_for_self_play(
    checkpoint: str | Path | None,
    *,
    board_size: int,
    device: str,
    model_kwargs: dict[str, int] | None = None,
) -> AlphaNet | None:
    if checkpoint is None:
        return None
    kwargs = dict(model_kwargs or {})
    kwargs["board_size"] = board_size
    model = AlphaNet(**kwargs).to(device)
    state = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model


def main() -> None:
    ai_config = get_ai_config()
    model_config = get_alphanet_kwargs()
    mcts_config = ai_config["mcts"]
    self_play_config = ai_config["self_play"]
    runtime_config = ai_config["runtime"]

    parser = argparse.ArgumentParser(description="Generate AlphaZero-style self-play data.")
    parser.add_argument("--output", default=self_play_config["output_path"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--board-size", type=int, default=model_config["board_size"])
    parser.add_argument("--games", type=int, default=self_play_config["games"])
    parser.add_argument("--simulations", type=int, default=mcts_config["num_simulations"])
    parser.add_argument("--self-play-batch-size", type=int, default=self_play_config["batch_size"])
    parser.add_argument("--self-play-workers", type=int, default=self_play_config["workers"])
    parser.add_argument("--device", default=runtime_config["device"])
    parser.add_argument("--seed", type=int, default=self_play_config["seed"])
    parser.add_argument("--no-root-noise", action="store_true")
    add_alphanet_model_args(parser, model_config)
    args = parser.parse_args()

    device = resolve_torch_device(args.device)
    model_config = apply_alphanet_arg_overrides(model_config, args)
    model = load_model_for_self_play(
        args.checkpoint,
        board_size=args.board_size,
        device=device,
        model_kwargs=model_config,
    )
    dataset = generate_self_play_dataset(
        model=model,
        device=device,
        config=SelfPlayConfig(
            board_size=args.board_size,
            games=args.games,
            num_simulations=args.simulations,
            c_puct=mcts_config["c_puct"],
            dirichlet_alpha=mcts_config["dirichlet_alpha"],
            dirichlet_epsilon=mcts_config["dirichlet_epsilon"],
            temperature=self_play_config["temperature"],
            temperature_drop_move=self_play_config["temperature_drop_move"],
            seed=args.seed,
            add_root_noise=(mcts_config["add_root_noise"] and not args.no_root_noise),
            model_version=ai_config["model"]["version"],
            batch_size=args.self_play_batch_size,
            workers=args.self_play_workers,
        ),
        model_kwargs=model_config,
        save_path=args.output,
    )
    print(f"Saved {len(dataset)} samples to {args.output}")


if __name__ == "__main__":
    main()
