"""Self-play data generation for the AlphaZero-style training pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import random
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
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
    score,
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
    game_id: int
    ply: int
    move_action: int
    temperature: float
    root_value: float
    root_visit_count: int
    chosen_visit_count: int
    chosen_prior: float
    chosen_q: float
    policy_entropy: float
    top_policy: float
    legal_count: int
    own_count: int
    opponent_count: int
    empty_count: int
    current_margin: int
    flipped_count: int


@dataclass
class RecordedGame:
    game_id: int
    samples: list[RecordedSample]
    winner: int
    move_count: int
    pass_count: int
    terminal_p1_count: int
    terminal_p2_count: int
    terminal_empty_count: int
    final_margin_p1: int


@dataclass
class _ActiveSelfPlayGame:
    game_id: int
    state: GameState
    samples: list[RecordedSample]
    move_index: int
    pass_count: int
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
        generator=_generator_metadata(config),
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
        generator=_generator_metadata(config),
    )


def play_self_play_games(
    *,
    evaluator: UniformEvaluator | NeuralEvaluator,
    config: SelfPlayConfig,
    rng: random.Random,
    on_game_done: Callable[[int, int], None] | None = None,
) -> list[RecordedGame]:
    results: list[RecordedGame] = []
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
                    game_id=games_started,
                    state=new_game(config.board_size),
                    samples=[],
                    move_index=0,
                    pass_count=0,
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
                game.pass_count += 1
                game.state = pass_turn(game.state).state

            if is_terminal(game.state):
                completed.append(game)
            else:
                playable.append(game)

        if completed:
            for game in completed:
                game_winner = winner(game.state)
                counts = score(game.state)
                results.append(
                    RecordedGame(
                        game_id=game.game_id,
                        samples=game.samples,
                        winner=game_winner,
                        move_count=game.move_index,
                        pass_count=game.pass_count,
                        terminal_p1_count=counts[1],
                        terminal_p2_count=counts[-1],
                        terminal_empty_count=counts[0],
                        final_margin_p1=counts[1] - counts[-1],
                    )
                )
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

            child = root.children[action]
            own_count = sum(row.count(player) for row in game.state.board)
            opponent_count = sum(row.count(-player) for row in game.state.board)
            empty_count = sum(row.count(0) for row in game.state.board)
            policy_entropy = _policy_entropy(policy)
            top_policy = max(policy) if policy else 0.0
            move_result = apply_move(
                game.state,
                player,
                (action // game.state.size, action % game.state.size),
                validate=False,
            )

            game.samples.append(
                RecordedSample(
                    state=encode_state(game.state, player),
                    legal_mask=mask,
                    policy=policy,
                    player=player,
                    game_id=game.game_id,
                    ply=game.move_index,
                    move_action=action,
                    temperature=search_temperature,
                    root_value=root.value_mean,
                    root_visit_count=root.visit_count,
                    chosen_visit_count=child.visit_count,
                    chosen_prior=child.prior,
                    chosen_q=-child.value_mean,
                    policy_entropy=policy_entropy,
                    top_policy=top_policy,
                    legal_count=sum(1 for allowed in mask if allowed),
                    own_count=own_count,
                    opponent_count=opponent_count,
                    empty_count=empty_count,
                    current_margin=own_count - opponent_count,
                    flipped_count=len(move_result.flipped),
                )
            )

            game.state = move_result.state
            game.move_index += 1

    return results


def _make_dataset_from_game_results(
    game_results: list[RecordedGame],
    *,
    board_size: int,
    model_version: str,
    generator: dict[str, Any],
) -> DODataset:
    states: list[torch.Tensor] = []
    legal_masks: list[torch.Tensor] = []
    policies: list[torch.Tensor] = []
    values: list[float] = []
    sample_metadata: dict[str, list[int | float]] = {
        "game_id": [],
        "ply": [],
        "absolute_player": [],
        "move_action": [],
        "temperature": [],
        "root_value": [],
        "root_visit_count": [],
        "chosen_visit_count": [],
        "chosen_prior": [],
        "chosen_q": [],
        "policy_entropy": [],
        "top_policy": [],
        "legal_count": [],
        "own_count": [],
        "opponent_count": [],
        "empty_count": [],
        "current_margin": [],
        "flipped_count": [],
    }
    game_metadata: dict[str, list[int]] = {
        "game_id": [],
        "first_player": [],
        "winner": [],
        "move_count": [],
        "pass_count": [],
        "terminal_p1_count": [],
        "terminal_p2_count": [],
        "terminal_empty_count": [],
        "final_margin_p1": [],
        "sample_start": [],
        "sample_count": [],
    }

    sample_start = 0
    for game in game_results:
        game_metadata["game_id"].append(game.game_id)
        game_metadata["first_player"].append(1)
        game_metadata["winner"].append(game.winner)
        game_metadata["move_count"].append(game.move_count)
        game_metadata["pass_count"].append(game.pass_count)
        game_metadata["terminal_p1_count"].append(game.terminal_p1_count)
        game_metadata["terminal_p2_count"].append(game.terminal_p2_count)
        game_metadata["terminal_empty_count"].append(game.terminal_empty_count)
        game_metadata["final_margin_p1"].append(game.final_margin_p1)
        game_metadata["sample_start"].append(sample_start)
        game_metadata["sample_count"].append(len(game.samples))
        sample_start += len(game.samples)

        for sample in game.samples:
            states.append(torch.tensor(sample.state, dtype=torch.float32))
            legal_masks.append(torch.tensor(sample.legal_mask, dtype=torch.bool))
            policies.append(torch.tensor(sample.policy, dtype=torch.float32))
            if game.winner == 0:
                values.append(0.0)
            else:
                values.append(1.0 if game.winner == sample.player else -1.0)
            sample_metadata["game_id"].append(sample.game_id)
            sample_metadata["ply"].append(sample.ply)
            sample_metadata["absolute_player"].append(sample.player)
            sample_metadata["move_action"].append(sample.move_action)
            sample_metadata["temperature"].append(sample.temperature)
            sample_metadata["root_value"].append(sample.root_value)
            sample_metadata["root_visit_count"].append(sample.root_visit_count)
            sample_metadata["chosen_visit_count"].append(sample.chosen_visit_count)
            sample_metadata["chosen_prior"].append(sample.chosen_prior)
            sample_metadata["chosen_q"].append(sample.chosen_q)
            sample_metadata["policy_entropy"].append(sample.policy_entropy)
            sample_metadata["top_policy"].append(sample.top_policy)
            sample_metadata["legal_count"].append(sample.legal_count)
            sample_metadata["own_count"].append(sample.own_count)
            sample_metadata["opponent_count"].append(sample.opponent_count)
            sample_metadata["empty_count"].append(sample.empty_count)
            sample_metadata["current_margin"].append(sample.current_margin)
            sample_metadata["flipped_count"].append(sample.flipped_count)

    if not states:
        raise ValueError("Self-play produced no trainable samples.")

    dataset = make_dataset(
        torch.stack(states),
        torch.stack(legal_masks),
        torch.stack(policies),
        torch.tensor(values, dtype=torch.float32),
        board_size=board_size,
        model_version=model_version,
        sample_metadata=_tensor_sample_metadata(sample_metadata),
        game_metadata=_tensor_game_metadata(game_metadata),
        generator=generator,
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
    generator: dict[str, Any],
) -> DODataset:
    if not payloads:
        raise ValueError("Self-play produced no shards.")
    sample_metadata_parts: list[dict[str, torch.Tensor]] = []
    game_metadata_parts: list[dict[str, torch.Tensor]] = []
    game_offset = 0
    sample_offset = 0
    for payload in payloads:
        sample_metadata = {
            field: tensor.clone()
            for field, tensor in payload["sample_metadata"].items()
        }
        game_metadata = {
            field: tensor.clone()
            for field, tensor in payload["game_metadata"].items()
        }
        sample_metadata["game_id"] += game_offset
        game_metadata["game_id"] += game_offset
        game_metadata["sample_start"] += sample_offset
        sample_metadata_parts.append(sample_metadata)
        game_metadata_parts.append(game_metadata)
        game_offset += int(game_metadata["game_id"].numel())
        sample_offset += int(payload["states"].shape[0])

    dataset = make_dataset(
        torch.cat([payload["states"] for payload in payloads], dim=0),
        torch.cat([payload["legal_masks"] for payload in payloads], dim=0),
        torch.cat([payload["policies"] for payload in payloads], dim=0),
        torch.cat([payload["values"] for payload in payloads], dim=0),
        board_size=board_size,
        model_version=model_version,
        sample_metadata=_cat_metadata(sample_metadata_parts),
        game_metadata=_cat_metadata(game_metadata_parts),
        generator=generator,
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


def _generator_metadata(config: SelfPlayConfig) -> dict[str, Any]:
    return asdict(config)


def _tensor_sample_metadata(data: dict[str, list[int | float]]) -> dict[str, torch.Tensor]:
    long_fields = {
        "game_id",
        "ply",
        "absolute_player",
        "move_action",
        "root_visit_count",
        "chosen_visit_count",
        "legal_count",
        "own_count",
        "opponent_count",
        "empty_count",
        "current_margin",
        "flipped_count",
    }
    return {
        field: torch.tensor(values, dtype=torch.long if field in long_fields else torch.float32)
        for field, values in data.items()
    }


def _tensor_game_metadata(data: dict[str, list[int]]) -> dict[str, torch.Tensor]:
    return {field: torch.tensor(values, dtype=torch.long) for field, values in data.items()}


def _cat_metadata(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not parts:
        return {}
    return {
        field: torch.cat([part[field] for part in parts], dim=0)
        for field in parts[0]
    }


def _policy_entropy(policy: list[float]) -> float:
    entropy = 0.0
    for probability in policy:
        if probability > 0.0:
            entropy -= probability * math.log(probability)
    return entropy


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
) -> RecordedGame:
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
