"""Self-play data generation for the AlphaZero-style training pipeline."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from src.dataset.dataset import DODataset, make_dataset, save_dataset, validate_dataset
from src.config import get_ai_config, get_alphanet_kwargs, resolve_torch_device
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


@dataclass
class RecordedSample:
    state: list[list[list[int]]]
    legal_mask: list[bool]
    policy: list[float]
    player: int


def generate_self_play_dataset(
    *,
    model: torch.nn.Module | None = None,
    device: str = "cpu",
    config: SelfPlayConfig | None = None,
    save_path: str | Path | None = None,
    show_progress: bool = False,
    progress_desc: str = "Self-play",
) -> DODataset:
    config = config or SelfPlayConfig()
    rng = random.Random(config.seed)
    evaluator = NeuralEvaluator(model, device=device) if model is not None else UniformEvaluator()

    states: list[torch.Tensor] = []
    legal_masks: list[torch.Tensor] = []
    policies: list[torch.Tensor] = []
    values: list[float] = []

    game_iter = range(config.games)
    progress_bar = None
    if show_progress:
        from tqdm import tqdm

        progress_bar = tqdm(game_iter, desc=progress_desc, unit="game")
        game_iter = progress_bar

    for _ in game_iter:
        game_rng = random.Random(rng.randrange(2**63))
        samples, game_winner = play_self_play_game(
            evaluator=evaluator,
            config=config,
            rng=game_rng,
        )
        for sample in samples:
            states.append(torch.tensor(sample.state, dtype=torch.float32))
            legal_masks.append(torch.tensor(sample.legal_mask, dtype=torch.bool))
            policies.append(torch.tensor(sample.policy, dtype=torch.float32))
            if game_winner == 0:
                values.append(0.0)
            else:
                values.append(1.0 if game_winner == sample.player else -1.0)
        if progress_bar is not None:
            progress_bar.set_postfix(samples=len(states))

    if not states:
        raise ValueError("Self-play produced no trainable samples.")

    dataset = make_dataset(
        torch.stack(states),
        torch.stack(legal_masks),
        torch.stack(policies),
        torch.tensor(values, dtype=torch.float32),
        board_size=config.board_size,
    )
    validate_dataset(dataset)
    if save_path is not None:
        save_dataset(dataset, save_path)
    return dataset


def play_self_play_game(
    *,
    evaluator: UniformEvaluator | NeuralEvaluator,
    config: SelfPlayConfig,
    rng: random.Random,
) -> tuple[list[RecordedSample], int]:
    state = new_game(config.board_size)
    samples: list[RecordedSample] = []
    move_index = 0
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

    while not is_terminal(state):
        player = state.current_player
        if not legal_moves(state, player):
            state = pass_turn(state).state
            continue

        root = mcts.search(state, add_root_noise=config.add_root_noise)
        search_temperature = config.temperature if move_index < config.temperature_drop_move else 0.0
        policy = mcts.visit_distribution(root, temperature=1.0)
        action_distribution = mcts.visit_distribution(root, temperature=search_temperature)
        action = choose_action_from_distribution(
            action_distribution,
            temperature=1.0,
            rng=rng,
        )

        mask = legal_mask(state, player)
        if not mask[action]:
            raise RuntimeError("MCTS selected an illegal move.")

        samples.append(
            RecordedSample(
                state=encode_state(state, player),
                legal_mask=mask,
                policy=policy,
                player=player,
            )
        )

        size = state.size
        state = apply_move(state, player, (action // size, action % size)).state
        move_index += 1

    return samples, winner(state)


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
    parser.add_argument("--device", default=runtime_config["device"])
    parser.add_argument("--seed", type=int, default=self_play_config["seed"])
    parser.add_argument("--no-root-noise", action="store_true")
    args = parser.parse_args()

    device = resolve_torch_device(args.device)
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
        ),
        save_path=args.output,
    )
    print(f"Saved {len(dataset)} samples to {args.output}")


if __name__ == "__main__":
    main()
