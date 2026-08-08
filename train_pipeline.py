"""Run self-play data generation followed by AlphaNet training."""

from __future__ import annotations

import argparse

from src.config import (
    add_alphanet_model_args,
    apply_alphanet_arg_overrides,
    get_ai_config,
    get_alphanet_kwargs,
    resolve_torch_device,
)
from src.train.selfplay import (
    SelfPlayConfig,
    generate_self_play_dataset,
    load_model_for_self_play,
)
from src.train.train import train_from_dataset


def parse_args() -> argparse.Namespace:
    ai_config = get_ai_config()
    model_config = get_alphanet_kwargs()
    runtime_config = ai_config["runtime"]
    mcts_config = ai_config["mcts"]
    self_play_config = ai_config["self_play"]
    train_config = ai_config["train"]

    parser = argparse.ArgumentParser(
        description="Generate self-play data and train a pygame-compatible AlphaNet model."
    )
    parser.add_argument("--checkpoint", default=None, help="Optional model used by self-play MCTS.")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional model weights used to initialize training.",
    )
    parser.add_argument(
        "--train-from-scratch",
        action="store_true",
        help="Do not initialize training from --checkpoint.",
    )
    parser.add_argument("--dataset", default=self_play_config["output_path"])
    parser.add_argument("--output", default=train_config["output_path"])
    parser.add_argument("--board-size", type=int, default=model_config["board_size"])
    parser.add_argument("--device", default=runtime_config["device"])
    parser.add_argument("--games", type=int, default=self_play_config["games"])
    parser.add_argument("--simulations", type=int, default=mcts_config["num_simulations"])
    parser.add_argument("--epochs", type=int, default=train_config["epochs"])
    parser.add_argument("--batch-size", type=int, default=train_config["batch_size"])
    parser.add_argument("--lr", type=float, default=train_config["lr"])
    parser.add_argument("--weight-decay", type=float, default=train_config["weight_decay"])
    parser.add_argument("--seed", type=int, default=self_play_config["seed"])
    parser.add_argument("--temperature", type=float, default=self_play_config["temperature"])
    parser.add_argument(
        "--temperature-drop-move",
        type=int,
        default=self_play_config["temperature_drop_move"],
    )
    parser.add_argument("--no-root-noise", action="store_true")
    add_alphanet_model_args(parser, model_config)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ai_config = get_ai_config()
    model_config = apply_alphanet_arg_overrides(get_alphanet_kwargs(), args)
    mcts_config = ai_config["mcts"]
    device = resolve_torch_device(args.device)
    init_checkpoint = None
    if not args.train_from_scratch:
        init_checkpoint = args.init_checkpoint or args.checkpoint

    print(f"Using device: {device}")
    if args.checkpoint is not None:
        print(f"Self-play checkpoint: {args.checkpoint}")
    if init_checkpoint is not None:
        print(f"Training init checkpoint: {init_checkpoint}")
    elif args.train_from_scratch:
        print("Training init checkpoint: none (from scratch)")

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
            temperature=args.temperature,
            temperature_drop_move=args.temperature_drop_move,
            seed=args.seed,
            add_root_noise=(mcts_config["add_root_noise"] and not args.no_root_noise),
            model_version=ai_config["model"]["version"],
        ),
        save_path=args.dataset,
        show_progress=True,
        progress_desc="Self-play",
    )
    print(f"Saved {len(dataset)} samples to {args.dataset}")

    metadata = train_from_dataset(
        dataset_path=args.dataset,
        output_path=args.output,
        init_checkpoint=init_checkpoint,
        board_size=args.board_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        model_kwargs=model_config,
        show_progress=True,
    )
    print(
        "Saved model to {output_path} after {steps} steps "
        "(loss={last_loss:.4f}, policy={last_policy_loss:.4f}, value={last_value_loss:.4f})".format(
            **metadata
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
