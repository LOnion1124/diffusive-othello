"""Train AlphaNet from versioned self-play datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import get_ai_config, get_alphanet_kwargs, resolve_torch_device
from src.dataset.dataset import load_dataset, save_dataset, validate_dataset
from src.model.alphanet.network import AlphaNet
from src.selfplay.selfplay import SelfPlayConfig, generate_self_play_dataset
from src.train.train_utils import train_step


def train_from_dataset(
    *,
    dataset_path: str | Path,
    output_path: str | Path = "model/latest.pth",
    board_size: int | None = None,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    model_kwargs: dict[str, int] | None = None,
) -> dict[str, float | int | str]:
    dataset = load_dataset(dataset_path)
    validate_dataset(dataset)

    inferred_board_size = dataset.metadata.board_size
    if board_size is not None and board_size != inferred_board_size:
        raise ValueError(
            f"Requested board_size={board_size}, but dataset uses {inferred_board_size}."
        )
    board_size = inferred_board_size

    kwargs = dict(model_kwargs or {})
    kwargs["board_size"] = board_size
    model = AlphaNet(**kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    last_loss = 0.0
    last_policy_loss = 0.0
    last_value_loss = 0.0
    steps = 0
    for _ in range(epochs):
        for batch in dataloader:
            last_loss, last_policy_loss, last_value_loss = train_step(
                model,
                optimizer,
                batch,
                device=device,
            )
            steps += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)

    metadata = {
        "board_size": board_size,
        "dataset_path": str(dataset_path),
        "dataset_format_version": dataset.metadata.format_version,
        "rule_version": dataset.metadata.rule_version,
        "sample_count": len(dataset),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "optimizer": "Adam",
        "steps": steps,
        "output_path": str(output_path),
        "last_loss": last_loss,
        "last_policy_loss": last_policy_loss,
        "last_value_loss": last_value_loss,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    ai_config = get_ai_config()
    model_config = get_alphanet_kwargs()
    runtime_config = ai_config["runtime"]
    mcts_config = ai_config["mcts"]
    self_play_config = ai_config["self_play"]
    train_config = ai_config["train"]

    parser = argparse.ArgumentParser(description="Train AlphaNet for pygame inference.")
    parser.add_argument("--dataset", default=train_config["dataset_path"])
    parser.add_argument("--output", default=train_config["output_path"])
    parser.add_argument("--board-size", type=int, default=model_config["board_size"])
    parser.add_argument("--epochs", type=int, default=train_config["epochs"])
    parser.add_argument("--batch-size", type=int, default=train_config["batch_size"])
    parser.add_argument("--lr", type=float, default=train_config["lr"])
    parser.add_argument("--weight-decay", type=float, default=train_config["weight_decay"])
    parser.add_argument("--device", default=runtime_config["device"])
    parser.add_argument("--generate-games", type=int, default=0)
    parser.add_argument("--generated-dataset", default=self_play_config["output_path"])
    parser.add_argument("--simulations", type=int, default=mcts_config["num_simulations"])
    parser.add_argument("--seed", type=int, default=self_play_config["seed"])
    args = parser.parse_args()

    device = resolve_torch_device(args.device)
    dataset_path = args.dataset
    if args.generate_games > 0:
        dataset = generate_self_play_dataset(
            device=device,
            config=SelfPlayConfig(
                board_size=args.board_size,
                games=args.generate_games,
                num_simulations=args.simulations,
                c_puct=mcts_config["c_puct"],
                dirichlet_alpha=mcts_config["dirichlet_alpha"],
                dirichlet_epsilon=mcts_config["dirichlet_epsilon"],
                temperature=self_play_config["temperature"],
                temperature_drop_move=self_play_config["temperature_drop_move"],
                seed=args.seed,
                add_root_noise=mcts_config["add_root_noise"],
            ),
        )
        save_dataset(dataset, args.generated_dataset)
        dataset_path = args.generated_dataset
    elif dataset_path is None:
        raise SystemExit("Provide --dataset or use --generate-games N.")

    metadata = train_from_dataset(
        dataset_path=dataset_path,
        output_path=args.output,
        board_size=args.board_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        model_kwargs=model_config,
    )
    print(
        "Saved model to {output_path} after {steps} steps "
        "(loss={last_loss:.4f}, policy={last_policy_loss:.4f}, value={last_value_loss:.4f})".format(
            **metadata
        )
    )


if __name__ == "__main__":
    main()
