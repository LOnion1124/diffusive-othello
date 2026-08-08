"""Train AlphaNet from versioned self-play datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import get_ai_config, get_alphanet_kwargs, resolve_torch_device
from src.train.dataset import load_dataset, save_dataset, validate_dataset
from src.model.alphanet.network import AlphaNet
from src.train.selfplay import SelfPlayConfig, generate_self_play_dataset
from src.train.train_utils import train_step


def train_from_dataset(
    *,
    dataset_path: str | Path,
    output_path: str | Path = "models/latest.pth",
    init_checkpoint: str | Path | None = None,
    board_size: int | None = None,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    model_kwargs: dict[str, int] | None = None,
    show_progress: bool = False,
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
    if init_checkpoint is not None:
        state_dict = load_model_state_dict(init_checkpoint, device=device)
        model.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    last_loss = 0.0
    last_policy_loss = 0.0
    last_value_loss = 0.0
    steps = 0

    epoch_iter = range(1, epochs + 1)
    epoch_bar = None
    if show_progress:
        from tqdm import tqdm

        epoch_bar = tqdm(epoch_iter, desc="Training", unit="epoch")
        epoch_iter = epoch_bar

    for epoch in epoch_iter:
        batch_iter = dataloader
        batch_bar = None
        if show_progress:
            from tqdm import tqdm

            batch_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch}/{epochs}",
                unit="batch",
                leave=False,
            )
            batch_iter = batch_bar

        for batch in batch_iter:
            last_loss, last_policy_loss, last_value_loss = train_step(
                model,
                optimizer,
                batch,
                device=device,
            )
            steps += 1
            if batch_bar is not None:
                batch_bar.set_postfix(
                    loss=f"{last_loss:.4f}",
                    policy=f"{last_policy_loss:.4f}",
                    value=f"{last_value_loss:.4f}",
                )

        if epoch_bar is not None:
            epoch_bar.set_postfix(
                loss=f"{last_loss:.4f}",
                policy=f"{last_policy_loss:.4f}",
                value=f"{last_value_loss:.4f}",
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)

    metadata = {
        "board_size": board_size,
        "dataset_path": str(dataset_path),
        "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
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


def load_model_state_dict(path: str | Path, *, device: str = "cpu") -> dict[str, torch.Tensor]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain an AlphaNet state dict.")
    return checkpoint


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
    parser.add_argument("--init-checkpoint", default=None)
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
        init_checkpoint=args.init_checkpoint,
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
