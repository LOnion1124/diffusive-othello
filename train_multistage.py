"""Run a multi-stage AlphaZero-style training schedule."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import get_ai_config, get_alphanet_kwargs, resolve_torch_device
from src.train.selfplay import (
    SelfPlayConfig,
    generate_self_play_dataset,
    load_model_for_self_play,
)
from src.train.train import train_from_dataset


@dataclass(frozen=True)
class StageSpec:
    index: int
    name: str
    games: int
    simulations: int
    epochs: int
    batch_size: int
    lr: float


SMOKE_STAGE = StageSpec(
    index=0,
    name="smoke",
    games=5,
    simulations=8,
    epochs=1,
    batch_size=64,
    lr=1e-3,
)

DEFAULT_STAGES = (
    StageSpec(1, "bootstrap", games=300, simulations=64, epochs=15, batch_size=128, lr=1e-3),
    StageSpec(2, "iter01", games=500, simulations=96, epochs=10, batch_size=128, lr=5e-4),
    StageSpec(3, "iter02", games=1000, simulations=128, epochs=8, batch_size=256, lr=3e-4),
    StageSpec(4, "iter03", games=1500, simulations=192, epochs=6, batch_size=256, lr=2e-4),
    StageSpec(5, "final", games=2000, simulations=256, epochs=5, batch_size=256, lr=1e-4),
)


def parse_args() -> argparse.Namespace:
    ai_config = get_ai_config()
    model_config = get_alphanet_kwargs()
    runtime_config = ai_config["runtime"]
    train_config = ai_config["train"]
    self_play_config = ai_config["self_play"]

    parser = argparse.ArgumentParser(
        description="Run multi-stage self-play and continued AlphaNet training."
    )
    parser.add_argument("--device", default=runtime_config["device"])
    parser.add_argument("--board-size", type=int, default=model_config["board_size"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--prefix", default="stage")
    parser.add_argument("--initial-checkpoint", default=None)
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--end-stage", type=int, default=5)
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--promote-latest", action="store_true")
    parser.add_argument("--latest-path", default=ai_config["runtime"]["model_path"])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--weight-decay", type=float, default=train_config["weight_decay"])
    parser.add_argument("--temperature", type=float, default=self_play_config["temperature"])
    parser.add_argument(
        "--temperature-drop-move",
        type=int,
        default=self_play_config["temperature_drop_move"],
    )
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--no-root-noise", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_stages = select_stages(
        include_smoke=args.include_smoke,
        start_stage=args.start_stage,
        end_stage=args.end_stage,
    )
    if not selected_stages:
        raise SystemExit("No stages selected.")

    device = resolve_torch_device(args.device)
    ai_config = get_ai_config()
    model_config = get_alphanet_kwargs()
    mcts_config = ai_config["mcts"]
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    manifest_path = Path(args.manifest) if args.manifest else model_dir / f"{args.prefix}_manifest.json"

    print(f"Using device: {device}", flush=True)
    print(
        "Selected stages: "
        + ", ".join(f"{stage.index}:{stage.name}" for stage in selected_stages),
        flush=True,
    )

    previous_checkpoint = resolve_initial_checkpoint(
        args.initial_checkpoint,
        selected_stages[0],
        include_smoke=args.include_smoke,
        model_dir=model_dir,
        prefix=args.prefix,
    )
    records: list[dict] = []

    for stage in selected_stages:
        dataset_path, output_path = stage_paths(stage, data_dir=data_dir, model_dir=model_dir, prefix=args.prefix)
        checkpoint = previous_checkpoint

        if output_path.exists() and args.resume:
            print(f"\n[{stage.index}:{stage.name}] existing model found, skipping: {output_path}", flush=True)
            previous_checkpoint = str(output_path)
            records.append(stage_record(stage, dataset_path, output_path, checkpoint, skipped=True))
            continue

        ensure_outputs_are_available(
            dataset_path,
            output_path,
            overwrite=args.overwrite,
        )

        print(f"\n[{stage.index}:{stage.name}] self-play", flush=True)
        if checkpoint is not None:
            print(f"Checkpoint: {checkpoint}", flush=True)
        else:
            print("Checkpoint: none (uniform bootstrap)", flush=True)

        self_play_model = load_model_for_self_play(
            checkpoint,
            board_size=args.board_size,
            device=device,
            model_kwargs=model_config,
        )
        dataset = generate_self_play_dataset(
            model=self_play_model,
            device=device,
            config=SelfPlayConfig(
                board_size=args.board_size,
                games=stage.games,
                num_simulations=stage.simulations,
                c_puct=mcts_config["c_puct"],
                dirichlet_alpha=mcts_config["dirichlet_alpha"],
                dirichlet_epsilon=mcts_config["dirichlet_epsilon"],
                temperature=args.temperature,
                temperature_drop_move=args.temperature_drop_move,
                seed=args.seed_base + stage.index,
                add_root_noise=(mcts_config["add_root_noise"] and not args.no_root_noise),
            ),
            save_path=dataset_path,
            show_progress=True,
            progress_desc=f"Stage {stage.index} self-play",
        )
        print(f"[{stage.index}:{stage.name}] saved {len(dataset)} samples to {dataset_path}", flush=True)

        print(f"[{stage.index}:{stage.name}] training", flush=True)
        metadata = train_from_dataset(
            dataset_path=dataset_path,
            output_path=output_path,
            init_checkpoint=checkpoint,
            board_size=args.board_size,
            epochs=stage.epochs,
            batch_size=stage.batch_size,
            lr=stage.lr,
            weight_decay=args.weight_decay,
            device=device,
            model_kwargs=model_config,
            show_progress=True,
        )
        print(
            "[{index}:{name}] saved {output_path} "
            "(steps={steps}, loss={loss:.4f}, policy={policy:.4f}, value={value:.4f})".format(
                index=stage.index,
                name=stage.name,
                output_path=output_path,
                steps=metadata["steps"],
                loss=metadata["last_loss"],
                policy=metadata["last_policy_loss"],
                value=metadata["last_value_loss"],
            ),
            flush=True,
        )

        records.append(stage_record(stage, dataset_path, output_path, checkpoint, skipped=False, metadata=metadata))
        previous_checkpoint = str(output_path)

    write_manifest(manifest_path, records)
    print(f"\nWrote manifest to {manifest_path}", flush=True)

    if args.promote_latest and previous_checkpoint is not None:
        latest_path = Path(args.latest_path)
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous_checkpoint, latest_path)
        metadata_source = Path(previous_checkpoint).with_suffix(Path(previous_checkpoint).suffix + ".json")
        if metadata_source.exists():
            shutil.copy2(metadata_source, latest_path.with_suffix(latest_path.suffix + ".json"))
        print(f"Promoted {previous_checkpoint} to {latest_path}", flush=True)

    return 0


def select_stages(*, include_smoke: bool, start_stage: int, end_stage: int) -> list[StageSpec]:
    stages = ((SMOKE_STAGE,) if include_smoke else ()) + DEFAULT_STAGES
    return [stage for stage in stages if start_stage <= stage.index <= end_stage]


def stage_paths(
    stage: StageSpec,
    *,
    data_dir: Path,
    model_dir: Path,
    prefix: str,
) -> tuple[Path, Path]:
    stem = f"{prefix}{stage.index}_{stage.name}"
    return data_dir / f"{stem}.pt", model_dir / f"{stem}.pth"


def resolve_initial_checkpoint(
    initial_checkpoint: str | None,
    first_stage: StageSpec,
    *,
    include_smoke: bool,
    model_dir: Path,
    prefix: str,
) -> str | None:
    if initial_checkpoint:
        return initial_checkpoint
    if first_stage.index <= 1:
        return None

    earlier = select_stages(
        include_smoke=include_smoke,
        start_stage=0 if include_smoke else 1,
        end_stage=first_stage.index - 1,
    )
    if not earlier:
        return None
    _, inferred = stage_paths(earlier[-1], data_dir=Path("data"), model_dir=model_dir, prefix=prefix)
    if inferred.exists():
        return str(inferred)
    raise SystemExit(
        f"Stage {first_stage.index} needs an initial checkpoint. "
        f"Pass --initial-checkpoint or run previous stages first."
    )


def ensure_outputs_are_available(
    dataset_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    existing = [str(path) for path in (dataset_path, output_path) if path.exists()]
    if existing:
        raise SystemExit(
            "Refusing to overwrite existing outputs. Use --resume to skip completed "
            f"stages or --overwrite to replace them: {', '.join(existing)}"
        )


def stage_record(
    stage: StageSpec,
    dataset_path: Path,
    output_path: Path,
    checkpoint: str | None,
    *,
    skipped: bool,
    metadata: dict | None = None,
) -> dict:
    record = {
        "stage": asdict(stage),
        "dataset_path": str(dataset_path),
        "output_path": str(output_path),
        "checkpoint": checkpoint,
        "skipped": skipped,
    }
    if metadata is not None:
        record["training"] = metadata
    return record


def write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stages": records}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
