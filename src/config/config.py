"""Project configuration loading and AI defaults."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

DEFAULT_CFG: dict[str, Any] = {
    "game": {
        "board_size": 9,
    },
    "ai": {
        "runtime": {
            "device": "auto",
            "model_path": "models/latest.pth",
        },
        "model": {
            "architecture": "alphanet",
            "version": "alphanet-v2",
            "board_size": 9,
            "in_channels": 3,
            "num_filters": 96,
            "num_res_blocks": 6,
            "value_hidden_dim": 128,
        },
        "mcts": {
            "num_simulations": 64,
            "c_puct": 1.5,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
            "add_root_noise": True,
        },
        "arena": {
            "games": 40,
            "move_temperature": 1.0,
            "seed": 0,
            "seed_base": 9000,
            "minimum_score": 0.5,
        },
        "self_play": {
            "output_path": "data/selfplay.pt",
            "games": 100,
            "batch_size": 1,
            "workers": 1,
            "temperature": 1.0,
            "temperature_drop_move": 20,
            "seed": None,
        },
        "train": {
            "dataset_path": "data/selfplay.pt",
            "output_path": "models/latest.pth",
            "epochs": 10,
            "batch_size": 128,
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
    },
}


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return deepcopy(DEFAULT_CFG)
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config.yaml must contain a mapping at the top level.")
    config = _deep_merge(deepcopy(DEFAULT_CFG), _normalize_legacy_config(loaded))
    _sync_model_board_size(config)
    return config


def get_game_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return (config or cfg)["game"]


def get_ai_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return (config or cfg)["ai"]


def get_alphanet_kwargs(config: dict[str, Any] | None = None) -> dict[str, int]:
    model_config = get_ai_config(config)["model"]
    if model_config.get("architecture") != "alphanet":
        raise ValueError(f"Unsupported model architecture: {model_config.get('architecture')}")
    return {
        "board_size": int(model_config["board_size"]),
        "in_channels": int(model_config["in_channels"]),
        "num_filters": int(model_config["num_filters"]),
        "num_res_blocks": int(model_config["num_res_blocks"]),
        "value_hidden_dim": int(model_config["value_hidden_dim"]),
    }


def add_alphanet_model_args(
    parser: argparse.ArgumentParser,
    model_config: dict[str, int],
) -> None:
    parser.add_argument("--num-filters", type=int, default=model_config["num_filters"])
    parser.add_argument("--num-res-blocks", type=int, default=model_config["num_res_blocks"])
    parser.add_argument("--value-hidden-dim", type=int, default=model_config["value_hidden_dim"])


def apply_alphanet_arg_overrides(
    model_config: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, int]:
    updated = dict(model_config)
    updated["num_filters"] = int(args.num_filters)
    updated["num_res_blocks"] = int(args.num_res_blocks)
    updated["value_hidden_dim"] = int(args.value_hidden_dim)
    return updated


def resolve_torch_device(requested: str | None = None, config: dict[str, Any] | None = None) -> str:
    device = requested or get_ai_config(config)["runtime"]["device"]
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def _normalize_legacy_config(loaded: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(loaded)
    ai = config.setdefault("ai", {})
    runtime = ai.setdefault("runtime", {})
    model = ai.setdefault("model", {})

    if "board_size" in config:
        config.setdefault("game", {}).setdefault("board_size", config["board_size"])
        model.setdefault("board_size", config["board_size"])
    if "model_path" in config:
        runtime.setdefault("model_path", config["model_path"])
    if "use_cuda" in config:
        runtime.setdefault("device", "auto" if config["use_cuda"] else "cpu")
    if "mcts" in config:
        ai.setdefault("mcts", config["mcts"])
    if "train" in config and "data_path" in config["train"]:
        ai.setdefault("train", {}).setdefault("dataset_path", config["train"]["data_path"])
    return config


def _sync_model_board_size(config: dict[str, Any]) -> None:
    game_size = int(config["game"]["board_size"])
    model = config["ai"]["model"]
    if model.get("board_size") is None:
        model["board_size"] = game_size


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


cfg = load_config()

# Backward-compatible export for older modules that imported src.config.args.
args = argparse.Namespace()
