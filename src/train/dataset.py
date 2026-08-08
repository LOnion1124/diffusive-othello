"""Versioned AlphaZero-style training datasets for Diffusive Othello."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

DATASET_FORMAT_VERSION = "az-do-dataset-v1"
RULE_VERSION = "diffusive-othello-rules-v2"
DEFAULT_MODEL_VERSION = "alphanet-v2"


@dataclass(frozen=True)
class DatasetMetadata:
    format_version: str = DATASET_FORMAT_VERSION
    rule_version: str = RULE_VERSION
    board_size: int = 9
    model_version: str = DEFAULT_MODEL_VERSION
    sample_count: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetMetadata":
        return cls(
            format_version=str(data.get("format_version", DATASET_FORMAT_VERSION)),
            rule_version=str(data.get("rule_version", RULE_VERSION)),
            board_size=int(data["board_size"]),
            model_version=str(data.get("model_version", DEFAULT_MODEL_VERSION)),
            sample_count=int(data.get("sample_count", 0)),
        )


class DODataset(Dataset):
    """Training samples in current-player perspective.

    Each sample contains:
    - state: (3, S, S) tensor from src.game.state.encode_state
    - legal_mask: (S*S,) bool tensor
    - policy: (S*S,) visit-count distribution
    - value: scalar final outcome from the encoded player's perspective
    """

    def __init__(
        self,
        states: torch.Tensor,
        legal_masks: torch.Tensor,
        policies: torch.Tensor,
        values: torch.Tensor,
        metadata: DatasetMetadata | dict[str, Any],
    ) -> None:
        self.states = states.detach().cpu().float()
        self.legal_masks = legal_masks.detach().cpu().bool()
        self.policies = policies.detach().cpu().float()
        self.values = values.detach().cpu().float().view(-1)
        self.metadata = (
            metadata if isinstance(metadata, DatasetMetadata) else DatasetMetadata.from_dict(metadata)
        )
        validate_tensors(
            self.states,
            self.legal_masks,
            self.policies,
            self.values,
            self.metadata.board_size,
        )

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        legal_mask = self.legal_masks[idx]
        return {
            "state": self.states[idx],
            "legal_mask": legal_mask,
            "mask": legal_mask,
            "policy": self.policies[idx],
            "value": self.values[idx],
        }

    def to_payload(self) -> dict[str, Any]:
        metadata = DatasetMetadata(
            format_version=self.metadata.format_version,
            rule_version=self.metadata.rule_version,
            board_size=self.metadata.board_size,
            model_version=self.metadata.model_version,
            sample_count=len(self),
        )
        return {
            "metadata": asdict(metadata),
            "states": self.states,
            "legal_masks": self.legal_masks,
            "policies": self.policies,
            "values": self.values,
        }


def make_dataset(
    states: torch.Tensor,
    legal_masks: torch.Tensor,
    policies: torch.Tensor,
    values: torch.Tensor,
    *,
    board_size: int,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> DODataset:
    metadata = DatasetMetadata(
        board_size=board_size,
        model_version=model_version,
        sample_count=int(states.shape[0]),
    )
    return DODataset(states, legal_masks, policies, values, metadata)


def save_dataset(dataset: DODataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_dataset(dataset)
    torch.save(dataset.to_payload(), path)


def load_dataset(path: str | Path) -> DODataset:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(payload, DODataset):
        validate_dataset(payload)
        return payload
    if not isinstance(payload, dict):
        raise ValueError("Dataset file must contain a dict payload.")

    metadata = DatasetMetadata.from_dict(payload["metadata"])
    return DODataset(
        payload["states"],
        payload["legal_masks"],
        payload["policies"],
        payload["values"],
        metadata,
    )


def validate_dataset(dataset: DODataset) -> None:
    validate_tensors(
        dataset.states,
        dataset.legal_masks,
        dataset.policies,
        dataset.values,
        dataset.metadata.board_size,
    )


def validate_tensors(
    states: torch.Tensor,
    legal_masks: torch.Tensor,
    policies: torch.Tensor,
    values: torch.Tensor,
    board_size: int,
) -> None:
    if board_size < 2:
        raise ValueError("board_size must be at least 2.")

    expected_actions = board_size * board_size
    if states.ndim != 4 or states.shape[1:] != (3, board_size, board_size):
        raise ValueError(
            f"states must have shape (N, 3, {board_size}, {board_size}); got {tuple(states.shape)}."
        )

    sample_count = states.shape[0]
    if legal_masks.shape != (sample_count, expected_actions):
        raise ValueError(
            f"legal_masks must have shape ({sample_count}, {expected_actions}); "
            f"got {tuple(legal_masks.shape)}."
        )
    if policies.shape != (sample_count, expected_actions):
        raise ValueError(
            f"policies must have shape ({sample_count}, {expected_actions}); got {tuple(policies.shape)}."
        )
    if values.shape != (sample_count,):
        raise ValueError(f"values must have shape ({sample_count},); got {tuple(values.shape)}.")

    if sample_count == 0:
        raise ValueError("Dataset must contain at least one sample.")
    if not torch.isfinite(states).all():
        raise ValueError("states contains non-finite values.")
    if not torch.isfinite(policies).all():
        raise ValueError("policies contains non-finite values.")
    if not torch.isfinite(values).all():
        raise ValueError("values contains non-finite values.")
    if (policies < -1e-6).any():
        raise ValueError("policies must be non-negative.")
    if ((values < -1.0 - 1e-6) | (values > 1.0 + 1e-6)).any():
        raise ValueError("values must be in [-1, 1].")

    legal_masks = legal_masks.bool()
    legal_counts = legal_masks.sum(dim=1)
    if (legal_counts == 0).any():
        raise ValueError("Every recorded sample must have at least one legal move.")

    illegal_policy_mass = policies.masked_fill(legal_masks, 0.0).abs().amax()
    if illegal_policy_mass > 1e-5:
        raise ValueError("Illegal moves must have zero policy probability.")

    legal_policy_sums = policies.masked_fill(~legal_masks, 0.0).sum(dim=1)
    if not torch.allclose(
        legal_policy_sums,
        torch.ones_like(legal_policy_sums),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError("Policy probabilities must sum to 1 over legal moves.")
