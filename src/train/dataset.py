"""Versioned AlphaZero-style training datasets for Diffusive Othello."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

DATASET_FORMAT_VERSION = "az-do-dataset-v2"
RULE_VERSION = "diffusive-othello-rules-v2"
DEFAULT_MODEL_VERSION = "alphanet-v2"

SAMPLE_METADATA_FIELDS = (
    "game_id",
    "ply",
    "absolute_player",
    "move_action",
    "temperature",
    "root_value",
    "root_visit_count",
    "chosen_visit_count",
    "chosen_prior",
    "chosen_q",
    "policy_entropy",
    "top_policy",
    "legal_count",
    "own_count",
    "opponent_count",
    "empty_count",
    "current_margin",
    "flipped_count",
)

GAME_METADATA_FIELDS = (
    "game_id",
    "first_player",
    "winner",
    "move_count",
    "pass_count",
    "terminal_p1_count",
    "terminal_p2_count",
    "terminal_empty_count",
    "final_margin_p1",
    "sample_start",
    "sample_count",
)

SAMPLE_LONG_FIELDS = {
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

@dataclass(frozen=True)
class DatasetMetadata:
    format_version: str = DATASET_FORMAT_VERSION
    rule_version: str = RULE_VERSION
    board_size: int = 9
    model_version: str = DEFAULT_MODEL_VERSION
    sample_count: int = 0
    game_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    generator: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetMetadata":
        return cls(
            format_version=str(data.get("format_version", DATASET_FORMAT_VERSION)),
            rule_version=str(data.get("rule_version", RULE_VERSION)),
            board_size=int(data["board_size"]),
            model_version=str(data.get("model_version", DEFAULT_MODEL_VERSION)),
            sample_count=int(data.get("sample_count", 0)),
            game_count=int(data.get("game_count", 0)),
            created_at=str(data.get("created_at") or datetime.now(UTC).isoformat()),
            generator=dict(data.get("generator") or {}),
        )


class DODataset(Dataset):
    """Training samples in current-player perspective.

    Each training sample contains:
    - state: (3, S, S) tensor from src.game.state.encode_state
    - legal_mask: (S*S,) bool tensor
    - policy: (S*S,) visit-count distribution
    - value: scalar final outcome from the encoded player's perspective

    Dataset files also contain per-sample and per-game metadata for analysis.
    The training DataLoader receives only the four training tensors.
    """

    def __init__(
        self,
        states: torch.Tensor,
        legal_masks: torch.Tensor,
        policies: torch.Tensor,
        values: torch.Tensor,
        metadata: DatasetMetadata | dict[str, Any],
        sample_metadata: dict[str, torch.Tensor] | None = None,
        game_metadata: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.states = states.detach().cpu().float()
        self.legal_masks = legal_masks.detach().cpu().bool()
        self.policies = policies.detach().cpu().float()
        self.values = values.detach().cpu().float().view(-1)
        self.metadata = (
            metadata if isinstance(metadata, DatasetMetadata) else DatasetMetadata.from_dict(metadata)
        )
        self.sample_metadata = normalize_sample_metadata(
            sample_metadata,
            self.states,
            self.legal_masks,
            self.policies,
        )
        self.game_metadata = normalize_game_metadata(
            game_metadata,
            sample_count=int(self.states.shape[0]),
        )
        validate_dataset(self)

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
            format_version=DATASET_FORMAT_VERSION,
            rule_version=self.metadata.rule_version,
            board_size=self.metadata.board_size,
            model_version=self.metadata.model_version,
            sample_count=len(self),
            game_count=int(self.game_metadata["game_id"].numel()),
            created_at=self.metadata.created_at,
            generator=dict(self.metadata.generator),
        )
        return {
            "metadata": asdict(metadata),
            "states": self.states,
            "legal_masks": self.legal_masks,
            "policies": self.policies,
            "values": self.values,
            "sample_metadata": self.sample_metadata,
            "game_metadata": self.game_metadata,
        }


def make_dataset(
    states: torch.Tensor,
    legal_masks: torch.Tensor,
    policies: torch.Tensor,
    values: torch.Tensor,
    *,
    board_size: int,
    model_version: str = DEFAULT_MODEL_VERSION,
    sample_metadata: dict[str, torch.Tensor] | None = None,
    game_metadata: dict[str, torch.Tensor] | None = None,
    generator: dict[str, Any] | None = None,
) -> DODataset:
    metadata = DatasetMetadata(
        board_size=board_size,
        model_version=model_version,
        sample_count=int(states.shape[0]),
        game_count=_metadata_game_count(game_metadata),
        generator=dict(generator or {}),
    )
    return DODataset(
        states,
        legal_masks,
        policies,
        values,
        metadata,
        sample_metadata=sample_metadata,
        game_metadata=game_metadata,
    )


def save_dataset(dataset: DODataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_dataset(dataset)
    torch.save(dataset.to_payload(), path)


def load_dataset(path: str | Path) -> DODataset:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if hasattr(payload, "to_payload"):
        payload = payload.to_payload()
    if not isinstance(payload, dict):
        raise ValueError("Dataset file must contain a dict payload.")

    metadata = DatasetMetadata.from_dict(payload["metadata"])
    if metadata.format_version != DATASET_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported dataset format {metadata.format_version!r}; "
            f"expected {DATASET_FORMAT_VERSION!r}."
        )
    return DODataset(
        payload["states"],
        payload["legal_masks"],
        payload["policies"],
        payload["values"],
        metadata,
        sample_metadata=payload.get("sample_metadata"),
        game_metadata=payload.get("game_metadata"),
    )


def normalize_sample_metadata(
    sample_metadata: dict[str, torch.Tensor] | None,
    states: torch.Tensor,
    legal_masks: torch.Tensor,
    policies: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if sample_metadata is None:
        return _default_sample_metadata(states, legal_masks, policies)
    return {
        field: _metadata_tensor(
            sample_metadata[field],
            dtype=torch.long if field in SAMPLE_LONG_FIELDS else torch.float32,
        )
        for field in SAMPLE_METADATA_FIELDS
    }


def normalize_game_metadata(
    game_metadata: dict[str, torch.Tensor] | None,
    *,
    sample_count: int,
) -> dict[str, torch.Tensor]:
    if game_metadata is None:
        return _default_game_metadata(sample_count)
    return {
        field: _metadata_tensor(game_metadata[field], dtype=torch.long)
        for field in GAME_METADATA_FIELDS
    }


def validate_dataset(dataset: DODataset) -> None:
    validate_tensors(
        dataset.states,
        dataset.legal_masks,
        dataset.policies,
        dataset.values,
        dataset.metadata.board_size,
    )
    validate_metadata(
        dataset.sample_metadata,
        dataset.game_metadata,
        sample_count=len(dataset),
        action_count=dataset.metadata.board_size * dataset.metadata.board_size,
        legal_masks=dataset.legal_masks,
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


def validate_metadata(
    sample_metadata: dict[str, torch.Tensor],
    game_metadata: dict[str, torch.Tensor],
    *,
    sample_count: int,
    action_count: int,
    legal_masks: torch.Tensor,
) -> None:
    for field in SAMPLE_METADATA_FIELDS:
        if field not in sample_metadata:
            raise ValueError(f"sample_metadata missing {field!r}.")
        tensor = sample_metadata[field]
        if tensor.shape != (sample_count,):
            raise ValueError(
                f"sample_metadata[{field!r}] must have shape ({sample_count},); "
                f"got {tuple(tensor.shape)}."
            )
        if tensor.dtype.is_floating_point and not torch.isfinite(tensor).all():
            raise ValueError(f"sample_metadata[{field!r}] contains non-finite values.")

    game_count = None
    for field in GAME_METADATA_FIELDS:
        if field not in game_metadata:
            raise ValueError(f"game_metadata missing {field!r}.")
        tensor = game_metadata[field]
        if tensor.ndim != 1:
            raise ValueError(f"game_metadata[{field!r}] must be one-dimensional.")
        if game_count is None:
            game_count = int(tensor.numel())
        elif int(tensor.numel()) != game_count:
            raise ValueError("All game_metadata fields must have the same length.")

    if game_count is None or game_count <= 0:
        raise ValueError("game_metadata must contain at least one game.")

    move_actions = sample_metadata["move_action"].long()
    if ((move_actions < 0) | (move_actions >= action_count)).any():
        raise ValueError("sample_metadata['move_action'] contains out-of-range actions.")
    if not legal_masks.gather(1, move_actions.view(-1, 1)).all():
        raise ValueError("sample_metadata['move_action'] must be legal for every sample.")

    game_ids = game_metadata["game_id"].long()
    sample_game_ids = sample_metadata["game_id"].long()
    if int(game_ids.unique().numel()) != game_count:
        raise ValueError("game_metadata['game_id'] values must be unique.")
    game_id_set = set(int(value) for value in game_ids.tolist())
    if any(int(value) not in game_id_set for value in sample_game_ids.tolist()):
        raise ValueError("sample_metadata['game_id'] references an unknown game.")

    if int(game_metadata["sample_count"].sum().item()) != sample_count:
        raise ValueError("game_metadata['sample_count'] must sum to the dataset sample count.")
    if (game_metadata["sample_start"] < 0).any():
        raise ValueError("game_metadata['sample_start'] cannot be negative.")
    if (game_metadata["sample_count"] <= 0).any():
        raise ValueError("Each game must contribute at least one training sample.")


def _metadata_tensor(value: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype).detach().cpu().view(-1)


def _default_sample_metadata(
    states: torch.Tensor,
    legal_masks: torch.Tensor,
    policies: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sample_count = int(states.shape[0])
    own = states[:, 1].sum(dim=(1, 2)).long()
    opponent = states[:, 2].sum(dim=(1, 2)).long()
    empty = states[:, 0].sum(dim=(1, 2)).long()
    legal_count = legal_masks.sum(dim=1).long()
    policy_floor = policies.clamp_min(1e-12)
    policy_entropy = -(policy_floor * policy_floor.log()).sum(dim=1).float()
    top_policy = policies.amax(dim=1).float()
    return {
        "game_id": torch.zeros(sample_count, dtype=torch.long),
        "ply": torch.arange(sample_count, dtype=torch.long),
        "absolute_player": torch.ones(sample_count, dtype=torch.long),
        "move_action": policies.argmax(dim=1).long(),
        "temperature": torch.ones(sample_count, dtype=torch.float32),
        "root_value": torch.zeros(sample_count, dtype=torch.float32),
        "root_visit_count": torch.zeros(sample_count, dtype=torch.long),
        "chosen_visit_count": torch.zeros(sample_count, dtype=torch.long),
        "chosen_prior": torch.zeros(sample_count, dtype=torch.float32),
        "chosen_q": torch.zeros(sample_count, dtype=torch.float32),
        "policy_entropy": policy_entropy,
        "top_policy": top_policy,
        "legal_count": legal_count,
        "own_count": own,
        "opponent_count": opponent,
        "empty_count": empty,
        "current_margin": own - opponent,
        "flipped_count": torch.zeros(sample_count, dtype=torch.long),
    }


def _default_game_metadata(sample_count: int) -> dict[str, torch.Tensor]:
    return {
        "game_id": torch.tensor([0], dtype=torch.long),
        "first_player": torch.tensor([1], dtype=torch.long),
        "winner": torch.tensor([0], dtype=torch.long),
        "move_count": torch.tensor([sample_count], dtype=torch.long),
        "pass_count": torch.tensor([0], dtype=torch.long),
        "terminal_p1_count": torch.tensor([0], dtype=torch.long),
        "terminal_p2_count": torch.tensor([0], dtype=torch.long),
        "terminal_empty_count": torch.tensor([0], dtype=torch.long),
        "final_margin_p1": torch.tensor([0], dtype=torch.long),
        "sample_start": torch.tensor([0], dtype=torch.long),
        "sample_count": torch.tensor([sample_count], dtype=torch.long),
    }


def _metadata_game_count(game_metadata: dict[str, torch.Tensor] | None) -> int:
    if not game_metadata:
        return 1
    return int(torch.as_tensor(game_metadata["game_id"]).numel())
