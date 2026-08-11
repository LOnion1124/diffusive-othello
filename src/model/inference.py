"""Torch-backed inference adapter for the AlphaNet policy and value heads."""

from __future__ import annotations

import torch

from src.config import get_ai_config, get_alphanet_kwargs, resolve_torch_device
from src.game.state import Move, encode_state, legal_mask, state_from_board
from src.model.alphanet.network import AlphaNet

class GameAI:
    def __init__(self, device: str | None = None):
        ai_config = get_ai_config()
        requested_device = device or ai_config["runtime"]["device"]
        self.device = resolve_torch_device(requested_device)
        self.model_config = get_alphanet_kwargs()
        self.board_size = self.model_config["board_size"]
        self.model = AlphaNet(**self.model_config).to(self.device)
        checkpoint = torch.load(
            ai_config["runtime"]["model_path"],
            map_location=self.device,
            weights_only=False,
        )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        self.model.load_state_dict(checkpoint)
        self.model.eval()
    
    def inference(self, board: list[list[int]], player: int) -> dict:
        # board: provided by logic.board.getGrids()
        board_size = len(board)
        if board_size != self.board_size:
            raise ValueError(
                f"Model expects board size {self.board_size}, but got board size {board_size}."
            )
        game_state = state_from_board(board, current_player=player)
        state = torch.tensor(
            encode_state(game_state, player),
            dtype=torch.get_default_dtype(),
            device=self.device,
        )
        mask = torch.tensor(
            legal_mask(game_state, player),
            dtype=torch.bool,
            device=self.device,
        )
        
        with torch.inference_mode():
            log_policy, value = self.model(state.unsqueeze(0), legal_mask=mask.unsqueeze(0))

        target_idx = log_policy.view(-1).argmax().item()
        x, y = target_idx // board_size, target_idx % board_size
        value = value.item()
        scores_list = log_policy.view(-1).exp().tolist()
        mask_list = mask.tolist()

        return {"pos": (x, y), "value": value, "scores": scores_list, "mask": mask_list}

    def suggest_moves(
        self,
        board: list[list[int]],
        player: int,
        *,
        limit: int = 3,
    ) -> tuple[tuple[Move, float], ...]:
        """Return the highest-probability legal policy moves for a position.

        This derives suggestions from the existing masked policy output; it does
        not alter the network or its inference constraints.
        """
        prediction = self.inference(board, player)
        return self.suggestions_from_prediction(prediction, board_size=len(board), limit=limit)

    @staticmethod
    def suggestions_from_prediction(
        prediction: dict,
        *,
        board_size: int,
        limit: int = 3,
    ) -> tuple[tuple[Move, float], ...]:
        """Extract sorted legal move probabilities from an inference result."""
        if limit < 1:
            raise ValueError("limit must be at least 1.")

        scores = prediction["scores"]
        mask = prediction["mask"]
        expected_length = board_size * board_size
        if len(scores) != expected_length or len(mask) != expected_length:
            raise ValueError("Prediction size does not match the board size.")

        legal_suggestions = [
            ((index // board_size, index % board_size), float(probability))
            for index, (probability, is_legal) in enumerate(zip(scores, mask))
            if is_legal
        ]
        legal_suggestions.sort(key=lambda suggestion: (-suggestion[1], suggestion[0]))
        return tuple(legal_suggestions[:limit])

