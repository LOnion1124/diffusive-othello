import pytest

from src.game.state import PLAYER_ONE, PLAYER_TWO, legal_moves, state_from_board
from src.ui.game_controller import GameController, MODE_PVE, MODE_PVP


class FailingPolicy:
    def select_move(self, state, player):
        raise RuntimeError("boom")


class RecordingFallbackPolicy:
    def __init__(self):
        self.calls = []

    def select_move(self, state, player):
        self.calls.append((state, player))
        return legal_moves(state, player)[0]


def test_pvp_controller_does_not_use_ai_policy():
    controller = GameController(
        mode=MODE_PVP,
        board_size=4,
        ai_policy=FailingPolicy(),
        error_sink=None,
    )

    controller.start_game()
    assert controller.play_human_move((0, 1))
    assert controller.current_player == PLAYER_TWO
    assert controller.play_human_move((2, 0))
    assert controller.current_player == PLAYER_ONE


def test_pve_controller_falls_back_when_ai_inference_fails(monkeypatch):
    monkeypatch.setattr("src.ui.game_controller.random.choice", lambda players: PLAYER_ONE)
    errors = []
    fallback = RecordingFallbackPolicy()
    controller = GameController(
        mode=MODE_PVE,
        board_size=4,
        ai_policy=FailingPolicy(),
        fallback_policy=fallback,
        error_sink=errors.append,
    )

    controller.start_game()
    assert controller.info == "Your turn."
    controller.play_human_move((0, 1))

    assert controller.current_player == PLAYER_TWO
    assert controller.info == "Thinking..."
    assert controller.play_ai_turn()
    assert fallback.calls
    assert "PVE AI inference failed" in errors[0]
    assert controller.current_player == PLAYER_ONE


def test_pve_controller_can_assign_ai_as_first_player(monkeypatch):
    monkeypatch.setattr("src.ui.game_controller.random.choice", lambda players: PLAYER_TWO)
    controller = GameController(
        mode=MODE_PVE,
        board_size=4,
        ai_policy=FailingPolicy(),
        fallback_policy=RecordingFallbackPolicy(),
        error_sink=None,
    )

    controller.start_game()

    assert controller.human_player == PLAYER_TWO
    assert controller.ai_player == PLAYER_ONE
    assert controller.current_player == PLAYER_ONE
    assert controller.is_ai_turn
    assert controller.info == "Thinking..."


def test_unknown_ui_mode_is_rejected():
    with pytest.raises(ValueError):
        GameController(mode="solo")
