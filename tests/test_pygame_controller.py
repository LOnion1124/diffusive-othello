import pytest

from src.game.state import PLAYER_ONE, PLAYER_TWO, legal_moves, state_from_board
from src.ui.game_controller import (
    GameController,
    GameSnapshot,
    MODE_PVE,
    MODE_PVP,
    PHASE_END,
    PHASE_GAME,
    PHASE_START,
)
from src.ui.pygame_renderer import PygameRenderer


class FailingPolicy:
    def select_move(self, state, player):
        raise RuntimeError("boom")


class RecordingFallbackPolicy:
    def __init__(self):
        self.calls = []

    def select_move(self, state, player):
        self.calls.append((state, player))
        return legal_moves(state, player)[0]


class PredictingPolicy:
    def __init__(self, win_rate=0.73):
        self.win_rate = win_rate
        self.move_calls = []
        self.prediction_calls = []

    def select_move(self, state, player):
        self.move_calls.append((state, player))
        return legal_moves(state, player)[0]

    def predict_player_win_rate(self, state, *, current_player, target_player):
        self.prediction_calls.append((state, current_player, target_player))
        return self.win_rate


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


def test_pve_controller_exposes_first_player_win_rate_prediction(monkeypatch):
    monkeypatch.setattr("src.ui.game_controller.random.choice", lambda players: PLAYER_ONE)
    policy = PredictingPolicy(win_rate=0.73)
    controller = GameController(
        mode=MODE_PVE,
        board_size=4,
        ai_policy=policy,
        error_sink=None,
    )

    controller.start_game()

    snapshot = controller.snapshot()
    assert snapshot.first_player_win_rate == pytest.approx(0.73)
    assert policy.prediction_calls
    _, current_player, target_player = policy.prediction_calls[-1]
    assert current_player == PLAYER_ONE
    assert target_player == PLAYER_ONE


def test_pvp_controller_requests_win_rate_prediction_without_using_ai_moves():
    policy = PredictingPolicy()
    controller = GameController(
        mode=MODE_PVP,
        board_size=4,
        ai_policy=policy,
        error_sink=None,
    )

    controller.start_game()
    controller.play_human_move((0, 1))

    assert controller.snapshot().first_player_win_rate == pytest.approx(policy.win_rate)
    assert policy.prediction_calls
    assert policy.move_calls == []


def test_controller_marks_win_rate_invalid_when_prediction_is_unavailable():
    controller = GameController(
        mode=MODE_PVP,
        board_size=4,
        ai_policy=FailingPolicy(),
        error_sink=None,
    )

    controller.start_game()

    snapshot = controller.snapshot()
    assert snapshot.first_player_win_rate == pytest.approx(0.5)
    assert snapshot.win_rate_invalid


def test_start_click_requires_mode_selection():
    controller = GameController(
        mode=MODE_PVP,
        board_size=4,
        ai_policy=FailingPolicy(),
        error_sink=None,
    )

    assert not controller.handle_click(True)
    assert controller.phase == PHASE_START

    assert controller.handle_click(True, start_mode=MODE_PVE)
    assert controller.phase == PHASE_GAME
    assert controller.mode == MODE_PVE


def test_end_click_returns_to_start_screen():
    controller = GameController(
        mode=MODE_PVP,
        board_size=4,
        ai_policy=FailingPolicy(),
        error_sink=None,
    )
    controller.state = state_from_board(
        [
            [PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE],
            [PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE],
            [PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE],
            [PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE],
        ]
    )
    controller._end_game()

    assert controller.phase == PHASE_END
    assert controller.handle_click(True)
    assert controller.phase == PHASE_START
    assert controller.state is None
    assert controller.info == "Diffusive Othello"


def test_pve_winner_name_uses_human_and_computer_labels():
    controller = GameController(
        mode=MODE_PVE,
        board_size=4,
        ai_policy=FailingPolicy(),
        error_sink=None,
    )
    controller.human_player = PLAYER_ONE
    controller.ai_player = PLAYER_TWO
    controller.state = state_from_board(
        [
            [PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE],
            [PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE],
            [PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE],
            [PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_TWO],
        ]
    )

    controller._end_game()
    assert controller.winner_name == "You"

    controller.state = state_from_board(
        [
            [PLAYER_TWO, PLAYER_TWO, PLAYER_TWO, PLAYER_TWO],
            [PLAYER_TWO, PLAYER_TWO, PLAYER_TWO, PLAYER_TWO],
            [PLAYER_TWO, PLAYER_TWO, PLAYER_TWO, PLAYER_TWO],
            [PLAYER_TWO, PLAYER_TWO, PLAYER_TWO, PLAYER_ONE],
        ]
    )

    controller._end_game()
    assert controller.winner_name == "Computer"


def test_renderer_only_shows_win_rate_bar_during_gameplay():
    renderer = object.__new__(PygameRenderer)
    renderer.show_winrate_bar = True
    snapshot = GameSnapshot(
        phase=PHASE_START,
        board_size=4,
        state=None,
        current_player=PLAYER_ONE,
        scores={0: 16, PLAYER_ONE: 0, PLAYER_TWO: 0},
        winner=0,
        winner_name="",
        info="Diffusive Othello",
        mode=MODE_PVE,
    )

    assert not renderer._should_draw_winrate_bar(snapshot)

    end_snapshot = GameSnapshot(
        phase=PHASE_END,
        board_size=4,
        state=None,
        current_player=PLAYER_ONE,
        scores={0: 0, PLAYER_ONE: 10, PLAYER_TWO: 6},
        winner=PLAYER_ONE,
        winner_name="Player1",
        info="Game over.",
        mode=MODE_PVE,
    )
    assert not renderer._should_draw_winrate_bar(end_snapshot)

    playing_snapshot = GameSnapshot(
        phase=PHASE_GAME,
        board_size=4,
        state=None,
        current_player=PLAYER_ONE,
        scores={0: 16, PLAYER_ONE: 0, PLAYER_TWO: 0},
        winner=0,
        winner_name="",
        info="Your turn.",
        mode=MODE_PVE,
    )
    assert renderer._should_draw_winrate_bar(playing_snapshot)

    pvp_snapshot = GameSnapshot(
        phase=PHASE_GAME,
        board_size=4,
        state=None,
        current_player=PLAYER_ONE,
        scores={0: 16, PLAYER_ONE: 0, PLAYER_TWO: 0},
        winner=0,
        winner_name="",
        info="Player1's turn.",
        mode=MODE_PVP,
    )
    assert renderer._should_draw_winrate_bar(pvp_snapshot)


def test_renderer_maps_start_buttons_to_modes():
    renderer = object.__new__(PygameRenderer)
    renderer.board_left_top = (0, 80)
    renderer.board_length = 240

    pvp_rect = renderer._start_button_rects()[MODE_PVP]
    pve_rect = renderer._start_button_rects()[MODE_PVE]

    assert renderer.start_mode_at(_rect_center(pvp_rect)) == MODE_PVP
    assert renderer.start_mode_at(_rect_center(pve_rect)) == MODE_PVE
    assert renderer.start_mode_at((0, 0)) is None


def test_renderer_formats_win_rate_label_as_score_pair():
    assert PygameRenderer._format_winrate_label(0.73) == "73 : 27"
    assert PygameRenderer._format_winrate_label(None) == "50 : 50"
    assert PygameRenderer._format_winrate_label(0.5, invalid=True) == "Invalid"


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


def _rect_center(rect):
    left, top, width, height = rect
    return (left + width // 2, top + height // 2)
