import pytest

from src.game.state import PLAYER_ONE, PLAYER_TWO, legal_moves, state_from_board
from src.ui.game_controller import (
    GameController,
    GameSnapshot,
    MODE_PVE,
    MODE_PVP,
    MoveSuggestion,
    PHASE_END,
    PHASE_GAME,
    PHASE_LOADING,
    PHASE_START,
    SIDEBAR_ACTION_END_MATCH,
    SIDEBAR_ACTION_TOGGLE_SUGGESTIONS,
    SIDEBAR_ACTION_TOGGLE_WIN_RATE,
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


class SuggestingPolicy(PredictingPolicy):
    def __init__(self):
        super().__init__()
        self.suggestion_calls = []

    def suggest_moves(self, state, player, *, limit=3):
        self.suggestion_calls.append((state, player, limit))
        moves = legal_moves(state, player)
        suggestions = [
            MoveSuggestion(move=(99, 99), probability=0.99),
            *(
                MoveSuggestion(move=move, probability=0.50 - index * 0.10)
                for index, move in enumerate(moves)
            ),
        ]
        return tuple(suggestions[: limit + 1])


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
    assert controller.info == "YOU TO MOVE."
    controller.play_human_move((0, 1))

    assert controller.current_player == PLAYER_TWO
    assert controller.info == "COMPUTER TO MOVE."
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


def test_controller_exposes_three_legal_ai_move_suggestions():
    policy = SuggestingPolicy()
    controller = GameController(
        mode=MODE_PVP,
        board_size=4,
        ai_policy=policy,
        error_sink=None,
    )

    controller.start_game()

    snapshot = controller.snapshot()
    assert policy.suggestion_calls[-1][1:] == (PLAYER_ONE, 3)
    assert len(snapshot.move_suggestions) == 3
    assert tuple(suggestion.move for suggestion in snapshot.move_suggestions) == tuple(
        legal_moves(snapshot.state, PLAYER_ONE)[:3]
    )
    assert tuple(suggestion.probability for suggestion in snapshot.move_suggestions) == (
        0.50,
        0.40,
        0.30,
    )


def test_pve_controller_hides_suggestions_during_the_ai_turn(monkeypatch):
    monkeypatch.setattr("src.ui.game_controller.random.choice", lambda players: PLAYER_TWO)
    policy = SuggestingPolicy()
    controller = GameController(
        mode=MODE_PVE,
        board_size=4,
        ai_policy=policy,
        error_sink=None,
    )

    controller.start_game()

    assert controller.is_ai_turn
    assert controller.snapshot().legal_place_moves == ()
    assert controller.snapshot().move_suggestions == ()
    assert policy.suggestion_calls == []


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
    assert controller.phase == PHASE_LOADING
    assert controller.mode == MODE_PVE
    assert controller.finish_loading()
    assert controller.phase == PHASE_GAME
    assert controller.mode == MODE_PVE


def test_finish_loading_only_starts_a_pending_game():
    controller = GameController(
        mode=MODE_PVP,
        board_size=4,
        ai_policy=FailingPolicy(),
        error_sink=None,
    )

    assert not controller.finish_loading()
    controller.begin_loading(MODE_PVP)
    assert controller.snapshot().legal_place_moves == ()
    assert controller.finish_loading()
    assert controller.phase == PHASE_GAME


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
    assert controller.winner_name == "YOU"

    controller.state = state_from_board(
        [
            [PLAYER_TWO, PLAYER_TWO, PLAYER_TWO, PLAYER_TWO],
            [PLAYER_TWO, PLAYER_TWO, PLAYER_TWO, PLAYER_TWO],
            [PLAYER_TWO, PLAYER_TWO, PLAYER_TWO, PLAYER_TWO],
            [PLAYER_TWO, PLAYER_TWO, PLAYER_TWO, PLAYER_ONE],
        ]
    )

    controller._end_game()
    assert controller.winner_name == "COMPUTER"


def test_pve_snapshot_uses_you_and_computer_for_each_assigned_side(monkeypatch):
    monkeypatch.setattr("src.ui.game_controller.random.choice", lambda players: PLAYER_TWO)
    controller = GameController(
        mode=MODE_PVE,
        board_size=4,
        ai_policy=FailingPolicy(),
        error_sink=None,
    )

    controller.start_game()

    snapshot = controller.snapshot()
    assert snapshot.player_names == {
        PLAYER_ONE: "COMPUTER",
        PLAYER_TWO: "YOU",
    }
    assert controller.info == "COMPUTER TO MOVE."


def test_sidebar_actions_toggle_analysis_and_settle_current_score():
    policy = SuggestingPolicy()
    controller = GameController(
        mode=MODE_PVP,
        board_size=4,
        ai_policy=policy,
        error_sink=None,
    )
    controller.start_game()

    assert controller.handle_sidebar_action(SIDEBAR_ACTION_TOGGLE_WIN_RATE)
    snapshot = controller.snapshot()
    assert not snapshot.show_win_rate_prediction
    assert snapshot.first_player_win_rate is None

    assert controller.handle_sidebar_action(SIDEBAR_ACTION_TOGGLE_SUGGESTIONS)
    snapshot = controller.snapshot()
    assert not snapshot.show_move_suggestions
    assert snapshot.move_suggestions == ()

    assert controller.play_human_move((0, 1))
    expected_scores = dict(controller.snapshot().scores)
    assert controller.handle_sidebar_action(SIDEBAR_ACTION_END_MATCH)
    snapshot = controller.snapshot()
    assert snapshot.phase == PHASE_END
    assert snapshot.scores == expected_scores
    assert snapshot.winner == PLAYER_ONE
    assert snapshot.info == "Match ended early; current score settled."


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
    renderer.content_rect = (24, 112, 484, 240)

    pvp_rect = renderer._start_button_rects()[MODE_PVP]
    pve_rect = renderer._start_button_rects()[MODE_PVE]

    assert renderer.start_mode_at(_rect_center(pvp_rect)) == MODE_PVP
    assert renderer.start_mode_at(_rect_center(pve_rect)) == MODE_PVE
    assert renderer.start_mode_at((0, 0)) is None


def test_renderer_labels_pve_sides_and_maps_sidebar_controls():
    renderer = object.__new__(PygameRenderer)
    renderer.show_winrate_bar = True
    renderer.sidebar_rect = (0, 0, 220, 540)
    snapshot = GameSnapshot(
        phase=PHASE_GAME,
        board_size=4,
        state=None,
        current_player=PLAYER_ONE,
        scores={0: 16, PLAYER_ONE: 0, PLAYER_TWO: 0},
        winner=0,
        winner_name="",
        info="YOU TO MOVE.",
        mode=MODE_PVE,
        player_names={PLAYER_ONE: "YOU", PLAYER_TWO: "COMPUTER"},
    )

    assert renderer._player_label(snapshot, PLAYER_ONE) == "YOU"
    assert renderer._player_label(snapshot, PLAYER_TWO) == "COMPUTER"
    assert renderer._end_title("YOU") == "YOU WIN"
    assert renderer._end_title("COMPUTER") == "COMPUTER WINS"

    controls = renderer._control_rects(snapshot)
    assert (
        renderer.sidebar_action_at(
            _rect_center(controls[SIDEBAR_ACTION_TOGGLE_WIN_RATE]),
            snapshot,
        )
        == SIDEBAR_ACTION_TOGGLE_WIN_RATE
    )
    assert (
        renderer.sidebar_action_at(
            _rect_center(controls[SIDEBAR_ACTION_TOGGLE_SUGGESTIONS]),
            snapshot,
        )
        == SIDEBAR_ACTION_TOGGLE_SUGGESTIONS
    )
    assert (
        renderer.sidebar_action_at(
            _rect_center(controls[SIDEBAR_ACTION_END_MATCH]),
            snapshot,
        )
        == SIDEBAR_ACTION_END_MATCH
    )


def test_renderer_uses_board_plus_sidebar_layout():
    assert PygameRenderer.screen_size(9, show_winrate_bar=False) == (832, 676)
    assert PygameRenderer.screen_size(9, show_winrate_bar=True) == (832, 676)


def test_renderer_formats_win_rate_label_as_score_pair():
    assert PygameRenderer._format_winrate_label(0.73) == "73 : 27"
    assert PygameRenderer._format_winrate_label(None) == "50 : 50"
    assert PygameRenderer._format_winrate_label(0.5, invalid=True) == "Invalid"


def test_renderer_formats_suggestion_probability_as_percentage():
    assert PygameRenderer._format_suggestion_probability(0.732) == "73.2%"
    assert PygameRenderer._format_suggestion_probability(2.0) == "100.0%"


def test_renderer_uses_distinct_shades_for_primary_and_other_suggestions():
    assert PygameRenderer.COLOR_MOVE_HINT == PygameRenderer.COLOR_SUGGESTION_SECONDARY
    assert PygameRenderer._suggestion_color(1) == PygameRenderer.COLOR_SUGGESTION_PRIMARY
    assert PygameRenderer._suggestion_color(2) == PygameRenderer.COLOR_SUGGESTION_SECONDARY
    assert PygameRenderer._suggestion_color(3) == PygameRenderer.COLOR_SUGGESTION_SECONDARY
    assert PygameRenderer._suggestion_hint_radius(0.0) == 22
    assert PygameRenderer._suggestion_hint_radius(1.0) == 25
    assert PygameRenderer.SUGGESTION_HINT_MAX_RADIUS == PygameRenderer.GRID_SIZE // 2 - 5


def test_renderer_tracks_piece_flips_between_snapshots():
    before = state_from_board(
        [
            [PLAYER_ONE, 0, 0, 0],
            [0, PLAYER_TWO, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    after = state_from_board(
        [
            [PLAYER_ONE, 0, 0, 0],
            [0, PLAYER_ONE, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    renderer = object.__new__(PygameRenderer)
    renderer.board_size = 4
    renderer._previous_board = tuple(tuple(column) for column in before.board)
    renderer._piece_entered_at = {}
    renderer._piece_flip_at = {}
    snapshot = GameSnapshot(
        phase=PHASE_GAME,
        board_size=4,
        state=after,
        current_player=PLAYER_ONE,
        scores={0: 14, PLAYER_ONE: 2, PLAYER_TWO: 0},
        winner=0,
        winner_name="",
        info="Player1's turn.",
        mode=MODE_PVP,
    )

    renderer._update_piece_animations(snapshot, now_ms=100)

    assert renderer._piece_flip_at[(1, 1)] == (PLAYER_TWO, 100)
    assert renderer._flip_visual_state((1, 1), PLAYER_ONE, 100) == (PLAYER_TWO, 1.0)
    player, width = renderer._flip_visual_state(
        (1, 1),
        PLAYER_ONE,
        100 + PygameRenderer.PIECE_FLIP_ANIMATION_MS // 2,
    )
    assert player == PLAYER_ONE
    assert width == pytest.approx(0.12)


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
    assert controller.info == "COMPUTER TO MOVE."


def test_unknown_ui_mode_is_rejected():
    with pytest.raises(ValueError):
        GameController(mode="solo")


def _rect_center(rect):
    left, top, width, height = rect
    return (left + width // 2, top + height // 2)
