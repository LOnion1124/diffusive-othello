import pytest

from src.game.logic import GameLogic
from src.game.state import (
    EMPTY,
    PLAYER_ONE,
    PLAYER_TWO,
    apply_move,
    encode_state,
    is_terminal,
    legal_mask,
    legal_moves,
    new_game,
    pass_turn,
    score,
    state_from_board,
    winner,
)


def test_new_game_initial_position_and_score():
    state = new_game(4)

    assert state.board == (
        (PLAYER_ONE, EMPTY, EMPTY, PLAYER_ONE),
        (EMPTY, EMPTY, EMPTY, EMPTY),
        (EMPTY, EMPTY, EMPTY, EMPTY),
        (PLAYER_TWO, EMPTY, EMPTY, PLAYER_TWO),
    )
    assert score(state) == {EMPTY: 12, PLAYER_ONE: 2, PLAYER_TWO: 2}


def test_legal_moves_and_flat_mask_for_initial_position():
    state = new_game(4)

    assert set(legal_moves(state, PLAYER_ONE)) == {(0, 1), (0, 2), (1, 0), (1, 3)}
    assert set(legal_moves(state, PLAYER_TWO)) == {(2, 0), (2, 3), (3, 1), (3, 2)}

    mask = legal_mask(state, PLAYER_ONE)
    assert len(mask) == 16
    assert [idx for idx, allowed in enumerate(mask) if allowed] == [1, 2, 4, 7]


def test_apply_move_flips_all_surrounding_opponents():
    state = state_from_board(
        [
            [0, -1, -1],
            [1, -1, -1],
            [-1, -1, -1],
        ]
    )

    result = apply_move(state, PLAYER_ONE, (0, 0))

    assert set(result.flipped) == {(0, 1), (1, 1)}
    assert result.state.board == (
        (1, 1, -1),
        (1, 1, -1),
        (-1, -1, -1),
    )
    assert result.state.current_player == PLAYER_TWO


def test_apply_move_rejects_invalid_inputs():
    state = new_game(4)

    with pytest.raises(ValueError):
        apply_move(state, PLAYER_ONE, (2, 2))

    with pytest.raises(ValueError):
        apply_move(state, 0, (0, 1))


def test_pass_turn_tracks_current_player_and_pass_count():
    state = state_from_board(
        [
            [1, -1, 0],
            [-1, -1, 0],
            [0, 0, -1],
        ],
        current_player=PLAYER_ONE,
    )

    result = pass_turn(state)

    assert result.passed is True
    assert result.move is None
    assert result.player == PLAYER_ONE
    assert result.state.current_player == PLAYER_TWO
    assert result.state.consecutive_passes == 1


def test_pass_turn_rejects_pass_when_moves_exist():
    with pytest.raises(ValueError):
        pass_turn(new_game(4))


def test_terminal_score_and_winner_are_explicit():
    state = state_from_board(
        [
            [1, 1, 1],
            [1, -1, -1],
            [1, 1, -1],
        ]
    )

    assert is_terminal(state)
    assert score(state) == {EMPTY: 0, PLAYER_ONE: 6, PLAYER_TWO: 3}
    assert winner(state) == PLAYER_ONE


def test_encode_state_uses_current_player_perspective():
    state = new_game(4)

    encoded = encode_state(state, PLAYER_TWO)

    assert encoded[0][1][1] == 1
    assert encoded[1][3][0] == 1
    assert encoded[2][0][0] == 1


def test_state_json_round_trip():
    state = state_from_board(
        [
            [1, -1, 0],
            [-1, -1, 0],
            [0, 0, -1],
        ],
        current_player=PLAYER_ONE,
    )
    state = pass_turn(state).state

    assert state_from_board(state.to_json()["board"], current_player=-1, consecutive_passes=1) == state
    assert type(state).from_json(state.to_json()) == state


def test_legacy_game_logic_delegates_to_canonical_rules():
    logic = GameLogic(board_size=4)
    logic.startGame()

    assert logic.board.checkValidMove(PLAYER_ONE, (0, 1))
    logic.board.move(PLAYER_ONE, (0, 1))

    assert logic.board.getGrids()[0][1] == PLAYER_ONE
    assert logic.board.grid_count == {EMPTY: 11, PLAYER_ONE: 3, PLAYER_TWO: 2}
