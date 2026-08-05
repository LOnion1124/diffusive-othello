"""Authoritative pure game rules for Diffusive Othello."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

EMPTY = 0
PLAYER_ONE = 1
PLAYER_TWO = -1
PLAYERS = (PLAYER_ONE, PLAYER_TWO)

Move = tuple[int, int]
BoardData = tuple[tuple[int, ...], ...]

ORTHOGONAL_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ALL_DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True)
class GameState:
    board: BoardData
    current_player: int = PLAYER_ONE
    consecutive_passes: int = 0

    @property
    def size(self) -> int:
        return len(self.board)

    def copy(self) -> "GameState":
        return GameState(
            board=tuple(tuple(row) for row in self.board),
            current_player=self.current_player,
            consecutive_passes=self.consecutive_passes,
        )

    def to_json(self) -> dict:
        return {
            "size": self.size,
            "board": [list(row) for row in self.board],
            "current_player": self.current_player,
            "consecutive_passes": self.consecutive_passes,
        }

    @classmethod
    def from_json(cls, data: dict) -> "GameState":
        return state_from_board(
            data["board"],
            current_player=data.get("current_player", PLAYER_ONE),
            consecutive_passes=data.get("consecutive_passes", 0),
        )


@dataclass(frozen=True)
class MoveResult:
    state: GameState
    player: int
    move: Move | None
    flipped: tuple[Move, ...] = ()
    passed: bool = False


def new_game(size: int = 9) -> GameState:
    if size < 2:
        raise ValueError("Board size must be at least 2.")

    board = [[EMPTY for _ in range(size)] for _ in range(size)]
    board[0][0] = PLAYER_ONE
    board[0][size - 1] = PLAYER_ONE
    board[size - 1][0] = PLAYER_TWO
    board[size - 1][size - 1] = PLAYER_TWO
    return state_from_board(board)


def state_from_board(
    board: Iterable[Iterable[int]],
    *,
    current_player: int = PLAYER_ONE,
    consecutive_passes: int = 0,
) -> GameState:
    normalized = tuple(tuple(int(cell) for cell in row) for row in board)
    _validate_board(normalized)
    _validate_player(current_player)
    if consecutive_passes < 0:
        raise ValueError("consecutive_passes cannot be negative.")
    return GameState(normalized, current_player, consecutive_passes)


def legal_moves(state: GameState, player: int) -> list[Move]:
    _validate_player(player)
    moves = []
    for x in range(state.size):
        for y in range(state.size):
            move = (x, y)
            if _is_legal_move(state, player, move):
                moves.append(move)
    return moves


def legal_mask(
    state: GameState,
    player: int,
    *,
    flatten: bool = True,
) -> list[bool] | list[list[bool]]:
    moves = set(legal_moves(state, player))
    mask_2d = [[(x, y) in moves for y in range(state.size)] for x in range(state.size)]
    if not flatten:
        return mask_2d
    return [mask_2d[x][y] for x in range(state.size) for y in range(state.size)]


def apply_move(state: GameState, player: int, move: Move) -> MoveResult:
    _validate_player(player)
    if not _is_legal_move(state, player, move):
        raise ValueError(f"Illegal move {move} for player {player}.")

    x, y = move
    board = [list(row) for row in state.board]
    board[x][y] = player

    flipped: list[Move] = []
    for nx, ny in _neighbors(state.size, move, ALL_DIRECTIONS):
        if board[nx][ny] == -player:
            board[nx][ny] = player
            flipped.append((nx, ny))

    next_state = state_from_board(
        board,
        current_player=-player,
        consecutive_passes=0,
    )
    return MoveResult(next_state, player=player, move=move, flipped=tuple(flipped))


def pass_turn(state: GameState) -> MoveResult:
    player = state.current_player
    if legal_moves(state, player):
        raise ValueError(f"Player {player} cannot pass while legal moves exist.")
    next_state = GameState(
        board=state.board,
        current_player=-player,
        consecutive_passes=state.consecutive_passes + 1,
    )
    return MoveResult(next_state, player=player, move=None, passed=True)


def is_terminal(state: GameState) -> bool:
    return not legal_moves(state, PLAYER_ONE) and not legal_moves(state, PLAYER_TWO)


def score(state: GameState) -> dict[int, int]:
    counts = {EMPTY: 0, PLAYER_ONE: 0, PLAYER_TWO: 0}
    for row in state.board:
        for cell in row:
            counts[cell] += 1
    return counts


def winner(state: GameState) -> int:
    counts = score(state)
    if counts[PLAYER_ONE] > counts[PLAYER_TWO]:
        return PLAYER_ONE
    if counts[PLAYER_TWO] > counts[PLAYER_ONE]:
        return PLAYER_TWO
    return EMPTY


def encode_state(state: GameState, player: int) -> list[list[list[int]]]:
    _validate_player(player)
    empty = []
    own = []
    opponent = []
    for row in state.board:
        empty.append([1 if cell == EMPTY else 0 for cell in row])
        own.append([1 if cell == player else 0 for cell in row])
        opponent.append([1 if cell == -player else 0 for cell in row])
    return [empty, own, opponent]


def _is_legal_move(state: GameState, player: int, move: Move) -> bool:
    x, y = move
    if not _inside(state.size, x, y):
        return False
    if state.board[x][y] != EMPTY:
        return False
    return any(
        state.board[nx][ny] == player
        for nx, ny in _neighbors(state.size, move, ORTHOGONAL_DIRECTIONS)
    )


def _neighbors(size: int, move: Move, directions: Iterable[Move]) -> Iterable[Move]:
    x, y = move
    for dx, dy in directions:
        nx = x + dx
        ny = y + dy
        if _inside(size, nx, ny):
            yield (nx, ny)


def _inside(size: int, x: int, y: int) -> bool:
    return 0 <= x < size and 0 <= y < size


def _validate_board(board: BoardData) -> None:
    if not board:
        raise ValueError("Board cannot be empty.")
    size = len(board)
    if any(len(row) != size for row in board):
        raise ValueError("Board must be square.")
    invalid_cells = {
        cell
        for row in board
        for cell in row
        if cell not in (EMPTY, PLAYER_ONE, PLAYER_TWO)
    }
    if invalid_cells:
        raise ValueError(f"Board contains invalid cell values: {sorted(invalid_cells)}.")


def _validate_player(player: int) -> None:
    if player not in PLAYERS:
        raise ValueError("Player must be 1 or -1.")
