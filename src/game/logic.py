from src.game.state import (
    EMPTY,
    PLAYER_ONE,
    PLAYER_TWO,
    GameState,
    apply_move,
    is_terminal,
    legal_moves,
    new_game,
    score,
    state_from_board,
    winner,
)


class Grid:
    def __init__(self, pos: tuple[int, int], board_size: int) -> None:
        self.status = EMPTY
        self.pos = pos
        self.board_size = board_size

    def getAdjacent(self) -> list[tuple[int, int]]:
        result = []
        x, y = self.pos
        size = self.board_size
        if x > 0:
            result.append((x - 1, y))
        if x < size - 1:
            result.append((x + 1, y))
        if y > 0:
            result.append((x, y - 1))
        if y < size - 1:
            result.append((x, y + 1))
        return result

    def getAround(self) -> list[tuple[int, int]]:
        result = []
        x, y = self.pos
        size = self.board_size
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    result.append((nx, ny))
        return result


class Board:
    def __init__(self, size: int) -> None:
        self.size = size
        self.num_grids = size * size
        self.grids = [
            [Grid(pos=(i, j), board_size=size) for j in range(size)]
            for i in range(size)
        ]
        self.state = state_from_board([[EMPTY for _ in range(size)] for _ in range(size)])
        self.grid_count = {EMPTY: self.num_grids, PLAYER_ONE: 0, PLAYER_TWO: 0}

    def initialize(self) -> None:
        self.state = new_game(self.size)
        self._sync_from_state()

    def checkValidMove(self, player: int, pos: tuple[int, int]) -> bool:
        return pos in legal_moves(self.state, player)

    def updateCount(self) -> None:
        self.grid_count = score(self.state)

    def move(self, player: int, pos: tuple[int, int]) -> None:
        result = apply_move(self.state, player, pos)
        self.state = result.state
        self._sync_from_state()

    def canMove(self, player: int) -> bool:
        return bool(legal_moves(self.state, player))

    def isTerminal(self) -> bool:
        return is_terminal(self.state)

    def setState(self, state: GameState) -> None:
        if state.size != self.size:
            raise ValueError("Cannot set board state with a different size.")
        self.state = state
        self._sync_from_state()

    def __str__(
        self,
        symbol: dict[int, str] | None = None,
        sep: str = " ",
    ) -> str:
        symbols = symbol or {EMPTY: ".", PLAYER_ONE: "o", PLAYER_TWO: "x"}
        lines = []
        for y in range(self.size):
            line = ""
            for x in range(self.size):
                line += symbols[self.state.board[x][y]]
                line += sep
            lines.append(line.rstrip(sep))
        return "\n".join(lines)

    def getGrids(self) -> list[list[int]]:
        return [list(row) for row in self.state.board]

    def _sync_from_state(self) -> None:
        for i in range(self.size):
            for j in range(self.size):
                self.grids[i][j].status = self.state.board[i][j]
        self.updateCount()


class GameLogic:
    def __init__(
        self,
        player_names: tuple[str, str] = ("Player1", "Player2"),
        board_size: int = 9,
    ) -> None:
        self.state = "start"
        self.game_state = ""
        self.board_size = board_size
        self.board = Board(self.board_size)
        self.winner = EMPTY
        self.winner_name = ""
        self.input_buffer = None
        self.player_names = player_names

    def startGame(self) -> None:
        self.board.initialize()
        self.state = "game"
        self.game_state = "player1"
        self.winner = EMPTY
        self.winner_name = ""

    def endGame(self) -> None:
        game_winner = winner(self.board.state)
        self.winner = game_winner
        if game_winner == EMPTY:
            self.winner_name = "Draw"
        elif game_winner == PLAYER_ONE:
            self.winner_name = self.player_names[0]
        else:
            self.winner_name = self.player_names[1]

        self.state = "end"
        self.game_state = ""

    def switchTurn(self) -> None:
        match self.game_state:
            case "player1":
                self.game_state = "player2"
                self.board.state = GameState(
                    self.board.state.board,
                    current_player=PLAYER_TWO,
                    consecutive_passes=self.board.state.consecutive_passes,
                )
            case "player2":
                self.game_state = "player1"
                self.board.state = GameState(
                    self.board.state.board,
                    current_player=PLAYER_ONE,
                    consecutive_passes=self.board.state.consecutive_passes,
                )
