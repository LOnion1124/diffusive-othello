"""Game-flow controller shared by the pygame desktop client."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Protocol

from src.game.state import (
    EMPTY,
    PLAYER_ONE,
    PLAYER_TWO,
    PLAYERS,
    GameState,
    Move,
    apply_move,
    is_terminal,
    legal_moves,
    new_game,
    pass_turn,
    score,
    winner,
)


PHASE_START = "start"
PHASE_GAME = "game"
PHASE_END = "end"

MODE_PVP = "PVP"
MODE_PVE = "PVE"


class MovePolicy(Protocol):
    def select_move(self, state: GameState, player: int) -> Move:
        """Return a legal move for the given state and player."""


@dataclass(frozen=True)
class GameSnapshot:
    phase: str
    board_size: int
    state: GameState | None
    current_player: int
    scores: dict[int, int]
    winner: int
    winner_name: str
    info: str
    mode: str
    legal_place_moves: tuple[Move, ...] = ()


class FirstLegalMovePolicy:
    """Deterministic fallback policy used when AI inference is unavailable."""

    def select_move(self, state: GameState, player: int) -> Move:
        moves = legal_moves(state, player)
        if not moves:
            raise ValueError(f"Player {player} has no legal moves.")
        return moves[0]


class LazyAiPolicy:
    """Load the torch-backed AI only when a PVE game reaches the AI turn."""

    def __init__(self) -> None:
        self._ai = None

    def select_move(self, state: GameState, player: int) -> Move:
        if self._ai is None:
            try:
                from src.model.inference import GameAI
            except Exception as exc:
                raise RuntimeError(
                    "AI dependencies could not be imported. Install requirements-ai.txt "
                    "and ensure the configured model is available."
                ) from exc

            try:
                self._ai = GameAI()
            except Exception as exc:
                raise RuntimeError(
                    "AI model could not be loaded. Check config.yaml and the checkpoint path."
                ) from exc

        prediction = self._ai.inference(
            board=[list(row) for row in state.board],
            player=player,
        )
        move = tuple(prediction["pos"])
        if move not in legal_moves(state, player):
            raise RuntimeError(f"AI returned illegal move {move} for player {player}.")
        return move


class GameController:
    def __init__(
        self,
        *,
        mode: str = MODE_PVP,
        board_size: int = 9,
        ai_policy: MovePolicy | None = None,
        fallback_policy: MovePolicy | None = None,
        error_sink: Callable[[str], None] | None = print,
    ) -> None:
        normalized_mode = mode.upper()
        if normalized_mode not in (MODE_PVP, MODE_PVE):
            raise ValueError("mode must be PVP or PVE.")

        self.mode = normalized_mode
        self.board_size = board_size
        self.player_names = {PLAYER_ONE: "Player1", PLAYER_TWO: "Player2"}
        self.human_player = PLAYER_ONE
        self.ai_player = PLAYER_TWO
        self.ai_policy = ai_policy if ai_policy is not None else LazyAiPolicy()
        self.fallback_policy = (
            fallback_policy if fallback_policy is not None else FirstLegalMovePolicy()
        )
        self.error_sink = error_sink

        self.phase = PHASE_START
        self.state: GameState | None = None
        self.winner = EMPTY
        self.winner_name = ""
        self.info = "Diffusive Othello"

    @property
    def current_player(self) -> int:
        if self.state is None:
            return PLAYER_ONE
        return self.state.current_player

    def snapshot(self) -> GameSnapshot:
        current_score = score(self.state) if self.state is not None else {
            EMPTY: self.board_size * self.board_size,
            PLAYER_ONE: 0,
            PLAYER_TWO: 0,
        }
        legal_place_moves: tuple[Move, ...] = ()
        if (
            self.phase == PHASE_GAME
            and self.state is not None
        ):
            legal_place_moves = tuple(legal_moves(self.state, self.current_player))
        return GameSnapshot(
            phase=self.phase,
            board_size=self.board_size,
            state=self.state,
            current_player=self.current_player,
            scores=current_score,
            winner=self.winner,
            winner_name=self.winner_name,
            info=self.info,
            mode=self.mode,
            legal_place_moves=legal_place_moves,
        )

    def start_game(self) -> None:
        self.state = new_game(self.board_size)
        if self.mode == MODE_PVE:
            self.human_player = random.choice(PLAYERS)
            self.ai_player = -self.human_player
        else:
            self.human_player = PLAYER_ONE
            self.ai_player = PLAYER_TWO
        self.phase = PHASE_GAME
        self.winner = EMPTY
        self.winner_name = ""
        self.info = self._turn_info(self.current_player)
        self._settle_turn()

    def handle_click(self, clicked: bool, move: Move | None = None) -> bool:
        if not clicked:
            return False

        if self.phase in (PHASE_START, PHASE_END):
            self.start_game()
            return True

        if self.phase != PHASE_GAME or self.state is None:
            return False

        if self.is_ai_turn:
            return False

        if move is None:
            return False

        return self.play_human_move(move)

    def play_human_move(self, move: Move) -> bool:
        if self.phase != PHASE_GAME or self.state is None:
            return False

        player = self.current_player
        if move not in legal_moves(self.state, player):
            self.info = "Try another position."
            return False

        self.state = apply_move(self.state, player, move).state
        self._settle_turn()
        return True

    def play_ai_turn(self) -> bool:
        if not self.is_ai_turn or self.state is None:
            return False

        player = self.current_player
        fallback_used = False
        try:
            move = self.ai_policy.select_move(self.state, player)
        except Exception as exc:
            self._report_ai_error(exc)
            move = self.fallback_policy.select_move(self.state, player)
            self.info = "AI unavailable; fallback move used."
            fallback_used = True

        self.state = apply_move(self.state, player, move).state
        self._settle_turn(keep_existing_info=fallback_used)
        return True

    @property
    def is_ai_turn(self) -> bool:
        return (
            self.phase == PHASE_GAME
            and self.mode == MODE_PVE
            and self.current_player == self.ai_player
        )

    def _settle_turn(self, *, keep_existing_info: bool = False) -> None:
        if self.state is None:
            return

        if self._is_terminal():
            self._end_game()
            return

        player = self.current_player
        if not self._can_current_player_act():
            skipped_name = self.player_names[player]
            self.state = pass_turn(self.state).state
            if self._is_terminal():
                self._end_game()
                return
            self.info = f"No place left for {skipped_name}."
            return

        if not keep_existing_info:
            self.info = self._turn_info(player)

    def _turn_info(self, player: int) -> str:
        if self.mode == MODE_PVE:
            return "Thinking..." if player == self.ai_player else "Your turn."
        return f"{self.player_names[player]}'s turn."

    def _is_terminal(self) -> bool:
        if self.state is None:
            return False
        return is_terminal(self.state)

    def _can_current_player_act(self) -> bool:
        if self.state is None:
            return False
        player = self.current_player
        return bool(legal_moves(self.state, player))

    def _end_game(self) -> None:
        if self.state is None:
            return

        game_winner = winner(self.state)
        self.winner = game_winner
        self.winner_name = (
            "Draw" if game_winner == EMPTY else self.player_names[game_winner]
        )
        self.phase = PHASE_END
        self.info = "Game over."

    def _report_ai_error(self, error: Exception) -> None:
        if self.error_sink is not None:
            self.error_sink(f"PVE AI inference failed: {error}")
