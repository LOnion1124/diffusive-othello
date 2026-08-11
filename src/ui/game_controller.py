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
PHASE_LOADING = "loading"
PHASE_GAME = "game"
PHASE_END = "end"

MODE_PVP = "PVP"
MODE_PVE = "PVE"

SIDEBAR_ACTION_TOGGLE_WIN_RATE = "toggle_win_rate"
SIDEBAR_ACTION_TOGGLE_SUGGESTIONS = "toggle_suggestions"
SIDEBAR_ACTION_END_MATCH = "end_match"


class MovePolicy(Protocol):
    def select_move(self, state: GameState, player: int) -> Move:
        """Return a legal move for the given state and player."""


class WinRatePolicy(Protocol):
    def predict_player_win_rate(
        self,
        state: GameState,
        *,
        current_player: int,
        target_player: int,
    ) -> float:
        """Return the estimated target-player win rate for the current state."""


@dataclass(frozen=True)
class MoveSuggestion:
    """One legal move together with its masked policy probability."""

    move: Move
    probability: float


class MoveSuggestionPolicy(Protocol):
    def suggest_moves(
        self,
        state: GameState,
        player: int,
        *,
        limit: int = 3,
    ) -> tuple[MoveSuggestion, ...]:
        """Return up to ``limit`` legal policy suggestions for the position."""


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
    first_player_win_rate: float | None = None
    win_rate_invalid: bool = False
    legal_place_moves: tuple[Move, ...] = ()
    move_suggestions: tuple[MoveSuggestion, ...] = ()
    player_names: dict[int, str] | None = None
    show_win_rate_prediction: bool = True
    show_move_suggestions: bool = True


class FirstLegalMovePolicy:
    """Deterministic fallback policy used when AI inference is unavailable."""

    def select_move(self, state: GameState, player: int) -> Move:
        moves = legal_moves(state, player)
        if not moves:
            raise ValueError(f"Player {player} has no legal moves.")
        return moves[0]


class LazyAiPolicy:
    """Load the torch-backed AI only when a feature needs inference."""

    def __init__(self) -> None:
        self._ai = None
        self._load_error: RuntimeError | None = None
        self._prediction_cache_key: tuple[GameState, int] | None = None
        self._prediction_cache: dict | None = None

    def predict_player_win_rate(
        self,
        state: GameState,
        *,
        current_player: int,
        target_player: int,
    ) -> float:
        prediction = self._predict(state, current_player)
        value = float(prediction["value"])
        target_value = value if current_player == target_player else -value
        return max(0.0, min(1.0, (target_value + 1.0) / 2.0))

    def select_move(self, state: GameState, player: int) -> Move:
        prediction = self._predict(state, player)
        move = tuple(prediction["pos"])
        if move not in legal_moves(state, player):
            raise RuntimeError(f"AI returned illegal move {move} for player {player}.")
        return move

    def suggest_moves(
        self,
        state: GameState,
        player: int,
        *,
        limit: int = 3,
    ) -> tuple[MoveSuggestion, ...]:
        ai = self._load_ai()
        prediction = self._predict(state, player)
        suggestions = ai.suggestions_from_prediction(
            prediction,
            board_size=state.size,
            limit=limit,
        )
        return tuple(
            MoveSuggestion(move=move, probability=probability)
            for move, probability in suggestions
        )

    def preload(self) -> None:
        """Load the optional model before the first rendered game frame."""
        self._load_ai()

    def _predict(self, state: GameState, player: int) -> dict:
        cache_key = (state, player)
        if self._prediction_cache_key != cache_key:
            ai = self._load_ai()
            self._prediction_cache = ai.inference(
                board=[list(row) for row in state.board],
                player=player,
            )
            self._prediction_cache_key = cache_key
        if self._prediction_cache is None:
            raise RuntimeError("AI prediction cache was unexpectedly empty.")
        return self._prediction_cache

    def _load_ai(self):
        if self._load_error is not None:
            raise self._load_error
        if self._ai is None:
            try:
                from src.model.inference import GameAI
            except Exception as exc:
                self._load_error = RuntimeError(
                    "AI dependencies could not be imported. Install requirements-ai.txt "
                    "and ensure the configured model is available."
                )
                raise self._load_error from exc

            try:
                self._ai = GameAI()
            except Exception as exc:
                self._load_error = RuntimeError(
                    "AI model could not be loaded. Check config.yaml and the checkpoint path."
                )
                raise self._load_error from exc
        return self._ai


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
        self.mode = self._normalize_mode(mode)
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
        self.first_player_win_rate: float | None = None
        self.win_rate_invalid = False
        self.move_suggestions: tuple[MoveSuggestion, ...] = ()
        self.show_win_rate_prediction = True
        self.show_move_suggestions = True
        self._win_rate_error_reported = False
        self._pending_mode: str | None = None

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
            and not self.is_ai_turn
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
            first_player_win_rate=self.first_player_win_rate,
            win_rate_invalid=self.win_rate_invalid,
            legal_place_moves=legal_place_moves,
            move_suggestions=self.move_suggestions,
            player_names=dict(self.player_names),
            show_win_rate_prediction=self.show_win_rate_prediction,
            show_move_suggestions=self.show_move_suggestions,
        )

    def start_game(self, mode: str | None = None) -> None:
        if mode is not None:
            self.mode = self._normalize_mode(mode)
        self._pending_mode = None
        self.state = new_game(self.board_size)
        if self.mode == MODE_PVE:
            self.human_player = random.choice(PLAYERS)
            self.ai_player = -self.human_player
            self.player_names = {
                self.human_player: "YOU",
                self.ai_player: "COMPUTER",
            }
        else:
            self.human_player = PLAYER_ONE
            self.ai_player = PLAYER_TWO
            self.player_names = {PLAYER_ONE: "Player1", PLAYER_TWO: "Player2"}
        self.phase = PHASE_GAME
        self.winner = EMPTY
        self.winner_name = ""
        self.first_player_win_rate = None
        self.win_rate_invalid = False
        self.move_suggestions = ()
        self._win_rate_error_reported = False
        self.info = self._turn_info(self.current_player)
        self._settle_turn()
        self._refresh_ai_analysis()

    def begin_loading(self, mode: str | None = None) -> None:
        """Enter the pre-game loading state without blocking the UI loop."""
        if mode is not None:
            self.mode = self._normalize_mode(mode)
        self._pending_mode = self.mode
        self.phase = PHASE_LOADING
        self.state = None
        self.winner = EMPTY
        self.winner_name = ""
        self.info = "Loading game model..."
        self.first_player_win_rate = None
        self.win_rate_invalid = False
        self.move_suggestions = ()
        self._win_rate_error_reported = False

    def finish_loading(self) -> bool:
        """Start the selected match after background preloading has finished."""
        if self.phase != PHASE_LOADING:
            return False
        self.start_game(self._pending_mode)
        return True

    def return_to_start(self) -> None:
        self.phase = PHASE_START
        self.state = None
        self._pending_mode = None
        self.winner = EMPTY
        self.winner_name = ""
        self.info = "Diffusive Othello"
        self.first_player_win_rate = None
        self.win_rate_invalid = False
        self.move_suggestions = ()
        self._win_rate_error_reported = False

    def handle_click(
        self,
        clicked: bool,
        move: Move | None = None,
        *,
        start_mode: str | None = None,
    ) -> bool:
        if not clicked:
            return False

        if self.phase == PHASE_START:
            if start_mode is None:
                return False
            self.begin_loading(start_mode)
            return True

        if self.phase == PHASE_END:
            self.return_to_start()
            return True

        if self.phase != PHASE_GAME or self.state is None:
            return False

        if self.is_ai_turn:
            return False

        if move is None:
            return False

        return self.play_human_move(move)

    def handle_sidebar_action(self, action: str | None) -> bool:
        """Apply a sidebar control action during an active game."""
        if self.phase != PHASE_GAME or action is None:
            return False

        if action == SIDEBAR_ACTION_TOGGLE_WIN_RATE:
            self.show_win_rate_prediction = not self.show_win_rate_prediction
            self._refresh_first_player_win_rate()
            return True

        if action == SIDEBAR_ACTION_TOGGLE_SUGGESTIONS:
            self.show_move_suggestions = not self.show_move_suggestions
            self._refresh_move_suggestions()
            return True

        if action == SIDEBAR_ACTION_END_MATCH:
            return self.end_match_early()

        return False

    def end_match_early(self) -> bool:
        """End an active match and settle the winner from the current board score."""
        if self.phase != PHASE_GAME or self.state is None:
            return False
        self._end_game(info="Match ended early; current score settled.")
        return True

    def play_human_move(self, move: Move) -> bool:
        if self.phase != PHASE_GAME or self.state is None:
            return False

        player = self.current_player
        if move not in legal_moves(self.state, player):
            self.info = "Try another position."
            return False

        self.state = apply_move(self.state, player, move).state
        self._settle_turn()
        self._refresh_ai_analysis()
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
        self._refresh_ai_analysis()
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
            return f"{self.player_names[player]} TO MOVE."
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

    def _end_game(self, *, info: str = "Game over.") -> None:
        if self.state is None:
            return

        game_winner = winner(self.state)
        self.winner = game_winner
        self.winner_name = self._winner_name(game_winner)
        self.phase = PHASE_END
        self.info = info
        self.win_rate_invalid = False
        self.move_suggestions = ()
        if game_winner == EMPTY:
            self.first_player_win_rate = 0.5
        else:
            self.first_player_win_rate = 1.0 if game_winner == PLAYER_ONE else 0.0

    def _report_ai_error(self, error: Exception) -> None:
        if self.error_sink is not None:
            self.error_sink(f"PVE AI inference failed: {error}")

    def _refresh_first_player_win_rate(self) -> None:
        if self.state is None or not self.show_win_rate_prediction:
            self.first_player_win_rate = None
            self.win_rate_invalid = False
            return

        if self.phase == PHASE_END:
            return

        predictor = getattr(self.ai_policy, "predict_player_win_rate", None)
        if predictor is None:
            self.first_player_win_rate = 0.5
            self.win_rate_invalid = True
            return

        try:
            self.first_player_win_rate = predictor(
                self.state,
                current_player=self.current_player,
                target_player=PLAYER_ONE,
            )
            self.win_rate_invalid = False
            self._win_rate_error_reported = False
        except Exception as exc:
            self.first_player_win_rate = 0.5
            self.win_rate_invalid = True
            if self.error_sink is not None and not self._win_rate_error_reported:
                self.error_sink(f"Win-rate prediction failed: {exc}")
                self._win_rate_error_reported = True

    def _refresh_move_suggestions(self) -> None:
        if (
            self.phase != PHASE_GAME
            or self.state is None
            or not self.show_move_suggestions
        ):
            self.move_suggestions = ()
            return

        if self.mode == MODE_PVE and self.current_player == self.ai_player:
            self.move_suggestions = ()
            return

        suggester = getattr(self.ai_policy, "suggest_moves", None)
        if not callable(suggester):
            self.move_suggestions = ()
            return

        try:
            suggestions = tuple(
                suggester(self.state, self.current_player, limit=3)
            )
        except Exception:
            self.move_suggestions = ()
            return

        legal = set(legal_moves(self.state, self.current_player))
        self.move_suggestions = tuple(
            suggestion
            for suggestion in suggestions
            if suggestion.move in legal
        )[:3]

    def _refresh_ai_analysis(self) -> None:
        """Refresh both UI analysis views from the current model prediction."""
        self._refresh_first_player_win_rate()
        self._refresh_move_suggestions()

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized_mode = mode.upper()
        if normalized_mode not in (MODE_PVP, MODE_PVE):
            raise ValueError("mode must be PVP or PVE.")
        return normalized_mode

    def _winner_name(self, game_winner: int) -> str:
        if game_winner == EMPTY:
            return "Draw"
        if self.mode == MODE_PVE:
            return "YOU" if game_winner == self.human_player else "COMPUTER"
        return self.player_names[game_winner]
