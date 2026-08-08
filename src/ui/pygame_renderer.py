"""Pygame renderer for Diffusive Othello."""

from __future__ import annotations

from src.game.state import PLAYER_ONE, PLAYER_TWO
from src.ui.game_controller import GameSnapshot, PHASE_END, PHASE_GAME, PHASE_START


class PygameRenderer:
    GRID_SIZE = 60
    SCOREBOARD_HEIGHT = 40
    INFOBOX_HEIGHT = 40

    COLOR_BACKGROUND = (128, 128, 128)
    COLOR_BORDER = (255, 255, 255)
    COLOR_PLAYER1 = (255, 165, 0)
    COLOR_PLAYER2 = (0, 255, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_PANEL = (165, 42, 42)
    COLOR_MOVE_HINT = (240, 240, 240)

    def __init__(self, pygame_module, screen, *, board_size: int) -> None:
        self.pygame = pygame_module
        self.screen = screen
        self.board_size = board_size
        self.board_left_top = (0, self.SCOREBOARD_HEIGHT)
        self.board_length = self.board_size * self.GRID_SIZE
        self.piece_radius = round(self.GRID_SIZE * 0.4)

        self.font_large = self.pygame.font.SysFont("microsoftyahei", 48)
        self.font_middle = self.pygame.font.SysFont("microsoftyahei", 36)
        self.font_small = self.pygame.font.SysFont("microsoftyahei", 24)

    @classmethod
    def screen_size(cls, board_size: int) -> tuple[int, int]:
        board_length = board_size * cls.GRID_SIZE
        return (
            board_length,
            cls.SCOREBOARD_HEIGHT + board_length + cls.INFOBOX_HEIGHT,
        )

    def draw(self, snapshot: GameSnapshot) -> None:
        if snapshot.phase == PHASE_START:
            self._draw_start()
        elif snapshot.phase == PHASE_GAME:
            self._draw_game(snapshot)
        elif snapshot.phase == PHASE_END:
            self._draw_end(snapshot)

        self._draw_scoreboard(snapshot)
        self._draw_info_box(snapshot.info)

    def _draw_start(self) -> None:
        self._draw_board_background(self.COLOR_PANEL)
        self._draw_title("CLICK TO START")

    def _draw_game(self, snapshot: GameSnapshot) -> None:
        self._draw_board_background(self.COLOR_BACKGROUND)
        self._draw_grid()
        self._draw_place_hints(snapshot)
        self._draw_pieces(snapshot)
        self._draw_flip_hints(snapshot)

    def _draw_end(self, snapshot: GameSnapshot) -> None:
        self._draw_board_background(self.COLOR_PANEL)
        title = "Draw" if snapshot.winner_name == "Draw" else f"Winner: {snapshot.winner_name}"
        self._draw_title(title)

    def _draw_board_background(self, color: tuple[int, int, int]) -> None:
        left, top = self.board_left_top
        self.pygame.draw.rect(
            self.screen,
            color,
            (left, top, self.board_length, self.board_length),
        )

    def _draw_grid(self) -> None:
        left, top = self.board_left_top
        for idx in range(self.board_size + 1):
            x = left + idx * self.GRID_SIZE
            y = top + idx * self.GRID_SIZE
            self.pygame.draw.line(
                self.screen,
                self.COLOR_BORDER,
                (x, top),
                (x, top + self.board_length),
            )
            self.pygame.draw.line(
                self.screen,
                self.COLOR_BORDER,
                (left, y),
                (left + self.board_length, y),
            )

    def _draw_pieces(self, snapshot: GameSnapshot) -> None:
        if snapshot.state is None:
            return

        left, top = self.board_left_top
        for x in range(self.board_size):
            for y in range(self.board_size):
                status = snapshot.state.board[x][y]
                if status not in (PLAYER_ONE, PLAYER_TWO):
                    continue
                color = self.COLOR_PLAYER1 if status == PLAYER_ONE else self.COLOR_PLAYER2
                center = (
                    left + (self.GRID_SIZE // 2) + x * self.GRID_SIZE,
                    top + (self.GRID_SIZE // 2) + y * self.GRID_SIZE,
                )
                self.pygame.draw.circle(self.screen, color, center, self.piece_radius)

    def _draw_place_hints(self, snapshot: GameSnapshot) -> None:
        left, top = self.board_left_top
        hint_radius = max(4, round(self.GRID_SIZE * 0.11))
        for x, y in snapshot.legal_place_moves:
            center = (
                left + (self.GRID_SIZE // 2) + x * self.GRID_SIZE,
                top + (self.GRID_SIZE // 2) + y * self.GRID_SIZE,
            )
            self.pygame.draw.circle(
                self.screen,
                self.COLOR_MOVE_HINT,
                center,
                hint_radius,
            )

    def _draw_flip_hints(self, snapshot: GameSnapshot) -> None:
        color = (
            self.COLOR_PLAYER1
            if snapshot.current_player == PLAYER_ONE
            else self.COLOR_PLAYER2
        )
        left, top = self.board_left_top
        ring_radius = min(self.GRID_SIZE // 2 - 5, self.piece_radius + 5)
        for x, y in snapshot.legal_flip_moves:
            center = (
                left + (self.GRID_SIZE // 2) + x * self.GRID_SIZE,
                top + (self.GRID_SIZE // 2) + y * self.GRID_SIZE,
            )
            self.pygame.draw.circle(
                self.screen,
                self.COLOR_MOVE_HINT,
                center,
                ring_radius,
                4,
            )
            self.pygame.draw.circle(self.screen, color, center, ring_radius - 4, 3)

    def _draw_title(self, text: str) -> None:
        center = (
            self.board_left_top[0] + (self.board_length // 2),
            self.board_left_top[1] + (self.board_length // 2),
        )
        title_text = self.font_large.render(text, True, self.COLOR_TEXT)
        title_rect = title_text.get_rect(center=center)
        self.screen.blit(title_text, title_rect)

    def _draw_scoreboard(self, snapshot: GameSnapshot) -> None:
        self.pygame.draw.rect(
            self.screen,
            self.COLOR_PANEL,
            (0, 0, self.board_length, self.SCOREBOARD_HEIGHT),
        )
        score1 = str(snapshot.scores[PLAYER_ONE]) if snapshot.phase != PHASE_START else ""
        score2 = str(snapshot.scores[PLAYER_TWO]) if snapshot.phase != PHASE_START else ""
        score_centers = (
            (self.board_length // 4, self.SCOREBOARD_HEIGHT // 2),
            (3 * self.board_length // 4, self.SCOREBOARD_HEIGHT // 2),
        )
        for score_text, center in zip((score1, score2), score_centers):
            text_surface = self.font_middle.render(score_text, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=center)
            self.screen.blit(text_surface, text_rect)

    def _draw_info_box(self, info: str) -> None:
        top = self.SCOREBOARD_HEIGHT + self.board_length
        self.pygame.draw.rect(
            self.screen,
            self.COLOR_PANEL,
            (0, top, self.board_length, self.INFOBOX_HEIGHT),
        )
        info_text = self.font_small.render(info, True, self.COLOR_TEXT)
        info_rect = info_text.get_rect(
            center=(self.board_length // 2, top + self.INFOBOX_HEIGHT // 2)
        )
        self.screen.blit(info_text, info_rect)
