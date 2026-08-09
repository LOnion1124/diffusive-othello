"""Pygame renderer for Diffusive Othello."""

from __future__ import annotations

from src.game.state import PLAYER_ONE, PLAYER_TWO
from src.ui.game_controller import (
    GameSnapshot,
    MODE_PVE,
    MODE_PVP,
    PHASE_END,
    PHASE_GAME,
    PHASE_START,
)


class PygameRenderer:
    GRID_SIZE = 60
    SCOREBOARD_HEIGHT = 40
    WINRATE_BAR_HEIGHT = 40
    INFOBOX_HEIGHT = 40

    COLOR_BACKGROUND = (128, 128, 128)
    COLOR_BORDER = (255, 255, 255)
    COLOR_PLAYER1 = (255, 165, 0)
    COLOR_PLAYER2 = (0, 255, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_PANEL = (165, 42, 42)
    COLOR_BUTTON = COLOR_PANEL
    COLOR_BUTTON_BORDER = (235, 235, 235)
    COLOR_MOVE_HINT = (240, 240, 240)
    COLOR_WINRATE_BACKGROUND = COLOR_PANEL
    COLOR_WINRATE_BORDER = (230, 230, 230)

    def __init__(
        self,
        pygame_module,
        screen,
        *,
        board_size: int,
        show_winrate_bar: bool = False,
    ) -> None:
        self.pygame = pygame_module
        self.screen = screen
        self.board_size = board_size
        self.show_winrate_bar = show_winrate_bar
        winrate_height = self.WINRATE_BAR_HEIGHT if self.show_winrate_bar else 0
        self.board_left_top = (0, self.SCOREBOARD_HEIGHT + winrate_height)
        self.board_length = self.board_size * self.GRID_SIZE
        self.piece_radius = round(self.GRID_SIZE * 0.4)

        self.font_large = self.pygame.font.SysFont("microsoftyahei", 48)
        self.font_middle = self.pygame.font.SysFont("microsoftyahei", 36)
        self.font_small = self.pygame.font.SysFont("microsoftyahei", 24)
        self.font_tiny = self.pygame.font.SysFont("microsoftyahei", 18)

    @classmethod
    def screen_size(
        cls,
        board_size: int,
        *,
        show_winrate_bar: bool = False,
    ) -> tuple[int, int]:
        board_length = board_size * cls.GRID_SIZE
        winrate_height = cls.WINRATE_BAR_HEIGHT if show_winrate_bar else 0
        return (
            board_length,
            cls.SCOREBOARD_HEIGHT + winrate_height + board_length + cls.INFOBOX_HEIGHT,
        )

    def draw(self, snapshot: GameSnapshot) -> None:
        if snapshot.phase == PHASE_START:
            self._draw_start()
        elif snapshot.phase == PHASE_GAME:
            self._draw_game(snapshot)
        elif snapshot.phase == PHASE_END:
            self._draw_end(snapshot)

        self._draw_scoreboard(snapshot)
        if self._should_draw_winrate_bar(snapshot):
            self._draw_winrate_bar(snapshot)
        self._draw_info_box(snapshot.info)

    def _draw_start(self) -> None:
        self._draw_panel_background()
        top = self.SCOREBOARD_HEIGHT
        height = self.board_left_top[1] - self.SCOREBOARD_HEIGHT + self.board_length
        title_center = (
            self.board_left_top[0] + (self.board_length // 2),
            top + round(height * 0.28),
        )
        title_text = self.font_large.render("Diffusive Othello", True, self.COLOR_TEXT)
        title_rect = title_text.get_rect(center=title_center)
        self.screen.blit(title_text, title_rect)

        button_labels = {
            MODE_PVP: "Play PvP",
            MODE_PVE: "Play PvE",
        }
        for mode, rect in self._start_button_rects().items():
            self._draw_start_button(button_labels[mode], rect)

    def _draw_game(self, snapshot: GameSnapshot) -> None:
        self._draw_board_background(self.COLOR_BACKGROUND)
        self._draw_grid()
        self._draw_place_hints(snapshot)
        self._draw_pieces(snapshot)

    def _draw_end(self, snapshot: GameSnapshot) -> None:
        self._draw_panel_background()
        title = "Draw" if snapshot.winner_name == "Draw" else f"Winner: {snapshot.winner_name}"
        self._draw_panel_title(title)

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

    def _draw_winrate_bar(self, snapshot: GameSnapshot) -> None:
        top = self.SCOREBOARD_HEIGHT
        self.pygame.draw.rect(
            self.screen,
            self.COLOR_WINRATE_BACKGROUND,
            (0, top, self.board_length, self.WINRATE_BAR_HEIGHT),
        )

        margin = 10
        bar_width = self.board_length - 2 * margin
        bar_height = 12
        bar_left = margin
        bar_top = top + self.WINRATE_BAR_HEIGHT - bar_height - 6
        first_rate = snapshot.first_player_win_rate
        fill_rate = self._winrate_fill_rate(snapshot)
        first_width = round(bar_width * fill_rate)
        second_width = bar_width - first_width

        self.pygame.draw.rect(
            self.screen,
            self.COLOR_PLAYER1,
            (bar_left, bar_top, first_width, bar_height),
        )
        self.pygame.draw.rect(
            self.screen,
            self.COLOR_PLAYER2,
            (bar_left + first_width, bar_top, second_width, bar_height),
        )
        self.pygame.draw.rect(
            self.screen,
            self.COLOR_WINRATE_BORDER,
            (bar_left, bar_top, bar_width, bar_height),
            width=1,
        )

        label = self._format_winrate_label(first_rate, invalid=snapshot.win_rate_invalid)
        label_text = self.font_tiny.render(label, True, self.COLOR_TEXT)
        label_rect = label_text.get_rect(
            center=(self.board_length // 2, top + 11)
        )
        self.screen.blit(label_text, label_rect)

    def _draw_info_box(self, info: str) -> None:
        top = self.SCOREBOARD_HEIGHT + self.board_length
        if self.show_winrate_bar:
            top += self.WINRATE_BAR_HEIGHT
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

    def _should_draw_winrate_bar(self, snapshot: GameSnapshot) -> bool:
        return (
            self.show_winrate_bar
            and snapshot.phase == PHASE_GAME
        )

    def start_mode_at(self, position: tuple[int, int] | None) -> str | None:
        if position is None:
            return None
        for mode, rect in self._start_button_rects().items():
            if self._point_in_rect(position, rect):
                return mode
        return None

    @staticmethod
    def _format_winrate_label(
        first_rate: float | None,
        *,
        invalid: bool = False,
    ) -> str:
        if invalid:
            return "Invalid"
        if first_rate is None:
            first_rate = 0.5
        fill_rate = max(0.0, min(1.0, first_rate))
        first_percent = round(fill_rate * 100)
        second_percent = 100 - first_percent
        return f"{first_percent:d} : {second_percent:d}"

    @staticmethod
    def _winrate_fill_rate(snapshot: GameSnapshot) -> float:
        if snapshot.win_rate_invalid or snapshot.first_player_win_rate is None:
            return 0.5
        return max(0.0, min(1.0, snapshot.first_player_win_rate))

    def _draw_panel_background(self) -> None:
        left = self.board_left_top[0]
        top = self.SCOREBOARD_HEIGHT
        height = self.board_left_top[1] - self.SCOREBOARD_HEIGHT + self.board_length
        self.pygame.draw.rect(
            self.screen,
            self.COLOR_PANEL,
            (left, top, self.board_length, height),
        )

    def _draw_panel_title(self, text: str) -> None:
        top = self.SCOREBOARD_HEIGHT
        height = self.board_left_top[1] - self.SCOREBOARD_HEIGHT + self.board_length
        center = (
            self.board_left_top[0] + (self.board_length // 2),
            top + (height // 2),
        )
        title_text = self.font_large.render(text, True, self.COLOR_TEXT)
        title_rect = title_text.get_rect(center=center)
        self.screen.blit(title_text, title_rect)

    def _draw_start_button(self, label: str, rect: tuple[int, int, int, int]) -> None:
        self.pygame.draw.rect(self.screen, self.COLOR_BUTTON, rect)
        self.pygame.draw.rect(self.screen, self.COLOR_BUTTON_BORDER, rect, width=2)
        label_text = self.font_small.render(label, True, self.COLOR_TEXT)
        x, y, width, height = rect
        label_rect = label_text.get_rect(center=(x + width // 2, y + height // 2))
        self.screen.blit(label_text, label_rect)

    def _start_button_rects(self) -> dict[str, tuple[int, int, int, int]]:
        panel_top = self.SCOREBOARD_HEIGHT
        panel_height = self.board_left_top[1] - self.SCOREBOARD_HEIGHT + self.board_length
        button_width = min(220, max(140, self.board_length - 80))
        button_height = 46
        left = self.board_left_top[0] + (self.board_length - button_width) // 2
        centers = (
            panel_top + round(panel_height * 0.48),
            panel_top + round(panel_height * 0.60),
        )
        return {
            MODE_PVP: (
                left,
                centers[0] - button_height // 2,
                button_width,
                button_height,
            ),
            MODE_PVE: (
                left,
                centers[1] - button_height // 2,
                button_width,
                button_height,
            ),
        }

    @staticmethod
    def _point_in_rect(
        position: tuple[int, int],
        rect: tuple[int, int, int, int],
    ) -> bool:
        x_pos, y_pos = position
        left, top, width, height = rect
        return left <= x_pos < left + width and top <= y_pos < top + height
