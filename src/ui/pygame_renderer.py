"""Polished pygame renderer for Diffusive Othello."""

from __future__ import annotations

import math

from src.game.state import PLAYER_ONE, PLAYER_TWO
from src.ui.game_controller import (
    GameSnapshot,
    MODE_PVE,
    MODE_PVP,
    PHASE_END,
    PHASE_GAME,
    PHASE_LOADING,
    PHASE_START,
)


class PygameRenderer:
    """Draw the desktop client without owning any game-flow state."""

    GRID_SIZE = 60
    SCOREBOARD_HEIGHT = 40
    WINRATE_BAR_HEIGHT = 40
    INFOBOX_HEIGHT = 40

    OUTER_PADDING = 24
    HEADER_HEIGHT = 68
    HEADER_GAP = 20
    SIDEBAR_WIDTH = 220
    CONTENT_GAP = 24
    CARD_RADIUS = 16
    PIECE_ANIMATION_MS = 170
    PIECE_FLIP_ANIMATION_MS = 260
    SUGGESTION_HINT_MAX_RADIUS = GRID_SIZE // 2 - 5
    SUGGESTION_HINT_MIN_RADIUS = SUGGESTION_HINT_MAX_RADIUS - 3

    # Warm desktop surfaces, a jade board, and blue/orange player colors make
    # the board legible without relying on the red-versus-green distinction.
    COLOR_BACKGROUND = (239, 238, 232)
    COLOR_SURFACE = (255, 253, 248)
    COLOR_SURFACE_MUTED = (232, 235, 228)
    COLOR_SHADOW = (204, 210, 200)
    COLOR_BORDER = (210, 218, 208)
    COLOR_TEXT = (31, 47, 47)
    COLOR_TEXT_MUTED = (101, 117, 113)
    COLOR_BOARD = (33, 104, 84)
    COLOR_BOARD_ALT = (38, 112, 90)
    COLOR_BOARD_EDGE = (21, 75, 61)
    COLOR_GRID = (23, 82, 67)
    COLOR_PLAYER1 = (241, 139, 75)
    COLOR_PLAYER1_DARK = (190, 84, 38)
    COLOR_PLAYER1_LIGHT = (255, 213, 176)
    COLOR_PLAYER2 = (92, 166, 220)
    COLOR_PLAYER2_DARK = (43, 108, 161)
    COLOR_PLAYER2_LIGHT = (200, 232, 250)
    COLOR_ACCENT = (214, 171, 92)
    COLOR_OVERLAY = (16, 42, 37)
    COLOR_BUTTON = COLOR_SURFACE
    COLOR_BUTTON_BORDER = COLOR_BORDER
    COLOR_WINRATE_BACKGROUND = COLOR_SURFACE_MUTED
    COLOR_WINRATE_BORDER = COLOR_BORDER
    COLOR_SUGGESTION_PRIMARY = (242, 245, 244)
    COLOR_SUGGESTION_SECONDARY = (191, 203, 200)
    COLOR_MOVE_HINT = COLOR_SUGGESTION_SECONDARY

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
        self.board_length = self.board_size * self.GRID_SIZE
        self.window_width, self.window_height = self.screen_size(
            self.board_size,
            show_winrate_bar=self.show_winrate_bar,
        )
        self.board_left_top = (
            self.OUTER_PADDING,
            self.OUTER_PADDING + self.HEADER_HEIGHT + self.HEADER_GAP,
        )
        self.sidebar_rect = (
            self.board_left_top[0] + self.board_length + self.CONTENT_GAP,
            self.board_left_top[1],
            self.SIDEBAR_WIDTH,
            self.board_length,
        )
        self.content_rect = (
            self.OUTER_PADDING,
            self.board_left_top[1],
            self.window_width - 2 * self.OUTER_PADDING,
            self.board_length,
        )
        self.piece_radius = round(self.GRID_SIZE * 0.36)

        self.font_display = self.pygame.font.SysFont("microsoftyahei", 40, bold=True)
        self.font_large = self.pygame.font.SysFont("microsoftyahei", 30, bold=True)
        self.font_middle = self.pygame.font.SysFont("microsoftyahei", 24, bold=True)
        self.font_small = self.pygame.font.SysFont("microsoftyahei", 18)
        self.font_tiny = self.pygame.font.SysFont("microsoftyahei", 14, bold=True)

        self._previous_board: tuple[tuple[int, ...], ...] | None = None
        self._piece_entered_at: dict[tuple[int, int], int] = {}
        self._piece_flip_at: dict[tuple[int, int], tuple[int, int]] = {}
        self._displayed_winrate: float | None = None

    @classmethod
    def screen_size(
        cls,
        board_size: int,
        *,
        show_winrate_bar: bool = False,
    ) -> tuple[int, int]:
        """Return a stable board-plus-sidebar desktop layout.

        ``show_winrate_bar`` remains part of the public API for compatibility;
        the projection now lives in the sidebar rather than changing height.
        """
        board_length = board_size * cls.GRID_SIZE
        return (
            2 * cls.OUTER_PADDING + board_length + cls.CONTENT_GAP + cls.SIDEBAR_WIDTH,
            2 * cls.OUTER_PADDING + cls.HEADER_HEIGHT + cls.HEADER_GAP + board_length,
        )

    def draw(self, snapshot: GameSnapshot, *, now_ms: int | None = None) -> None:
        """Render one frame. The optional timestamp keeps animation testable."""
        if now_ms is None:
            now_ms = self.pygame.time.get_ticks()

        self._draw_window_background()
        self._draw_header(snapshot)
        if snapshot.phase == PHASE_START:
            self._reset_animations()
            self._draw_start(now_ms)
            return
        if snapshot.phase == PHASE_LOADING:
            self._reset_animations()
            self._draw_loading(now_ms)
            return

        self._update_piece_animations(snapshot, now_ms)
        self._draw_game(snapshot, now_ms)
        self._draw_sidebar(snapshot)
        if snapshot.phase == PHASE_END:
            self._draw_end(snapshot)

    def _draw_window_background(self) -> None:
        self.screen.fill(self.COLOR_BACKGROUND)

    def _draw_header(self, snapshot: GameSnapshot) -> None:
        title_x = self.OUTER_PADDING
        title_y = self.OUTER_PADDING + 18
        self._blit_text(
            "DIFFUSIVE OTHELLO",
            self.font_middle,
            self.COLOR_TEXT,
            topleft=(title_x, title_y),
        )
        self._blit_text(
            "DESKTOP BOARD GAME",
            self.font_tiny,
            self.COLOR_TEXT_MUTED,
            topleft=(title_x, title_y + 29),
        )

        if snapshot.phase == PHASE_START:
            self._draw_pill(
                "CHOOSE A MODE",
                (self.window_width - self.OUTER_PADDING - 142, self.OUTER_PADDING + 18, 142, 28),
                fill=self.COLOR_SURFACE_MUTED,
                text_color=self.COLOR_TEXT_MUTED,
            )
            return
        if snapshot.phase == PHASE_LOADING:
            self._draw_pill(
                "LOADING MODEL",
                (self.window_width - self.OUTER_PADDING - 142, self.OUTER_PADDING + 18, 142, 28),
                fill=self.COLOR_SURFACE_MUTED,
                text_color=self.COLOR_TEXT_MUTED,
            )
            return

        mode_width = 52
        self._draw_pill(
            snapshot.mode,
            (self.window_width - self.OUTER_PADDING - 176, self.OUTER_PADDING + 18, mode_width, 28),
            fill=self.COLOR_SURFACE_MUTED,
            text_color=self.COLOR_TEXT,
        )
        player_color = self._player_color(snapshot.current_player)
        self.pygame.draw.circle(
            self.screen,
            player_color,
            (self.window_width - self.OUTER_PADDING - 108, self.OUTER_PADDING + 32),
            6,
        )
        turn_label = (
            f"P{1 if snapshot.current_player == PLAYER_ONE else 2} TO MOVE"
            if snapshot.phase == PHASE_GAME
            else "FINISHED"
        )
        self._blit_text(
            turn_label,
            self.font_tiny,
            self.COLOR_TEXT_MUTED,
            midleft=(self.window_width - self.OUTER_PADDING - 96, self.OUTER_PADDING + 32),
        )

    def _draw_start(self, now_ms: int) -> None:
        self._draw_card(self.content_rect, fill=self.COLOR_SURFACE)
        content_left, content_top, content_width, content_height = self.content_rect
        center_x = content_left + content_width // 2

        pulse = 0.5 + 0.5 * math.sin(now_ms / 700)
        halo_radius = round(46 + 4 * pulse)
        self.pygame.draw.circle(
            self.screen,
            self.COLOR_SURFACE_MUTED,
            (center_x, content_top + round(content_height * 0.22)),
            halo_radius,
        )
        self.pygame.draw.circle(
            self.screen,
            self.COLOR_BOARD,
            (center_x, content_top + round(content_height * 0.22)),
            32,
        )
        self.pygame.draw.circle(
            self.screen,
            self.COLOR_PLAYER1,
            (center_x - 11, content_top + round(content_height * 0.22) - 6),
            11,
        )
        self.pygame.draw.circle(
            self.screen,
            self.COLOR_PLAYER2,
            (center_x + 11, content_top + round(content_height * 0.22) + 6),
            11,
        )

        self._blit_text(
            "Pick your match",
            self.font_display,
            self.COLOR_TEXT,
            center=(center_x, content_top + round(content_height * 0.40)),
        )
        self._blit_text(
            "Choose a mode to begin a diffusion battle.",
            self.font_small,
            self.COLOR_TEXT_MUTED,
            center=(center_x, content_top + round(content_height * 0.47)),
        )

        labels = {
            MODE_PVP: ("Play PvP", "Two players, one board", "VS"),
            MODE_PVE: ("Play PvE", "Challenge the computer", "AI"),
        }
        for mode, rect in self._start_button_rects().items():
            label, description, badge = labels[mode]
            self._draw_start_button(label, description, badge, rect)

    def _draw_loading(self, now_ms: int) -> None:
        self._draw_card(self.content_rect, fill=self.COLOR_SURFACE)
        content_left, content_top, content_width, content_height = self.content_rect
        center_x = content_left + content_width // 2
        center_y = content_top + content_height // 2 - 22

        spinner_rect = (center_x - 34, center_y - 34, 68, 68)
        angle = (now_ms % 900) / 900 * math.tau
        self.pygame.draw.arc(
            self.screen,
            self.COLOR_SURFACE_MUTED,
            spinner_rect,
            0,
            math.tau,
            width=6,
        )
        self.pygame.draw.arc(
            self.screen,
            self.COLOR_BOARD,
            spinner_rect,
            angle,
            angle + math.tau * 0.72,
            width=6,
        )
        self.pygame.draw.circle(
            self.screen,
            self.COLOR_PLAYER1,
            (center_x, center_y),
            13,
        )

        self._blit_text(
            "Preparing your match",
            self.font_display,
            self.COLOR_TEXT,
            center=(center_x, center_y + 92),
        )
        self._blit_text(
            "Loading the game model...",
            self.font_small,
            self.COLOR_TEXT_MUTED,
            center=(center_x, center_y + 126),
        )
        self._blit_text(
            "The first launch can take a little longer.",
            self.font_tiny,
            self.COLOR_TEXT_MUTED,
            center=(center_x, center_y + 152),
        )

        active_dot = (now_ms // 180) % 3
        for index in range(3):
            color = self.COLOR_PLAYER2 if index == active_dot else self.COLOR_SURFACE_MUTED
            self.pygame.draw.circle(
                self.screen,
                color,
                (center_x - 18 + index * 18, center_y + 181),
                4,
            )

    def _draw_game(self, snapshot: GameSnapshot, now_ms: int) -> None:
        self._draw_board_background()
        self._draw_grid()
        self._draw_place_hints(snapshot, now_ms)
        self._draw_pieces(snapshot, now_ms)

    def _draw_board_background(self) -> None:
        left, top = self.board_left_top
        shadow_rect = (left - 5, top + 7, self.board_length + 10, self.board_length + 10)
        self.pygame.draw.rect(
            self.screen,
            self.COLOR_SHADOW,
            shadow_rect,
            border_radius=self.CARD_RADIUS + 3,
        )
        self.pygame.draw.rect(
            self.screen,
            self.COLOR_BOARD_EDGE,
            (left - 4, top - 4, self.board_length + 8, self.board_length + 8),
            border_radius=self.CARD_RADIUS,
        )
        for x in range(self.board_size):
            for y in range(self.board_size):
                color = self.COLOR_BOARD if (x + y) % 2 == 0 else self.COLOR_BOARD_ALT
                self.pygame.draw.rect(
                    self.screen,
                    color,
                    (left + x * self.GRID_SIZE, top + y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE),
                )

    def _draw_grid(self) -> None:
        left, top = self.board_left_top
        for idx in range(self.board_size + 1):
            x = left + idx * self.GRID_SIZE
            y = top + idx * self.GRID_SIZE
            self.pygame.draw.line(
                self.screen,
                self.COLOR_GRID,
                (x, top),
                (x, top + self.board_length),
                width=1,
            )
            self.pygame.draw.line(
                self.screen,
                self.COLOR_GRID,
                (left, y),
                (left + self.board_length, y),
                width=1,
            )
        self.pygame.draw.rect(
            self.screen,
            self.COLOR_BOARD_EDGE,
            (left, top, self.board_length, self.board_length),
            width=3,
        )

    def _draw_pieces(self, snapshot: GameSnapshot, now_ms: int) -> None:
        if snapshot.state is None:
            return

        left, top = self.board_left_top
        for x in range(self.board_size):
            for y in range(self.board_size):
                status = snapshot.state.board[x][y]
                if status not in (PLAYER_ONE, PLAYER_TWO):
                    continue
                center = (
                    left + (self.GRID_SIZE // 2) + x * self.GRID_SIZE,
                    top + (self.GRID_SIZE // 2) + y * self.GRID_SIZE,
                )
                visual_status, horizontal_scale = self._flip_visual_state((x, y), status, now_ms)
                scale = 1.0 if horizontal_scale < 1.0 else self._piece_scale((x, y), now_ms)
                radius = max(6, round(self.piece_radius * scale))
                self._draw_piece(visual_status, center, radius, horizontal_scale=horizontal_scale)

    def _draw_piece(
        self,
        player: int,
        center: tuple[int, int],
        radius: int,
        *,
        horizontal_scale: float = 1.0,
    ) -> None:
        horizontal_radius = max(2, round(radius * horizontal_scale))
        center_x, center_y = center
        shadow_rect = (
            center_x - horizontal_radius + 2,
            center_y - radius + 3,
            horizontal_radius * 2,
            radius * 2,
        )
        outline_rect = (
            center_x - horizontal_radius - 2,
            center_y - radius - 2,
            horizontal_radius * 2 + 4,
            radius * 2 + 4,
        )
        piece_rect = (
            center_x - horizontal_radius,
            center_y - radius,
            horizontal_radius * 2,
            radius * 2,
        )
        self.pygame.draw.ellipse(self.screen, self.COLOR_BOARD_EDGE, shadow_rect)
        self.pygame.draw.ellipse(self.screen, self._player_dark_color(player), outline_rect)
        self.pygame.draw.ellipse(self.screen, self._player_color(player), piece_rect)
        if horizontal_radius >= 5:
            highlight_radius = max(2, round(radius * 0.23))
            self.pygame.draw.circle(
                self.screen,
                self._player_light_color(player),
                (center_x - round(horizontal_radius * 0.30), center_y - round(radius * 0.30)),
                highlight_radius,
            )

    def _draw_place_hints(self, snapshot: GameSnapshot, now_ms: int) -> None:
        left, top = self.board_left_top
        hint_color = self._player_color(snapshot.current_player)
        pulse = 0.5 + 0.5 * math.sin(now_ms / 250)
        base_hint_radius = max(7, round(self.GRID_SIZE * (0.13 + 0.035 * pulse)))
        suggestions = {
            suggestion.move: (rank, suggestion.probability)
            for rank, suggestion in enumerate(snapshot.move_suggestions, start=1)
        }
        for x, y in snapshot.legal_place_moves:
            center = (
                left + (self.GRID_SIZE // 2) + x * self.GRID_SIZE,
                top + (self.GRID_SIZE // 2) + y * self.GRID_SIZE,
            )
            suggestion = suggestions.get((x, y))
            if suggestion is None:
                self.pygame.draw.circle(
                    self.screen,
                    hint_color,
                    center,
                    base_hint_radius,
                    width=2,
                )
                self.pygame.draw.circle(self.screen, self.COLOR_MOVE_HINT, center, 3)
                continue

            rank, probability = suggestion
            label = self._format_suggestion_probability(probability)
            suggestion_color = self._suggestion_color(rank)
            self.pygame.draw.circle(
                self.screen,
                hint_color,
                center,
                self._suggestion_hint_radius(pulse),
                width=2,
            )
            self._blit_text(label, self.font_tiny, suggestion_color, center=center)

    def _draw_sidebar(self, snapshot: GameSnapshot) -> None:
        left, top, width, height = self.sidebar_rect
        score_height = 160
        self._draw_score_card(snapshot, (left, top, width, score_height))

        next_top = top + score_height + 16
        if self._should_draw_winrate_bar(snapshot):
            projection_height = 126
            self._draw_projection_card(snapshot, (left, next_top, width, projection_height))
            next_top += projection_height + 16
        self._draw_status_card(snapshot, (left, next_top, width, top + height - next_top))

    def _draw_score_card(self, snapshot: GameSnapshot, rect: tuple[int, int, int, int]) -> None:
        left, top, width, _ = rect
        self._draw_card(rect, fill=self.COLOR_SURFACE)
        self._blit_text("SCORE", self.font_tiny, self.COLOR_TEXT_MUTED, topleft=(left + 18, top + 16))
        self._draw_score_row(
            snapshot,
            player=PLAYER_ONE,
            rect=(left + 18, top + 47, width - 36, 34),
        )
        self._draw_score_row(
            snapshot,
            player=PLAYER_TWO,
            rect=(left + 18, top + 96, width - 36, 34),
        )

    def _draw_score_row(
        self,
        snapshot: GameSnapshot,
        *,
        player: int,
        rect: tuple[int, int, int, int],
    ) -> None:
        left, top, width, height = rect
        is_current = snapshot.phase == PHASE_GAME and snapshot.current_player == player
        if is_current:
            self.pygame.draw.rect(
                self.screen,
                self.COLOR_SURFACE_MUTED,
                rect,
                border_radius=10,
            )
        center_y = top + height // 2
        self.pygame.draw.circle(self.screen, self._player_dark_color(player), (left + 12, center_y + 1), 10)
        self.pygame.draw.circle(self.screen, self._player_color(player), (left + 12, center_y), 9)
        self._blit_text(
            self._player_label(player),
            self.font_tiny,
            self.COLOR_TEXT if is_current else self.COLOR_TEXT_MUTED,
            midleft=(left + 34, center_y),
        )
        self._blit_text(
            str(snapshot.scores[player]),
            self.font_middle,
            self.COLOR_TEXT,
            midright=(left + width - 3, center_y),
        )

    def _draw_projection_card(
        self,
        snapshot: GameSnapshot,
        rect: tuple[int, int, int, int],
    ) -> None:
        left, top, width, _ = rect
        self._draw_card(rect, fill=self.COLOR_SURFACE)
        self._blit_text("WIN PROJECTION", self.font_tiny, self.COLOR_TEXT_MUTED, topleft=(left + 18, top + 16))

        fill_rate = self._animated_winrate(snapshot)
        bar_left = left + 18
        bar_top = top + 52
        bar_width = width - 36
        bar_height = 14
        first_width = round(bar_width * fill_rate)
        self.pygame.draw.rect(
            self.screen,
            self.COLOR_WINRATE_BACKGROUND,
            (bar_left, bar_top, bar_width, bar_height),
            border_radius=bar_height // 2,
        )
        if first_width:
            self.pygame.draw.rect(
                self.screen,
                self.COLOR_PLAYER1,
                (bar_left, bar_top, first_width, bar_height),
                border_radius=bar_height // 2,
            )
        if first_width < bar_width:
            self.pygame.draw.rect(
                self.screen,
                self.COLOR_PLAYER2,
                (bar_left + first_width, bar_top, bar_width - first_width, bar_height),
                border_radius=bar_height // 2,
            )
        self.pygame.draw.rect(
            self.screen,
            self.COLOR_WINRATE_BORDER,
            (bar_left, bar_top, bar_width, bar_height),
            width=1,
            border_radius=bar_height // 2,
        )

        first_percent = round(fill_rate * 100)
        self._blit_text("P1", self.font_tiny, self.COLOR_PLAYER1_DARK, topleft=(bar_left, top + 77))
        self._blit_text(
            f"{first_percent}%",
            self.font_tiny,
            self.COLOR_TEXT,
            topleft=(bar_left + 24, top + 77),
        )
        self._blit_text(
            f"{100 - first_percent}%",
            self.font_tiny,
            self.COLOR_TEXT,
            topright=(bar_left + bar_width - 24, top + 77),
        )
        self._blit_text("P2", self.font_tiny, self.COLOR_PLAYER2_DARK, topright=(bar_left + bar_width, top + 77))
        label = "Model unavailable" if snapshot.win_rate_invalid else "Estimated outcome"
        self._blit_text(label, self.font_tiny, self.COLOR_TEXT_MUTED, topleft=(bar_left, top + 99))

    def _draw_status_card(self, snapshot: GameSnapshot, rect: tuple[int, int, int, int]) -> None:
        left, top, width, height = rect
        self._draw_card(rect, fill=self.COLOR_SURFACE)
        self._blit_text("STATUS", self.font_tiny, self.COLOR_TEXT_MUTED, topleft=(left + 18, top + 16))
        lines = self._wrap_text(snapshot.info, self.font_small, width - 36)
        line_height = self.font_small.get_linesize()
        text_top = top + max(49, (height - line_height * len(lines)) // 2 + 10)
        for index, line in enumerate(lines):
            self._blit_text(
                line,
                self.font_small,
                self.COLOR_TEXT,
                center=(left + width // 2, text_top + index * line_height),
            )
        if snapshot.phase == PHASE_GAME:
            helper_text = (
                "AI Top 3: probability + shade"
                if snapshot.move_suggestions
                else (
                    "Computer is choosing a move"
                    if snapshot.mode == MODE_PVE and not snapshot.legal_place_moves
                    else "Select a glowing position"
                )
            )
            self._blit_text(
                helper_text,
                self.font_tiny,
                self.COLOR_TEXT_MUTED,
                center=(left + width // 2, top + height - 23),
            )

    def _draw_end(self, snapshot: GameSnapshot) -> None:
        overlay = self.pygame.Surface((self.window_width, self.window_height), self.pygame.SRCALPHA)
        overlay.fill((*self.COLOR_OVERLAY, 138))
        self.screen.blit(overlay, (0, 0))

        modal_width = min(420, max(200, self.board_length - 72))
        modal_height = 218
        left = self.board_left_top[0] + (self.board_length - modal_width) // 2
        top = self.board_left_top[1] + (self.board_length - modal_height) // 2
        self._draw_card((left, top, modal_width, modal_height), fill=self.COLOR_SURFACE)

        title = "DRAW" if snapshot.winner_name == "Draw" else f"{snapshot.winner_name.upper()} WINS"
        self._blit_text(title, self.font_large, self.COLOR_TEXT, center=(left + modal_width // 2, top + 55))
        score_text = f"{snapshot.scores[PLAYER_ONE]}  —  {snapshot.scores[PLAYER_TWO]}"
        self._blit_text(score_text, self.font_display, self.COLOR_TEXT, center=(left + modal_width // 2, top + 111))
        self._blit_text("Final score", self.font_tiny, self.COLOR_TEXT_MUTED, center=(left + modal_width // 2, top + 140))
        self._blit_text(
            "Click anywhere to return home",
            self.font_small,
            self.COLOR_TEXT_MUTED,
            center=(left + modal_width // 2, top + 180),
        )

    def _draw_card(
        self,
        rect: tuple[int, int, int, int],
        *,
        fill: tuple[int, int, int],
        border: tuple[int, int, int] | None = None,
        radius: int | None = None,
    ) -> None:
        left, top, width, height = rect
        card_radius = self.CARD_RADIUS if radius is None else radius
        self.pygame.draw.rect(
            self.screen,
            self.COLOR_SHADOW,
            (left, top + 5, width, height),
            border_radius=card_radius,
        )
        self.pygame.draw.rect(
            self.screen,
            fill,
            rect,
            border_radius=card_radius,
        )
        if border is not None:
            self.pygame.draw.rect(
                self.screen,
                border,
                rect,
                width=1,
                border_radius=card_radius,
            )

    def _draw_pill(
        self,
        text: str,
        rect: tuple[int, int, int, int],
        *,
        fill: tuple[int, int, int],
        text_color: tuple[int, int, int],
    ) -> None:
        self.pygame.draw.rect(self.screen, fill, rect, border_radius=rect[3] // 2)
        self._blit_text(
            text,
            self.font_tiny,
            text_color,
            center=(rect[0] + rect[2] // 2, rect[1] + rect[3] // 2),
        )

    def _draw_start_button(
        self,
        label: str,
        description: str,
        badge: str,
        rect: tuple[int, int, int, int],
    ) -> None:
        hovered = self._point_in_rect(self.pygame.mouse.get_pos(), rect)
        left, top, width, height = rect
        visual_top = top - 4 if hovered else top
        visual_rect = (left, visual_top, width, height)
        fill = self.COLOR_SURFACE_MUTED if hovered else self.COLOR_BUTTON
        border = self.COLOR_BOARD if hovered else self.COLOR_BUTTON_BORDER
        self._draw_card(visual_rect, fill=fill, border=border)
        self._draw_pill(
            badge,
            (left + 18, visual_top + 18, 38, 26),
            fill=self.COLOR_BOARD if badge == "VS" else self.COLOR_PLAYER2_DARK,
            text_color=self.COLOR_SURFACE,
        )
        self._blit_text(label, self.font_middle, self.COLOR_TEXT, topleft=(left + 18, visual_top + 58))
        self._blit_text(
            description,
            self.font_small,
            self.COLOR_TEXT_MUTED,
            topleft=(left + 18, visual_top + 90),
        )

    def _update_piece_animations(self, snapshot: GameSnapshot, now_ms: int) -> None:
        if snapshot.state is None:
            return
        board = tuple(tuple(column) for column in snapshot.state.board)
        if self._previous_board is None or len(self._previous_board) != len(board):
            self._piece_flip_at.clear()
            self._piece_entered_at = {
                (x, y): now_ms
                for x in range(self.board_size)
                for y in range(self.board_size)
                if board[x][y] in (PLAYER_ONE, PLAYER_TWO)
            }
        else:
            for x in range(self.board_size):
                for y in range(self.board_size):
                    current = board[x][y]
                    previous = self._previous_board[x][y]
                    if current not in (PLAYER_ONE, PLAYER_TWO) or current == previous:
                        continue
                    if previous in (PLAYER_ONE, PLAYER_TWO):
                        self._piece_flip_at[(x, y)] = (previous, now_ms)
                        self._piece_entered_at.pop((x, y), None)
                    else:
                        self._piece_entered_at[(x, y)] = now_ms
        self._previous_board = board

    def _reset_animations(self) -> None:
        self._previous_board = None
        self._piece_entered_at.clear()
        self._piece_flip_at.clear()
        self._displayed_winrate = None

    def _piece_scale(self, position: tuple[int, int], now_ms: int) -> float:
        started_at = self._piece_entered_at.get(position)
        if started_at is None:
            return 1.0
        progress = min(1.0, max(0.0, (now_ms - started_at) / self.PIECE_ANIMATION_MS))
        if progress >= 1.0:
            self._piece_entered_at.pop(position, None)
            return 1.0
        eased = 1.0 - (1.0 - progress) ** 3
        return 0.76 + 0.24 * eased

    def _flip_visual_state(
        self,
        position: tuple[int, int],
        current_player: int,
        now_ms: int,
    ) -> tuple[int, float]:
        animation = self._piece_flip_at.get(position)
        if animation is None:
            return current_player, 1.0
        previous_player, started_at = animation
        progress = min(1.0, max(0.0, (now_ms - started_at) / self.PIECE_FLIP_ANIMATION_MS))
        if progress >= 1.0:
            self._piece_flip_at.pop(position, None)
            return current_player, 1.0
        if progress < 0.5:
            return previous_player, max(0.12, 1.0 - 2.0 * progress)
        return current_player, max(0.12, 2.0 * progress - 1.0)

    def _animated_winrate(self, snapshot: GameSnapshot) -> float:
        target = self._winrate_fill_rate(snapshot)
        if self._displayed_winrate is None:
            self._displayed_winrate = target
        else:
            self._displayed_winrate += (target - self._displayed_winrate) * 0.16
        return self._displayed_winrate

    def _should_draw_winrate_bar(self, snapshot: GameSnapshot) -> bool:
        return self.show_winrate_bar and snapshot.phase == PHASE_GAME

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
    def _format_suggestion_probability(probability: float) -> str:
        return f"{max(0.0, min(1.0, probability)) * 100:.1f}%"

    @classmethod
    def _suggestion_color(cls, rank: int) -> tuple[int, int, int]:
        return (
            cls.COLOR_SUGGESTION_PRIMARY
            if rank == 1
            else cls.COLOR_SUGGESTION_SECONDARY
        )

    @classmethod
    def _suggestion_hint_radius(cls, pulse: float) -> int:
        normalized_pulse = max(0.0, min(1.0, pulse))
        return round(
            cls.SUGGESTION_HINT_MIN_RADIUS
            + (cls.SUGGESTION_HINT_MAX_RADIUS - cls.SUGGESTION_HINT_MIN_RADIUS)
            * normalized_pulse
        )

    @staticmethod
    def _winrate_fill_rate(snapshot: GameSnapshot) -> float:
        if snapshot.win_rate_invalid or snapshot.first_player_win_rate is None:
            return 0.5
        return max(0.0, min(1.0, snapshot.first_player_win_rate))

    def _start_button_rects(self) -> dict[str, tuple[int, int, int, int]]:
        content_left, content_top, content_width, content_height = self.content_rect
        gap = 20
        button_width = min(300, max(145, (content_width - 60 - gap) // 2))
        button_height = min(132, max(88, round(content_height * 0.25)))
        total_width = 2 * button_width + gap
        left = content_left + (content_width - total_width) // 2
        top = content_top + round(content_height * 0.66) - button_height // 2
        return {
            MODE_PVP: (left, top, button_width, button_height),
            MODE_PVE: (left + button_width + gap, top, button_width, button_height),
        }

    @staticmethod
    def _point_in_rect(
        position: tuple[int, int],
        rect: tuple[int, int, int, int],
    ) -> bool:
        x_pos, y_pos = position
        left, top, width, height = rect
        return left <= x_pos < left + width and top <= y_pos < top + height

    def _player_label(self, player: int) -> str:
        return "PLAYER 1" if player == PLAYER_ONE else "PLAYER 2"

    def _player_color(self, player: int) -> tuple[int, int, int]:
        return self.COLOR_PLAYER1 if player == PLAYER_ONE else self.COLOR_PLAYER2

    def _player_dark_color(self, player: int) -> tuple[int, int, int]:
        return self.COLOR_PLAYER1_DARK if player == PLAYER_ONE else self.COLOR_PLAYER2_DARK

    def _player_light_color(self, player: int) -> tuple[int, int, int]:
        return self.COLOR_PLAYER1_LIGHT if player == PLAYER_ONE else self.COLOR_PLAYER2_LIGHT

    def _blit_text(
        self,
        text: str,
        font,
        color: tuple[int, int, int],
        *,
        center: tuple[int, int] | None = None,
        topleft: tuple[int, int] | None = None,
        midleft: tuple[int, int] | None = None,
        midright: tuple[int, int] | None = None,
        topright: tuple[int, int] | None = None,
    ) -> None:
        surface = font.render(text, True, color)
        rect = surface.get_rect()
        if center is not None:
            rect.center = center
        elif topleft is not None:
            rect.topleft = topleft
        elif midleft is not None:
            rect.midleft = midleft
        elif midright is not None:
            rect.midright = midright
        elif topright is not None:
            rect.topright = topright
        self.screen.blit(surface, rect)

    @staticmethod
    def _wrap_text(text: str, font, max_width: int) -> list[str]:
        words = text.split()
        if not words:
            return [""]
        lines: list[str] = []
        current_line = words[0]
        for word in words[1:]:
            candidate = f"{current_line} {word}"
            if font.size(candidate)[0] <= max_width:
                current_line = candidate
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        return lines
