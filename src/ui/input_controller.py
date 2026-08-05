"""Pygame input helpers for the desktop client."""

from __future__ import annotations

from dataclasses import dataclass

from src.game.state import Move


@dataclass(frozen=True)
class PointerInput:
    quit_requested: bool = False
    clicked: bool = False
    move: Move | None = None


class PygameInputController:
    def __init__(
        self,
        pygame_module,
        *,
        board_left_top: tuple[int, int],
        grid_size: int,
        board_size: int,
    ) -> None:
        self.pygame = pygame_module
        self.board_left_top = board_left_top
        self.grid_size = grid_size
        self.board_size = board_size

    def poll(self) -> PointerInput:
        clicked = False
        move = None
        quit_requested = False

        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                quit_requested = True
            elif event.type == self.pygame.MOUSEBUTTONDOWN:
                clicked = True
                move = self.pos_to_grid(*self.pygame.mouse.get_pos())

        return PointerInput(
            quit_requested=quit_requested,
            clicked=clicked,
            move=move,
        )

    def pos_to_grid(self, x_pos: int, y_pos: int) -> Move | None:
        left, top = self.board_left_top
        x = (x_pos - left) // self.grid_size
        y = (y_pos - top) // self.grid_size
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            return (x, y)
        return None
