"""Pygame desktop application entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import get_game_config
from src.ui.game_controller import GameController, MODE_PVE, MODE_PVP, PHASE_START
from src.ui.input_controller import PygameInputController
from src.ui.pygame_renderer import PygameRenderer


FPS = 60
AI_COOLDOWN_MS = 300
GUI_ICON_PATH = Path(__file__).resolve().parents[2] / "assets" / "gui" / "favicon-32x32.png"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    game_config = get_game_config()
    parser = argparse.ArgumentParser(description="Play Diffusive Othello.")
    parser.add_argument(
        "--mode",
        choices=(MODE_PVP, MODE_PVE),
        default=None,
        help="Optional quick start mode. Omit to choose from the start screen.",
    )
    parser.add_argument("--board-size", type=int, default=game_config["board_size"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import pygame

    pygame.init()
    icon = pygame.image.load(GUI_ICON_PATH)
    controller = GameController(mode=args.mode or MODE_PVP, board_size=args.board_size)
    if args.mode is not None:
        controller.start_game(args.mode)
    show_winrate_bar = True
    pygame.display.set_icon(icon)
    screen = pygame.display.set_mode(
        PygameRenderer.screen_size(
            args.board_size,
            show_winrate_bar=show_winrate_bar,
        )
    )
    pygame.display.set_caption("Diffusive Othello")

    renderer = PygameRenderer(
        pygame,
        screen,
        board_size=args.board_size,
        show_winrate_bar=show_winrate_bar,
    )
    input_controller = PygameInputController(
        pygame,
        board_left_top=renderer.board_left_top,
        grid_size=renderer.GRID_SIZE,
        board_size=args.board_size,
    )
    clock = pygame.time.Clock()
    ai_ready_at = pygame.time.get_ticks()

    try:
        while True:
            pointer = input_controller.poll()
            if pointer.quit_requested:
                return 0

            previous_phase = controller.phase
            previous_player = controller.current_player
            start_mode = None
            if controller.phase == PHASE_START:
                start_mode = renderer.start_mode_at(pointer.position)
            controller.handle_click(pointer.clicked, pointer.move, start_mode=start_mode)
            if (
                controller.phase != previous_phase
                or controller.current_player != previous_player
            ):
                ai_ready_at = pygame.time.get_ticks()

            if (
                controller.is_ai_turn
                and pygame.time.get_ticks() - ai_ready_at >= AI_COOLDOWN_MS
            ):
                controller.play_ai_turn()
                ai_ready_at = pygame.time.get_ticks()

            renderer.draw(controller.snapshot())
            pygame.display.update()
            clock.tick(FPS)
    finally:
        pygame.quit()
