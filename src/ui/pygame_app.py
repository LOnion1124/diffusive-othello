"""Pygame desktop application entry point."""

from __future__ import annotations

import argparse

from src.ui.game_controller import GameController, MODE_PVE, MODE_PVP
from src.ui.input_controller import PygameInputController
from src.ui.pygame_renderer import PygameRenderer


FPS = 60
AI_COOLDOWN_MS = 300


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Diffusive Othello.")
    parser.add_argument("--mode", choices=(MODE_PVP, MODE_PVE), default=MODE_PVP)
    parser.add_argument("--board-size", type=int, default=9)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import pygame

    pygame.init()
    controller = GameController(mode=args.mode, board_size=args.board_size)
    screen = pygame.display.set_mode(PygameRenderer.screen_size(args.board_size))
    pygame.display.set_caption("Diffusive Othello")

    renderer = PygameRenderer(pygame, screen, board_size=args.board_size)
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

            previous_player = controller.current_player
            controller.handle_click(pointer.clicked, pointer.move)
            if controller.current_player != previous_player:
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
