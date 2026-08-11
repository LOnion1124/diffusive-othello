"""Pygame desktop application entry point."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from src.config import get_game_config
from src.ui.game_controller import (
    GameController,
    MODE_PVE,
    MODE_PVP,
    PHASE_LOADING,
    PHASE_START,
)
from src.ui.input_controller import PygameInputController
from src.ui.pygame_renderer import PygameRenderer


FPS = 60
AI_COOLDOWN_MS = 300
GUI_ICON_PATH = Path(__file__).resolve().parents[2] / "assets" / "gui" / "favicon-32x32.png"


def _preload_ai_policy(policy) -> None:
    preload = getattr(policy, "preload", None)
    if callable(preload):
        preload()


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
        controller.begin_loading(args.mode)
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
    loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-loader")
    loading_future: Future[None] | None = None

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
            sidebar_action = renderer.sidebar_action_at(
                pointer.position,
                controller.snapshot(),
            )
            if not controller.handle_sidebar_action(sidebar_action):
                controller.handle_click(pointer.clicked, pointer.move, start_mode=start_mode)

            if controller.phase == PHASE_LOADING:
                if loading_future is None:
                    loading_future = loader.submit(_preload_ai_policy, controller.ai_policy)
                elif loading_future.done():
                    try:
                        loading_future.result()
                    except Exception:
                        # The controller renders the existing Invalid 50:50 state
                        # after the load failure, so the game remains playable.
                        pass
                    controller.finish_loading()
                    loading_future = None
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

            renderer.draw(controller.snapshot(), now_ms=pygame.time.get_ticks())
            pygame.display.update()
            clock.tick(FPS)
    finally:
        loader.shutdown(wait=False, cancel_futures=True)
        pygame.quit()
