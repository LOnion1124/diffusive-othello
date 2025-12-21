import pygame
import sys
from src.game.logic import GameLogic
from src.model.inference import GameAI
from src.config import cfg, args

# Predefined colors
BLACK   = (0, 0, 0)
WHITE   = (255, 255, 255)
RED     = (255, 0, 0)
GREEN   = (0, 255, 0)
BLUE    = (0, 0, 255)
YELLOW  = (255, 255, 0)
CYAN    = (0, 255, 255)
MAGENTA = (255, 0, 255)
GRAY    = (128, 128, 128)
ORANGE  = (255, 165, 0)
PURPLE  = (128, 0, 128)
PINK    = (255, 192, 203)
BROWN   = (165, 42, 42)

pygame.init()

SCREEN_WIDTH = 540
SCREEN_HEIGHT = 620
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
screen.fill(GRAY)
pygame.display.set_caption("Diffusive Othello")

FPS = 60
FramePerSec = pygame.time.Clock()

# game assets
logic = GameLogic()
playerAI = GameAI()
game_mode = args[0].mode # "PVP" or "PVE"
last_time = pygame.time.get_ticks() # for AI cool down

grid_size = 60
grid_num = logic.board_size

peice_radius = round(grid_size * 0.4)

color_background = GRAY
color_border = WHITE
color_player1 = ORANGE
color_player2 = CYAN
color_text = WHITE

color_start = BROWN
color_end = BROWN
color_scoreboard = BROWN
color_infobox = BROWN

font_large = pygame.font.SysFont("microsoftyahei", 48)
font_middle = pygame.font.SysFont("microsoftyahei", 36)
font_small = pygame.font.SysFont("microsoftyahei", 24)

board_left_top = (0, 40)
board_lenth = grid_num * grid_size
board_center = (board_left_top[0] + (board_lenth // 2),
                board_left_top[1] + (board_lenth // 2))

scoreboard_left_top = (0, 0)
scoreboard_width = board_lenth
scoreboard_height = board_left_top[1]
scoreboard_center1 = (scoreboard_left_top[0] + (scoreboard_width // 4),
                      scoreboard_left_top[1] + (scoreboard_height // 2))
scoreboard_center2 = (scoreboard_left_top[0] + (3 * scoreboard_width // 4),
                      scoreboard_left_top[1] + (scoreboard_height // 2))

infobox_left_top = (board_left_top[0], board_left_top[1] + board_lenth)
infobox_width = board_lenth
infobox_height = SCREEN_HEIGHT - infobox_left_top[1]
infobox_center = (infobox_left_top[0] + (infobox_width // 2),
                  infobox_left_top[1] + (infobox_height // 2))


def drawBackground(color: tuple[int, int, int] = color_background) -> None:
    left, top = board_left_top
    pygame.draw.rect(screen, color, (left, top, board_lenth, board_lenth))

def drawGrid() -> None:
    left, top = board_left_top
    xs = [left + i * grid_size for i in range(grid_num + 1)]
    ys = [top + i * grid_size for i in range(grid_num + 1)]
    for x in xs:
        pygame.draw.line(screen, color_border, (x, top), (x, top + board_lenth))
    for y in ys:
        pygame.draw.line(screen, color_border, (left, y), (left + board_lenth, y))

def drawPiece() -> None:
    left, top = board_left_top
    grids = logic.board.grids
    xs = [left + (grid_size // 2) + i * grid_size for i in range(grid_num)]
    ys = [top + (grid_size // 2) + i * grid_size for i in range(grid_num)]
    for i in range(grid_num):
        for j in range(grid_num):
            x = xs[i]
            y = ys[j]
            status = grids[i][j].status
            if status == 1:
                pygame.draw.circle(screen, color_player1, (x, y), peice_radius)
            if status == -1:
                pygame.draw.circle(screen, color_player2, (x, y), peice_radius)

def drawGame() -> None:
    drawBackground()
    drawGrid()
    drawPiece()

def drawTitle(text: str) -> None:
    title_text = font_large.render(text, True, color_text)
    title_rect = title_text.get_rect(center=board_center)
    screen.blit(title_text, title_rect)

def drawStart() -> None:
    drawBackground(color=color_start)
    drawTitle(text="CLICK TO START")

def drawEnd() -> None:
    drawBackground(color=color_end)
    winner_name = logic.winner_name
    if winner_name == "Draw":
        title = "Draw"
    else:
        title = "Winner: " + winner_name
    drawTitle(text=title)

def drawInfoBox(info: str) -> None:
    left, top = infobox_left_top
    pygame.draw.rect(screen, color_infobox, (left, top, infobox_width, infobox_height))
    info_text = font_small.render(info, True, color_text)
    info_rect = info_text.get_rect(center=infobox_center)
    screen.blit(info_text, info_rect)

def pos2grid(x_pos: int, y_pos: int) -> tuple[int, int]:
    left, top = board_left_top
    x = (x_pos - left) // grid_size
    y = (y_pos - top) // grid_size
    if x >= 0 and x < grid_num and y >= 0 and y < grid_num:
        return (x, y)
    return (-1, -1)

def drawScoreboard() -> None:
    left, top = scoreboard_left_top
    pygame.draw.rect(screen, color_scoreboard, (left, top, scoreboard_width, scoreboard_height))
    score1 = str(logic.board.grid_count[1]) if logic.state == "game" else ""
    score2 = str(logic.board.grid_count[-1]) if logic.state == "game" else ""
    score1_text = font_middle.render(score1, True, color_text)
    score2_text = font_middle.render(score2, True, color_text)
    score1_rect = score1_text.get_rect(center=scoreboard_center1)
    score2_rect = score2_text.get_rect(center=scoreboard_center2)
    screen.blit(score1_text, score1_rect)
    screen.blit(score2_text, score2_rect)


def handle_events():
    mouse_click = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_click = pygame.mouse.get_pos()
    return mouse_click

def update_game_state(mouse_click):
    global last_time
    match logic.state:
        case "start":
            if mouse_click:
                logic.startGame()
                drawInfoBox("Game started.")

        case "game":
            if not logic.board.canMove(1) and not logic.board.canMove(-1):
                logic.endGame()
            else:
                if logic.game_state == "player1":
                    if not logic.board.canMove(1):
                        drawInfoBox("No place left for Player1.")
                        logic.switchTurn()
                        last_time = pygame.time.get_ticks()
                    else:
                        drawInfoBox("Player1's turn.")
                        if mouse_click:
                            x, y = pos2grid(*mouse_click)
                            if (x, y) != (-1, -1) and logic.board.checkValidMove(player=1, pos=(x, y)):
                                logic.board.move(player=1, pos=(x, y))
                                logic.switchTurn()
                                last_time = pygame.time.get_ticks()
                            else:
                                drawInfoBox("Try another position.")
                elif logic.game_state == "player2":
                    if not logic.board.canMove(-1):
                        drawInfoBox("No place left for Player2.")
                        logic.switchTurn()
                    else:
                        drawInfoBox("Player2's turn.")
                        if game_mode == "PVP":
                            if mouse_click:
                                x, y = pos2grid(*mouse_click)
                                if (x, y) != (-1, -1) and logic.board.checkValidMove(player=-1, pos=(x, y)):
                                    logic.board.move(player=-1, pos=(x, y))
                                    logic.switchTurn()
                                else:
                                    drawInfoBox("Try another position.")
                        else:
                            # player2 is AI
                            if pygame.time.get_ticks() - last_time >= 300:
                                grids = logic.board.getGrids()
                                pred = playerAI.inference(board=grids, player=-1)
                                logic.board.move(player=-1, pos=pred["pos"])
                                logic.switchTurn()
        case "end":
            drawEnd()
            if mouse_click:
                logic.startGame()
                drawInfoBox("Game restarted.")

def render():
    match logic.state:
        case "start":
            drawStart()
            drawInfoBox("Diffusive Othello")
            drawScoreboard()
        case "game":
            drawGame()
            drawScoreboard()
        case "end":
            drawEnd()
            drawScoreboard()

drawStart()
drawInfoBox("Diffusive Othello")
drawScoreboard()

while True:
    mouse_click = handle_events()
    update_game_state(mouse_click)
    render()
    pygame.display.update()
    FramePerSec.tick(FPS)