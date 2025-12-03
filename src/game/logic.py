class Grid:
    def __init__(self, pos: tuple[int, int], board_size: int) -> None:
        self.status = 0 # 0: empty, 1 / -1: occupied
        self.pos = pos # 2 dimension tuple
        self.board_size = board_size
    
    def getAdjacent(self) -> list[tuple[int, int]]:
        # Return a list of positions of adjacent grids (4 directions)
        result = []
        pos = self.pos
        size = self.board_size
        if pos[0] > 0:
            result.append((pos[0] - 1, pos[1]))
        if pos[0] < size - 1:
            result.append((pos[0] + 1, pos[1]))
        if pos[1] > 0:
            result.append((pos[0], pos[1] - 1))
        if pos[1] < size - 1:
            result.append((pos[0], pos[1] + 1))
        
        return result
    
    def getAround(self) -> list[tuple[int, int]]:
        # Return a list of positions of around grids (8 directions)
        result = []
        pos = self.pos
        size = self.board_size
        dxs = [-1, 0, 1]
        dys = [-1, 0, 1]
        for dx in dxs:
            for dy in dys:
                if dx == 0 and dy == 0:
                    continue

                x = pos[0] + dx
                y = pos[1] + dy
                if x < 0 or x >= size:
                    continue
                if y < 0 or y >= size:
                    continue
                
                result.append((x, y))
        
        return result

class Board:
    def __init__(self, size: int) -> None:
        self.size = size
        self.num_grids = size * size
        self.grids = [[Grid(pos=(i, j), board_size=size)
                       for j in range(size)] for i in range(size)]
    
    def initialize(self) -> None:
        for i in range(self.size):
            for j in range(self.size):
                self.grids[i][j].status = 0

        self.grids[0][0].status = 1
        self.grids[0][self.size - 1].status = 1
        self.grids[self.size - 1][0].status = -1
        self.grids[self.size - 1][self.size - 1].status = -1

        self.grid_count = [self.num_grids - 4, 2, 2]
    
    def checkValidMove(self, player: int, pos: tuple[int, int]) -> bool:
        x, y = pos
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        
        grid = self.grids[x][y]
        if grid.status != 0:
            return False
        
        adjacents = grid.getAdjacent()
        flag = False
        for adj in adjacents:
            stat = self.grids[adj[0]][adj[1]].status
            if stat == player:
                flag = True
                break
        
        return flag
    
    def updateCount(self) -> None:
        self.grid_count = [0, 0, 0]
        for i in range(self.size):
            for j in range(self.size):
                status = self.grids[i][j].status
                self.grid_count[status] += 1
    
    def move(self, player: int, pos: tuple[int, int]) -> None:
        # Assume move is valid
        grid = self.grids[pos[0]][pos[1]]
        grid.status = player

        arounds = grid.getAround()
        for around in arounds:
            cur = self.grids[around[0]][around[1]]
            if cur.status == -player:
                cur.status = player
        # Update scores
        self.updateCount()
    
    def canMove(self, player: int) -> bool:
        for i in range(self.size):
            for j in range(self.size):
                if self.checkValidMove(player=player, pos=(i, j)):
                    return True
        return False
    
    def __str__(
            self,
            symbol: tuple[str, str, str] = ('.', 'o', 'x'),
            sep: str = ' '
            ) -> str:
        result = ""
        for j in range(self.size):
            for i in range(self.size):
                status = self.grids[i][j].status
                result += symbol[status]
                result += sep

            if j < self.size - 1:
                result += "\n"

        return result
    
    def getGrids(self) -> list[list[int]]:
        # return 2d list representing current grids
        result = [[0 for _ in range(self.size)] for _ in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                result[i][j] = self.grids[i][j].status
        return result


class GameLogic:
    def __init__(
            self,
            player_names : tuple[str, str] = ("Player1", "Player2"),
            board_size: int = 9
        ) -> None:
        self.state = "start"
        self.game_state = ""
        self.board_size = board_size
        self.board = Board(self.board_size)
        self.winner = 0 # 0: draw, 1: player1, -1: player2
        self.winner_name = ""
        self.input_buffer = None # Input API
        self.player_names = player_names

    
    def startGame(self) -> None:
        self.board.initialize()
        self.state = "game"
        self.game_state = "player1"

    def endGame(self) -> None:
        cnt1 = self.board.grid_count[1]
        cnt2 = self.board.grid_count[2]
        if cnt1 == cnt2:
            self.winner = 0
            self.winner_name = "Draw"
        elif cnt1 > cnt2:
            self.winner = 1
            self.winner_name = self.player_names[0]
        else:
            self.winner = -1
            self.winner_name = self.player_names[1]

        self.state = "end"
        self.game_state = ""
    
    def switchTurn(self) -> None:
        match self.game_state:
            case "player1":
                self.game_state = "player2"
            case "player2":
                self.game_state = "player1"