from mcts_simple import *
from src.config import cfg, args

class DO(Game):
    def __init__(self):
        # load config
        mcts_cfg = cfg['mcts']
        self.save_data = mcts_cfg['save_data']
        self.verbose = mcts_cfg['verbose']

        # create board
        S = cfg['board_size']
        self.board_size = S
        self.board = [[0 for _ in range(S)] for _ in range(S)]
        # initial pieces
        self.board[0][0] = 1
        self.board[0][S - 1] = 1
        self.board[S - 1][0] = -1
        self.board[S - 1][S - 1] = -1

        # player symbol and ID
        self.players = [1, -1]
        self.cur_player_id = 0

        # selfplay data
        self.gamedata = {'state': [], 'policy': [], 'value': []}

    def board2str(self):
        S = self.board_size
        first_line = '|'.join(f"{num:3d}" for num in self.board[0])
        sep = '-' * len(first_line)

        lines = []
        for i, row in enumerate(self.board):
            line = '|'.join(f"{num:3d}" for num in row)
            lines.append(line)
            if i != S - 1:
                lines.append(sep)

        return "\n".join(lines) + "\n"
    
    def render(self):
        if self.verbose:
            board_str = self.board2str()
            print(board_str)

    def get_state(self):
        pass

    def number_of_players(self):
        return len(self.players)
    
    def current_player(self):
        return self.cur_player_id
    
    def possible_actions(self):
        S = self.board_size
        player = self.players[self.cur_player_id]

        # get valid cells for current move
        valid_cells = [[0 for _ in range(S)] for _ in range(S)]
        for x in range(S):
            for y in range(S):
                if valid_cells[x][y] == 1:
                    continue
                if self.board[x][y] == player:
                    dxs = []
                    dys = []
                    if x - 1 >= 0:
                        dxs.append(-1)
                    if x + 1 < S:
                        dxs.append(1)
                    if y - 1 >= 0:
                        dys.append(-1)
                    if y + 1 < S:
                        dys.append(1)

                    for dx in dxs:
                        if self.board[x + dx][y] == 0:
                            valid_cells[x + dx][y] = 1
                    for dy in dys:
                        if self.board[x][y + dy] == 0:
                            valid_cells[x][y + dy] = 1

        # encode valid cells
        res = [x * S + y for x in range(S) for y in range(S) if valid_cells[x][y] == 1]
        if len(res) == 0:
            res = [-1] # no valid move
        return res
    
    def take_action(self, action):
        if action == -1:
            # directly switch to opponent's turn if no valid move
            self.cur_player_id = 1 - self.cur_player_id
            return

        player = self.players[self.cur_player_id]
        opponent = self.players[1 - self.cur_player_id]
        S = self.board_size
        x, y = action // S, action % S

        # save game data
        if self.save_data:
            # make 3-layer state
            state_empty = [[1 if status == 0 else 0 for status in row] for row in self.board]
            state_player = [[1 if status == player else 0 for status in row] for row in self.board]
            state_opponent = [[1 if status == opponent else 0 for status in row] for row in self.board]
            self.gamedata['state'].append([state_empty, state_player, state_opponent])

            # make one-hot policy
            policy = [[0 for _ in range(S)] for _ in range(S)]
            policy[x][y] = 1
            self.gamedata['policy'].append(policy)
        
        # play move and update
        self.board[x][y] = player
        dxs = [0]
        dys = [0]
        if x - 1 >= 0:
            dxs.append(-1)
        if x + 1 < S:
            dxs.append(1)
        if y - 1 >= 0:
            dys.append(-1)
        if y + 1 < S:
            dys.append(1)
        for dx in dxs:
            for dy in dys:
                if self.board[x + dx][y + dy] == opponent:
                    self.board[x + dx][y + dy] = player
        # update current player
        self.cur_player_id = 1 - self.cur_player_id
    
    def has_outcome(self):
        # end game when board is full
        return not any(0 in row for row in self.board)

    def winner(self):
        cnt_p0 = sum(1 if status == self.players[0] else 0 for row in self.board for status in row)
        cnt_p1 = sum(1 if status == self.players[1] else 0 for row in self.board for status in row)
        winners = []
        if cnt_p0 > cnt_p1:
            winners.append(0)
        elif cnt_p0 < cnt_p1:
            winners.append(1)
        else:
            # draw, return all players
            winners.append(0)
            winners.append(1)
        return winners