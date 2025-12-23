from mcts_simple import *
from src.config import cfg, args
from src.model.inference import GameAI

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
        if self.save_data:
            self.game_data = {'state': [], 'mask': [], 'policy': [], 'player': []}
            self.move_cnt = 0

    def board2str(self):
        S = self.board_size
        first_line = '|'.join(" O " if num == 1 else (" X " if num == -1 else "   ") for num in self.board[0])
        sep = '-' * len(first_line)

        lines = []
        for i, row in enumerate(self.board):
            line = '|'.join(" O " if num == 1 else (" X " if num == -1 else "   ") for num in row)
            lines.append(line)
            if i != S - 1:
                lines.append(sep)

        return "\n".join(lines) + "\n"
    
    def render(self):
        if self.verbose:
            board_str = self.board2str()
            print(board_str)
            input()

    def get_state(self):
        return self.board

    def number_of_players(self):
        return len(self.players)
    
    def current_player(self):
        return self.cur_player_id
    
    def generate_mask(self):
        S = self.board_size
        player = self.players[self.cur_player_id]

        # get valid cells for current move
        mask = [[0 for _ in range(S)] for _ in range(S)]
        for x in range(S):
            for y in range(S):
                if mask[x][y] == 1:
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
                            mask[x + dx][y] = 1
                    for dy in dys:
                        if self.board[x][y + dy] == 0:
                            mask[x][y + dy] = 1
        return mask
    
    def possible_actions(self):
        res = []
        S = self.board_size
        mask = self.generate_mask()
        # encode valid cells
        res = [x * S + y for x in range(S) for y in range(S) if mask[x][y] == 1]
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
            self.move_cnt += 1

            # make 3-layer state
            state_empty = [[1 if status == 0 else 0 for status in row] for row in self.board]
            state_player = [[1 if status == player else 0 for status in row] for row in self.board]
            state_opponent = [[1 if status == opponent else 0 for status in row] for row in self.board]
            self.game_data['state'].append([state_empty, state_player, state_opponent])

            # get mask
            mask = self.generate_mask()
            self.game_data['mask'].append(mask)

            # make one-hot policy
            policy = [[0 for _ in range(S)] for _ in range(S)]
            policy[x][y] = 1
            self.game_data['policy'].append(policy)

            # record player
            self.game_data['player'].append(self.cur_player_id)
        
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

def choose_distributed_action(node: Node, temperature: float = 1.0) -> int:
    """
    Choose next action base on distribution with temperature for exploration
    """
    
    if not node.children:
        return node.rng.choice(list(node.children.keys()))
    
    n_total = sum(child.n for child in node.children.values())
    
    if n_total == 0:
        return node.rng.choice(list(node.children.keys()))
    
    actions = []
    probs = []
    for action, child in node.children.items():
        actions.append(action)
        # Apply temperature to make distribution softer
        probs.append((child.n / n_total) ** (1.0 / temperature) if temperature > 0 else child.n / n_total)
    
    # Normalize probabilities after temperature scaling
    prob_sum = sum(probs)
    probs = [p / prob_sum for p in probs]

    return node.rng.choices(actions, weights=probs, k=1)[0]

class MyMCTS(MCTS):
    """
    A special version of MCTS enabling saving selfplay data to file
    """
    def __init__(self, game, training = True, seed = None):
        super().__init__(game, allow_transpositions=False, training=training, seed=seed)

        mcts_cfg = cfg['mcts']
        self.save_data = mcts_cfg['save_data']
        self.data_path = mcts_cfg['data_path']
        if self.save_data:
            self.game_data = {'state': [], 'mask': [], 'policy': [], 'value': []}
            self.total_move_cnt = 0
        
        self.use_model = mcts_cfg['use_model']
        if self.use_model:
            device = "cuda" if cfg['use_cuda'] else "cpu"
            self.model = GameAI(device)

    # override self_play to save selfplay data
    def step(self) -> None:
        if self.training is True:
            self.backpropagation(self.simulation(self.expansion(self.selection(self.root))))
        else:
            node = self.root
            while not self.copied_game.has_outcome():
                self.copied_game.render()
                if self.use_model:
                    # make choice based on inference
                    board = self.copied_game.get_state()
                    player = self.copied_game.players[self.copied_game.current_player()]
                    pred_dict = self.model.inference(board=board, player=player)
                    scores, mask = pred_dict['scores'], pred_dict['mask']
                    actions = [i for i, m in enumerate(mask) if m]
                    probs = [scores[i] for i in actions]
                    if len(actions) == 0:
                        action = -1
                    else:
                        action = self.rng.choices(actions, weights=probs, k=1)[0]
                elif len(node.children) > 0:
                    # Use temperature > 1.0 to add exploration randomness
                    action = choose_distributed_action(node, temperature=1.5)
                    node = node.children[action]
                else:
                    action = self.rng.choice(self.copied_game.possible_actions())

                self.copied_game.take_action(action)
            self.copied_game.render()

            # record game data
            if self.save_data:
                self.game_data['state'] += self.copied_game.game_data['state']
                self.game_data['mask'] += self.copied_game.game_data['mask']
                self.game_data['policy'] += self.copied_game.game_data['policy']
                winner = self.copied_game.winner()[0]
                players = self.copied_game.game_data['player']
                values = [1 if player == winner else 0 for player in players]
                self.game_data['value'] += values
                self.total_move_cnt += self.copied_game.move_cnt
            
        self.copied_game = deepcopy(self.game)

    def self_play(self, iterations: int = 1) -> None:
        if self.save_data:
            for key in self.game_data:
                self.game_data[key] = []
            self.total_move_cnt = 0
        
        desc = "Training" if self.training is True else "Evaluating"
        for _ in tqdm(range(iterations), desc = desc):
            self.step()
        
        # save game data to file
        if not self.training and self.save_data:
            import json
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(self.game_data, f)