from mcts_simple import *
from tqdm import tqdm
from src.config import cfg, args
from src.model.inference import GameAI
from src.game.state import (
    apply_move,
    encode_state,
    is_terminal,
    legal_mask,
    new_game,
    pass_turn,
    score,
)

class DO(Game):
    def __init__(self):
        # load config
        mcts_cfg = cfg['mcts']
        self.record_data = mcts_cfg['record_data']
        self.verbose = mcts_cfg['verbose']

        S = cfg['board_size']
        self.board_size = S
        self.state = new_game(S)

        # player symbol and ID
        self.players = [1, -1]
        self.cur_player_id = 0

        # selfplay data
        if self.record_data:
            self.game_data = {'state': [], 'mask': [], 'policy': [], 'player': []}
            self.move_cnt = 0

    def board2str(self):
        S = self.board_size
        board = self.get_state()
        first_line = '|'.join(" O " if num == 1 else (" X " if num == -1 else "   ") for num in board[0])
        sep = '-' * len(first_line)

        lines = []
        for i, row in enumerate(board):
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
        return [list(row) for row in self.state.board]

    def number_of_players(self):
        return len(self.players)
    
    def current_player(self):
        return self.cur_player_id
    
    def generate_mask(self):
        player = self.players[self.cur_player_id]
        return [
            [1 if cell else 0 for cell in row]
            for row in legal_mask(self.state, player, flatten=False)
        ]
    
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
            self.state = pass_turn(self.state).state
            self.cur_player_id = 1 - self.cur_player_id
            self.record_data = False # stop recording move data afterward
            return

        player = self.players[self.cur_player_id]
        opponent = self.players[1 - self.cur_player_id]
        S = self.board_size
        x, y = action // S, action % S

        # record game data
        if self.record_data:
            self.move_cnt += 1

            self.game_data['state'].append(encode_state(self.state, player))

            # get mask
            mask = self.generate_mask()
            self.game_data['mask'].append(mask)

            # make one-hot policy
            policy = [[0 for _ in range(S)] for _ in range(S)]
            policy[x][y] = 1
            self.game_data['policy'].append(policy)

            # record player
            self.game_data['player'].append(self.cur_player_id)
        
        self.state = apply_move(self.state, player, (x, y)).state
        # update current player
        self.cur_player_id = 1 - self.cur_player_id
    
    def has_outcome(self):
        return is_terminal(self.state)

    def winner(self):
        counts = score(self.state)
        cnt_p0 = counts[self.players[0]]
        cnt_p1 = counts[self.players[1]]
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
        raise ValueError("Node has no children to choose from")
    
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
    def __init__(self, game, training = True, seed = None, use_model = False):
        super().__init__(game, allow_transpositions=False, training=training, seed=seed)

        mcts_cfg = cfg['mcts']
        self.record_data = mcts_cfg['record_data']
        if self.record_data:
            self.game_data = {'state': [], 'mask': [], 'policy': [], 'value': []}
            self.total_move_cnt = 0
        self.save_data = mcts_cfg['save_data']
        if self.save_data:
            self.data_path = mcts_cfg['data_path']
        
        self.use_model = use_model
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
                    # Note: when using model, we skip MCTS tree traversal
                elif len(node.children) > 0:
                    # Use temperature > 1.0 to add exploration randomness
                    action = choose_distributed_action(node, temperature=1.5)
                    node = node.children[action]
                else:
                    action = self.rng.choice(self.copied_game.possible_actions())

                self.copied_game.take_action(action)
            self.copied_game.render()

            # record game data
            if self.record_data:
                self.game_data['state'] += self.copied_game.game_data['state']
                self.game_data['mask'] += self.copied_game.game_data['mask']
                self.game_data['policy'] += self.copied_game.game_data['policy']
                winners = self.copied_game.winner()
                players = self.copied_game.game_data['player']
                # Handle win/loss/draw: winner gets 1, loser gets -1, draw gets 0
                if len(winners) == 1:
                    # One winner, assign 1 to winner and -1 to loser
                    values = [1 if player == winners[0] else -1 for player in players]
                else:
                    # Draw: all players get 0
                    values = [0 for _ in players]
                self.game_data['value'] += values
                self.total_move_cnt += self.copied_game.move_cnt
            
        self.copied_game = deepcopy(self.game)

    def self_play(self, iterations: int = 1) -> None:
        if not self.training and self.record_data:
            for key in self.game_data:
                self.game_data[key] = []
            self.total_move_cnt = 0
        
        desc = "Training" if self.training is True else "Evaluating"
        for _ in tqdm(range(iterations), desc = desc):
            self.step()
        
        # save game data to file
        if not self.training and self.record_data and self.save_data:
            import json
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(self.game_data, f)
