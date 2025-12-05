import torch
from model.alphanet.network import AlphaNet
from src.game.logic import GameLogic
from torch.utils.data import Dataset, DataLoader

class MoveData:
    def __init__(
            self,
            player: int,
            board: list[list[int]],
            pos: tuple[int, int] = None,
            device = "cuda"
        ) -> None:
        self.player = player
        self.board = torch.tensor(board, dtype=torch.get_default_dtype(), device=device)

        # state
        # translate board state
        # board: size * size 0/1/-1
        empty = torch.zeros_like(self.board)
        empty[self.board == 0] = 1
        own_side = torch.zeros_like(self.board)
        own_side[self.board == player] = 1
        opp_side = torch.zeros_like(self.board)
        opp_side[self.board == -player] = 1
        self.state = torch.stack([empty, own_side, opp_side]) # (3, size, size), [empty; own; opp]

        # mask
        # Create shifted versions of the board (up, down, left, right)
        up = torch.zeros_like(self.board)
        down = torch.zeros_like(self.board)
        left = torch.zeros_like(self.board)
        right = torch.zeros_like(self.board)
        up[1:] = self.board[:-1]       # shift up
        down[:-1] = self.board[1:]     # shift down
        left[:, 1:] = self.board[:, :-1]   # shift left
        right[:, :-1] = self.board[:, 1:]  # shift right
        # any neighbor equals player
        adj = (up == player) | (down == player) | (left == player) | (right == player)
        # legal move = empty cell & adjacent to player
        self.mask = (self.board == 0) & adj # (size, size), 1 if grid is possible legal move
        self.mask = self.mask.view(-1) # (size*size, )

        self.policy = None
        if pos is not None:
            # get policy from position
            self.policy = torch.zeros_like(self.board)
            x, y = pos
            self.policy[x, y] = 1 # size * size one-hot
            self.policy = self.policy.view(-1) # (size*size, )
        
        # value
        # assess value of current state (unrelated to move pos)
        # not the final value of this move (will be updated based on game ending)
        size = self.board.shape[0]
        num_grids = size * size
        piece_cnt = self.state.view(3, -1).sum(dim=-1) # (3, )
        own_rate = piece_cnt[1].item() / num_grids
        opp_rate = piece_cnt[2].item() / num_grids
        self.value = own_rate * (1 - opp_rate) # scalar, always positive

class GameRecorder:
    def __init__(self) -> None:
        self.num_moves = 0 # n

        self.states = [] # state for n moves, n * (3, size, size)
        self.masks = [] # mask for n moves, n * (size, size)
        self.policies = [] # policy for n moves, n * (size, size)

        self.player_turns = [] # player for each move
        self.values = [] # value for n moves, n scalar
    
    def recordMove(self, move: MoveData) -> None:
        self.num_moves += 1

        self.states.append(move.state)
        self.masks.append(move.mask)
        self.policies.append(move.policy)
        self.values.append(move.value)

        self.player_turns.append(move.player)
    
    def recordEnding(self, winner: int) -> None:
        # game ending (0: draw, 1: player1 win, -1: player2 win)

        # first stack datas into one tensor
        self.states = torch.stack(self.states) # (n, 3, size, size)
        self.masks = torch.stack(self.masks) # (n, size, size)
        self.policies = torch.stack(self.policies) # (n, size, size)
        self.values = torch.tensor(self.values) # was list of float, now (n, )

        # adjust each move value based on game ending by multiplying an ending factor
        # draw: 1, win: 2, lose: 0.5
        player_turns = torch.tensor(self.player_turns)
        if winner != 0: # not draw
            winner_side = player_turns == winner
            loser_side = player_turns == -winner
            ending_factor = torch.ones(self.num_moves)
            ending_factor[winner_side] = 2.0
            ending_factor[loser_side] = 0.5
            self.values = self.values * ending_factor
        # normalize to [0, 1] using min-max
        min_val = self.values.min()
        max_val = self.values.max()
        if max_val > min_val:
            self.values = (self.values - min_val) / (max_val - min_val)
        else:
            self.values = torch.zeros_like(self.values)

class OthelloDataset(Dataset):
    def __init__(self, recorders: list[GameRecorder]):
        self.states = []
        self.masks = []
        self.policies = []
        self.values = []
        for recorder in recorders:
            self.states.append(recorder.states)
            self.masks.append(recorder.masks)
            self.policies.append(recorder.policies)
            self.values.append(recorder.values)

        # # switch to tensor
        self.states = torch.concat(self.states, dim=0) # (N, 3, size, size) do not flat for CNN architecture
        self.masks = torch.concat(self.masks, dim=0)
        self.policies = torch.concat(self.policies, dim=0)
        self.values = torch.concat(self.values, dim=0) # (N, )

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'mask': self.masks[idx],
            'policy': self.policies[idx],
            'value': self.values[idx]
        }

def selfPlayOneGame(model: AlphaNet, device: str = "cuda") -> GameRecorder:
    model.eval()
    logic = GameLogic()
    recorder = GameRecorder()
    logic.startGame()

    while logic.state == "game":
        if not logic.board.canMove(1) and not logic.board.canMove(-1):
            logic.endGame()
            break

        if logic.game_state == "player1" and not logic.board.canMove(1):
            logic.switchTurn()
            continue

        if logic.game_state == "player2" and not logic.board.canMove(-1):
            logic.switchTurn()
            continue

        player = 1 if logic.game_state == "player1" else -1
        board = logic.board.getGrids()
        dummy_move = MoveData(player, board, device=device) # to get current board mask
        mask = dummy_move.mask # (size * size, )

        # when producing train data, we choose the max value position for each move
        # this is for reinforcement-like training logic
        # so we'll try all possible moves here
        choices = {} # dict of pos: tuple(int, int) -> value (float)
        size = len(board)
        for i in range(size):
            for j in range(size):
                if mask[i * size + j] == False:
                    continue
                board_try = board.copy()
                board_try[i][j] = player  # play possible move
                move_try = MoveData(player, board_try)
                state_try, mask_try = move_try.state, move_try.mask
                with torch.no_grad():
                    _, value_try = model.forward(x=state_try.unsqueeze(0), legal_mask=mask_try.unsqueeze(0))
                choices[(i, j)] = float(value_try.squeeze().item()) if torch.is_tensor(value_try) else float(value_try)

        # pick the position with the highest value
        pos = max(choices.items(), key=lambda kv: kv[1])[0]

        logic.board.move(player, pos=pos)
        move = MoveData(player, board, pos, device=device)
        recorder.recordMove(move)
        logic.switchTurn()

    winner = logic.winner
    recorder.recordEnding(winner)

    return recorder

def selfPlay(
        model: AlphaNet,
        num_play: int = 1,
        save_to_file: bool = False,
        file_name: str = "dataset.pt",
        device: str = "cuda",
        verbose: bool = False
) -> OthelloDataset:
    game_list = []
    for i in range(num_play):
        game = selfPlayOneGame(model, device)
        game_list.append(game)
        if verbose and (i + 1) % (max(num_play // 5, 1)) == 0:
            print(f"playing... {i + 1}/{num_play}")
    if verbose:
        print("done")
    dataset = OthelloDataset(game_list)
    if save_to_file:
        torch.save(dataset, file_name)
    return dataset

if __name__ == "__main__":
    model = AlphaNet().to("cuda")
    dataset = selfPlay(model, num_play=5, save_to_file=True, verbose=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in dataloader:
        print(batch['state'].shape, batch['mask'].shape, batch['policy'].shape, batch['value'].shape)
        # print(batch['mask'][0:8])
        break