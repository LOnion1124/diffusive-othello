import torch
from model import AlphaNet
from logic import GameLogic
from torch.utils.data import Dataset, DataLoader

class MoveData:
    def __init__(
            self,
            board: list[list[int]],
            pos: tuple[int, int],
            player: int
        ) -> None:
        self.player = player
        self.state = MoveData.generateState(board, player) # 3 * size * size, [empty; player1; player2]
        self.mask = MoveData.generateMask(board, player) # size * size, 1 if grid is possible legal move
        self.policy = MoveData.generatePolicy(board, pos) # size * size one-hot
    
    @classmethod
    def generateState(cls, board: list[list[int]], player: int) -> list:
        # translate board state
        # board: size * size 0/1/-1
        size = len(board)
        empty = [[0 for _ in range(size)] for _ in range(size)]
        this_side = [[0 for _ in range(size)] for _ in range(size)]
        that_side = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if board[i][j] == 0:
                    empty[i][j] = 1
                elif board[i][j] == player:
                    this_side[i][j] = 1
                else:
                    that_side[i][j] = 1
        
        return [empty, this_side, that_side]
    
    @classmethod
    def generateMask(cls, board: list[list[int]], player: int) -> list:
        # mask for legal moves
        size = len(board)
        mask = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                adjacents = []
                if i > 0:
                    adjacents.append(board[i - 1][j])
                if i < size - 1:
                    adjacents.append(board[i + 1][j])
                if j > 0:
                    adjacents.append(board[i][j - 1])
                if j < size - 1:
                    adjacents.append(board[i][j + 1])

                if board[i][j] == 0 and player in adjacents:
                    mask[i][j] = 1
        return mask

    @classmethod
    def generatePolicy(cls, board: list[list[int]], pos: tuple[int, int]):
        # get policy from position
        size = len(board)
        policy = [[0 for _ in range(size)] for _ in range(size)]
        policy[pos[0]][pos[1]] = 1
        return policy

class GameRecorder:
    def __init__(self) -> None:
        self.num_moves = 0 # n

        self.states = [] # state for n moves, n * 3 * size * size
        self.masks = [] # mask for n moves, n * size * size
        self.policies = [] # policy for n moves, n * size * size

        self.player_turns = [] # player for each move
        self.values = [] # value for n moves, n scalar
    
    def recordMove(self, move: MoveData) -> None:
        self.num_moves += 1

        self.states.append(move.state)
        self.masks.append(move.mask)
        self.policies.append(move.policy)

        self.player_turns.append(move.player)
    
    def recordEnd(self, winner: int) -> None:
        # uppack all moves at the end of the game
        # game ending (0: draw, 1: player1 win, -1: player2 win)
        if winner == 0: # draw
            self.values = [0 for _ in range(self.num_moves)]
        else:
            self.values = [1 if self.player_turns[i] == winner else -1 for i in range(self.num_moves)]

class OthelloDataset(Dataset):
    def __init__(self, recorders: list[GameRecorder]):
        self.states = []
        self.masks = []
        self.policies = []
        self.values = []
        for recorder in recorders:
            self.states += recorder.states
            self.masks += recorder.masks
            self.policies += recorder.policies
            self.values += recorder.values

        self.N = len(self.states)

        # switch to tensor
        self.states = torch.tensor(self.states, dtype=torch.get_default_dtype())
        self.masks = torch.tensor(self.masks, dtype=torch.bool).view(self.N, -1)
        self.policies = torch.tensor(self.policies, dtype=torch.get_default_dtype()).view(self.N, -1)
        self.values = torch.tensor(self.values, dtype=torch.get_default_dtype())

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'mask': self.masks[idx],
            'policy': self.policies[idx],
            'value': self.values[idx]
        }


def selfPlayOneGame(model: AlphaNet, device: str = "cuda", verbose: bool = False) -> GameRecorder:
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
        state = MoveData.generateState(board, player)
        state_tensor = torch.tensor(state, dtype=torch.get_default_dtype(), device=device).unsqueeze(0)
        mask = MoveData.generateMask(board, player)
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device).view(1, -1)
        with torch.no_grad():
            policy, value = model.forward(x=state_tensor, legal_mask=mask_tensor)
        max_index = torch.argmax(policy)
        x = max_index // logic.board_size
        y = max_index % logic.board_size
        pos = (int(x), int(y))
        logic.board.move(player, pos=pos)
        move = MoveData(board, pos, player)
        recorder.recordMove(move)
        logic.switchTurn()
        if verbose:
            print("game state: " + logic.game_state)
            print(f"move: ({pos[0]}, {pos[1]})")
            print("board:")
            print(str(logic.board))
            print(f"score: {logic.board.grid_count[1]} : {logic.board.grid_count[-1]}\n")
    winner = logic.winner
    recorder.recordEnd(winner)
    if verbose:
        print(f"winner: {logic.winner}\n\n")
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
        game = selfPlayOneGame(model, device, verbose=verbose)
        game_list.append(game)
        if ((i + 1) % (max(num_play // 5, 1)) == 0):
            print(f"playing... {i + 1}/{num_play}")
    print("done")
    dataset = OthelloDataset(game_list)
    if save_to_file:
        torch.save(dataset, file_name)
    return dataset

if __name__ == "__main__":
    model = AlphaNet().to("cuda")
    dataset = selfPlay(model, num_play=5, save_to_file=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in dataloader:
        print(batch['state'].shape, batch['mask'].shape, batch['policy'].shape, batch['value'].shape)