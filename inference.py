import torch
from model import AlphaNet
from selfplay import MoveData

class GameAI:
    def __init__(self, device="cuda"):
        self.model = AlphaNet().to(device)
        self.model.load_state_dict(torch.load("model.pth"))
    
    def inference(self, board: list[list[int]], player: int, device="cuda"):
        # board: provided by logic.board.getGrids()
        board_size = len(board)
        # translate board
        state = MoveData.generateState(board, player)
        state = torch.tensor(state, dtype=torch.get_default_dtype(), device=device).unsqueeze(0) # (1, 3, board_size, board_size)
        mask = MoveData.generateMask(board, player)
        mask = torch.tensor(mask, dtype=torch.bool, device=device).view(1, -1) # (1, board_size * board_size)

        self.model.eval()
        log_policy, value = self.model(state, legal_mask=mask)
        target_idx = torch.argmax(log_policy.view(-1))
        value = value.item
        x, y = target_idx // board_size, target_idx % board_size

        return {"pos": (x, y), "value": value}

