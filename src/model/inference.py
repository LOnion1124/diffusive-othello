import torch
from src.model.model import AlphaNet
from src.selfplay.selfplay import MoveData
from src.config import config

class GameAI:
    def __init__(self, device="cuda"):
        self.model = AlphaNet().to(device)
        self.model.load_state_dict(torch.load(config["model_path"]))
    
    def inference(self, board: list[list[int]], player: int, device="cuda"):
        # board: provided by logic.board.getGrids()
        board_size = len(board)
        # translate board
        dummy_move = MoveData(player, board)
        state, mask = (
            dummy_move.state,
            dummy_move.mask
        )
        # state = MoveData.generateState(board, player)
        # state = torch.tensor(state, dtype=torch.get_default_dtype(), device=device).unsqueeze(0) # (1, 3, board_size, board_size)
        # mask = MoveData.generateMask(board, player)
        # mask = torch.tensor(mask, dtype=torch.bool, device=device).view(1, -1) # (1, board_size * board_size)

        self.model.eval()
        log_policy, value = self.model(state.unsqueeze(0), legal_mask=mask.unsqueeze(0))
        target_idx = torch.argmax(log_policy.view(-1))
        value = value.item
        x, y = target_idx // board_size, target_idx % board_size

        return {"pos": (x, y), "value": value}

