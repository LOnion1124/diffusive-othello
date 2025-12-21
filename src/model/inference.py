import torch
from src.model.alphanet.network import AlphaNet
from src.selfplay.selfplay import MoveData
from src.config import cfg

class GameAI:
    def __init__(self, device="cuda"):
        if not torch.cuda.is_available() or cfg["use_cuda"] == False:
            device = "cpu"
        self.device = device
        self.model = AlphaNet().to(device)
        self.model.load_state_dict(torch.load(cfg["model_path"]))
    
    def inference(self, board: list[list[int]], player: int):
        # board: provided by logic.board.getGrids()
        board_size = len(board)
        # translate board
        dummy_move = MoveData(player, board, device=self.device)
        state, mask = (
            dummy_move.state,
            dummy_move.mask
        )
        
        self.model.eval()
        log_policy, value = self.model(state.unsqueeze(0), legal_mask=mask.unsqueeze(0))

        target_idx = log_policy.view(-1).argmax().item()
        x, y = target_idx // board_size, target_idx % board_size
        value = value.item()
        scores_list = log_policy.view(-1).exp().tolist()
        mask_list = mask.tolist()

        return {"pos": (x, y), "value": value, "scores": scores_list, "mask": mask_list}

