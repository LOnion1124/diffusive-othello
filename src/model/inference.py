import torch
from src.model.alphanet.network import AlphaNet
from src.game.state import encode_state, legal_mask, state_from_board
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
        game_state = state_from_board(board, current_player=player)
        state = torch.tensor(
            encode_state(game_state, player),
            dtype=torch.get_default_dtype(),
            device=self.device,
        )
        mask = torch.tensor(
            legal_mask(game_state, player),
            dtype=torch.bool,
            device=self.device,
        )
        
        self.model.eval()
        log_policy, value = self.model(state.unsqueeze(0), legal_mask=mask.unsqueeze(0))

        target_idx = log_policy.view(-1).argmax().item()
        x, y = target_idx // board_size, target_idx % board_size
        value = value.item()
        scores_list = log_policy.view(-1).exp().tolist()
        mask_list = mask.tolist()

        return {"pos": (x, y), "value": value, "scores": scores_list, "mask": mask_list}

