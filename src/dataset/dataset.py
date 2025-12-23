"""Make dataset for AlphaNet training from MCTS game result"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
from src.config import cfg, args

class DODataset(Dataset):
    def __init__(self, data_dict: dict[str]):
        self.states = data_dict['states']
        self.masks = data_dict['masks']
        self.policies = data_dict['policies']
        self.values = data_dict['values']
    
    def __len__(self):
        return self.states.shape[0]
    
    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'mask': self.masks[idx],
            'policy': self.policies[idx],
            'value': self.values[idx]
        }

class DatasetMaker:
    def __init__(self):
        mcts_data_path = cfg['mcts']['data_path']
        with open(mcts_data_path, 'r', encoding='utf-8') as f:
            self.mcts_data = json.load(f)
        self.data_path = cfg['train']['data_path']

        self.device = 'cuda' if cfg['use_cuda'] else 'cpu'
    
    def makeDataset(self, dtype = torch.float32):
        states = torch.tensor(self.mcts_data['state'], dtype=dtype, device=self.device)
        masks = torch.tensor(self.mcts_data['mask'], dtype=dtype, device=self.device)
        policies = torch.tensor(self.mcts_data['policy'], dtype=dtype, device=self.device)
        values = torch.tensor(self.mcts_data['value'], dtype=dtype, device=self.device)
        
        dataset = DODataset({
            "states": states,
            "masks": masks,
            "policies": policies,
            "values": values
        })

        torch.save(dataset, self.data_path)

        return dataset