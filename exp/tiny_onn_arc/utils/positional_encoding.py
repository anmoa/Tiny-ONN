import math
import torch

def get_2d_sinusoidal_embedding(height: int, width: int, hidden_size: int) -> torch.Tensor:
    if hidden_size % 2 != 0:
        raise ValueError(f"Cannot create sinusoidal embedding for odd hidden size {hidden_size}")

    position_embedding = torch.zeros(height, width, hidden_size)
    half_dim = hidden_size // 2
    
    div_term = torch.exp(torch.arange(0, half_dim, 2).float() * -(math.log(10000.0) / half_dim))

    pos_w = torch.arange(0, width).unsqueeze(1)
    pos_h = torch.arange(0, height).unsqueeze(1)

    position_embedding[:, :, 0:half_dim:2] = torch.sin(pos_w * div_term).unsqueeze(0).repeat(height, 1, 1)
    position_embedding[:, :, 1:half_dim:2] = torch.cos(pos_w * div_term).unsqueeze(0).repeat(height, 1, 1)
    position_embedding[:, :, half_dim::2] = torch.sin(pos_h * div_term).unsqueeze(1).repeat(1, width, 1)
    position_embedding[:, :, half_dim + 1::2] = torch.cos(pos_h * div_term).unsqueeze(1).repeat(1, width, 1)
    
    return position_embedding.unsqueeze(0)