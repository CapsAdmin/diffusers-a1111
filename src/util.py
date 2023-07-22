import shared
from safetensors.torch import load_file
import torch
    
def load_state_dict(path):
    if path.endswith(".safetensors"):
       return load_file(path, device=shared.device)
    else:
        return torch.load(path, map_location=shared.device)
