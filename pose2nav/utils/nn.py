import torch
from torch import nn
from pathlib import Path

import shutil
import os

def save_checkpoint(state: dict, is_best: bool, save_dir: str, name: str, epoch: int):
    """
    Save {save_dir}/{name}.pth always.
    If is_best=True, also save {save_dir}/{name}_best.pth.
    Returns the path to {save_dir}/{name}.pth (the non-best path).
    """
    os.makedirs(save_dir, exist_ok=True)

    base_path = os.path.join(save_dir, f"{name}_e{epoch}.pth")
    torch.save(state, base_path)
    path = base_path

    if is_best:
        best_path = os.path.join(save_dir, f"{name}_best.pth")
        # avoid copying file onto itself
        if os.path.abspath(best_path) != os.path.abspath(base_path):
            shutil.copyfile(base_path, best_path)
        path = best_path

    return path


def load_checkpoint(save: str, device: str):
    """Loads model parameters (state_dict) from file_path.

    Args:
        save: (str) directory of the saved checkpoint
        device: (str) map location
    """
    if not os.path.exists(save):
        raise f"File doesn't exist {save}"
    return torch.load(save, map_location=device, weights_only=False)

@torch.no_grad()
def check_grad_norm(net: nn.Module):
    """Compute and return the grad norm of all parameters of the network.
    To check whether gradients flowing in the network or not
    """
    total_norm = 0
    for p in list(filter(lambda p: p.grad is not None, net.parameters())):
        param_norm = p.grad.detach().norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm
