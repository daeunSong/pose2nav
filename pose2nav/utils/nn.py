import torch
from pathlib import Path

import shutil
import os
import datetime

def save_checkpoint(state: dict, is_best: bool, save_dir: str, name: str):
    """
    Save {save_dir}/{name}.pth always.
    If is_best=True, also save {save_dir}/{name}_best.pth.
    Returns the path to {save_dir}/{name}.pth (the non-best path).
    """
    os.makedirs(save_dir, exist_ok=True)

    base_path = os.path.join(save_dir, f"{name}.pth")
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
    return torch.load(save, map_location=device)
