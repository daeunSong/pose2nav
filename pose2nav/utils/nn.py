import torch
from pathlib import Path

def save_checkpoint(model, optimizer, epoch, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

# --- robust ckpt loader: handles various formats produced in your repo ---
def load_checkpoint(path, map_location=None):
    """
    Returns a plain model state_dict compatible with model.load_state_dict(...).
    Supports formats:
      - {"model_state_dict": ..., "optimizer_state_dict": ..., "epoch": ...}
      - {"state_dict": ...}
      - {"model": ...}
      - raw state_dict
    Also strips common prefixes like "module." or "model." if present.
    """
    ckpt = torch.load(path, map_location=map_location)

    # 1) Pick the model weights blob
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        else:
            # might already be a raw state_dict
            sd = ckpt
    else:
        sd = ckpt

    # 2) Strip common prefixes (DDP/DataParallel or nested wrappers)
    def strip_prefix(sd_in, prefix):
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd_in.items()}

    for pref in ("module.", "model."):
        if any(k.startswith(pref) for k in sd.keys()):
            sd = strip_prefix(sd, pref)

    return sd
