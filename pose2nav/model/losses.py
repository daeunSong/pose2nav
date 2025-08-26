import torch
import torch.nn.functional as F

# ---------- helpers ----------

def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# ---------- Barlow Twins ----------

def barlow_loss(z1: torch.Tensor, z2: torch.Tensor, lambd: float = 5e-3, eps: float = 1e-12) -> torch.Tensor:
    """
    z1, z2: [B, D] (any dtype)  -- internally cast to float32
    lambd: off-diagonal scaling factor
    """
    z1 = z1.float()
    z2 = z2.float()
    B, D = z1.shape

    # Normalize each feature dim across the batch
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + eps)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + eps)

    # Cross-correlation
    c = (z1.T @ z2) / B  # [D, D]

    on_diag = (torch.diagonal(c) - 1).pow(2).sum()
    off_diag = _off_diagonal(c).pow(2).sum()
    return on_diag + lambd * off_diag

# ---------- VICReg ----------

def vicreg_loss(z1: torch.Tensor, z2: torch.Tensor, sim_w: float = 25.0, var_w: float = 25.0, cov_w: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """
    z1, z2: [B, D] (any dtype) -- internally cast to float32
    """
    z1 = z1.float()
    z2 = z2.float()

    # Invariance term (MSE)
    inv = ((z1 - z2) ** 2).mean()

    # Variance term (promote per-dim std >= 1)
    def variance(z):
        z = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        return torch.relu(1.0 - std).mean()

    v = variance(z1) + variance(z2)

    # Covariance term (decorrelate features)
    def covariance(z):
        z = z - z.mean(dim=0, keepdim=True)
        n = z.size(0)
        if n <= 1:
            return z.new_tensor(0.0)
        c = (z.T @ z) / (n - 1)  # [D, D]
        off = c - torch.diag(torch.diag(c))
        return (off ** 2).sum() / z.size(1)

    c = covariance(z1) + covariance(z2)
    return sim_w * inv + var_w * v + cov_w * c

# ---------- selector ----------

def get_loss_fn(name: str):
    name = name.lower()
    if name == "barlow":
        return barlow_loss
    if name == "vicreg":
        return vicreg_loss
    raise ValueError(f"Unsupported loss: {name}")
