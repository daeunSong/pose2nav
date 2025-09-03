import torch
import torch.nn.functional as F

# ---------- helpers ----------
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# ---------- VICReg ----------

# def vicreg_loss(z1: torch.Tensor, z2: torch.Tensor, sim_w: float = 25.0, var_w: float = 25.0, cov_w: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
#     """
#     z1, z2: [B, D] (any dtype) -- internally cast to float32
#     """
#     z1 = z1.float()
#     z2 = z2.float()

#     # Invariance term (MSE)
#     inv = ((z1 - z2) ** 2).mean()

#     # Variance term (promote per-dim std >= 1)
#     def variance(z):
#         z = z - z.mean(dim=0, keepdim=True)
#         std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
#         return torch.relu(1.0 - std).mean()

#     v = variance(z1) + variance(z2)

#     # Covariance term (decorrelate features)
#     def covariance(z):
#         z = z - z.mean(dim=0, keepdim=True)
#         n = z.size(0)
#         if n <= 1:
#             return z.new_tensor(0.0)
#         c = (z.T @ z) / (n - 1)  # [D, D]
#         off = c - torch.diag(torch.diag(c))
#         return (off ** 2).sum() / z.size(1)

#     c = covariance(z1) + covariance(z2)
#     return sim_w * inv + var_w * v + cov_w * c

def vicreg_loss(
    z1, z2, sim_coeff: float = 25.0, std_coeff: float = 25.0, cov_coeff: float = 1.0
):
    repr_loss = F.mse_loss(z1, z2)

    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)

    std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
    std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

    cov_x = (z1.T @ z1) / (z1.shape[0] - 1)
    cov_y = (z2.T @ z2) / (z2.shape[0] - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div_(z1.shape[1]) + off_diagonal(
        cov_y
    ).pow_(2).sum().div_(z2.shape[1])

    loss = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss
    return loss, (repr_loss, std_loss, cov_loss)

# ---------- selector ----------

def get_loss_fn(name: str):
    name = name.lower()
    # if name == "barlow":
    #     return barlow_loss
    if name == "vicreg":
        return vicreg_loss
    raise ValueError(f"Unsupported loss: {name}")
