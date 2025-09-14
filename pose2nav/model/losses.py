import torch
import torch.nn.functional as F

# ---------- helpers ----------
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# ---------- VICReg ----------

def vicreg_loss(
    z1, z2, sim_coef: float = 25.0, std_coef: float = 25.0, cov_coef: float = 5.0
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

    loss = sim_coef * repr_loss + std_coef * std_loss + cov_coef * cov_loss
    return loss, (repr_loss, std_loss, cov_loss)

# ---------- selector ----------

def get_loss_fn(name: str):
    name = name.lower()
    # if name == "barlow":
    #     return barlow_loss
    if name == "vicreg":
        return vicreg_loss
    raise ValueError(f"Unsupported loss: {name}")

