import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from model.pretext_model import PretextModel


class TrajHeadWithGoalAbs(nn.Module):
    """
    Goal-conditioned late-fusion head that predicts ABSOLUTE future positions.
    Pipeline: goal [B,2] --MLP--> g_enc [B,d_goal]; fused=[z_obs; g_enc] --> MLP --> [B,T_pred,2]
    """
    def __init__(self, d: int, T_pred: int, d_goal: Optional[int] = None, hidden_mult: int = 2, num_layers: int = 3):
        super().__init__()
        self.T_pred = T_pred
        d_goal = d if d_goal is None else d_goal

        self.goal_enc = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, d_goal), nn.LeakyReLU(negative_slope=0.2),
            nn.LayerNorm(d_goal),
        )

        fuse_in = d + d_goal
        hidden_dim = hidden_mult * d

        # build deeper MLP head
        layers = [nn.LayerNorm(fuse_in)]
        in_dim = fuse_in
        for i in range(num_layers - 1):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.LayerNorm(hidden_dim),
            ]
            in_dim = hidden_dim
        # final output layer
        layers.append(nn.Linear(in_dim, 2 * T_pred))
        self.head = nn.Sequential(*layers)

    def forward(self, z_obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        g = self.goal_enc(goal)                       # [B, d_goal]
        fused = torch.cat([z_obs, g], dim=-1)         # [B, d + d_goal]
        y = self.head(fused)                          # [B, 2*T_pred]
        return y.view(-1, self.T_pred, 2)             # [B, T_pred, 2]


class DownstreamTrajPredictor(nn.Module):
    """
    Wraps the pretrained observation backbone (PretextModel) and a goal-conditioned MLP head.
    Forward returns absolute future positions: [B, T_pred, 2].
    """
    def __init__(self, cfg):
        super().__init__()
        self.backbone = PretextModel(cfg)
        d = getattr(cfg.model, "d", 256)
        T_pred = cfg.model.input.T_pred
        hidden_mult = int(getattr(cfg.model, "head_hidden_mult", 2))
        d_goal = getattr(cfg.model, "d_goal", None)
        num_layers = getattr(cfg.model, "num_layers", 3)

        self.head = TrajHeadWithGoalAbs(d=d, T_pred=T_pred, d_goal=d_goal, hidden_mult=hidden_mult, num_layers=num_layers)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.backbone.train()

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        obs_batch = {
            "past_frames":  batch["past_frames"],
            "past_kp_2d":   batch["past_kp_2d"],
            # "past_root_3d": batch["past_root_3d"],
        }
        bb_out = self.backbone(obs_batch, use_future=False)  # <-- no future at inference
        z_obs = bb_out["z_obs"]

        if "goal" not in batch:
            raise KeyError("DownstreamTrajPredictor: 'goal' is required (no fallback to GT).")
        goal = batch["goal"].float()

        return self.head(z_obs, goal)                    # [B, T_pred, 2]
