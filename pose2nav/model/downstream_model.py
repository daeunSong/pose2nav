import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from model.pretext_model import PretextModel


class MLPHead(nn.Module):
    """
    Goal-conditioned late-fusion head that predicts ABSOLUTE future positions.
    Pipeline: goal [B,2] --MLP--> g_enc [B,d_goal]; fused=[z_obs; g_enc] --> MLP --> [B,T_pred,2]
    """
    def __init__(self, cfg):
        super().__init__()
        self.T_pred = cfg.model.input.T_pred
        gc = cfg.model.downstream.goal_encoder
        goal_dims = gc.dims
        d = cfg.model.d

        # self.goal_enc = nn.Sequential(
        #     nn.Linear(2, goal_dims[0], bias=gc.bias),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.LayerNorm(goal_dims[0]),
        #     nn.Dropout(gc.dropout),

        #     nn.Linear(goal_dims[0], goal_dims[1], bias=gc.bias),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.LayerNorm(goal_dims[1]),
        #     nn.Dropout(gc.dropout),

        #     nn.Linear(goal_dims[1], d, bias=gc.bias),   # d = obs_context_size (e.g., 512)
        # )

        self.goal_enc = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, 64), 
            nn.LeakyReLU(negative_slope=0.2),
            nn.LayerNorm(64),
        )

        # fuse_in = d * 2
        fuse_in = d + 64

        hc = cfg.model.downstream.head
        head_dims = hc.dims

        # build MLP head
        self.head = nn.Sequential(
            # nn.Linear(fuse_in, head_dims[0], bias=hc.bias),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.LayerNorm(head_dims[0]),
            # nn.Dropout(hc.dropout),

            # nn.Linear(head_dims[0], head_dims[1], bias=hc.bias),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.LayerNorm(head_dims[1]),
            # nn.Dropout(hc.dropout),

            # nn.Linear(head_dims[1], 2 * self.T_pred, bias=hc.bias),  # final layer, no norm/act

            nn.LayerNorm(fuse_in),

            nn.Linear(fuse_in, head_dims[0]),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(head_dims[0]),
            
            nn.Linear(head_dims[0], head_dims[1]),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(head_dims[1]),

            nn.Linear(head_dims[1], 2 * self.T_pred),  # final layer, no norm/act
        )

    def forward(self, z_obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        g = self.goal_enc(goal)                       # [B, d_goal]
        x = torch.cat([z_obs, g], dim=-1)             # [B, 2d]
        y = self.head(x)                              # [B, 2*T_pred]
        return y.view(-1, self.T_pred, 2)             # [B, T_pred, 2]


class DownstreamTrajPredictor(nn.Module):
    """
    Wraps the pretrained observation backbone (PretextModel) and a goal-conditioned MLP head.
    Forward returns absolute future positions: [B, T_pred, 2].
    """
    def __init__(self, cfg):
        super().__init__()
        self.model = PretextModel(cfg)
        self.head = MLPHead(cfg)
        self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def unfreeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train()

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        obs_batch = {
            "past_frames":  batch["past_frames"],
            "past_kp_2d":   batch["past_kp_2d"],
            # "past_root_3d": batch["past_root_3d"],
        }
        bb_out = self.model(obs_batch, use_future=False)  

        z_obs = bb_out["h_cls"]   # [B, d]
        goal = batch["goal"].float()

        return self.head(z_obs, goal)                    # [B, T_pred, 2]
