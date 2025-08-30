import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from model.encoders import (
    ImageEncoder,
    TrajectoryEncoder,
    KeypointEncoder2D,
    # RootPointEncoder2D,   # REMOVED
)

class PretextModel(nn.Module):
    """
    Observation tokens: per-frame IMAGE + per-frame pooled-HUMANS (from 2D keypoints only).
    Adds a learnable CLS token (VANP-style) and uses its output as the observation summary.

    Inputs (batch):
      past_frames : [B, T, 3, H, W] or list[T] of [B, 3, H, W]
      past_kp_2d  : [B, T, N, 17, 2]
      future_pos  : [B, T_pred, 2]   (optional; if provided and use_future=True)

    Outputs:
      z_obs   : [B, d]    (from CLS)
      z_traj  : [B, d]    (projected future trajectory embedding, if computed)
      + intermediates
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # --- core dims / inputs
        self.d        = int(cfg.model.d)
        self.T_obs    = int(cfg.model.input.T_obs)
        self.T_pred   = int(cfg.model.input.T_pred)
        self.N_human  = int(cfg.model.input.N_human)
        self.N_joints = int(cfg.model.input.N_joints)
        d = self.d

        # === Encoders ===
        self.image_encoder = ImageEncoder(
            backbone=cfg.model.image_encoder.name,
            pretrained=cfg.model.image_encoder.pretrained,
            output_dim=d,
        )
        # self.root2d_encoder = RootPointEncoder2D(
        #     seq_len=self.T_obs, num_humans=self.N_human, input_dim=2, output_dim=d
        # )
        self.kp_encoder = KeypointEncoder2D(
            seq_len=self.T_obs, num_humans=self.N_human, num_joints=self.N_joints,
            coord_dim=2, output_dim=d
        )
        self.future_traj_encoder = TrajectoryEncoder(
            output_dim=d,
            hidden=int(cfg.model.future_traj_encoder.hidden),
            num_layers=int(cfg.model.future_traj_encoder.num_layers),
        )

        # === Per-human pooling across humans per frame (linear-softmax on keypoint embeddings only) ===
        self.human_pool_w   = nn.Parameter(torch.randn(d))
        self.human_pool_temp = getattr(cfg.model, "human_pool_temp", 1.0)

        # --- CLS + learned positional table
        self.cls_token = nn.Parameter(torch.randn(1, 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.max_len   = 1 + 2 * self.T_obs
        self.pos_table = nn.Parameter(torch.randn(1, self.max_len, d))
        nn.init.trunc_normal_(self.pos_table, std=0.02)

        # === Observation Transformer (temporal) ===
        otc = cfg.model.obs_transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(otc.heads),
            dim_feedforward=int(otc.ff_mult) * d,
            dropout=float(otc.dropout),
            activation="relu",
            batch_first=True,
            norm_first=(str(otc.norm).lower() == "preln"),
        )
        self.observation_encoder = nn.TransformerEncoder(enc_layer, num_layers=int(otc.layers))
        self.final_ln = nn.LayerNorm(d)

        # === Projection heads ===
        pc = cfg.model.projector
        prj_d_out = int(pc.d_out)
        prj_hidden_w = int(pc.hidden)

        def projector(d_in: int, d_hidden=2048, d_out=128) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_in, d_hidden, bias=False),
                nn.BatchNorm1d(d_hidden, eps=1e-5, affine=True),
                nn.ReLU(inplace=True),
                nn.Linear(d_hidden, d_hidden, bias=False),
                nn.BatchNorm1d(d_hidden, eps=1e-5, affine=True),
                nn.ReLU(inplace=True),
                nn.Linear(d_hidden, d_out, bias=False),
            )
        self.proj_obs    = projector(d, prj_hidden_w, prj_d_out)
        self.proj_future = projector(d, prj_hidden_w, prj_d_out)

    def forward(self, batch: Dict[str, Any], use_future: bool = True) -> Dict[str, torch.Tensor]:
        # ---- Unpack
        past_frames = batch["past_frames"]
        kp_2d       = batch["past_kp_2d"].float()          # [B, T, N, 17, 2]

        future_positions = batch.get("future_positions", None)
        z_traj = None
        if use_future and (future_positions is not None):
            future_positions = future_positions.float()
            z_traj_raw = self.future_traj_encoder(future_positions)  # [B,d]
            z_traj = self.proj_future(z_traj_raw)                    # [B,d]

        # ---- Ensure frames tensor shape
        if isinstance(past_frames, list):
            past_frames = torch.stack(past_frames, dim=1)  # [B, T, 3, H, W]
        B, T, C, H, W = past_frames.shape
        assert T == self.T_obs, f"Expected T_obs={self.T_obs}, got {T}"
        d = self.d

        # =========================
        # 1) Per-modality encoders
        # =========================
        # Image → [B,T,d]
        frames_flat = past_frames.view(B * T, C, H, W)
        img_feats = self.image_encoder(frames_flat)      # [B*T, d]
        X_img = img_feats.view(B, T, d)                  # [B, T, d]

        # Keypoints → [B,T,N,d]
        X_pose_tn = self.kp_encoder(kp_2d)               # [B, T, N, d]

        # =========================
        # 2) Per-frame pooling across humans (linear-softmax) on keypoint embeddings
        # =========================
        # scores: [B,T,N]
        scores = (X_pose_tn * self.human_pool_w).sum(dim=-1) / float(self.human_pool_temp)
        alpha  = scores.softmax(dim=2).unsqueeze(-1)     # [B,T,N,1]
        X_hum  = (alpha * X_pose_tn).sum(dim=2)          # [B,T,d]
        X_hum  = F.layer_norm(X_hum, (d,))               # stabilize

        # =========================
        # 3) Tokens + learned positional embeddings
        # =========================
        body_tokens = torch.stack([X_img, X_hum], dim=2).reshape(B, 2 * T, d)         # [B,2T,d]
        cls_tok     = self.cls_token.expand(B, 1, d)                                   # [B,1,d]
        S = torch.cat([cls_tok, body_tokens], dim=1)                                   # [B,1+2T,d]
        S = S + self.pos_table[:, : S.size(1), :]                                      # add positions

        # =========================
        # 4) Observation transformer (temporal reasoning)
        # =========================
        U = self.observation_encoder(S)   # [B,1+2T,d]
        U = self.final_ln(U)
        h_cls = U[:, 0, :]                # CLS
        z_obs = self.proj_obs(h_cls)      # [B,d]

        return {
            "z_obs": z_obs,
            "z_traj": z_traj,
            # Intermediates
            "X_img": X_img,                 # [B,T,d]
            "X_pose_tn": X_pose_tn,         # [B,T,N,d]
            "hum_alpha": alpha.squeeze(-1), # [B,T,N]
            "X_hum": X_hum,                 # [B,T,d]
            "tokens": S,                     # [B,1+2T,d] (with pos added)
        }
