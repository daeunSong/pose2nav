import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from model.encoders import (
    ImageEncoder,
    TrajectoryEncoder,
    KeypointEncoder2D,
    RootPointEncoder2D,
)


class PretextModel(nn.Module):
    """
    Two-stream observation with per-frame humans pooling (linear-softmax) → 2T tokens.
    - No CLS, learned time embeddings only.
    - No goal branch.
    - Projection heads for both obs and future.

    Inputs (batch):
      past_frames : [B, T, 3, H, W] or list[T] of [B, 3, H, W]
      past_kp_2d  : [B, T, N, 17, 2]
      past_root_2d: [B, T, N, 2]
      future_pos  : [B, T_pred, 2]

    Outputs:
      z_obs   : [B, d]
      z_traj  : [B, d]
      plus useful intermediates
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.T_obs = cfg.model.input.T_obs
        self.T_pred = cfg.model.input.T_pred
        self.N_human = cfg.model.input.N_human
        self.N_joints = 17

        # model dims
        self.d = getattr(cfg.model, "d", 256)
        d = self.d

        # === Encoders ===
        self.image_encoder = ImageEncoder(
            backbone=cfg.model.image_encoder.name,
            pretrained=cfg.model.image_encoder.pretrained,
            output_dim=d,
        )
        self.kp_encoder = KeypointEncoder2D(
            seq_len=self.T_obs, num_humans=self.N_human, num_joints=self.N_joints, coord_dim=2, output_dim=d
        )
        self.root2d_encoder = RootPointEncoder2D(
            seq_len=self.T_obs, num_humans=self.N_human, input_dim=2, output_dim=d
        )
        self.future_traj_encoder = TrajectoryEncoder(output_dim=d)

        # === Per-human fusion: (pose, root) -> human vector ===
        self.human_fuse = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d),
        )

        # === Linear-softmax pooling across humans per frame ===
        # score vector w ∈ R^d and temperature (optional)
        self.human_pool_w = nn.Parameter(torch.randn(d))
        self.human_pool_temp = getattr(cfg.model, "human_pool_temp", 1.0)

        # === Time positional embeddings (learned) ===
        # We reuse each time embedding twice per frame (img, humans)
        self.time_pos = nn.Parameter(torch.randn(1, self.T_obs, d))
        nn.init.trunc_normal_(self.time_pos, std=0.02)

        # === Observation Transformer (temporal) ===
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=8,
            dim_feedforward=4 * d,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.observation_encoder = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.final_ln = nn.LayerNorm(d)

        # === Projection heads ===
        def projector(d_in: int, d_out: Optional[int] = None) -> nn.Sequential:
            d_out = d_out or d_in
            return nn.Sequential(
                nn.LayerNorm(d_in),
                nn.Linear(d_in, 2 * d_in),
                nn.GELU(),
                nn.Linear(2 * d_in, d_out),
            )

        self.proj_obs = projector(d)
        self.proj_future = projector(d)

    def forward(self, batch: Dict[str, Any], use_future: bool=True) -> Dict[str, torch.Tensor]:
        # ---- Unpack
        past_frames = batch["past_frames"]
        kp_2d = batch["past_kp_2d"].float()                   # [B, T, N, 17, 2]
        root_2d = batch["past_root_3d"][..., :2].float()      # [B, T, N, 2]

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
        N = self.N_human

        # =========================
        # 1) Per-modality encoders
        # =========================
        # Image → [B,T,d]
        frames_flat = past_frames.view(B * T, C, H, W)
        img_feats = self.image_encoder(frames_flat)      # [B*T, d]
        X_img = img_feats.view(B, T, d)                  # [B, T, d]

        # Keypoints / Root → [B,T,N,d]
        X_pose_tn = self.kp_encoder(kp_2d)               # [B, T, N, d]
        X_root_tn = self.root2d_encoder(root_2d)         # [B, T, N, d]

        # =========================
        # 2) Per-human fusion: (pose, root) -> one human vector per (t,n)
        # =========================
        Z_h = torch.cat([X_pose_tn, X_root_tn], dim=-1)  # [B, T, N, 2d]
        H_tn = self.human_fuse(Z_h)                      # [B, T, N, d]
        # cheap residual to retain modality detail
        H_tn = H_tn + 0.5 * (X_pose_tn + X_root_tn)

        # =========================
        # 3) Per-frame pooling across humans (linear-softmax)
        # =========================
        # scores: [B,T,N]
        scores = (H_tn * self.human_pool_w).sum(dim=-1) / float(self.human_pool_temp)
        alpha = scores.softmax(dim=2).unsqueeze(-1)      # [B,T,N,1]
        X_hum = (alpha * H_tn).sum(dim=2)                # [B,T,d]
        X_hum = F.layer_norm(X_hum, (d,))                # stabilize

        # =========================
        # 4) Build 2T-token sequence with time embeddings (no type emb, no CLS)
        # =========================
        S = torch.stack([X_img, X_hum], dim=2).reshape(B, 2 * T, d)  # [B,2T,d]
        time_pos = self.time_pos[:, :T, :].repeat_interleave(2, dim=1)  # [1,2T,d]
        S = S + time_pos

        # =========================
        # 5) Observation transformer (temporal reasoning)
        # =========================
        U = self.observation_encoder(S)   # [B,2T,d]
        U = self.final_ln(U)              # final LN (Pre-LN style)
        h_obs = U.mean(dim=1)             # or learned pooling
        z_obs = self.proj_obs(h_obs)      # [B,d]

        # =========================
        # Trajectory branch
        # =========================
        # z_traj_raw = self.future_traj_encoder(future_positions)  # [B,d]
        # z_traj = self.proj_future(z_traj_raw)                    # [B,d]

        return {
            "z_obs": z_obs,
            "z_traj": z_traj,
            # Intermediates
            "X_img": X_img,                 # [B,T,d]
            "X_pose_tn": X_pose_tn,         # [B,T,N,d]
            "X_root_tn": X_root_tn,         # [B,T,N,d]
            "H_tn": H_tn,                   # [B,T,N,d]
            "hum_alpha": alpha.squeeze(-1), # [B,T,N]
            "X_hum": X_hum,                 # [B,T,d]
            "S": S,                         # [B,2T,d]
            "U": U,                         # [B,2T,d]
        }
