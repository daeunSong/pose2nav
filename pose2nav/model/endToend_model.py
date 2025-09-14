import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoders import (
    ImageEncoder,
    KeypointEncoder2D
)
from model.attention import HumanCrossAttPool

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


class EndToEndModel(nn.Module):
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
            frozen=getattr(cfg.model.image_encoder, "frozen", False),
            output_dim=d,
        )
        # self.root2d_encoder = RootPointEncoder2D(
        #     seq_len=self.T_obs, num_humans=self.N_human, input_dim=2, output_dim=d
        # )
        self.kp_encoder = KeypointEncoder2D(
            seq_len=self.T_obs, num_humans=self.N_human, num_joints=self.N_joints,
            coord_dim=2, output_dim=d
        )

        # === Per-human pooling across humans per frame (linear-softmax on keypoint embeddings only) ===
        # self.human_pool_w   = nn.Parameter(torch.randn(d))
        # self.human_pool_temp = getattr(cfg.model, "human_pool_temp", 1.0)
        self.hum_pool = HumanCrossAttPool(d_model=self.d, n_heads=4, dropout=0.0)

        # --- CLS + learned positional table
        self.cls_token = nn.Parameter(torch.randn(1, 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.max_len   = 1 + self.T_obs
        self.pos_table = nn.Parameter(torch.randn(1, self.max_len, d))
        nn.init.trunc_normal_(self.pos_table, std=0.02)

        # === Observation Transformer (temporal) ===
        otc = cfg.model.obs_transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(otc.heads),
            dim_feedforward= 2 * d,
            dropout=float(otc.dropout),
            activation="relu",
            batch_first=True,
            norm_first=True, #preln
        )
        self.observation_encoder = nn.TransformerEncoder(enc_layer, num_layers=int(otc.layers))
        self.final_ln = nn.LayerNorm(d)

        self.head = MLPHead(cfg)


    def forward(self, batch):
        # ---- Unpack
        past_frames = batch["past_frames"]
        kp_2d       = batch["past_kp_2d"].float()          # [B, T, N, 17, 2]

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
        pose_mask = None # masking method..

        # =========================
        # 2) Cross-attention Pooling (이미지 Q ↔ 사람들 K/V)
        # =========================
        X_hum = self.hum_pool(X_pose_tn, mask=pose_mask, X_img=X_img)   # [B, T, d]
        ## early fusion
        body_tokens = X_img + X_hum     # [B, T, d]

        cls_tok     = self.cls_token.expand(B, 1, d)                                   # [B,1,d]
        S = torch.cat([cls_tok, body_tokens], dim=1)                                   # [B,1+T,d]
        S = S + self.pos_table[:, : S.size(1), :]                                      # add positions

        # =========================
        # 4) Observation transformer (temporal reasoning)
        # =========================
        U = self.observation_encoder(S)   # [B,1+T,d]
        U = self.final_ln(U)
        z_obs = U[:, 0, :]                # CLS

        goal = batch["goal"].float()

        return self.head(z_obs, goal)                    # [B, T_pred, 2]
