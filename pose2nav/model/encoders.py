import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from typing import Optional

# ---------- Helpers ----------

def mlp_block(input_dim: int, hidden_dim: int, output_dim: int, use_ln_in: bool = True, use_ln_out: bool = True) -> nn.Sequential:
    layers = []
    if use_ln_in:
        layers.append(nn.LayerNorm(input_dim))
    layers += [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, output_dim)]
    if use_ln_out:
        layers.append(nn.LayerNorm(output_dim))
    return nn.Sequential(*layers)

# ---------- Image Encoder ----------
class ImageEncoder(nn.Module):
    """
    Returns a single vector per image: input [B, 3, H, W] -> output [B, output_dim]
    """
    def __init__(self, backbone: str = "custom", output_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.output_dim = output_dim

        if self.backbone_name == "custom":
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            cnn_out_dim = 256
            self.fc = nn.Sequential(nn.Flatten(), nn.Linear(cnn_out_dim, output_dim), nn.LayerNorm(output_dim))

        elif self.backbone_name in ["resnet18", "resnet34", "resnet50"]:
            self.resnet = self._load_resnet(self.backbone_name, pretrained)
            self.encoder = nn.Sequential(*list(self.resnet.children())[:-1])  # remove FC
            resnet_out_dim = self._get_resnet_output_dim(self.backbone_name)
            self.fc = nn.Sequential(nn.Flatten(), nn.Linear(resnet_out_dim, output_dim), nn.LayerNorm(output_dim))
        
        elif self.backbone_name == "dino":
            self.encoder = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14_reg"
            )
            dino_out_dim = 384
            self.fc = nn.Sequential(
                nn.Linear(dino_out_dim, output_dim), nn.LeakyReLU()
            )
        
        else:
            raise ValueError(f"Unsupported image backbone: {self.backbone_name}")

    def _load_resnet(self, name: str, pretrained: bool):
        # Support both old (pretrained=True) and new (weights=...) torchvision APIs
        if name == "resnet18":
            from torchvision.models import ResNet18_Weights
            return resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        if name == "resnet34":
            from torchvision.models import ResNet34_Weights
            return resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        if name == "resnet50":
            from torchvision.models import ResNet50_Weights
            return resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)


    def _get_resnet_output_dim(self, name: str) -> int:
        return {"resnet18": 512, "resnet34": 512, "resnet50": 2048}[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, 3, H, W]
        feats = self.encoder(x)            # [B, C, 1, 1]
        return self.fc(feats)              # [B, output_dim]

# ---------- Trajectory Encoder ----------
class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, T_pred = 12, output_dim: int = 256):
        super().__init__()
        action_encoder_output_size = input_dim * T_pred
        feature_size = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(action_encoder_output_size, feature_size),
            nn.LeakyReLU()
        )

    def forward(self, traj: torch.Tensor) -> torch.Tensor:  # traj: [B, T, 2]
        B, T, D = traj.shape
        flat = traj.reshape(B, T * D)        # [B, T*2]
        return self.encoder(flat)            # [B, output_dim]

# ---------- Trajectory Encoder ----------
# class TrajectoryEncoder(nn.Module):
#     """
#     Encodes a 2D sequence [B, T, 2] into a vector [B, output_dim].
#     Uses a GRU for better temporal summarization.
#     """
#     def __init__(self, input_dim: int = 2, output_dim: int = 256, hidden: int = 256, num_layers: int = 1):
#         super().__init__()
#         self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden, num_layers=num_layers,
#                           batch_first=True, bidirectional=False)
#         self.proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, output_dim), nn.LayerNorm(output_dim))

#     def forward(self, traj: torch.Tensor) -> torch.Tensor:  # traj: [B, T, 2]
#         _, h = self.gru(traj)            # h: [num_layers, B, hidden]
#         h = h[-1]                        # [B, hidden]
#         return self.proj(h)              # [B, output_dim]

# ---------- Keypoint Encoder (2D) ----------
class KeypointEncoder2D(nn.Module):
    """
    Time-distributed, per-human keypoint encoder.
    Accepts either:
      - kp: [B, T, N, J, 2]
      - kp: [B, T, N, F]  where F = J*2
    Returns per (t, n) embeddings: [B, T, N, output_dim]
    Lazily builds the MLP on first forward to match the flattened joint dim.
    """
    def __init__(self, seq_len: Optional[int] = None, num_humans: Optional[int] = None,
                 num_joints: int = 17, coord_dim: int = 2, output_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.output_dim = output_dim
        self.hidden = hidden
        self.mlp: Optional[nn.Sequential] = None  # built on first forward

    def forward(self, kp: torch.Tensor) -> torch.Tensor:
        # kp: [B, T, N, J, 2] or [B, T, N, F]
        if kp.dim() == 5:
            B, T, N, J, C = kp.shape
            assert C == 2, "Expected last coord dim = 2 for (B,T,N,J,2)"
            F_in = J * C
            x = kp.view(B * T * N, F_in)
        elif kp.dim() == 4:
            B, T, N, F_in = kp.shape
            x = kp.view(B * T * N, F_in)
        else:
            raise ValueError(f"Unexpected kp shape: {kp.shape}")

        if self.mlp is None:
            self.mlp = nn.Sequential(
                nn.LayerNorm(F_in),
                nn.Linear(F_in, self.hidden), 
                nn.LeakyReLU(0.2),
                nn.Linear(self.hidden, self.output_dim),
                nn.LayerNorm(self.output_dim),
            ).to(x.device)

        out = self.mlp(x)                                # [B*T*N, d]
        return out.view(B, T, N, self.output_dim)        # [B, T, N, d]

# ---------- Root Point Encoder (2D) ----------
class RootPointEncoder2D(nn.Module):
    """
    Time-distributed, per-human root encoder.
    Accepts [B, T, N, 2] (or [B, T, N, F] with F=2).
    Returns per (t, n) embeddings: [B, T, N, output_dim]
    Lazily builds the MLP on first forward.
    """
    def __init__(self, seq_len: Optional[int] = None, num_humans: Optional[int] = None,
                 input_dim: int = 2, output_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.output_dim = output_dim
        self.hidden = hidden
        self.mlp: Optional[nn.Sequential] = None

        
    def forward(self, root_2d: torch.Tensor) -> torch.Tensor:
        if root_2d.dim() != 4:
            raise ValueError(f"Expected root_2d shape [B,T,N,F], got {root_2d.shape}")
        B, T, N, F_in = root_2d.shape
        x = root_2d.view(B * T * N, F_in)

        if self.mlp is None:
            self.mlp = nn.Sequential(
                nn.LayerNorm(F_in),
                nn.Linear(F_in, self.hidden), 
                nn.LeakyReLU(0.2),
                nn.Linear(self.hidden, self.output_dim),
                nn.LayerNorm(self.output_dim),
            ).to(x.device)

        out = self.mlp(x)                                # [B*T*N, d]
        return out.view(B, T, N, self.output_dim)        # [B, T, N, d]

# ---------- 3D variants kept for completeness ----------
class KeypointEncoder3D(nn.Module):
    def __init__(self, seq_len: int, num_humans: int = 6, num_joints: int = 17, coord_dim: int = 3, output_dim: int = 256):
        super().__init__()
        input_dim = seq_len * num_humans * num_joints * coord_dim
        self.mlp = mlp_block(input_dim, 512, output_dim)

    def forward(self, kp: torch.Tensor) -> torch.Tensor:  # kp: [B, T, N, J, 3]
        B = kp.shape[0]
        return self.mlp(kp.reshape(B, -1))  # [B, output_dim]

class RootPointEncoder3D(nn.Module):
    def __init__(self, seq_len: int, num_humans: int = 1, input_dim: int = 3, output_dim: int = 256):
        super().__init__()
        total_input_dim = seq_len * num_humans * input_dim
        self.mlp = mlp_block(total_input_dim, 256, output_dim)

    def forward(self, root_3d: torch.Tensor) -> torch.Tensor:  # [B, T, N, 3]
        B = root_3d.shape[0]
        return self.mlp(root_3d.reshape(B, -1))  # [B, output_dim]

# ---------- Goal Encoder (kept for compatibility) ----------
class GoalEncoder(nn.Module):
    """Encodes goal position [B, 2] into [B, output_dim]."""
    def __init__(self, input_dim: int = 2, output_dim: int = 256):
        super().__init__()
        self.mlp = mlp_block(input_dim, 128, output_dim)

    def forward(self, goal: torch.Tensor) -> torch.Tensor:  # goal: [B, 2]
        return self.mlp(goal)
