import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class HumanCrossAttPool(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.0, pad_tol=1e-8, use_img_query=True):
        super().__init__()
        self.d_model = d_model
        self.pad_tol = pad_tol
        self.dropout = nn.Dropout(dropout)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.use_img_query = use_img_query
        if use_img_query:
            self.q_from_img = nn.Linear(d_model, d_model, bias=False)
            self.q_token = None
        else:
            self.q_token = nn.Parameter(torch.randn(1, 1, d_model) / sqrt(d_model))
            self.q_from_img = None
        self.out_proj = nn.Identity()

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None = None, X_img: torch.Tensor | None = None):
        # X: [B,T,N,d], X_img: [B,T,d]
        B, T, N, d = X.shape
        if mask is None:
            mask = (X.abs().amax(dim=-1) > self.pad_tol)  # [B,T,N]
        kpm = (~mask).view(B*T, N)

        X_bt_n_d = X.view(B*T, N, d)
        if self.use_img_query:
            assert X_img is not None
            q_bt_1_d = self.q_from_img(X_img).reshape(B*T, 1, d)
        else:
            q_bt_1_d = self.q_token.expand(B*T, 1, d).to(X.device, X.dtype)

        all_masked = kpm.all(dim=1)
        if all_masked.any():
            kpm = kpm.clone(); kpm[all_masked, 0] = False

        pooled, _ = self.mha(q_bt_1_d, X_bt_n_d, X_bt_n_d, key_padding_mask=kpm, need_weights=False)
        pooled = self.dropout(pooled)

        if all_masked.any():
            pooled[all_masked] = q_bt_1_d[all_masked]

        return self.out_proj(pooled).squeeze(1).view(B, T, d)
