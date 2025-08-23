import argparse
import os
import math
import random
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb

from model.data_loader import SocialNavDataset
from model.downstream_model import DownstreamTrajPredictor
from utils.helpers import get_conf, tensor_stats, NaNGuard
from utils.nn import save_checkpoint


class LearnerDownstream:
    def __init__(self, cfg_path: str, use_wandb: bool = True, pretext_ckpt: Optional[str] = None):
        self.cfg = get_conf(cfg_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb

        # prefer CLI arg; otherwise read from YAML: directory.pretext_ckpt
        yaml_ckpt = getattr(getattr(self.cfg, "directory", {}), "pretext_ckpt", None)
        self.pretext_ckpt = pretext_ckpt or yaml_ckpt

        self.set_seed(self.cfg.train_params.seed)
        self.init_data()
        self.init_model()
        self.init_optimizer(stage=1)
        self.init_logger()

        amp_enabled = bool(self.cfg.train_params.get("amp", True))
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # -------------------- setup --------------------
    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def init_data(self):
        dataset = SocialNavDataset(
            index_file=getattr(self.cfg.dataset, "index_file", self.cfg.dataset.get("root", "")),
            train=True,
            only_human_visable=getattr(self.cfg.dataset, "only_human_visable", False),
            resize=tuple(self.cfg.dataset.get("resize", [224, 224])),
            metric_waypoint_spacing=self.cfg.dataset.get("metric_waypoint_spacing", 1.0),
        )

        ratio = float(self.cfg.train_params.get("sample_ratio", 1.0))
        if ratio < 1.0:
            total = len(dataset)
            k = max(1, math.ceil(total * ratio))
            sub_seed = self.cfg.train_params.get("sample_seed", None)
            rng = np.random.RandomState(sub_seed) if sub_seed is not None else np.random
            indices = rng.choice(total, size=k, replace=False)
            dataset = Subset(dataset, indices)

        self.train_loader = DataLoader(
            dataset,
            batch_size=self.cfg.train_params.get("batch_size", 32),
            shuffle=True,
            num_workers=self.cfg.train_params.get("num_workers", 0),
            pin_memory=True,
            drop_last=True,
        )

    def init_model(self):
        self.model = DownstreamTrajPredictor(self.cfg).to(self.device)
        self._maybe_load_pretext_ckpt()
        # Stage-1: freeze backbone (head only training)
        self.model.freeze_backbone()

    def _maybe_load_pretext_ckpt(self):
        if not self.pretext_ckpt:
            return
        ckpt_path = os.path.expanduser(self.pretext_ckpt)
        if not os.path.isfile(ckpt_path):
            print(f"[WARN] pretext ckpt not found: {ckpt_path}")
            return
        sd = torch.load(ckpt_path, map_location="cpu")
        model_sd = sd.get("model", sd)
        missing, unexpected = self.model.backbone.load_state_dict(model_sd, strict=False)
        print(f"[Load] Pretext ckpt: {ckpt_path}")
        if missing:
            print("  missing keys:", len(missing))
        if unexpected:
            print("  unexpected keys:", len(unexpected))

    def init_optimizer(self, stage: int):
        head_lr = float(getattr(self.cfg.train_params, "head_lr", 5e-4))
        bb_lr   = float(getattr(self.cfg.train_params, "backbone_lr", 1e-4))
        wd      = float(self.cfg.train_params.get("weight_decay", 1e-5))

        if stage == 1:
            params = [p for p in self.model.head.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(params, lr=head_lr, weight_decay=wd)
        else:
            params = [
                {"params": [p for p in self.model.head.parameters() if p.requires_grad], "lr": head_lr},
                {"params": [p for p in self.model.backbone.parameters() if p.requires_grad], "lr": bb_lr},
            ]
            self.optimizer = torch.optim.AdamW(params, weight_decay=wd)

        self._check_opt_params()

    def _check_opt_params(self):
        opt_params = {id(p) for g in self.optimizer.param_groups for p in g["params"]}
        missing = [n for n, p in self.model.named_parameters() if p.requires_grad and id(p) not in opt_params]
        if missing:
            print("[WARN] Params missing from optimizer:", missing)

    # -------------------- logging --------------------
    def _cfg_to_dict(self, cfg):
        try:
            from omegaconf import OmegaConf
            return OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            return cfg if isinstance(cfg, dict) else {"cfg": str(cfg)}

    def init_logger(self):
        if not self.use_wandb:
            self.run = None
            return
        project = getattr(self.cfg.logger, "project", "downstream")
        entity  = getattr(self.cfg.logger, "entity", None)
        mode    = getattr(self.cfg.logger, "mode", "online")
        exp     = getattr(self.cfg.logger, "experiment_name", "downstream_exp")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{exp}_{timestamp}"

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            mode=mode,
            config=self._cfg_to_dict(self.cfg),
            tags=getattr(self.cfg.logger, "tags", None),
        )
        wandb.watch(self.model, log="all", log_freq=200)

    # -------------------- utils --------------------
    def move_batch_to_device(self, batch: Dict[str, Any]):
        out = {}
        for k, v in batch.items():
            if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                out[k] = torch.stack(v, dim=1).to(self.device, non_blocking=True)
            elif isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    @staticmethod
    def ade_fde(pred: torch.Tensor, target: torch.Tensor):
        """
        pred/target: [B, T, 2]
        returns ADE, FDE
        """
        se = (pred - target).pow(2).sum(dim=-1)     # [B, T]
        de = torch.sqrt(se + 1e-8)                  # [B, T]
        ade = de.mean()
        fde = de[:, -1].mean()
        return ade, fde

    # -------------------- 1 epoch --------------------
    def train_one_epoch(self, epoch_idx: int = 0, stage: int = 1):
        # Head always in train(); backbone mode controlled by freeze/unfreeze
        total_loss, num_batches = 0.0, 0
        total_ade, total_fde = 0.0, 0.0
        guard = NaNGuard(self.model)

        pbar = tqdm(self.train_loader, desc=f"Downstream E{epoch_idx+1} (S{stage})", leave=False)
        for step, batch in enumerate(pbar, start=1):
            batch = self.move_batch_to_device(batch)

            if step == 1:
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        tensor_stats(f"batch.{k}", v)

            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                # 1) Truncate target to T_pred (handles dataset T_gt >= T_pred)
                T_pred = int(self.cfg.model.input.T_pred)
                future = batch["future_positions"].float()          # [B, T_gt, 2]
                T_gt   = future.size(1)
                T_use  = min(T_pred, T_gt)                          # guard if ever T_gt < T_pred
                target = future[:, :T_use, :]                       # [B, T_use, 2]

                # 2) Make goal consistent with the truncated horizon
                batch = dict(batch)                                 # shallow copy so we can override goal
                batch["goal"] = target[:, -1, :].contiguous()       # [B, 2]

                # 3) Forward  (pass obs-only + goal)
                inputs = {
                    "past_frames":  batch["past_frames"],
                    "past_kp_2d":   batch["past_kp_2d"],
                    "past_root_3d": batch["past_root_3d"],  # if you want strictly XY, use [..., :2]
                    "goal":         batch["goal"],
                }
                pred = self.model(inputs)                             # [B, T_pred, 2]

                # 4) Align pred to the same horizon as target (only matters if T_gt < T_pred)
                if pred.size(1) != T_use:
                    pred = pred[:, :T_use, :]                       # [B, T_use, 2]

                # 5) Loss
                loss = F.mse_loss(pred, target)

            if not torch.isfinite(loss):
                print("[BAD] loss NaN/Inf"); raise FloatingPointError("loss NaN/Inf")

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            # Grad diagnostics
            self.scaler.unscale_(self.optimizer)
            bad_grad = False
            total_norm_sq = 0.0
            for n, p in self.model.named_parameters():
                if p.grad is None: continue
                g = p.grad
                if not torch.isfinite(g).all():
                    print(f"[BAD] grad NaN/Inf at {n}"); bad_grad = True
                total_norm_sq += g.norm().item() ** 2
            if bad_grad:
                raise FloatingPointError("gradient NaN/Inf")
            total_norm = total_norm_sq ** 0.5

            # Clip
            clip = float(self.cfg.train_params.get("clip_grad_norm", 1.0))
            if clip and clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            ade, fde = self.ade_fde(pred.detach(), target.detach())
            total_loss += float(loss.item()); num_batches += 1
            total_ade  += float(ade.item())
            total_fde  += float(fde.item())
            avg = total_loss / num_batches
            avg_ade = total_ade / num_batches
            avg_fde = total_fde / num_batches

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ADE=f"{ade.item():.3f}",
                FDE=f"{fde.item():.3f}",
                avg=f"{avg:.4f}",
                avg_ADE=f"{avg_ade:.3f}",
                avg_FDE=f"{avg_fde:.3f}",
            )

            if self.use_wandb:
                wandb.log({
                    "downstream/loss_step": float(loss.item()),
                    "downstream/ADE_step": float(ade.item()),
                    "downstream/FDE_step": float(fde.item()),
                    "downstream/grad_norm": float(total_norm),
                    "downstream/lr_head": float(self.optimizer.param_groups[0]["lr"]),
                    "downstream/lr_backbone": float(self.optimizer.param_groups[-1]["lr"]) if len(self.optimizer.param_groups) > 1 else 0.0,
                    "downstream/epoch": epoch_idx + 1,
                    "downstream/step": step,
                })

        guard.close()
        return (total_loss / max(1, num_batches),
                total_ade  / max(1, num_batches),
                total_fde  / max(1, num_batches))

    # -------------------- train --------------------
    def train(self):
        print(f"Using device: {self.device}")

        total_epochs  = int(self.cfg.train_params.get("epochs", 20))
        stage1_epochs = int(getattr(self.cfg.train_params, "stage1_epochs", 5))
        stage1_epochs = max(0, min(stage1_epochs, total_epochs))
        stage2_epochs = max(0, total_epochs - stage1_epochs)

        save_every = int(self.cfg.train_params.get("save_every", 5))
        best_loss = float("inf")

        model_dir = (
            getattr(self.cfg.directory, "model_dir", None)
            or "checkpoints/downstream"
        )
        os.makedirs(model_dir, exist_ok=True)

        # -------- Stage 1: freeze backbone --------
        for epoch in range(stage1_epochs):
            self.model.freeze_backbone()  # ensure frozen
            loss, ade, fde = self.train_one_epoch(epoch_idx=epoch, stage=1)
            print(f"[Stage-1] Epoch {epoch+1}/{stage1_epochs} - Loss: {loss:.4f} | ADE: {ade:.3f} | FDE: {fde:.3f}")

            if self.use_wandb:
                wandb.log({"downstream/loss_epoch": float(loss),
                           "downstream/ADE_epoch": float(ade),
                           "downstream/FDE_epoch": float(fde),
                           "downstream/stage": 1, "downstream/epoch": epoch + 1})

            is_best = loss < best_loss
            if is_best:
                best_loss = loss
            if is_best or ((epoch + 1) % save_every == 0):
                timestamp = datetime.now().strftime("%Y%m%d")
                tag = "best_s1" if is_best else f"s1_e{epoch+1}"
                ckpt = os.path.join(model_dir, f"{self.cfg.logger.experiment_name}_{timestamp}_{tag}.pth")
                save_checkpoint(self.model, self.optimizer, epoch + 1, ckpt)
                print(f"[Checkpoint] Saved to {ckpt}")
                if self.use_wandb and os.path.isfile(ckpt):
                    artifact = wandb.Artifact(
                        name=f"{self.cfg.logger.experiment_name}_{tag}",
                        type="model",
                        metadata={"epoch": epoch + 1, "loss": float(loss), "ADE": float(ade), "FDE": float(fde)},
                    )
                    artifact.add_file(ckpt)
                    wandb.log_artifact(artifact)

        # -------- Stage 2: unfreeze & fine-tune --------
        if stage2_epochs > 0:
            self.model.unfreeze_backbone()
            self.init_optimizer(stage=2)  # two param groups

            for e2 in range(stage2_epochs):
                epoch = stage1_epochs + e2
                loss, ade, fde = self.train_one_epoch(epoch_idx=epoch, stage=2)
                print(f"[Stage-2] Epoch {epoch+1}/{total_epochs} - Loss: {loss:.4f} | ADE: {ade:.3f} | FDE: {fde:.3f}")

                if self.use_wandb:
                    wandb.log({"downstream/loss_epoch": float(loss),
                               "downstream/ADE_epoch": float(ade),
                               "downstream/FDE_epoch": float(fde),
                               "downstream/stage": 2, "downstream/epoch": epoch + 1})

                is_best = loss < best_loss
                if is_best:
                    best_loss = loss
                if is_best or ((epoch + 1) % save_every == 0):
                    timestamp = datetime.now().strftime("%Y%m%d")
                    tag = "best_s2" if is_best else f"s2_e{epoch+1}"
                    ckpt = os.path.join(model_dir, f"{self.cfg.logger.experiment_name}_{timestamp}_{tag}.pth")
                    save_checkpoint(self.model, self.optimizer, epoch + 1, ckpt)
                    print(f"[Checkpoint] Saved to {ckpt}")
                    if self.use_wandb and os.path.isfile(ckpt):
                        artifact = wandb.Artifact(
                            name=f"{self.cfg.logger.experiment_name}_{tag}",
                            type="model",
                            metadata={"epoch": epoch + 1, "loss": float(loss), "ADE": float(ade), "FDE": float(fde)},
                        )
                        artifact.add_file(ckpt)
                        wandb.log_artifact(artifact)

        if getattr(self, "run", None) is not None:
            self.run.finish()


# ---------- Entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--pretext_ckpt", type=str, default=None,
                        help="Path to pretrained pretext checkpoint to init the backbone")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    args = parser.parse_args()

    learner = LearnerDownstream(args.cfg, use_wandb=not args.no_wandb, pretext_ckpt=args.pretext_ckpt)
    learner.train()

# python pose2nav/train_downstream.py --cfg pose2nav/config/train_downstream.yaml