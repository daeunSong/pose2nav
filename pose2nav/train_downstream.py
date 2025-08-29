# train_downstream.py
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
from utils.helpers import get_conf
from utils.nn import save_checkpoint, load_checkpoint


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
        self.init_optimizer()   # head-only optimizer (backbone frozen)
        self.init_logger()

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
            only_human_visable=getattr(self.cfg.dataset, "only_human_visable", True),
            only_nonlinear=getattr(self.cfg.dataset, "only_nonlinear", True),
            resize=tuple(self.cfg.dataset.get("resize", [224, 224])),  # keep 224x224 for ViT/DINO
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

        nw = self.cfg.train_params.get("num_workers", 0)
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.cfg.train_params.get("batch_size", 32),
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            drop_last=True,
            persistent_workers=bool(nw > 0),
            prefetch_factor=self.cfg.train_params.get("prefetch_factor", 2) if nw > 0 else None,
        )

    def init_model(self):
        self.model = DownstreamTrajPredictor(self.cfg).to(self.device)
        self._load_backbone_from_pretext()  # robust loader for your rich pretext ckpt

        # Freeze the backbone (decoder-only training)
        self.model.freeze_backbone()
        for p in self.model.backbone.parameters():
            p.requires_grad = False

    # ---------- single-checkpoint backbone loader ----------
    def _load_backbone_from_pretext(self):
        if not self.pretext_ckpt:
            print("[Load] No pretext_ckpt provided (skipping).")
            return

        ckpt_path = os.path.expanduser(self.pretext_ckpt)
        if not os.path.isfile(ckpt_path):
            print(f"[WARN] pretext ckpt not found: {ckpt_path}")
            return

        raw = load_checkpoint(ckpt_path, self.device)

        if not isinstance(raw, dict):
            print("[Load] Unexpected checkpoint format.")
            return

        # ---- Prefer the full model state_dict you saved ----
        sd = None
        if "model" in raw and isinstance(raw["model"], dict):
            sd = raw["model"]

        # ---- Fallback: assemble from per-module dicts if present ----
        if sd is None:
            assembled = {}
            for prefix in (
                "image_encoder",
                "kp_encoder",
                "observation_encoder",
                "proj_obs",
                "proj_future",
                "final_ln",
                "future_traj_encoder",
            ):
                sub = raw.get(prefix, None)
                if isinstance(sub, dict):
                    for k, v in sub.items():
                        assembled[f"{prefix}.{k}"] = v
            sd = assembled if assembled else None

        # ---- Last-resort legacy key ----
        if sd is None and "state_dict" in raw and isinstance(raw["state_dict"], dict):
            sd = raw["state_dict"]

        if sd is None:
            print("[Load] No recognizable state_dict found in checkpoint.")
            return

        # Strip "module." if present
        if any(k.startswith("module.") for k in sd):
            sd = {k[7:]: v for k, v in sd.items()}

        # Filter to matching names & shapes for our downstream backbone (PretextModel)
        msd = self.model.backbone.state_dict()
        filtered = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}

        ret = self.model.backbone.load_state_dict(filtered, strict=False)
        num_loaded = len(filtered)
        print(f"[Load] Pretext checkpoint: {ckpt_path}")
        print(f"  backbone tensors loaded: {num_loaded}/{len(msd)}")

        miss = getattr(ret, "missing_keys", [])
        unex = getattr(ret, "unexpected_keys", [])
        if miss: print("  missing keys:", len(miss), " e.g.", miss[:8])
        if unex: print("  unexpected keys:", len(unex), " e.g.", unex[:8])

        if num_loaded < 0.8 * len(msd):
            print("[WARN] <80% of backbone weights loaded — check that checkpoint matches PretextModel config.")

    def init_optimizer(self):
        head_lr = float(getattr(self.cfg.train_params, "head_lr", 5e-4))
        wd      = float(self.cfg.train_params.get("weight_decay", 1e-5))

        # --- Head-only optimizer ---
        params = [p for p in self.model.head.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=head_lr, weight_decay=wd)

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

    # -------------------- 1 epoch --------------------
    def train_one_epoch(self, epoch_idx: int = 0):
        total_loss, num_batches = 0.0, 0
        total_ade, total_fde = 0.0, 0.0

        pbar = tqdm(self.train_loader, desc=f"Downstream E{epoch_idx+1}", leave=False)
        for step, batch in enumerate(pbar, start=1):
            batch = self.move_batch_to_device(batch)

            # 1) Truncate target to T_pred
            T_pred = int(self.cfg.model.input.T_pred)
            future = batch["future_positions"].float()          # [B, T_gt, 2]
            T_gt   = future.size(1)
            T_use  = min(T_pred, T_gt)
            target = future[:, :T_use, :]

            # quick sanity: catch all-zero targets (dataset issue)
            if torch.allclose(target, torch.zeros_like(target)):
                print("[WARN] target is all zeros this step")

            # 2) Derive goal from the truncated horizon
            batch = dict(batch)
            batch["goal"] = target[:, -1, :].contiguous()       # [B, 2]

            # 3) Forward (decoder-only training; backbone frozen)
            inputs = {
                "past_frames":  batch["past_frames"],
                "past_kp_2d":   batch["past_kp_2d"],
                "goal":         batch["goal"],
            }
            pred = self.model(inputs)                            # [B, T_pred, 2]

            # 4) Align horizon
            if pred.size(1) != T_use:
                pred = pred[:, :T_use, :]

            # 5) Loss
            loss = F.mse_loss(pred, target)

            if not torch.isfinite(loss):
                print("[BAD] loss NaN/Inf"); raise FloatingPointError("loss NaN/Inf")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # grad diagnostics + clipping
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

            clip = float(self.cfg.train_params.get("clip_grad_norm", 1.0))
            if clip and clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            self.optimizer.step()

            bsz = pred.size(0)
            with torch.no_grad():
                ade = (pred - target).norm(dim=-1).mean()             # mean over (B,T)
                fde = (pred[:, -1] - target[:, -1]).norm(dim=-1).mean()

            total_loss += float(loss.item())
            total_ade  += float(ade.item())
            total_fde  += float(fde.item())
            num_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if self.use_wandb:
                wandb.log({
                    "downstream/loss_step": float(loss.item()),
                    "downstream/ade_step": float(ade.item()),
                    "downstream/fde_step": float(fde.item()),
                    "downstream/grad_norm": float(total_norm),
                    "downstream/lr_head": float(self.optimizer.param_groups[0]["lr"]),
                    "downstream/epoch": epoch_idx + 1,
                    "downstream/step": step,
                })

        avg_loss = total_loss / num_batches

        return avg_loss


    # -------------------- train (single stage: head only) --------------------
    def train(self):
        print(f"Using device: {self.device}")

        epochs = int(self.cfg.train_params.get("epochs", 20))
        save_every   = int(self.cfg.train_params.get("save_every", 5))
        best_loss = float("inf")

        model_dir = (
            getattr(self.cfg.directory, "model_dir", None)
            or "checkpoints/downstream"
        )
        os.makedirs(model_dir, exist_ok=True)

        for epoch in range(epochs):
            loss = self.train_one_epoch(epoch_idx=epoch)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

            if self.use_wandb:
                wandb.log({"downstream/loss_epoch": float(loss),
                           "downstream/epoch": epoch + 1})

            is_best = loss < best_loss
            if is_best:
                best_loss = loss

            if is_best or ((epoch + 1) % save_every == 0):
                ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
                tag = "best" if is_best else f"e{epoch+1}"
                name = f"{self.cfg.logger.experiment_name}"

                # rich downstream checkpoint
                state = {
                    "time": ts,
                    "epoch": epoch + 1,
                    "best": float(best_loss),
                    "last_loss": float(loss),
                    "model_name": type(self.model).__name__,
                    "optimizer_name": type(self.optimizer).__name__,
                    "cfg": self._cfg_to_dict(self.cfg),
                    "model": self.model.state_dict(),              # full downstream model
                    "backbone": self.model.backbone.state_dict(),  # frozen pretext backbone
                    "head": self.model.head.state_dict(),          # prediction head
                    "optimizer": self.optimizer.state_dict(),
                }

                ckpt_path = save_checkpoint(state, is_best, model_dir, name)  # writes {name}.pth (+ -best.pth if best)
                print(f"[Checkpoint] Saved to {ckpt_path}")

                if self.use_wandb and os.path.isfile(ckpt_path):
                    artifact = wandb.Artifact(
                        name=f"{self.cfg.logger.experiment_name}_{tag}",
                        type="model",
                        metadata={"epoch": epoch + 1, "loss": float(loss), "saved_at": ts},
                    )
                    artifact.add_file(ckpt_path)
                    wandb.log_artifact(artifact)

        if getattr(self, "run", None) is not None:
            self.run.finish()


# ---------- Entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--pretext_ckpt", type=str, default=None,
                        help="Path to rich pretext checkpoint (*.pt/.pth). "
                             "This script will load matching weights into the downstream backbone.")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    args = parser.parse_args()

    learner = LearnerDownstream(args.cfg, use_wandb=not args.no_wandb, pretext_ckpt=args.pretext_ckpt)
    learner.train()

# Usage:
# python pose2nav/train_downstream.py --cfg pose2nav/config/train_downstream.yaml --pretext_ckpt path/to/pretext_*.pt
