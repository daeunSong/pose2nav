import argparse
import os
import math
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb  # W&B

from model.data_loader import SocialNavDataset
from model.pretext_model import PretextModel
from model.losses import get_loss_fn
from utils.helpers import get_conf, tensor_stats, NaNGuard
from utils.nn import save_checkpoint


class Learner:
    def __init__(self, cfg_path, use_wandb=True):
        self.cfg = get_conf(cfg_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb

        self.set_seed(self.cfg.train_params.seed)
        self.init_data()
        self.init_model()
        self.init_loss()
        self.init_optimizer()
        self.init_logger()  # W&B init

        amp_enabled = bool(self.cfg.train_params.get("amp", True))
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # -------------------- setup --------------------
    def set_seed(self, seed):
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

        # Optional: train on a subset (e.g., 10%)
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
        self.model = PretextModel(self.cfg).to(self.device)

    def init_loss(self):
        # Base loss (VICReg or Barlow) between z_obs and z_traj
        self.base_loss_fn = get_loss_fn(self.cfg.train_params.loss)

        self.loss_kwargs = {}
        lp = getattr(self.cfg, "loss_params", None)
        if lp and self.cfg.train_params.loss.lower() in lp:
            self.loss_kwargs = dict(lp[self.cfg.train_params.loss.lower()])


    def init_optimizer(self):
        lr = self.cfg.train_params.get("lr", 5e-4)
        wd = self.cfg.train_params.get("weight_decay", 0.03)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self._check_opt_params()

    def _check_opt_params(self):
        opt_params = {id(p) for g in self.optimizer.param_groups for p in g['params']}
        missing = [n for n,p in self.model.named_parameters() if p.requires_grad and id(p) not in opt_params]
        if missing:
            print("[WARN] Params missing from optimizer:", missing)

    # -------------------- W&B ----------------------
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
        project = getattr(self.cfg.logger, "project", "pretext")
        entity = getattr(self.cfg.logger, "entity", None)
        mode = getattr(self.cfg.logger, "mode", "online")  # "online"|"offline"|"disabled"
        exp = getattr(self.cfg.logger, "experiment_name", "exp")
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
        wandb.watch(self.model, log="all", log_freq=100)

    # -------------------- training -----------------
    def move_batch_to_device(self, batch):
        out = {}
        for k, v in batch.items():
            if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                out[k] = torch.stack(v, dim=1).to(self.device, non_blocking=True)
            elif isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def train_one_epoch(self, epoch_idx=0):
        self.model.train()
        total_loss, num_batches = 0.0, 0
        guard = NaNGuard(self.model)

        pbar = tqdm(self.train_loader, desc=f"Train E{epoch_idx+1}", leave=False)
        for step, batch in enumerate(pbar, start=1):
            batch = self.move_batch_to_device(batch)

            # Basic batch sanity once every N steps
            if step == 1:
                for k,v in batch.items():
                    if torch.is_tensor(v):
                        tensor_stats(f"batch.{k}", v)

            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                out = self.model(batch)
                z_obs  = out["z_obs"]
                z_traj = out["z_traj"]

                # Sanity stats
                tensor_stats("z_obs", z_obs)
                tensor_stats("z_traj", z_traj)

                # Single loss (VICReg / Barlow) between obs and traj
                loss = self.base_loss_fn(z_obs, z_traj, **self.loss_kwargs)
                
                if not torch.isfinite(loss):
                    print("[BAD] total loss NaN.", "debug dims:", z_obs.shape, z_traj.shape)
                    raise FloatingPointError("loss NaN")

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            # Gradient diagnostics
            self.scaler.unscale_(self.optimizer)
            bad_grad = False
            total_norm_sq = 0.0
            for n,p in self.model.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad
                if not torch.isfinite(g).all():
                    print(f"[BAD] grad NaN/Inf at {n}")
                    bad_grad = True
                total_norm_sq += g.norm().item() ** 2
            if bad_grad:
                raise FloatingPointError("gradient NaN")
            total_norm = total_norm_sq ** 0.5

            # Clip after checking
            clip = self.cfg.train_params.get("clip_grad_norm", 1.0)
            if clip and clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item()); num_batches += 1
            avg = total_loss / num_batches
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg:.4f}")

            if self.use_wandb:
                wandb.log({
                    "train/loss_step": float(loss.item()),
                    "train/grad_norm": float(total_norm),
                    "train/z_obs_norm": float(z_obs.norm(p=2, dim=1).mean().item()),
                    "train/z_traj_norm": float(z_traj.norm(p=2, dim=1).mean().item()),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch_idx + 1,
                    "train/step": step,
                })

        guard.close()
        return total_loss / max(1, num_batches)

    def train(self):
        print(f"Using device: {self.device}")
        epochs = int(self.cfg.train_params.epochs)
        save_every = int(self.cfg.train_params.get("save_every", 5))
        best_loss = float("inf")

        # Resolve save dir from config (fallbacks)
        model_dir = (
            getattr(self.cfg.directory, "model_dir", None)
            or getattr(self.cfg.directory, "save", None)
            or "checkpoints/pretext"
        )
        os.makedirs(model_dir, exist_ok=True)

        for epoch in range(epochs):
            avg_loss = self.train_one_epoch(epoch_idx=epoch)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

            if self.use_wandb:
                wandb.log({"train/loss_epoch": float(avg_loss), "train/epoch": epoch + 1})

            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss

            if is_best or ((epoch + 1) % save_every == 0):
                timestamp = datetime.now().strftime("%Y%m%d")
                tag = "best" if is_best else f"e{epoch+1}"
                ckpt = os.path.join(model_dir, f"{self.cfg.logger.experiment_name}_{timestamp}_{tag}.pth")
                save_checkpoint(self.model, self.optimizer, epoch + 1, ckpt)
                print(f"[Checkpoint] Saved to {ckpt}")

                if self.use_wandb and os.path.isfile(ckpt):
                    artifact = wandb.Artifact(
                        name=f"{self.cfg.logger.experiment_name}_{tag}",
                        type="model",
                        metadata={"epoch": epoch + 1, "loss": float(avg_loss)},
                    )
                    artifact.add_file(ckpt)
                    wandb.log_artifact(artifact)

        if self.run is not None:
            self.run.finish()


# ---------- Entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    args = parser.parse_args()

    learner = Learner(args.cfg, use_wandb=not args.no_wandb)
    learner.train()

# python pose2nav/train_pretext.py --cfg pose2nav/config/train_pretext.yaml
