# train_downstream.py
import argparse
import os
import math
import random
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb

from model.data_loader import SocialNavDataset
from model.downstream_model import DownstreamTrajPredictor
from utils.helpers import get_conf
from utils.nn import save_checkpoint, load_checkpoint, check_grad_norm


class LearnerDownstream:
    def __init__(self, cfg_path: str, use_wandb: bool = True):
        self.cfg = get_conf(cfg_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb

        self.ckpt_path = self.cfg.directory.pretext_ckpt

        self.global_step = 0

        self.set_seed(self.cfg.train_params.seed)
        self.init_data()
        self.init_model()
        self.init_optimizer()   # head-only optimizer (backbone frozen)
        self.init_logger()

    def _build_checkpoint(self, epoch:int, iteration:int, best:float, last_loss:float):
        model = self.model #self.uncompiled_model
        ckpt = {
            "time":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch":            epoch,
            "iteration":        iteration,
            "best":             float(best),
            "last_loss":        float(last_loss),
            "model_name":       type(model).__name__,
            "optimizer_name":   type(self.optimizer).__name__,
            "optimizer":        self.optimizer.state_dict(),
            "model":            model.state_dict(),  
            # "backbone":         model.backbone.state_dict(),  # frozen pretext backbone
            # "head":             model.head.state_dict(),          # prediction head
        }
        return ckpt
    # -------------------- setup --------------------
    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def init_data(self):
        dataset = SocialNavDataset(**self.cfg.dataset)
        self.train_loader = DataLoader(dataset, **self.cfg.dataloader)

    def init_model(self):
        self.model = DownstreamTrajPredictor(self.cfg).to(self.device)
        # self.uncompiled_model = self.model
        # self.model = torch.compile(self.model)

        ckpt = load_checkpoint(self.ckpt_path, self.device)
        state = ckpt.get("model", ckpt)
        # strip 'module.' if present
        state = {k.replace("module.", ""): v for k,v in state.items()}
        # drop projector weights (pretext-only)
        state = {k: v for k, v in state.items() if not k.startswith(("proj_obs", "proj_future"))}

        self.model.backbone.load_state_dict(state, strict=False)
        print(f"[✓] Loaded backbone: {self.ckpt_path}")

        # Freeze the backbone 
        self.model.freeze_backbone()
        for p in self.model.backbone.parameters():
            p.requires_grad = False


    def init_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), **self.cfg.optimizer.sgd)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-5)

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
        project = getattr(self.cfg.logger.downstream, "project", "downstream")
        entity  = getattr(self.cfg.logger.downstream, "entity", None)
        mode    = getattr(self.cfg.logger.downstream, "mode", "online")
        exp     = getattr(self.cfg.logger.downstream, "experiment_name", "exp2")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{exp}_{timestamp}"

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            mode=mode,
            config=self._cfg_to_dict(self.cfg),
            tags=getattr(self.cfg.logger.downstream, "tags", None),
        )
        wandb.watch(self.model, log="all", log_freq=200)

    # # -------------------- training -----------------
    def move_batch_to_device(self, batch: Dict[str, Any]):
        out = {}
        out["past_frames"] = [x.to(self.device) for x in batch["past_frames"]]
        out["future_positions"] = batch["future_positions"].to(device=self.device)
        out["past_kp_2d"] = batch["past_kp_2d"].to(device=self.device)
        out["goal"] = out["future_positions"][:, -1, :].contiguous()  # [B, 2]
        return out

    def train_one_epoch(self, epoch_idx: int = 0):
        total_loss, num_batches = 0.0, 0
        total_ade, total_fde = 0.0, 0.0

        pbar = tqdm(self.train_loader, desc=f"Downstream E{epoch_idx+1}", leave=False)
        for data in pbar:   # batch
            self.model.train()

            # move data to device
            batch = self.move_batch_to_device(data)
            pred = self.model(batch)                            # [B, T_pred, 2]

            target = batch["future_positions"].float()  # target future positions
            loss = torch.nn.functional.mse_loss(pred, target)

            if not torch.isfinite(loss):
                print("[BAD] loss NaN/Inf"); raise FloatingPointError("loss NaN/Inf")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = check_grad_norm(self.model)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if self.use_wandb:
                wandb.log({
                    "downstream/loss_step": float(loss.item()),
                    "downstream/grad_norm": grad_norm,
                    "downstream/lr_head": float(self.optimizer.param_groups[0]["lr"]),
                }, step=self.global_step)

            self.global_step += 1

        return total_loss / max(1, num_batches)


    # -------------------- train (single stage: head only) --------------------
    def train(self):
        print(f"Using device: {self.device}")

        epochs = int(self.cfg.train_params.get("epochs", 20))
        save_every   = int(self.cfg.train_params.get("save_every", 50))
        save_best = int(self.cfg.train_params.get("save_best", 150))
        best_loss = float("inf")

        model_dir = (
            getattr(self.cfg.directory, "downstream_model_dir", None)
            or "checkpoints/downstream"
        )
        os.makedirs(model_dir, exist_ok=True)

        for epoch in range(epochs):
            avg_loss = self.train_one_epoch(epoch_idx=epoch)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

            if self.use_wandb:
                wandb.log({"downstream/loss_epoch": float(avg_loss)})

            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss

            if (epoch + 1) % save_every == 0 or (is_best and (epoch + 1) >= save_best):
                timestamp  = datetime.now().strftime("%Y%m%d")
                tag = "best" if is_best else f"e{epoch+1}"
                name = f"{self.cfg.logger.downstream.experiment_name}_{timestamp}"

                checkpoint = self._build_checkpoint(
                    epoch=epoch+1,
                    iteration=self.global_step,
                    best=best_loss,
                    last_loss=avg_loss,
                )
                ckpt_path = save_checkpoint(checkpoint, is_best, model_dir, name, epoch+1)  # writes {name}.pth (+ -best.pth if best)
                print(f"[Checkpoint] Saved to {ckpt_path}")

                if self.use_wandb and os.path.isfile(ckpt_path):
                    artifact = wandb.Artifact(
                        name=f"{self.cfg.logger.downstream.experiment_name}_{tag}",
                        type="model",
                        metadata={"epoch": epoch + 1, "loss": float(avg_loss), "saved_at": timestamp},
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

    learner = LearnerDownstream(args.cfg, use_wandb=not args.no_wandb)
    learner.train()

# Usage:
# python pose2nav/train_downstream.py --cfg pose2nav/config/train.yaml
