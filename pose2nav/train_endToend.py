# train_endToend.py
import argparse
import os
import math
import random
from datetime import datetime
import time
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb  # W&B

from model.data_loader import SocialNavDataset
from model.endToend_model import EndToEndModel
from utils.helpers import get_conf
from utils.nn import save_checkpoint, load_checkpoint, check_grad_norm, get_param_groups


class EndtoEndLearner:
    def __init__(self, cfg_path, use_wandb=True, resume: str = ""):
        self.cfg = get_conf(cfg_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb

        self.global_step = 0
        self.start_epoch = 0
        self.best_loss = float("inf")

        self.set_seed(self.cfg.train_params.seed)
        self.init_data()
        self.init_model()
        self.init_optimizer()
        if resume:
            self.resume_from(resume)
        self.init_logger()  # W&B init

    def _build_checkpoint(self, epoch:int, iteration:int, best:float, last_loss:float):
        model = self.uncompiled_model 
        ckpt = {
            "time":                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch":                epoch,
            "iteration":            iteration,
            "best":                 float(best),
            "last_loss":            float(last_loss),
            "model_name":           type(model).__name__,
            "optimizer_name":       type(self.optimizer).__name__,
            "optimizer":            self.optimizer.state_dict(),
            "model":                model.state_dict(),  
            # "image_encoder":        model.image_encoder.state_dict(),
            # "kp_encoder":           model.kp_encoder.state_dict(),
            # "observation_encoder":  model.observation_encoder.state_dict(),
            # "proj_obs":             model.proj_obs.state_dict(),
            # "proj_future":          model.proj_future.state_dict(),
            # "final_ln":             model.final_ln.state_dict(),
            # "future_traj_encoder":  model.future_traj_encoder.state_dict(),
        }
        return ckpt

    def resume_from(self, ckpt_path: str):
        ckpt = load_checkpoint(ckpt_path, self.device)

        # 1) model weights
        self.uncompiled_model.load_state_dict(ckpt["model"], strict=True)

        # 2) optimizer state
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])

        # 3) bookkeeping
        self.start_epoch = int(ckpt.get("epoch", 0))
        self.global_step = int(ckpt.get("iteration", 0))
        self.best_loss   = float(ckpt.get("best", float("inf")))

        print(f"[✓] Resumed from {ckpt_path} | start_epoch={self.start_epoch}, "
              f"global_step={self.global_step}, best_loss={self.best_loss:.4f}")

    # -------------------- setup --------------------
    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def init_data(self):
        dataset = SocialNavDataset(**self.cfg.dataset)
        self.train_loader = DataLoader(dataset, **self.cfg.dataloader)

    def init_model(self):
        self.uncompiled_model = EndToEndModel(self.cfg).to(self.device)
        self.uncompiled_model.train()
        self.model = torch.compile(self.uncompiled_model)

    def init_optimizer(self):
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), **self.cfg.optimizer.adamw)
        param_groups = get_param_groups(self.uncompiled_model)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            **self.cfg.optimizer.adamw 
        )

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
        project = getattr(self.cfg.logger.endToend, "project", "endToend")
        entity  = getattr(self.cfg.logger.endToend, "entity", None)
        mode    = getattr(self.cfg.logger.endToend, "mode", "online")  # "online"|"offline"|"disabled"
        exp     = getattr(self.cfg.logger.endToend, "experiment_name", "exp2")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{exp}_{timestamp}"

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            mode=mode,
            config=self._cfg_to_dict(self.cfg),
            tags=getattr(self.cfg.logger.endToend, "tags", None),
        )
        wandb.watch(self.model, log="parameters", log_freq=100)

    # # -------------------- training -----------------
    def move_batch_to_device(self, batch):
        out = {}
        out["past_frames"] = [x.to(self.device) for x in batch["past_frames"]]
        out["future_positions"] = batch["future_positions"].to(device=self.device)
        out["past_kp_2d"] = batch["past_kp_2d"].to(device=self.device)
        # out["future_kp_2d"] = batch["future_kp_2d"].to(device=self.device)
        out["goal"] = out["future_positions"][:, -1, :].contiguous()  # [B, 2]
        return out

    def train_one_epoch(self, epoch_idx: int = 0, log_every: int = 50):
        loss_buf = []   
        total_loss_t = torch.zeros((), device="cuda") 
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Downstream E{epoch_idx+1}", leave=False)
        for step, data in enumerate(pbar):
            batch = self.move_batch_to_device(data)
            pred = self.model(batch)
            target = batch["future_positions"].float()
            loss = torch.nn.functional.mse_loss(pred, target)

            if not torch.isfinite(loss):
                print("[BAD] loss NaN/Inf"); raise FloatingPointError("loss NaN/Inf")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.uncompiled_model.parameters(), 1.0)
            self.optimizer.step()

            loss_buf.append(loss.detach())
            total_loss_t += loss.detach()
            num_batches += 1

            if (step + 1) % log_every == 0:
                avg_loss = torch.stack(loss_buf).mean().item()
                pbar.set_postfix_str(f"loss={avg_loss:.4f}")

                if self.use_wandb:
                    try:
                        grad_norm_val = float(check_grad_norm(self.uncompiled_model))
                    except Exception:
                        grad_norm_val = 0.0

                    wandb.log({
                        "train/loss_step": avg_loss,
                        "train/grad_norm": grad_norm_val,
                        "train/lr_head": float(self.optimizer.param_groups[0]["lr"]),
                    }, step=self.global_step)

                loss_buf.clear()
                
            self.global_step += 1

        epoch_avg = (total_loss_t / max(1, num_batches)).item()
        if self.use_wandb:
            wandb.log({"train/loss_epoch": epoch_avg}, step=self.global_step)
        return epoch_avg


    def train(self):
        print(f"Using device: {self.device}")
        epochs = int(self.cfg.train_params.get("epochs", 20))
        save_every   = int(self.cfg.train_params.get("save_every", 50))
        save_best = int(self.cfg.train_params.get("save_best", 150))
        log_every = int(self.cfg.train_params.get("log_every", 50))  #step

        model_dir = (
            getattr(self.cfg.directory, "endToend_model_dir", None)
            or "checkpoint/endToend"
        )
        os.makedirs(model_dir, exist_ok=True)

        for epoch in range(self.start_epoch, epochs):
            avg_loss = self.train_one_epoch(epoch_idx=epoch, log_every=log_every)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss

            if (epoch + 1) % save_every == 0 or (is_best and (epoch + 1) >= save_best):
                timestamp  = datetime.now().strftime("%Y%m%d")
                tag = "best" if is_best else f"e{epoch+1}"
                name = f"{self.cfg.logger.endToend.experiment_name}_{timestamp}"

                checkpoint = self._build_checkpoint(
                    epoch=epoch+1,
                    iteration=self.global_step,
                    best=self.best_loss,
                    last_loss=avg_loss,
                )
                ckpt_path = save_checkpoint(checkpoint, is_best, model_dir, name, epoch+1)   # writes {name}.pth (+ -best.pth if best)
                print(f"[Checkpoint] Saved to {ckpt_path}")

                if self.use_wandb and os.path.isfile(ckpt_path):
                    artifact = wandb.Artifact(
                        name=f"{self.cfg.logger.endToend.experiment_name}_{tag}",
                        type="model",
                        metadata={"epoch": epoch + 1, "loss": float(avg_loss), "best": float(self.best_loss), "saved_at": timestamp},
                    )
                    artifact.add_file(ckpt_path)
                    wandb.log_artifact(artifact)


        if self.run is not None:
            self.run.finish()

# ---------- Entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--resume", type=str, default="", help="Path to a .pth checkpoint to resume from (or 'auto')")
    args = parser.parse_args()

    learner = EndtoEndLearner(args.cfg, use_wandb=not args.no_wandb, resume=args.resume)
    learner.train()

# Usage:
# python pose2nav/train_endToend.py --cfg pose2nav/config/train.yaml
# python pose2nav/train_endToend.py --cfg pose2nav/config/train.yaml --resume checkpoint/endToend/socialnav_endtoend_..._best.pth
