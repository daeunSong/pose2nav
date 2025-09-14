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
from utils.nn import save_checkpoint, load_checkpoint, check_grad_norm, get_param_groups


class LearnerDownstream:
    def __init__(self, cfg_path: str, use_wandb: bool = True, resume: bool = False):
        self.cfg = get_conf(cfg_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb

        self.ckpt_path = self.cfg.directory.pretext_ckpt

        self.global_step = 0
        self.start_epoch = 0
        self.best_loss = float("inf")
        self.last_loss = float("inf")

        self.set_seed(self.cfg.train_params.seed)
        self.init_data()
        if resume:
            resume_path = getattr(self.cfg.directory, "resume_path", "")
            self.resume_from(resume_path)
        else:
            self.init_model()
        self.init_optimizer()   # head-only optimizer (backbone frozen)
        self.model = torch.compile(self.uncompiled_model)
        self.init_logger()

    def _build_checkpoint(self, epoch:int, iteration:int, best:float, last_loss:float):
        model = self.uncompiled_model

       # map id(param) -> name
        id2name = {id(p): n for n, p in model.named_parameters()}

        # record param names per optimizer group (in order)
        group_names = []
        for g in self.optimizer.param_groups:
            group_names.append([id2name.get(id(p), None) for p in g["params"]])

        ckpt = {
            "time":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch":            epoch,
            "iteration":        iteration,
            "best":             float(best),
            "last_loss":        float(last_loss),
            "model_name":       type(model).__name__,
            "optimizer_name":   type(self.optimizer).__name__,
            "optimizer":        self.optimizer.state_dict(),
            "optimizer_param_group_names": group_names,
            "model":            model.state_dict(),  
            # "backbone":         model.backbone.state_dict(),  # frozen pretext backbone
            # "head":             model.head.state_dict(),          # prediction head
        }
        return ckpt
    
    def resume_from(self, ckpt_path: str):
        self.uncompiled_model = DownstreamTrajPredictor(self.cfg).to(self.device)
        
        ckpt = load_checkpoint(ckpt_path, self.device)
        preload = ckpt.get("model", ckpt)  

        # Optimizer state (may fail if param groups changed)
        if "optimizer" in ckpt and ckpt["optimizer"]:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[resume] optimizer load failed; using fresh optimizer. Reason: {e}")

        target_sd = self.uncompiled_model.state_dict()
        filtered = {k: v for k, v in preload.items()
            if k in target_sd and v.shape == target_sd[k].shape}
        res = self.uncompiled_model.load_state_dict(filtered, strict=False)

        # Report Pretext loading
        matched = len(filtered) - len(res.missing_keys)
        print(f"[load/model] matched≈{matched}  missing={len(res.missing_keys)}  unexpected={len(res.unexpected_keys)}")
        if res.missing_keys:
            print("  missing(sample):", res.missing_keys[:10])
        if res.unexpected_keys:
            print("  unexpected(sample):", res.unexpected_keys[:10])

        # 3) Bookkeeping
        self.start_epoch = int(ckpt.get("epoch", 0))
        self.global_step = int(ckpt.get("iteration", 0))
        self.best_loss   = float(ckpt.get("best", float("inf")))
        last_loss        = float(ckpt.get("last_loss", float("inf")))

        print(f"[✓] Resumed from {ckpt_path} | start_epoch={self.start_epoch}, "
            f"global_step={self.global_step}, best_loss={self.best_loss:.4f}, last_loss={last_loss:.4f}")
        
        # Freeze Backbone
        for p in self.uncompiled_model.model.parameters():
            p.requires_grad = False
        # Train head
        for p in self.uncompiled_model.head.parameters():
            p.requires_grad = True

        self.uncompiled_model.freeze_backbone()
        self.uncompiled_model.model.eval()
        self.uncompiled_model.head.train()


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
        self.uncompiled_model = DownstreamTrajPredictor(self.cfg).to(self.device)

        # Load Pretext
        ckpt = load_checkpoint(self.ckpt_path, self.device)
        preload = ckpt.get("model", ckpt)  

        target_sd = self.uncompiled_model.model.state_dict()
        filtered = {k: v for k, v in preload.items()
            if k in target_sd and v.shape == target_sd[k].shape  and "projector" not in k}
        res = self.uncompiled_model.model.load_state_dict(filtered, strict=False)

        # Report Pretext loading
        matched = len(filtered) - len(res.missing_keys)
        print(f"[load/model] matched≈{matched}  missing={len(res.missing_keys)}  unexpected={len(res.unexpected_keys)}")
        if res.missing_keys:
            print("  missing(sample):", res.missing_keys[:10])
        if res.unexpected_keys:
            print("  unexpected(sample):", res.unexpected_keys[:10])

        # Freeze Backbone
        for p in self.uncompiled_model.model.parameters():
            p.requires_grad = False
        # Train head
        for p in self.uncompiled_model.head.parameters():
            p.requires_grad = True

        # Set submodule modes before compiling. The compiled model will respect these.
        self.uncompiled_model.model.eval()  # Backbone in eval mode (frozen)
        self.uncompiled_model.head.train()  # Head in train mode (fine-tuning)


    def init_optimizer(self):
        # self.optimizer = torch.optim.SGD(self.model.parameters(), **self.cfg.optimizer.sgd)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        param_groups = get_param_groups(self.uncompiled_model.head)
        # self.optimizer = torch.optim.AdamW(param_groups, lr=5e-4, weight_decay=1e-5)
        self.optimizer = torch.optim.SGD(param_groups, **self.cfg.optimizer.sgd)

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

    def train_one_epoch(self, epoch_idx: int = 0, log_every: int = 50):
        loss_buf = []   
        total_loss_t = torch.zeros((), device="cuda")  
        num_batches = 0

        # Ensure correct modes are set for each part of the model
        self.uncompiled_model.model.eval()
        self.uncompiled_model.head.train()

        pbar = tqdm(self.train_loader, desc=f"Downstream E{epoch_idx+1}", leave=False)
        for step, data in enumerate(pbar):   # batch

            # move data to device
            batch = self.move_batch_to_device(data)
            pred = self.model(batch)                            # [B, T_pred, 2]

            target = batch["future_positions"].float()  # target future positions
            loss = torch.nn.functional.mse_loss(pred, target)

            if not torch.isfinite(loss):
                print("[BAD] loss NaN/Inf"); raise FloatingPointError("loss NaN/Inf")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = check_grad_norm(self.uncompiled_model.head)
            torch.nn.utils.clip_grad_norm_(self.uncompiled_model.head.parameters(), 1.0)

            self.optimizer.step()

            loss_buf.append(loss.detach())
            total_loss_t += loss.detach()
            num_batches += 1

            if (step + 1) % log_every == 0:
                avg_loss = torch.stack(loss_buf).mean().item()
                pbar.set_postfix_str(f"loss={avg_loss:.4f}")

                # wandb
                if self.use_wandb:
                    if isinstance(grad_norm, (int, float)):
                        grad_val = float(grad_norm)
                    else:
                        try:
                            grad_val = float(grad_norm)
                        except Exception:
                            grad_val = 0.0

                    wandb.log({
                        "downstream/loss_step": avg_loss,
                        "downstream/grad_norm": grad_val,
                        "downstream/lr_head": float(self.optimizer.param_groups[0]["lr"]),
                    }, step=self.global_step)

                loss_buf.clear()

            self.global_step += 1

        epoch_avg = (total_loss_t / max(1, num_batches)).item()
        if self.use_wandb:
            wandb.log({"downstream/loss_epoch": epoch_avg}, step=self.global_step)
        return epoch_avg


    # -------------------- train (single stage: head only) --------------------
    def train(self):
        print(f"Using device: {self.device}")

        epochs = int(self.cfg.train_params.get("epochs", 20))
        save_every   = int(self.cfg.train_params.get("save_every", 50))
        save_best = int(self.cfg.train_params.get("save_best", 150))
        log_every = int(self.cfg.train_params.get("log_every", 50))  #step

        model_dir = (
            getattr(self.cfg.directory, "downstream_model_dir", None)
            or "checkpoints/downstream"
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
                name = f"{self.cfg.logger.downstream.experiment_name}_{timestamp}"

                checkpoint = self._build_checkpoint(
                    epoch=epoch+1,
                    iteration=self.global_step,
                    best=self.best_loss,
                    last_loss=avg_loss,
                )
                ckpt_path = save_checkpoint(checkpoint, is_best, model_dir, name, epoch+1)  # writes {name}.pth (+ -best.pth if best)
                print(f"[Checkpoint] Saved to {ckpt_path}")

                if self.use_wandb and os.path.isfile(ckpt_path):
                    artifact = wandb.Artifact(
                        name=f"{self.cfg.logger.downstream.experiment_name}_{tag}",
                        type="model",
                        metadata={"epoch": epoch + 1, "loss": float(avg_loss), "best": float(self.best_loss), "saved_at": timestamp},
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
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    learner = LearnerDownstream(args.cfg, use_wandb=not args.no_wandb, resume= args.resume)
    learner.train()

# Usage:
# python pose2nav/train_downstream.py --cfg pose2nav/config/train.yaml
