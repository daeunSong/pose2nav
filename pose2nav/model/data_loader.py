# pose2nav/model/data_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Union
from pathlib import Path
import numpy as np
import pickle, json, copy
from PIL import ImageOps, ImageFilter
from utils.data_utils import imread, cartesian_to_polar


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        return img


class SocialNavDataset(Dataset):
    def __init__(
        self,
        sample_file: str,
        train: bool = True,
        resize: Union[list, tuple] = (224, 224),
        metric_waypoint_spacing: float = 1.0,
        only_human_visable: bool = True,
        only_nonlinear: bool = True,         
        cache_size: int = 1                   # keep this small to save RAM
    ):
        """
        Dataloader that loads from an sample pkl file of shards.
        Loads shards lazily to avoid huge memory usage.
        """
        self.resize = resize
        self.metric_waypoint_spacing = metric_waypoint_spacing
        self.train = train
        self.only_human_visable = only_human_visable
        self.only_nonlinear = only_nonlinear  
        self.cache_size = cache_size
        self.shard_cache = {}  # shard_path -> loaded shard dict

        # read and store directories
        with Path(sample_file).open("rb") as f:
            self.data = pickle.load(f)

        # Define transformations
        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Resize(self.resize, antialias=True),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAutocontrast(p=0.4),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.6),
                Solarization(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.resize, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
            ])

        N = len(self.data["past_positions"])
        nonlin_mask = np.asarray(self.data["non_linear"])      
        humans_mask = np.asarray(self.data["has_humans"])      
        base    = np.ones(N, dtype=bool)

        # note: keep your config names; you used 'only_nonlinear' and 'only_human_visable'
        if self.only_nonlinear and self.only_human_visable:
            keep = nonlin_mask & humans_mask          # BOTH must be 1
        elif self.only_nonlinear:
            keep = nonlin_mask                        # non-linear only
        elif self.only_human_visable:
            keep = humans_mask                        # humans only
        else:
            keep = base                               # no filtering

        self._apply_mask_inplace(keep)
        print(f"nonlinear: {sum(nonlin_mask)}, humans: {sum(humans_mask)}, both: {sum(keep)}")

    def _apply_mask_inplace(self, keep_mask: np.ndarray):
        """keep_mask: bool array of shape [N]; applies to ALL per-sample lists."""
        keep_mask = np.asarray(keep_mask, dtype=bool).reshape(-1)
        N = len(self.data["past_positions"])
        assert keep_mask.size == N, f"Mask len {keep_mask.size} != data len {N}"

        idx = np.flatnonzero(keep_mask)
        assert idx.size > 0, "[DATA][ERR] Mask removed all samples."

        # sanity: all keys share same length before masking
        for k, v in self.data.items():
            if len(v) != N:
                raise ValueError(f"Length mismatch before masking: key={k} len={len(v)} vs {N}")

        # apply same indices to every key (preserves types)
        for k, v in list(self.data.items()):
            self.data[k] = [v[i] for i in idx]

        print(f"[DATA] kept {idx.size}/{keep_mask.size} samples after filtering.")

    def _apply_mask_inplace(self, keep_mask):
        # normalize mask → 1D bool of length N
        keep_mask = np.asarray(keep_mask).reshape(-1).astype(bool)

        N = len(self.data["past_positions"])
        assert keep_mask.size == N, f"Mask len {keep_mask.size} != data len {N}"

        kept = int(keep_mask.sum())
        assert kept > 0, "[DATA][ERR] Mask removed all samples."

        # sanity: all keys have the same length before masking
        for k, v in self.data.items():
            if len(v) != N:
                raise ValueError(f"Length mismatch before masking: key={k} len={len(v)} vs {N}")

        # filter every per-sample list with the same mask (preserves types)
        for k, v in list(self.data.items()):
            self.data[k] = [x for x, m in zip(v, keep_mask) if m]

        print(f"[DATA] kept {kept}/{N} samples after filtering.")


    def __len__(self):
        return len(self.data["past_positions"])

    def __getitem__(self, idx):
        # last past-frame path (string) for plotting later
        past_paths = self.data["past_frames"][idx]  # list[Path-like]
        last_past_frame_path = str(past_paths[-1]) if len(past_paths) > 0 else None

        past_frames = [
            self.transform(imread(str(p)).convert("RGB"))
            for p in past_paths
        ]

        future_paths = self.data["future_frames"][idx]
        future_frames = [
            self.transform(imread(str(p)).convert("RGB"))
            for p in future_paths
        ]

        sample = {
            "past_positions": self.data["past_positions"][idx],
            "future_positions": self.data["future_positions"][idx],
            "past_yaw": self.data["past_yaw"][idx],
            "future_yaw": self.data["future_yaw"][idx],
            "past_vw": self.data["past_vw"][idx],
            "future_vw": self.data["future_vw"][idx],
            "past_frames": past_frames,
            "future_frames": future_frames,
            "past_kp_3d": self.data["past_kp_3d"][idx],
            "future_kp_3d": self.data["future_kp_3d"][idx],
            "past_kp_2d": self.data["past_kp_2d"][idx],
            "future_kp_2d": self.data["future_kp_2d"][idx],
            "past_root_3d": self.data["past_root_3d"][idx],
            "future_root_3d": self.data["future_root_3d"][idx],

            # Single original RGB path for the last observed frame
            "last_past_frame_path": last_past_frame_path,
        }

        # Egocentric normalization
        def yaw_to_rot(yaw):
            c, s = np.cos(yaw), np.sin(yaw)
            return np.array([[c, -s],
                            [s,  c]], dtype=np.float32)    

        # Egocentric positions
        current = copy.deepcopy(sample["past_positions"][-1])
        current_rot = yaw_to_rot(sample["past_yaw"][-1])
        Rk_T = current_rot.T

        past_xy = np.array(sample["past_positions"],   dtype=np.float32)[:, :2]
        future_xy = np.array(sample["future_positions"], dtype=np.float32)[:, :2]
        past_yaw   = np.asarray(sample["past_yaw"], dtype=np.float32)
        future_yaw = np.asarray(sample["future_yaw"], dtype=np.float32)

        current = np.array(current, dtype=np.float32)[:2]
        sample["past_positions"]   = (past_xy   - current).dot(current_rot) * self.metric_waypoint_spacing
        sample["future_positions"] = (future_xy - current).dot(current_rot) * self.metric_waypoint_spacing

        # Egocentric keypoints
        for key, pos_xy, yaws in [
            ("past_kp_3d",     past_xy,   past_yaw),
            ("past_root_3d",   past_xy,   past_yaw),
            ("future_kp_3d",   future_xy, future_yaw),
            ("future_root_3d", future_xy, future_yaw),
        ]:
            arr = np.asarray(sample[key], dtype=np.float32)  # [T, ..., 3]
            T = arr.shape[0]
            for t in range(T):
                # R_t from yaw[t]
                Rt = yaw_to_rot(yaws[t])

                # Relative rotation: R_rel = R_k^T * R_t  (2x2)
                Rrel = Rk_T @ Rt

                # Translation in pinned frame: ((pos_xy[t] - current) * R_k) * scale
                t_rel_xy = ((pos_xy[t, :2] - current) @ current_rot) * self.metric_waypoint_spacing

                # Apply to all keypoints/roots at time t (rotate XY, keep Z, then translate)
                flat = arr[t].reshape(-1, 3)            # [M,3]
                xy   = flat[:, :2] @ Rrel.T             # rotate into pinned orientation
                xy  += t_rel_xy                         # translate into pinned coords
                flat[:, :2] = xy
                arr[t] = flat.reshape(arr[t].shape)

            sample[key] = arr

        # Goal direction
        dt = np.random.randint(low=len(sample["future_positions"]) // 2,
                               high=len(sample["future_positions"]))
        goal = copy.deepcopy(sample["future_positions"][dt])
        goal = cartesian_to_polar(goal[0], goal[1])
        goal = goal / (torch.norm(goal) + 0.000006)
        sample["goal_direction"] = goal
        sample["dt"] = torch.Tensor([1 - (1 / dt)])

        return sample


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_file", type=str, required=True,
                        help="Path to samples_train_index.json or samples_val_index.json")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--train", action="store_true", help="Use training augmentations")
    parser.add_argument("--only_human_visable", action="store_true", default=True)
    parser.add_argument("--only_nonlinear", action="store_true", default=False,  # <<< NEW
                        help="Keep only samples marked as non-linear.")
    args = parser.parse_args()

    dataset = SocialNavDataset(
        sample_file=args.sample_file,
        train=args.train,
        only_human_visable=args.only_human_visable,
        only_nonlinear=args.only_nonlinear,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    for batch_idx, sample in enumerate(loader):
        print(f"\n[Batch {batch_idx}]")
        for k, v in sample.items():
            if torch.is_tensor(v):
                print(f"{k}: {tuple(v.shape)}")
            elif isinstance(v, list) and v and torch.is_tensor(v[0]):
                print(f"{k}: list of tensors, each shape {tuple(v[0].shape)}")
            else:
                if k == "last_past_frame_path":
                    # v is a list of strings (one per item in batch)
                    preview = v[:3] if isinstance(v, list) else v
                    print(f"{k}: {preview}{'...' if isinstance(v, list) and len(v)>3 else ''}")
                else:
                    print(f"{k}: {type(v)}")
        if batch_idx >= 1:  # stop after 2 batches
            break
