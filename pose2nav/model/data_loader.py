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
        index_file: str,
        train: bool = True,
        resize: Union[list, tuple] = (224, 224),
        metric_waypoint_spacing: float = 1.0,
        only_human_visable: bool = True,
        only_nonlinear: bool = True,         
        cache_size: int = 1                   # keep this small to save RAM
    ):
        """
        Dataloader that loads from an index JSON file of shards.
        Loads shards lazily to avoid huge memory usage.
        """
        self.resize = resize
        self.metric_waypoint_spacing = metric_waypoint_spacing
        self.train = train
        self.only_human_visable = only_human_visable
        self.only_nonlinear = only_nonlinear  
        self.cache_size = cache_size
        self.shard_cache = {}  # shard_path -> loaded shard dict

        # Load the index JSON
        with open(index_file, "r") as f:
            self.index_info = json.load(f)

        # Build flat (shard_file, local_idx) list (apply filters per-shard)
        self.samples = []
        for shard in self.index_info:
            shard_path = Path(shard["file"])
            with shard_path.open("rb") as f:
                shard_data = pickle.load(f)

            total = len(shard_data["past_frames"])
            mask = np.ones(total, dtype=bool)

            if self.only_human_visable:
                hv = np.asarray(shard_data["has_humans"]).astype(bool)
                if hv.shape[0] != total:
                    raise ValueError(f"has_humans length mismatch in {shard_path}")
                mask &= hv

            if self.only_nonlinear:
                # parser stores key as "non_linear" (singular)
                nl = np.asarray(shard_data.get("non_linear", [False] * total)).astype(bool)
                if nl.shape[0] != total:
                    raise ValueError(f"non_linear length mismatch in {shard_path}")
                mask &= nl

            indices = np.nonzero(mask)[0]

            for local_idx in indices:
                self.samples.append((str(shard_path), int(local_idx)))

            # free immediately (don’t keep shard_data)
            del shard_data

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

    def __len__(self):
        return len(self.samples)

    def _load_shard(self, shard_path):
        """Load shard from disk with LRU cache."""
        if shard_path in self.shard_cache:
            return self.shard_cache[shard_path]

        with open(shard_path, "rb") as f:
            shard_data = pickle.load(f)

        # Manage cache size
        if len(self.shard_cache) >= self.cache_size:
            self.shard_cache.pop(next(iter(self.shard_cache)))
        self.shard_cache[shard_path] = shard_data
        return shard_data

    def __getitem__(self, idx):
        shard_path, local_idx = self.samples[idx]
        shard_data = self._load_shard(shard_path)

        # last past-frame path (string) for plotting later
        past_paths = shard_data["past_frames"][local_idx]  # list[Path-like]
        last_past_frame_path = str(past_paths[-1]) if len(past_paths) > 0 else None

        past_frames = [
            self.transform(imread(str(p)).convert("RGB"))
            for p in past_paths
        ]

        future_paths = shard_data["future_frames"][local_idx]
        future_frames = [
            self.transform(imread(str(p)).convert("RGB"))
            for p in future_paths
        ]

        sample = {
            "past_positions": shard_data["past_positions"][local_idx],
            "future_positions": shard_data["future_positions"][local_idx],
            "past_yaw": shard_data["past_yaw"][local_idx],
            "future_yaw": shard_data["future_yaw"][local_idx],
            "past_vw": shard_data["past_vw"][local_idx],
            "future_vw": shard_data["future_vw"][local_idx],
            "past_frames": past_frames,
            "future_frames": future_frames,
            "past_kp_3d": shard_data["past_kp_3d"][local_idx],
            "future_kp_3d": shard_data["future_kp_3d"][local_idx],
            "past_kp_2d": shard_data["past_kp_2d"][local_idx],
            "future_kp_2d": shard_data["future_kp_2d"][local_idx],
            "past_root_3d": shard_data["past_root_3d"][local_idx],
            "future_root_3d": shard_data["future_root_3d"][local_idx],

            # NEW: single original RGB path for the last observed frame
            "last_past_frame_path": last_past_frame_path,
        }

        # Egocentric normalization
        current = copy.deepcopy(sample["past_positions"][-1])
        rot = np.array([
            [np.cos(sample["past_yaw"][-1]), -np.sin(sample["past_yaw"][-1])],
            [np.sin(sample["past_yaw"][-1]),  np.cos(sample["past_yaw"][-1])],
        ], dtype=np.float32)
        sample["past_positions"]   = np.array(sample["past_positions"],   dtype=np.float32)[:, :2]
        sample["future_positions"] = np.array(sample["future_positions"], dtype=np.float32)[:, :2]
        current = np.array(current, dtype=np.float32)[:2]
        sample["past_positions"]   = (sample["past_positions"]   - current).dot(rot) * self.metric_waypoint_spacing
        sample["future_positions"] = (sample["future_positions"] - current).dot(rot) * self.metric_waypoint_spacing

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
    parser.add_argument("--index_file", type=str, required=True,
                        help="Path to samples_train_index.json or samples_val_index.json")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--train", action="store_true", help="Use training augmentations")
    parser.add_argument("--only_human_visable", action="store_true", default=True)
    parser.add_argument("--only_nonlinear", action="store_true", default=False,  # <<< NEW
                        help="Keep only samples marked as non-linear.")
    args = parser.parse_args()

    dataset = SocialNavDataset(
        index_file=args.index_file,
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
