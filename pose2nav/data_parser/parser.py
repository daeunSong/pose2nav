"""
Main script to parse bag files.
"""
import os
import argparse
import pickle
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map
import numpy as np
import json

from utils.helpers import get_conf
from utils.parser_utils import poly_fit
from collections import defaultdict

from skeleton.predict import PoseEstimatorNode
import cv2


def append_pickle(file_path, new_data):
    """Append new_data (list) to a pickle file containing a list."""
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    else:
        data = []
    data.extend(new_data)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
        
def create_samples(input_path, obs_window: int = 6, pred_window: int = 8, linear_threshold: int = 0.03, num_ped: int = 6, num_joints: int = 17) -> dict:
    """Create multiple samples from the parsed data folder

    input_path (PosixPath): directory of the parsed trajectory
    obs_window (int): observation window (history)
    pred_window (int): prediction window
    """
    print(f"input_path: {input_path}")
    with input_path.open("rb") as f:
        traj_data = pickle.load(f)
    
    # Load keypoints
    keypoints_path = input_path.parent / "keypoints" / "keypoints_data.pkl"
    with keypoints_path.open("rb") as f:
        keypoints_data = pickle.load(f)["keypoints"]

    all_frames = sorted(
        list([x for x in (input_path.parent / "rgb").iterdir()]), 
        key=lambda x: int(x.name.split(".")[0])
    )
    traj_len = len(traj_data["position"])
    seq_len = obs_window + pred_window

    # Lists to store sample chunks
    positions, goal_positions = [], []  # [T, 3]
    yaws, goal_yaws = [], []            # [T, 1]
    vws, goal_vws = [], []              # [T, 2]
    past_frames, goal_frames = [], []   # [T, 1]
    non_linears = []

    # Lists to store ketpoints
    past_kp_3d, future_kp_3d = [], []   # [T, N, 17, 3]
    past_kp_2d, future_kp_2d = [], []   # [T, N, 17, 2]    
    past_root_3d, future_root_3d = [], []   # [T, N, 3]
    has_humans = []                    

    non_linear_counter = 0
    has_human_counter = 0
    
    for i in range(traj_len - seq_len):

        # Reset tracking
        track_3d = defaultdict(lambda: [None] * seq_len)
        track_2d = defaultdict(lambda: [None] * seq_len)
        track_root = defaultdict(lambda: [None] * seq_len)
        has_human = False
        # Loop through all frames in the current window
        for t in range(seq_len):  # local frame index within window
            j = i + t
            frame_key = f"{j}.jpg"
            humans = keypoints_data.get(frame_key, [])

            for human in humans:
                label_id = human["label_id"]
                if "keypoints_3d" in human:
                    k3d = np.stack(human["keypoints_3d"], axis=0)  # (17, 3)
                    k2d = np.array(human["keypoints_2d"], dtype=np.float32)  # (17, 2)
                    root = np.array(human["root_3d"], dtype=np.float32)  # (3,)
                    track_3d[label_id][t] = k3d
                    track_2d[label_id][t] = k2d
                    track_root[label_id][t] = root
                    if t < obs_window:
                        has_human = True
            
        sorted_ids = sorted(track_3d.items(), key=lambda item: sum(x is not None for x in item[1][:obs_window]), reverse=True)
        selected_ids = [k for k, _ in sorted_ids[:num_ped]]

        # Initialize output tensors
        keypoints_3d = np.zeros((seq_len, num_ped, num_joints, 3), dtype=np.float32)
        keypoints_2d = np.zeros((seq_len, num_ped, num_joints, 2), dtype=np.float32)
        root_3d = np.zeros((seq_len, num_ped, 3), dtype=np.float32)

        for n, label_id in enumerate(selected_ids): # n = 0..num_ped-1
            for t in range(seq_len):    # t = 0..T-1
                if track_3d[label_id][t] is not None:
                    keypoints_3d[t, n] = track_3d[label_id][t]
                if track_2d[label_id][t] is not None:
                    keypoints_2d[t, n] = track_2d[label_id][t]
                if track_root[label_id][t] is not None:
                    root_3d[t, n] = track_root[label_id][t]

        # past and future positions
        positions.append(traj_data["position"][i : i + obs_window])
        goal_positions.append(traj_data["position"][i + obs_window : i + seq_len])
        # past and future yaw
        yaws.append(traj_data["yaw"][i : i + obs_window])
        goal_yaws.append(traj_data["yaw"][i + obs_window : i + seq_len])
        # past and future vw
        vws.append(traj_data["vw"][i : i + obs_window])
        goal_vws.append(traj_data["vw"][i + obs_window : i + seq_len])
        # store image addresses
        past_frames.append(all_frames[i : i + obs_window])
        goal_frames.append(all_frames[i + obs_window : i + seq_len])
        is_nonlinear = poly_fit(np.array(traj_data["position"])[i + obs_window : i + seq_len], pred_window, linear_threshold)
        non_linears.append(is_nonlinear)
        non_linear_counter += is_nonlinear
        has_human_counter += int(has_human)

        past_kp_3d.append(keypoints_3d[:obs_window])
        future_kp_3d.append(keypoints_3d[obs_window:])
        past_kp_2d.append(keypoints_2d[:obs_window])
        future_kp_2d.append(keypoints_2d[obs_window:])
        past_root_3d.append(root_3d[:obs_window])
        future_root_3d.append(root_3d[obs_window:])
        has_humans.append(int(has_human))
        

        # print(f"frames = {all_frames[i : i + obs_window]}")
        # frames = [PosixPath('data/processed/WW_FairOaks_9_neutral_0/rgb/436.jpg'), PosixPath('data/processed/WW_FairOaks_9_neutral_0/rgb/437.jpg'), PosixPath('data/processed/WW_FairOaks_9_neutral_0/rgb/438.jpg'), PosixPath('data/processed/WW_FairOaks_9_neutral_0/rgb/439.jpg'), PosixPath('data/processed/WW_FairOaks_9_neutral_0/rgb/440.jpg'), PosixPath('data/processed/WW_FairOaks_9_neutral_0/rgb/441.jpg')]

    post_processed = {
        "past_positions": positions,
        "future_positions": goal_positions,
        "past_yaw": yaws,
        "future_yaw": goal_yaws,
        "past_vw": vws,
        "future_vw": goal_vws,
        "past_frames": past_frames,
        "future_frames": goal_frames,
        "non_linear": non_linears,
        "past_kp_3d": past_kp_3d,
        "future_kp_3d": future_kp_3d,
        "past_kp_2d": past_kp_2d,
        "future_kp_2d": future_kp_2d,
        "past_root_3d": past_root_3d,
        "future_root_3d": future_root_3d,
        "has_humans": has_humans,
    }
    # print(f"{non_linear_counter = }")
    print(f"has_human_counter = {has_human_counter}/{len(has_humans)}")
    return post_processed


def merge(base_dict: dict, new_dict: dict):
    """Merges two dictionary together

    base_dict (dict): The base dictionary to be updated
    new_dict (dict): The new data to be added to the base dictionary
    """
    # assert base_dict is None, "Base dictionary cannot be None"
    assert (
        base_dict.keys() == new_dict.keys()
    ), "The two dictionaries must have the same keys"
    for key in base_dict.keys():
        base_dict[key].extend(new_dict[key])

    return base_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        default="scand",
        type=str,
        help="Dataset name.",
    )
    parser.add_argument(
        "-c",
        "--conf",
        default="../conf/parser",
        type=str,
        help="Config file address.",
    )
    parser.add_argument(
        "-cs",
        "--create_samples",
        action="store_true",
        help="Create samples. Applicable only after parsing bags.",
    )
    parser.add_argument(
        "-pk",
        "--parse_keypoints",
        action="store_true",
        help="Run skeletal keypoints parsing.",
    )
    args, _ = parser.parse_known_args()
    print("args:", args)

    cfg_dir = args.conf
    cfg = get_conf(cfg_dir)
    # dataset = "musohu" if "musohu" in cfg_dir.lower() else "scand"
    dataset = args.name

    if args.parse_keypoints:
        # Run skeletal keypoints parsing
        keypoint_model = PoseEstimatorNode()

        processed_dir = Path(cfg.parsed_dir)
        folders = [x for x in processed_dir.iterdir() if x.is_dir()]

        for folder in tqdm(folders):
            rgb_dir = folder / "rgb"
            rgb_images = sorted(rgb_dir.glob("*.jpg"))

            keypoints_output = {"keypoints": {}}
            keypoints_dir = folder / "keypoints"
            keypoints_dir.mkdir(exist_ok=True)
            topdown_img_dir = keypoints_dir / "topdown"
            topdown_img_dir.mkdir(exist_ok=True)

            for idx, img_path in enumerate(tqdm(rgb_images)):
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Failed to read {img_path}")
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # print(f"[Info] Working on {img_path}")

                try:
                    kp_3d, xyz_3d, kp_2d, track_id, output_img, topdown_img = keypoint_model.predict(img_rgb)

                    # Should all match the number of pedestrian
                    assert len(kp_3d) == len(xyz_3d) == len(kp_2d) == len(track_id) 

                    # Save output image as JPEG
                    output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
                    img_save_path = keypoints_dir / img_path.name
                    cv2.imwrite(str(img_save_path), output_img_bgr)

                    img_save_path = topdown_img_dir / img_path.name
                    cv2.imwrite(str(img_save_path), topdown_img)

                    # Prepare entry for current image
                    per_image_output = []
                    for i in range(len(kp_3d)):
                        per_image_output.append({
                            "label_id": track_id[i],
                            "keypoints_3d": kp_3d[i],     # (17, 3)
                            "root_3d": xyz_3d[i],         # (3,)
                            "keypoints_2d": kp_2d[i],     # (17, 2)
                        })

                    # Add to main dictionary
                    keypoints_output["keypoints"][img_path.name] = per_image_output

                except Exception as e:
                    print(f"Error processing {img_path.name}: {e}")
                    continue

            # Sort keypoints_output["keypoints"] by numeric prefix
            sorted_keypoints = dict(sorted(
                keypoints_output["keypoints"].items(),
                key=lambda item: int(item[0].split(".")[0])  # extract '0' from '0.jpg'
            ))
            keypoints_output["keypoints"] = sorted_keypoints

            # Save final dictionary as pickle
            save_path = keypoints_dir / "keypoints_data.pkl"
            with save_path.open("wb") as f:
                pickle.dump(keypoints_output, f)
                print(f"[Info] Done parisng {folder}")
            # save_path = keypoints_dir / "keypoints_data.json"
            # with save_path.open("w") as f:
            #     # Ensure NumPy arrays are converted to lists
            #     def tolist_handler(obj):
            #         if isinstance(obj, (np.ndarray,)):
            #             return obj.tolist()
            #         raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            #     json.dump(keypoints_output, f, indent=2, default=tolist_handler)
                print(f"[Info] Done parisng {folder}")
    else:                 
        if args.create_samples:
            save_dir = Path(cfg.parsed_dir)
            samples_dir = save_dir.parent / "samples"
            samples_dir.mkdir(exist_ok=True, parents=True)

            shard_size = 5000  # samples per shard file
            shard_idx = 0
            buffer = None
            shard_info = []  # will store {"file": ..., "count": ...} for each shard

            def save_shard(data_dict, shard_idx):
                shard_path = samples_dir / f"samples_{shard_idx:04d}.pkl"
                with shard_path.open("wb") as f:
                    pickle.dump(data_dict, f)
                shard_info.append({"file": str(shard_path), "count": len(data_dict["past_frames"])})
                print(f"[Info] Saved shard {shard_idx} with {len(data_dict['past_frames'])} samples")

            list_pickles = list(save_dir.glob("**/traj_data.pkl"))
            bar = tqdm(list_pickles, desc="Creating samples: ")

            for file_name in bar:
                bar.set_postfix(Trajectory=f"{file_name}")
                post_processed = create_samples(
                    file_name,
                    obs_window=cfg.obs_len,
                    pred_window=cfg.pred_len,
                    linear_threshold=cfg.linear_threshold,
                    num_ped=cfg.num_ped
                )

                if buffer is None:
                    buffer = post_processed
                else:
                    buffer = merge(buffer, post_processed)

                if len(buffer["past_frames"]) >= shard_size:
                    save_shard(buffer, shard_idx)
                    shard_idx += 1
                    buffer = None

            if buffer is not None and len(buffer["past_frames"]) > 0:
                save_shard(buffer, shard_idx)

            # 🔹 Randomly split shard files into train and val (80/20)
            indices = np.arange(len(shard_info))
            np.random.shuffle(indices)
            split_point = int(len(indices) * 0.8)
            train_shards = [shard_info[i] for i in indices[:split_point]]
            val_shards = [shard_info[i] for i in indices[split_point:]]

            # 🔹 Save train/val index JSON files
            train_index_path = samples_dir / "samples_train_index.json"
            val_index_path = samples_dir / "samples_val_index.json"
            with train_index_path.open("w") as f:
                json.dump(train_shards, f, indent=2)
            with val_index_path.open("w") as f:
                json.dump(val_shards, f, indent=2)

            print(f"[Info] Train index saved to {train_index_path}")
            print(f"[Info] Val index saved to {val_index_path}")

        # Data Parsing
        else:
            if dataset == "musohu":
                from data_parser.musohu_parser import MuSoHuParser
                cfg.musohu.update({"sample_rate": cfg.sample_rate})
                cfg.musohu.update({"save_dir": cfg.parsed_dir})
                data_parser = MuSoHuParser(cfg.musohu)
                bag_files = Path(data_parser.cfg.bags_dir).resolve()
                bag_files = [str(x) for x in bag_files.iterdir() if x.suffix == ".bag"]
                # if there are ram limitations, reduce the number of max_workers
                # process_map(data_parser.parse_bags, bag_files, max_workers=os.cpu_count() - 4)
                for bag in tqdm(bag_files):
                    try:
                        data_parser.parse_bags(bag)
                        print(f"[Info] Done parisng {bag}")
                    except Exception as e:
                        print(f"[ERROR] Crashed on {bag}")

            elif dataset == "scand":
                from data_parser.scand_parser import SCANDParser
                cfg.scand.update({"sample_rate": cfg.sample_rate})
                cfg.scand.update({"save_dir": cfg.parsed_dir})
                data_parser = SCANDParser(cfg.scand)
                bag_files = Path(data_parser.cfg.bags_dir).resolve()
                bag_files = [str(x) for x in bag_files.iterdir() if x.suffix == ".bag"]
                # process_map(data_parser.parse_bags, bag_files, max_workers=os.cpu_count() - 4)
                for bag in tqdm(bag_files):
                    try:
                        data_parser.parse_bags(bag)
                        print(f"[Info] Done parisng {bag}")
                    except Exception as e:
                        print(f"[ERROR] Crashed on {bag}")
            
            # TODO: need to fix it into ROS2
            elif dataset == "sbpd":
                from data_parser.sbpd_parser import SBPDParser
                cfg.scand.update({"sample_rate": cfg.sample_rate})
                cfg.scand.update({"save_dir": cfg.parsed_dir})
                data_parser = SBPDParser(cfg.sbpd)
                bag_files = Path(data_parser.cfg.bags_dir).resolve()
                bag_files = [str(x) for x in bag_files.iterdir() if x.suffix == ".db3"]
                # process_map(data_parser.parse_bags, bag_files, max_workers=os.cpu_count() - 4)
                for bag in tqdm(bag_files):
                    try:
                        data_parser.parse_bags(bag)
                        print(f"[Info] Done parisng {bag}")
                    except Exception as e:
                        print(f"[ERROR] Crashed on {bag}")
            else:
                raise Exception("Invalid dataset!")