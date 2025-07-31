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

from utils.helpers import get_conf
from data_parser.musohu_parser import MuSoHuParser
from data_parser.scand_parser import SCANDParser
from data_parser.sbpd_parser import SBPDParser
from utils.parser_utils import poly_fit

from skeleton.predict import PoseEstimatorNode
import cv2

def create_samples(input_path, obs_window: int = 6, pred_window: int = 8, linear_threshold: int = 0.03) -> dict:
    """Create multiple samples from the parsed data folder

    input_path (PosixPath): directory of the parsed trajectory
    obs_window (int): observation window (history)
    pred_window (int): prediction window
    """
    print(f"input_path: {input_path}")
    with input_path.open("rb") as f:
        data = pickle.load(f)

    all_frames = sorted(list([x for x in (input_path.parent / "rgb").iterdir()]), key=lambda x: int(x.name.split(".")[0]))
    traj_len = len(data["position"])
    seq_len = obs_window + pred_window
    positions = []
    goal_positions = []
    yaws = []
    goal_yaws = []
    vws = []
    goal_vws = []
    past_frames = []
    goal_frames = []
    non_linears = []
    non_linear_counter = 0
    for i in range(traj_len - seq_len):
        # past and future positions
        positions.append(data["position"][i : i + obs_window])
        goal_positions.append(data["position"][i + obs_window : i + seq_len])
        # past and future yaw
        yaws.append(data["yaw"][i : i + obs_window])
        goal_yaws.append(data["yaw"][i + obs_window : i + seq_len])
        # past and future vw
        vws.append(data["vw"][i : i + obs_window])
        goal_vws.append(data["vw"][i + obs_window : i + seq_len])
        # store image addresses
        past_frames.append(all_frames[i : i + obs_window])
        goal_frames.append(all_frames[i + obs_window : i + seq_len])
        non_linears.append(poly_fit(np.array(data["position"])[i + obs_window : i + seq_len], pred_window, linear_threshold))
        non_linear_counter += poly_fit(np.array(data["position"])[i + obs_window : i + seq_len], pred_window, linear_threshold)

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
    }
    # print(f"{non_linear_counter = }")
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
    if args.create_samples:
        print(f"Create Samples")
        # # Creating samples
        # save_path = Path(cfg.parsed_dir) / "samples.pkl"
        # # if (save_path).exists():
        # #     print("Save path exists!")
        # #     save_path.rename(f"{cfg.parsed_dir}/{save_path.stem}_new{save_path.suffix}")
        # # List all the pickle files
        # list_pickles = list(save_path.parent.glob("**/traj_data.pkl"))
        # # list_pickles = [x for x in Path(cfg.save_dir).iterdir() if x.suffix == '.pkl']
        # # Base dictionary to store data
        # base_dict = dict()
        # # Iterate over processed files and create samples from them
        # bar = tqdm(list_pickles, desc="Creating samples: ")
        # for file_name in bar:
        #     bar.set_postfix(Trajectory=f"{file_name}")
        #     post_processed = create_samples(
        #         file_name, obs_window=cfg.obs_len, pred_window=cfg.pred_len, linear_threshold=cfg.linear_threshold
        #     )
        #     if bool(base_dict):
        #         base_dict = merge(base_dict, post_processed)
        #     else:
        #         base_dict = post_processed

        # # Saving the final file
        # print(f"Created {len(base_dict['past_frames'])} samples from data in {save_path.parent} directory!")
        # with save_path.open("wb") as f:
        #     pickle.dump(base_dict, f)
    else:
        if dataset == "musohu":
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
        
        elif dataset == "sbpd":
            cfg.scand.update({"sample_rate": cfg.sample_rate})
            cfg.scand.update({"save_dir": cfg.parsed_dir})
            data_parser = SBPDParser(cfg.scenario_based)
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
    
    if args.parse_keypoints:
            # Run skeletal keypoints parsing
            keypoint_model = PoseEstimatorNode()

            processed_dir = Path(cfg.parsed_dir)
            folders = [x for x in processed_dir.iterdir() if x.is_dir()]

            for folder in tqdm(folders):
                rgb_dir = folder / "rgb"
                rgb_images = sorted(rgb_dir.glob("*.jpg"))

                keypoints_output = []
                keypoints_dir = folder / "keypoints"
                keypoints_dir.mkdir(exist_ok=True)

                for idx, img_path in enumerate(tqdm(rgb_images)):
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"Failed to read {img_path}")
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    try:
                        kp_3d, xyz_3d, kp_2d, output_img = keypoint_model.predict(img_rgb)

                        # Save output image as JPEG
                        output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
                        img_save_path = keypoints_dir / img_path.name
                        cv2.imwrite(str(img_save_path), output_img_bgr)

                        # Save keypoint data
                        keypoints_output.append({
                            "keypoints_3d": kp_3d,  # (t, n, 17, 3)
                            "root_3d": xyz_3d,      # (t, n, 3)
                            "keypoints_2d": kp_2d,  # (t, n, 17, 2)
                        })
                    except Exception as e:
                        print(f"Error processing {img_path.name}: {e}")
                        continue

                save_path = keypoints_dir / "keypoints_data.pkl"
                with save_path.open("wb") as f:
                    pickle.dump(keypoints_output, f)
                print(f"[Info] Done parisng {folder}")

