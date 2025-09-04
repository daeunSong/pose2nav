"""
Parses bag files. Thanks to "GNM: A General Navigation Model to Drive Any Robot" 
by Dhruv Shah et el. paper for open sourcing their code.
link: https://github.com/PrieureDeSion/drive-any-robot
"""
from typing import Any, Union, Callable
import os
from pathlib import Path
import pickle

import numpy as np
from rich import print
from pyntcloud import PyntCloud
from rosbags.rosbag2 import Reader as Rosbag2
from rosbags.serde import deserialize_cdr
from tqdm import tqdm
from scipy.signal import savgol_filter
import pandas as pd

from utils.parser_utils import *

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class SBPDParser:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def process_images(self, im_list: list, img_process_func: Callable) -> list:
        """
        Process image data from a topic that publishes ros images into a list of PIL images
        """
        images = []
        for img_msg in im_list:
            img = img_process_func(img_msg).convert('RGB')
            images.append(img)
        return images

    def process_odom(
        self,
        odom_list: list,
        action_list: list,
        odom_process_func: Any,
        ang_offset: float = 0.0,
    ) -> dict[np.ndarray, np.ndarray]:
        """
        Process odom data from a topic that publishes nav_msgs/Odometry into position and yaw
        """
        xys = []
        yaws = []
        vws = []
        angles = []
        for odom_msg, action_msg in zip(odom_list, action_list):
            xy, vw, yaw, angle = odom_process_func(odom_msg, action_msg, ang_offset)
            xys.append(xy)
            yaws.append(yaw)
            vws.append(vw)
            angles.append(angle)
        return {"position": np.array(xys, dtype=np.float32), "yaw": np.array(yaws, dtype=np.float32), "vw": np.array(vws, dtype=np.float32),
        "angle": np.array(angles, dtype=np.float32)}

    def parse_data(
        self,
        bag: Rosbag2,
        imtopics: Union[list[str], str],
        odomtopics: Union[list[str], str],
        lidartopics: Union[list[str], str],
        actiontopics: Union[list[str], str],
        img_process_func: Any,
        lidar_process_func: Any,
        odom_process_func: Any,
        rate: float = 2.0,
        ang_offset: float = 0.0,
    ):
        """
        Get image, depth, lidar and odom data from a bag file

        Args:
            bag (rosbag.Bag): bag file
            imtopics (Union[list[str], str]): topic name(s) for image data
            odomtopics (Union[list[str], str]): topic name(s) for odom data
            lidartopics (Union[list[str], str]): topic name(s) for lidar data
            actiontopics (Union[list[str], str]): topic name(s) for action data
            img_process_func (Any): function to process image data
            lidar_process_func (Any): function to process lidar data
            odom_process_func (Any): function to process odom data
            rate (float, optional): rate to sample data. Defaults to 4.0.
            ang_offset (float, optional): angle offset to add to odom data. Defaults to 0.0.
        Returns:
            img_data (list): list of PIL images
            lidar_data (list): list of np.arrays
            traj_data (list): list of odom and linear/angular velocity data
        """
        # check if bag has both topics
        imtopic = self.cfg.topics.rgb[0]
        odomtopic = self.cfg.topics.odom[0]
        actiontopic = self.cfg.topics.cmd_vel[0]
        pctopic = self.cfg.topics.lidar[0]

        # get start time of bag in seconds
        # currtime = bag.get_start_time()
        # starttime = currtime
        # print(f"{starttime = }")

        all_data = []
        curr_imdata = None
        curr_odomdata = None
        curr_actiondata = None
        curr_pcdata = None
        times = []

        starttime = None

        for connection, timestamp, rawdata in bag.messages():
            t = timestamp / 1e9
            if starttime == None:
                currtime = t
                starttime = currtime
            if t - starttime < self.cfg.skip_first_seconds:
                # skip the first few seconds
                continue
            
            topic = connection.topic
            try:
                if topic == imtopic:
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    curr_imdata = msg
                    curr_timestamp = t
                elif topic == odomtopic:
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    curr_odomdata = msg
                elif topic == actiontopic:
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    curr_actiondata = msg
                else:
                    # Shouldn't happen due to connections filter, but be defensive:
                    continue
            except Exception as e:
                print(f"[WARN] Skipping message on {topic} at {t:.3f}s: {e}")
                continue

            if all(x is not None for x in [curr_imdata, curr_odomdata, curr_actiondata]):
                all_data.append((curr_timestamp, curr_imdata, curr_odomdata, curr_actiondata))
                curr_imdata = curr_odomdata = curr_actiondata = curr_timestamp = None

        synced_imdata = []
        synced_odomdata = []
        synced_actiondata = []
        synced_pcdata = []
        last_time = None

        for t, im, odom, action in sorted(all_data, key=lambda x: x[0]):
            if last_time is None or (t - last_time) >= 1.0 / rate:
                synced_imdata.append(im)
                synced_odomdata.append(odom)
                synced_actiondata.append(action)
                # synced_pcdata.append(pc)
                last_time = t

        img_data = self.process_images(synced_imdata, img_process_func)
        # pc_data = synced_pcdata
        pc_data = None  
        traj_data = self.process_odom(
            synced_odomdata,
            synced_actiondata,
            odom_process_func,
            ang_offset=ang_offset,
        )

        # smooth pos and actions
        # traj_data["yaw"] = savgol_filter(
        #     traj_data["yaw"], window_length=31, polyorder=3, mode="nearest"
        # )
        traj_data["vw"][:, 0] = savgol_filter(
            traj_data["vw"][:, 0], window_length=31, polyorder=3, mode="nearest"
        )
        traj_data["vw"][:, 1] = savgol_filter(
            traj_data["vw"][:, 1], window_length=31, polyorder=3, mode="nearest"
        )
        traj_data["position"][:, 0] = savgol_filter(
            traj_data["position"][:, 0], window_length=31, polyorder=3, mode="nearest"
        )
        traj_data["position"][:, 1] = savgol_filter(
            traj_data["position"][:, 1], window_length=31, polyorder=3, mode="nearest"
        )

        return img_data, traj_data, pc_data

    def parse_bags(self, bag_path) -> None:
        # id = 0
        # bag_files = Path(self.cfg.bags_dir).resolve()
        save_dir = Path(self.cfg.save_dir)
        # bag_files = [str(x) for x in bag_files.iterdir() if x.suffix == ".bag"]

        try:
            b = Rosbag2(bag_path.parent)
            b.open()
            # print(f"Bag opened from path: {bag_path}")
        except Exception as e:
            print(e)
            print(f"Error loading {bag_path.parent}. Skipping...")
            return
        # name is that folders separated by _ and then the last part of the path
        traj_name = "_".join(bag_path.stem.split("_")[:-1])

        # parse data
        (
            bag_img_data,
            bag_traj_data,
            bag_pc_data,
        ) = self.parse_data(
            b,
            self.cfg.topics.rgb,
            self.cfg.topics.odom,
            self.cfg.topics.lidar,
            self.cfg.topics.cmd_vel,
            eval(self.cfg.functions.rgb),
            eval(self.cfg.functions.lidar),
            eval(self.cfg.functions.odom),
            rate=self.cfg.sample_rate / 2, # scout too slow
            ang_offset=self.cfg.ang_offset, 
        )

        if bag_img_data is None or bag_traj_data is None:
            print(
                f"{bag_path} did not have the topics we were looking for. Skipping..."
            )
            return
        # print(f"Working on bag: {bag_path}")
        # remove backwards movement
        cut_trajs = filter_backwards_scand(bag_img_data, bag_traj_data, bag_pc_data)
        # for i, (img_data_i, traj_data_i, pc_data_i) in enumerate(cut_trajs):
        for i, (img_data_i, traj_data_i) in enumerate(cut_trajs):
            if len(img_data_i) < self.cfg.skip_traj_shorter:
                # skip trajectories with less than 20 frames
                continue

            traj_name_i = f"{traj_name}_{i}"
            traj_folder_i = save_dir / traj_name_i
            output_rgb = str(traj_folder_i / "rgb")
            output_pc = str(traj_folder_i / "point_cloud")
            # attach frame names to the pickle file
            traj_data_i['img_path'] = [f"{Path(output_rgb).parent.stem}/rgb/{x}.jpg" for x, _ in enumerate(img_data_i)]
            # create a data frame with each row contains one sample
            refine_data = list(
                zip(range(len(traj_data_i['position'])), traj_data_i['position'], traj_data_i['yaw'], traj_data_i['vw'], traj_data_i['angle'], traj_data_i['img_path']))
            pd_frame = pd.DataFrame(refine_data, columns=['frame', 'position', 'yaw', 'vw', 'angle', 'img_path'])
            # make a folder for the traj
            if not os.path.exists(traj_folder_i):
                os.makedirs(traj_folder_i)
            if not os.path.exists(output_rgb):
                os.makedirs(output_rgb)
            if not os.path.exists(output_pc):
                os.makedirs(output_pc)
            with open(str(traj_folder_i / "traj_data.pkl"), "wb") as f:
                pickle.dump(traj_data_i, f)
            # also save csv version
            pd_frame.to_csv(Path(traj_folder_i / "traj_dataframe.csv").as_posix(), index=False)
            # save the image data to disk
            for j, img in enumerate(img_data_i):
                img.save(os.path.join(output_rgb, f"{j}.jpg"))
            # save the pc data to disk
            # for j, pc in enumerate(pc_data_i):
            #     pc = PyntCloud(pc)
            #     pc.to_file(os.path.join(output_pc, f"{j}.ply"))
        b.close()
