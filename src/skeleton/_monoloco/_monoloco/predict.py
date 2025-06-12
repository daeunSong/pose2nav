import torch
import argparse
import yaml
import numpy as np
import cv2
import time

from .network import Loco
from .process import preprocess_pifpaf, load_calibration
from .utils import pixel_to_camera_pt

import openpifpaf
from openpifpaf import decoder, network, visualizer, show, logger, Predictor

class MonoLocoPredictor:
    def __init__(self, args, device='cuda'):
        """
        MonoLoco++ Predictor Wrapper using .pkl model
        """
        self.args = args
        self.args.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"device = {self.args.device}")
        self.camera_intrinsic = self.load_intrinsics(self.args.conf, self.args.robot)

        self.image = None
        self.net = Loco(
            model=self.args.model,
            device=self.args.device,
            n_dropout=self.args.n_dropout,
            p_dropout=self.args.dropout)

        # config openpifpaf
        self.args.batch_size = 1
        self.args.force_complete_pose = True
        self.args.pin_memory = True if torch.cuda.is_available() else False

        decoder.configure(self.args)
        network.Factory.configure(self.args)
        Predictor.configure(self.args)

        # openpifpaf predictor
        self.predictor = Predictor(checkpoint=self.args.checkpoint)


    def predict (self, image = None):
        
        # if image == None:
        #     #### TEMPORARY
        #     # Read image using OpenCV (BGR format)
        #     image_bgr = cv2.imread('11.jpg')
        #     if image_bgr is None:
        #         raise FileNotFoundError(f"Image not found at {image_path}")
        #     self.image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # else:
        #     self.image = image
        self.image = image
        # self.net = Loco(
        #     model=self.args.model,
        #     device=self.args.device,
        #     n_dropout=self.args.n_dropout,
        #     p_dropout=self.args.dropout)
        
        # === openpifpaf 2D keypoints predictions ===
        pifpaf_outs = {}
        start = time.time()
        timing = []

        pred, _, meta = self.predictor.numpy_image(self.image)
        pifpaf_outs['pred'] = pred
        pifpaf_outs['data'] = [ann.json_data() for ann in pred]
        pifpaf_outs['width_height'] = meta['width_height']

        im_size = (float(pifpaf_outs['width_height'][0]), float(pifpaf_outs['width_height'][1]))
        kk = load_calibration(im_size, self.args.focal_length)

        print(f"im_size = {im_size}")

        # Preprocess pifpaf outputs and run monoloco
        boxes, keypoints = preprocess_pifpaf(
            pifpaf_outs['data'], im_size, min_conf=self.args.min_conf)   
        
        # === Monoloco++ 3D keypoints predictions ===
        dic_out = self.net.forward(keypoints, kk)
        fwd_time = (time.time()-start)*1000
        timing.append(fwd_time)  
        print(f"Forward time: {fwd_time:.0f} ms")

        dic_out = self.net.post_process(
            dic_out, boxes, keypoints, kk)

        xyz_pred = dic_out["xyz_pred"]
        keypoints = np.transpose(np.array(keypoints), (0, 2, 1)) ## [[x, y], [x, y], [x, y]]
        keypoints_3d = []

        # === All keypoints to 3D ===
        for i, keypoint in enumerate(keypoints):
            center_2d = dic_out['uv_centers'][i]
            xyz = xyz_pred[i]

            center_3d = pixel_to_camera_pt(center_2d[0], center_2d[1], xyz[2], self.camera_intrinsic, self.args.scale)
            translation_vector = [xyz[2], -xyz[0], xyz[1]] - center_3d
            keypoint_3d = []

            for j, point in enumerate(keypoint):
                pt = pixel_to_camera_pt(point[0], point[1], xyz[2], self.camera_intrinsic, self.args.scale)
                pt = pt + translation_vector
                keypoint_3d.append(pt)

            keypoints_3d.append(keypoint_3d)
        
        return keypoints_3d, xyz_pred

    def load_intrinsics(self, yaml_path, robot_name):
        """
        Load camera intrinsics from a YAML file.

        Args:
            yaml_path (str): Path to the YAML file.
            robot_name (str): Key name (e.g., 'scand_spot').

        Returns:
            np.ndarray: 3x3 intrinsics matrix
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if robot_name not in data:
            raise ValueError(f"Robot '{robot_name}' not found in YAML file.")

        intrinsics_list = data[robot_name]['intrinsics']

        return np.array(intrinsics_list, dtype=np.float32)

if __name__ == "__main__":

    from config import cli
    args = cli()
    
    # === Monoloco++ predictor ===
    predictor = MonoLocoPredictor(args)
    keypoints = predictor.predict()

