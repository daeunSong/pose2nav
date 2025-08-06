#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header 
from visualization_msgs.msg import Marker, MarkerArray
import cv2
import mmcv
import numpy as np
import time
import argparse

from _mmpose import *
from _mmpose.configs.arg import get_mmpose_parser
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

from _monoloco import *
from _monoloco.config.arg import get_monoloco_parser
from _monoloco.process import load_calibration
from _monoloco.utils import pixel_to_camera_pt
from _monoloco.printer import BirdEyeViewPlot, draw_orientation, draw_uncertainty

class PoseEstimatorNode:
    def __init__(self):
        # Parse ROS arguments
        # self.args = args
        self.img = None

        ## Arguent for MMPOSE
        self.pose_cnt = 0
        self.mmpose_parser = get_mmpose_parser()
        self.mmpose_args, _ = self.mmpose_parser.parse_known_args()
        self.mmpose_predictor = MMPosePredictor(self.mmpose_args)

        # Visualizer
        self.mmpose_visualizer = VISUALIZERS.build(self.mmpose_predictor.pose_estimator.cfg.visualizer)
        self.mmpose_visualizer.set_dataset_meta(self.mmpose_predictor.pose_estimator.dataset_meta)

        # Argment for MonoLoco
        self.monoloco_parser = get_monoloco_parser()
        self.monoloco_args, _ = self.monoloco_parser.parse_known_args()
        self.monoloco_predictor = MonoLocoPredictor(self.monoloco_args)
        self.monoloco_args.focal_length = 5.0
        
        print(f"[INFO] 3D Pose Estimator Node Initialized")

    def transform_points(self, points, translation_vector = [0., 0., 0.], angle_deg = -15):
        """Apply 3D rotation and translation to points.
        """
        angle_rad = np.radians(angle_deg)

        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)

        rotation_matrix = np.array([
            [cos_theta, 0, sin_theta],
            [0,         1, 0        ],
            [-sin_theta, 0, cos_theta]
        ])
        transformed_points = (points @ rotation_matrix.T) + translation_vector
        return transformed_points

    def xyxy_to_xywh(self, bbox, box_score):
        x_min, y_min, x_max, y_max = bbox
        x = x_min
        y = y_min
        w = x_max - x_min
        h = y_max - y_min
        return [x, y, w, h, box_score]

    def preprocess_monoloco(self, pose_est_result):
        # convert mmpose result to monoloco format 
        # boxes = [[xmin, ymin, xmax, ymax, box_score], ...]
        # keypoints = [[[x1, x2, .., x17], [y1, y2, .., y17], [c1, c2, ..., c17]], ...]
        # in image frame, c is a confidence score

        boxes = []
        keypoints = []
        track_id = []
        
        for i, data in enumerate(pose_est_result):
            xs = []
            ys = []
            cs = []

            pred_instances = data.pred_instances.cpu().numpy()
            box_score = pred_instances.bbox_scores[0]
            box = self.xyxy_to_xywh(pred_instances.bboxes[0], box_score)
            
            kps = pred_instances.keypoints[0]
            conf = pred_instances.keypoint_scores[0]

            for j, kp in enumerate(kps):
                xs.append(kp[0])
                ys.append(kp[1])
                cs.append(conf[j])
            boxes.append(box)
            keypoints.append([xs, ys, cs])
            track_id.append(data.track_id)

        # im_size = (data.img_shape[1], data.img_shape[0])

        return boxes, keypoints, track_id

    def generate_topdown_image(self, dic_out, len_3d_keypoints):
        """
        Generate a top-down (bird's-eye) view image from MonoLoco output.

        Args:
            dic_out (dict): Output dictionary from MonoLoco post-processing.
        
        Returns:
            np.ndarray: RGB image as NumPy array (H, W, 3)
        """
        # Create canvas
        z_max = 12 #min(10, 4 + max(el[1] for el in xz_centers))
        plotter = BirdEyeViewPlot(z_max)
        ax = plotter.get_axis()

        if len_3d_keypoints != 0:
            # Extract XZ centers from 3D predictions
            xz_centers = [[xyz[0], xyz[2]] for xyz in dic_out['xyz_pred']]  # [Z, X, Y] → [Z, Y] → [Z, X]

            # Orientation, uncertainty, and color info
            angles = dic_out['angles']
            stds = dic_out['stds_ale']
            colors = ['deepskyblue' for _ in dic_out['uv_heads']]

            # draw
            draw_orientation(ax, xz_centers, [], angles, colors, mode='bird')
            draw_uncertainty(ax, xz_centers, stds)

        # 5. Return image
        return plotter.render()    

    def predict(self, image):

        start = time.time()
        timing = []

        img_mmcv = mmcv.imread(image, channel_order='rgb')

        # STEP1: [mmpose] predict 2D sekeltal keypoints -> pose_est_result
        # + STEP3: [mmpose] 3D pose estimation -> pred_3d_instances
        pose_est_results_list = []
        pose_est_result, _, pred_3d_instances, _  = self.mmpose_predictor.process_one_image(self.mmpose_args, img_mmcv, pose_est_results_list=pose_est_results_list)

        if pred_3d_instances is None or len(pred_3d_instances.keypoints) == 0:
            boxes, keypoints_2d, track_id = [], [], []
            keypoints = np.zeros((0, 17, 3))
            keypoints_3d, xyzs = [], []
            dic_out = {"xyz_pred": []}
        
        else:
            # convert mmpose result to monoloco format -> xyz_pred
            boxes, keypoints_2d, track_id = self.preprocess_monoloco(pose_est_result)

            # STEP2: [monoloco] predict 3D depth
            kk = load_calibration((img_mmcv.shape[1],img_mmcv.shape[0]), self.monoloco_args.focal_length)

            dic_out = self.monoloco_predictor.net.forward(keypoints_2d, kk)
            dic_out = self.monoloco_predictor.net.post_process(
                dic_out, boxes, keypoints_2d, kk)

            xyz_pred = dic_out["xyz_pred"] # 3D depth
            keypoints_2d = np.array(keypoints_2d)[:, :2, :].transpose(0, 2, 1) # reshape (n, 3, 17) -> (n, 17, 2)

            ##### STEP4: post-processing
            # 3D keypoints
            keypoints = pred_3d_instances.keypoints
            keypoints = np.stack([-keypoints[..., 1], keypoints[..., 0], keypoints[..., 2]], axis=-1)   

            fwd_time = (time.time()-start)*1000
            timing.append(fwd_time)  

            xyzs = []
            # [mmpose] + [monoloco] All poses in 3D
            keypoints_3d = [] 
            for i, keypoint in enumerate(keypoints):
                xyz = xyz_pred[i]
                xyz = [xyz[2], -xyz[0], xyz[1] + 0.1]

                # if self.args.robot == 'jackal':
                #     xyz = self.transform_points(xyz, [0., 0., 0.], -15)

                keypoint_3d = []
                translation_vector = xyz #- keypoint[0]

                for j, point in enumerate(keypoint): 
                    kpt_3d = self.transform_points(point - keypoint[0], translation_vector)
                    keypoint_3d.append(kpt_3d)
                keypoints_3d.append(keypoint_3d)
                xyzs.append(xyz)

        topdown_img = self.generate_topdown_image(dic_out, len(keypoints_3d))

        # visualizing 2D output
        input_img = img_mmcv.copy()
        self.mmpose_visualizer.add_datasample(
            name='result',
            image=input_img,
            data_sample=merge_data_samples(pose_est_result),  
            draw_gt=False,
            draw_pred=True,
            show=False,
            draw_bbox=True,
            wait_time=0,
            out_file=None  
        )
        output_img = self.mmpose_visualizer.get_image()

        # draw bbox score
        for res in pose_est_result:
            if hasattr(res.pred_instances, 'bboxes') and hasattr(res.pred_instances, 'bbox_scores'):
                for bbox, score in zip(res.pred_instances.bboxes, res.pred_instances.bbox_scores):
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.putText(
                        output_img,
                        f'{score:.2f}',
                        (x1, y1 - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 255, 0),
                        thickness=1
                    )

        # save the result
        cv2.imwrite('vis_with_bbox_score.jpg', output_img)
        
        return keypoints_3d, xyzs, keypoints_2d, track_id, output_img, topdown_img
        # xyz: root position of humans


if __name__ == '__main__':
    try:
        # args = parse_arguments()
        node = PoseEstimatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
