#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage
from visualization_msgs.msg import Marker, MarkerArray
import cv2

import logging
import mimetypes
import os
import time
from functools import partial

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import (_track_by_iou, _track_by_oks,
                         convert_keypoint_definition, extract_pose_sequence,
                         inference_pose_lifter_model, inference_topdown,
                         init_model)
from mmpose.models.pose_estimators import PoseLifter
from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator
from mmpose.registry import VISUALIZERS
from mmpose.structures import (PoseDataSample, merge_data_samples,
                               split_instances)
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


"""
    Pipeline of the method:

                            frame
                                |
                                V
                        +-----------------+
                        |     detector    |
                        +-----------------+
                                |  det_result
                                V
                        +-----------------+
                        |  pose_estimator |
                        +-----------------+
                                |  pose_est_results
                                V
            +--------------------------------------------+
            |  convert 2d kpts into pose-lifting format  |
            +--------------------------------------------+
                                |  pose_est_results_list
                                V
                    +-----------------------+
                    | extract_pose_sequence |
                    +-----------------------+
                                |  pose_seq_2d
                                V
                        +-------------+
                        | pose_lifter |
                        +-------------+
                                |  pose_lift_results
                                V
                    +-----------------+
                    | post-processing |
                    +-----------------+
                                |  pred_3d_data_samples
                                V
                        +------------+
                        | visualizer |
                        +------------+
"""

class MMPosePredictor:
    def __init__(self, args):       
        ## Argument for MMPOSE
        self.pose_cnt = 0
        self.args = args

        self.detector = init_detector(
            self.args.det_config, self.args.det_checkpoint, device=self.args.device.lower())
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

        self.pose_estimator = init_model(
            self.args.pose_estimator_config,
            self.args.pose_estimator_checkpoint,
            device=self.args.device.lower())

        det_kpt_color = self.pose_estimator.dataset_meta.get('keypoint_colors', None)
        det_dataset_skeleton = self.pose_estimator.dataset_meta.get(
            'skeleton_links', None)
        self.det_dataset_link_color = self.pose_estimator.dataset_meta.get(
            'skeleton_link_colors', None)

        self.pose_lifter = init_model(
            self.args.pose_lifter_config,
            self.args.pose_lifter_checkpoint,
            device=self.args.device.lower())
        self.lift_dataset_link_color = self.pose_lifter.dataset_meta.get(
            'keypoint_colors', None)

        self.pose_lifter.cfg.visualizer.radius = self.args.radius
        self.pose_lifter.cfg.visualizer.line_width = self.args.thickness
        self.pose_lifter.cfg.visualizer.det_kpt_color = det_kpt_color
        self.pose_lifter.cfg.visualizer.det_dataset_skeleton = det_dataset_skeleton
        self.pose_lifter.cfg.visualizer.det_dataset_link_color = self.det_dataset_link_color

    def pose_est_2d(self, args, frame):
        """Visualize detected and predicted keypoints of one image.

        Pipeline of this function:

                                frame
                                    |
                                    V
                            +-----------------+
                            |     detector    |
                            +-----------------+
                                    |  det_result
                                    V
                            +-----------------+
                            |  pose_estimator |
                            +-----------------+
                                    |  pose_est_results
                                    V
        """
        # First stage: conduct 2D pose detection in a Topdown manner
        # use detector to obtain person bounding boxes
        det_result = inference_detector(self.detector, frame)
        pred_instance = det_result.pred_instances.cpu().numpy()

        # filter out the person instances with category and bbox threshold
        # e.g. 0 for person in COCO
        bboxes = pred_instance.bboxes
        bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                    pred_instance.scores > args.bbox_thr)]

        # estimate pose results for current image
        pose_est_results = inference_topdown(self.pose_estimator, frame, bboxes)

        return pose_est_results


    def pose_est_3d(self, args, frame, pose_est_results, frame_idx = 0, 
                        pose_est_results_last = [], pose_est_results_list = [], next_id = 0,
                        visualizer = None):
        """
                                |  pose_est_results_list
                                V
                    +-----------------------+
                    | extract_pose_sequence |
                    +-----------------------+
                                |  pose_seq_2d
                                V
                        +-------------+
                        | pose_lifter |
                        +-------------+
                                |  pose_lift_results
                                V
                    +-----------------+
                    | post-processing |
                    +-----------------+
                                |  pred_3d_data_samples
                                V
        """

        pose_lift_dataset = self.pose_lifter.cfg.test_dataloader.dataset
        pose_lift_dataset_name = self.pose_lifter.dataset_meta['dataset_name']

        pose_det_dataset_name = self.pose_estimator.dataset_meta['dataset_name']
        pose_est_results_converted = []

        # print(f"len(pose_est_results) = {len(pose_est_results)}")

        # convert 2d pose estimation results into the format for pose-lifting
        # such as changing the keypoint order, flipping the keypoint, etc.
        for i, data_sample in enumerate(pose_est_results):
            pred_instances = data_sample.pred_instances.cpu().numpy()
            keypoints = pred_instances.keypoints
            # calculate area and bbox
            if 'bboxes' in pred_instances:
                areas = np.array([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                for bbox in pred_instances.bboxes])
                pose_est_results[i].pred_instances.set_field(areas, 'areas')
            else:
                areas, bboxes = [], []
                for keypoint in keypoints:
                    xmin = np.min(keypoint[:, 0][keypoint[:, 0] > 0], initial=1e10)
                    xmax = np.max(keypoint[:, 0])
                    ymin = np.min(keypoint[:, 1][keypoint[:, 1] > 0], initial=1e10)
                    ymax = np.max(keypoint[:, 1])
                    areas.append((xmax - xmin) * (ymax - ymin))
                    bboxes.append([xmin, ymin, xmax, ymax])
                pose_est_results[i].pred_instances.areas = np.array(areas)
                pose_est_results[i].pred_instances.bboxes = np.array(bboxes)

            # track id
            _track = _track_by_oks
            track_id, pose_est_results_last, _ = _track(data_sample,
                                                        pose_est_results_last,
                                                        args.tracking_thr)
            if track_id == -1:
                if np.count_nonzero(keypoints[:, :, 1]) >= 3:
                    track_id = next_id
                    next_id += 1
                else:
                    # If the number of keypoints detected is small,
                    # delete that person instance.
                    keypoints[:, :, 1] = -10
                    pose_est_results[i].pred_instances.set_field(
                        keypoints, 'keypoints')
                    pose_est_results[i].pred_instances.set_field(
                        pred_instances.bboxes * 0, 'bboxes')
                    pose_est_results[i].set_field(pred_instances, 'pred_instances')
                    track_id = -1
            pose_est_results[i].set_field(track_id, 'track_id')

            # convert keypoints for pose-lifting
            pose_est_result_converted = PoseDataSample()
            pose_est_result_converted.set_field(
                pose_est_results[i].pred_instances.clone(), 'pred_instances')
            pose_est_result_converted.set_field(
                pose_est_results[i].gt_instances.clone(), 'gt_instances')
            keypoints = convert_keypoint_definition(keypoints,
                                                    pose_det_dataset_name,
                                                    pose_lift_dataset_name)
            pose_est_result_converted.pred_instances.set_field(
                keypoints, 'keypoints')
            pose_est_result_converted.set_field(pose_est_results[i].track_id,
                                                'track_id')
            pose_est_results_converted.append(pose_est_result_converted)

        pose_est_results_list.append(pose_est_results_converted.copy())

        # Second stage: Pose lifting
        # extract and pad input pose2d sequence
        pose_seq_2d = extract_pose_sequence(
            pose_est_results_list,
            frame_idx=frame_idx,
            causal=pose_lift_dataset.get('causal', False),
            seq_len=pose_lift_dataset.get('seq_len', 1),
            step=pose_lift_dataset.get('seq_step', 1))

        # conduct 2D-to-3D pose lifting
        norm_pose_2d = not args.disable_norm_pose_2d
        pose_lift_results = inference_pose_lifter_model(
            self.pose_lifter,
            pose_seq_2d,
            image_size=frame.shape[:2],
            norm_pose_2d=norm_pose_2d)

        # post-processing
        for idx, pose_lift_result in enumerate(pose_lift_results):
            pose_lift_result.track_id = pose_est_results[idx].get('track_id', 1e4)

            pred_instances = pose_lift_result.pred_instances
            keypoints = pred_instances.keypoints
            keypoint_scores = pred_instances.keypoint_scores
            if keypoint_scores.ndim == 3:
                keypoint_scores = np.squeeze(keypoint_scores, axis=1)
                pose_lift_results[
                    idx].pred_instances.keypoint_scores = keypoint_scores
            if keypoints.ndim == 4:
                keypoints = np.squeeze(keypoints, axis=1)

            keypoints = keypoints[..., [0, 2, 1]]
            keypoints[..., 0] = -keypoints[..., 0]
            keypoints[..., 2] = -keypoints[..., 2]

            # rebase height (z-axis)
            if not args.disable_rebase_keypoint:
                keypoints[..., 2] -= np.min(
                    keypoints[..., 2], axis=-1, keepdims=True)

            pose_lift_results[idx].pred_instances.keypoints = keypoints

        pose_lift_results = sorted(
            pose_lift_results, key=lambda x: x.get('track_id', 1e4))

        # print(f"pose_lift_results = {pose_lift_results}")

        pred_3d_data_samples = merge_data_samples(pose_lift_results)
        det_data_sample = merge_data_samples(pose_est_results)
        pred_3d_instances = pred_3d_data_samples.get('pred_instances', None)

        # print(f"pred_3d_instances = {pred_3d_instances.keypoints}")

        if args.num_instances < 0:
            args.num_instances = len(pose_lift_results)

        return pose_est_results, pose_est_results_list, pred_3d_instances, next_id


        
    def process_one_image(self, args, frame, frame_idx = 0, 
                        pose_est_results_last = [], pose_est_results_list = [], next_id = 0,
                        visualizer = None):
        """Visualize detected and predicted keypoints of one image.

        Args:
            args (Argument): Custom command-line arguments.
            detector (mmdet.BaseDetector): The mmdet detector.
            frame (np.ndarray): The image frame read from input image or video.
            frame_idx (int): The index of current frame.
            pose_estimator (TopdownPoseEstimator): The pose estimator for 2d pose.
            pose_est_results_last (list(PoseDataSample)): The results of pose
                estimation from the last frame for tracking instances.
            pose_est_results_list (list(list(PoseDataSample))): The list of all
                pose estimation results converted by
                ``convert_keypoint_definition`` from previous frames. In
                pose-lifting stage it is used to obtain the 2d estimation sequence.
            next_id (int): The next track id to be used.
            pose_lifter (PoseLifter): The pose-lifter for estimating 3d pose.
            visualize_frame (np.ndarray): The image for drawing the results on.
            visualizer (Visualizer): The visualizer for visualizing the 2d and 3d
                pose estimation results.

        Returns:
            pose_est_results (list(PoseDataSample)): The pose estimation result of
                the current frame.
            pose_est_results_list (list(list(PoseDataSample))): The list of all
                converted pose estimation results until the current frame.
            pred_3d_instances (InstanceData): The result of pose-lifting.
                Specifically, the predicted keypoints and scores are saved at
                ``pred_3d_instances.keypoints`` and
                ``pred_3d_instances.keypoint_scores``.
            next_id (int): The next track id to be used.
        """

        pose_lift_dataset = self.pose_lifter.cfg.test_dataloader.dataset
        pose_lift_dataset_name = self.pose_lifter.dataset_meta['dataset_name']

        # First stage: conduct 2D pose detection in a Topdown manner
        # use detector to obtain person bounding boxes
        det_result = inference_detector(self.detector, frame)
        pred_instance = det_result.pred_instances.cpu().numpy()

        # filter out the person instances with category and bbox threshold
        # e.g. 0 for person in COCO
        bboxes = pred_instance.bboxes
        bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                    pred_instance.scores > args.bbox_thr)]
        # print(f"num bboxes = {len(bboxes)}")

        if len(bboxes) == 0:
            # If there is no human detected
            return [], [], None, next_id

        # estimate pose results for current image
        pose_est_results = inference_topdown(self.pose_estimator, frame, bboxes)

        pose_det_dataset_name = self.pose_estimator.dataset_meta['dataset_name']
        pose_est_results_converted = []

        # convert 2d pose estimation results into the format for pose-lifting
        # such as changing the keypoint order, flipping the keypoint, etc.
        for i, data_sample in enumerate(pose_est_results):
            pred_instances = data_sample.pred_instances.cpu().numpy()
            keypoints = pred_instances.keypoints
            # calculate area and bbox
            if 'bboxes' in pred_instances:
                areas = np.array([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                for bbox in pred_instances.bboxes])
                pose_est_results[i].pred_instances.set_field(areas, 'areas')
            else:
                areas, bboxes = [], []
                for keypoint in keypoints:
                    xmin = np.min(keypoint[:, 0][keypoint[:, 0] > 0], initial=1e10)
                    xmax = np.max(keypoint[:, 0])
                    ymin = np.min(keypoint[:, 1][keypoint[:, 1] > 0], initial=1e10)
                    ymax = np.max(keypoint[:, 1])
                    areas.append((xmax - xmin) * (ymax - ymin))
                    bboxes.append([xmin, ymin, xmax, ymax])
                pose_est_results[i].pred_instances.areas = np.array(areas)
                pose_est_results[i].pred_instances.bboxes = np.array(bboxes)

            # track id
            _track = _track_by_iou
            track_id, pose_est_results_last, _ = _track(data_sample,
                                                        pose_est_results_last,
                                                        args.tracking_thr)
            if track_id == -1:
                if np.count_nonzero(keypoints[:, :, 1]) >= 3:
                    track_id = next_id
                    next_id += 1
                else:
                    # If the number of keypoints detected is small,
                    # delete that person instance.
                    keypoints[:, :, 1] = -10
                    pose_est_results[i].pred_instances.set_field(
                        keypoints, 'keypoints')
                    pose_est_results[i].pred_instances.set_field(
                        pred_instances.bboxes * 0, 'bboxes')
                    pose_est_results[i].set_field(pred_instances, 'pred_instances')
                    track_id = -1
            pose_est_results[i].set_field(track_id, 'track_id')

            # convert keypoints for pose-lifting
            pose_est_result_converted = PoseDataSample()
            pose_est_result_converted.set_field(
                pose_est_results[i].pred_instances.clone(), 'pred_instances')
            pose_est_result_converted.set_field(
                pose_est_results[i].gt_instances.clone(), 'gt_instances')
            keypoints = convert_keypoint_definition(keypoints,
                                                    pose_det_dataset_name,
                                                    pose_lift_dataset_name)
            pose_est_result_converted.pred_instances.set_field(
                keypoints, 'keypoints')
            pose_est_result_converted.set_field(pose_est_results[i].track_id,
                                                'track_id')
            pose_est_results_converted.append(pose_est_result_converted)

        pose_est_results_list.append(pose_est_results_converted.copy())

        # Second stage: Pose lifting
        # extract and pad input pose2d sequence
        pose_seq_2d = extract_pose_sequence(
            pose_est_results_list,
            frame_idx=frame_idx,
            causal=pose_lift_dataset.get('causal', False),
            seq_len=pose_lift_dataset.get('seq_len', 1),
            step=pose_lift_dataset.get('seq_step', 1))

        # conduct 2D-to-3D pose lifting
        norm_pose_2d = not args.disable_norm_pose_2d
        pose_lift_results = inference_pose_lifter_model(
            self.pose_lifter,
            pose_seq_2d,
            image_size=frame.shape[:2],
            norm_pose_2d=norm_pose_2d)

        # post-processing
        for idx, pose_lift_result in enumerate(pose_lift_results):
            pose_lift_result.track_id = pose_est_results[idx].get('track_id', 1e4)

            pred_instances = pose_lift_result.pred_instances
            keypoints = pred_instances.keypoints
            keypoint_scores = pred_instances.keypoint_scores
            if keypoint_scores.ndim == 3:
                keypoint_scores = np.squeeze(keypoint_scores, axis=1)
                pose_lift_results[
                    idx].pred_instances.keypoint_scores = keypoint_scores
            if keypoints.ndim == 4:
                keypoints = np.squeeze(keypoints, axis=1)

            keypoints = keypoints[..., [0, 2, 1]]
            keypoints[..., 0] = -keypoints[..., 0]
            keypoints[..., 2] = -keypoints[..., 2]

            # rebase height (z-axis)
            if not args.disable_rebase_keypoint:
                keypoints[..., 2] -= np.min(
                    keypoints[..., 2], axis=-1, keepdims=True)

            pose_lift_results[idx].pred_instances.keypoints = keypoints

        pose_lift_results = sorted(
            pose_lift_results, key=lambda x: x.get('track_id', 1e4))

        # print(f"pose_lift_results = {pose_lift_results}")

        pred_3d_data_samples = merge_data_samples(pose_lift_results)
        det_data_sample = merge_data_samples(pose_est_results)
        pred_3d_instances = pred_3d_data_samples.get('pred_instances', None)

        if args.num_instances < 0:
            args.num_instances = len(pose_lift_results)

        return pose_est_results, pose_est_results_list, pred_3d_instances, next_id
