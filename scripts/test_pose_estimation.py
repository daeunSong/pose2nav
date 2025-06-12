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
from _mmpose.configs.arg import parse_args
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

from _monoloco import *
from _monoloco.config.arg import cli
from _monoloco.process import load_calibration
from _monoloco.utils import pixel_to_camera_pt

def parse_arguments():
    """Parse command line arguments for the pose estimator node."""
    parser = argparse.ArgumentParser(description='3D Pose Estimator ROS Node')

    parser.add_argument('--robot', type=str, default='spot',
                        help='Robot name')
    
    return parser.parse_args()

class PoseEstimatorNode:
    def __init__(self, args):
        # Parse ROS arguments
        self.args = args

        rospy.init_node('pose_estimator_node')
        self.img = None

        ## Arguent for MMPOSE
        self.pose_cnt = 0
        self.mmpose_args = parse_args()
        self.mmpose_predictor = MMPosePredictor(self.mmpose_args)

        # Visualizer
        self.mmpose_visualizer = VISUALIZERS.build(self.mmpose_predictor.pose_estimator.cfg.visualizer)
        self.mmpose_visualizer.set_dataset_meta(self.mmpose_predictor.pose_estimator.dataset_meta)

        # Argment for MonoLoco
        self.monoloco_args = cli()
        self.monoloco_predictor = MonoLocoPredictor(self.monoloco_args)

        if self.args.robot == 'jackal':
            self.image_sub = rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.image_callback)
            self.monoloco_args.focal_length = 6.0
        if self.args.robot == 'spot':
            self.image_sub = rospy.Subscriber('/image_raw/compressed', CompressedImage, self.image_callback)    
            self.monoloco_args.focal_length = 5.0
        self.marker_pub = rospy.Publisher('/pose_markers', MarkerArray, queue_size=1)
        self.image_pub = rospy.Publisher('/pose_image', Image, queue_size=1)
        
        rospy.loginfo("3D Pose Estimator Node Initialized")

    def numpy_to_rosimg(self, img_np, encoding="bgr8"):
        """Convert numpy array (cv2 image) to ROS1 sensor_msgs/Image manually."""
        ros_img = Image()
        ros_img.header = Header()
        ros_img.header.stamp = rospy.Time.now()
        ros_img.height = img_np.shape[0]
        ros_img.width = img_np.shape[1]
        ros_img.encoding = encoding
        ros_img.is_bigendian = False
        ros_img.step = img_np.shape[1] * img_np.shape[2]  # width * channels
        ros_img.data = np.asarray(img_np).tobytes()
        return ros_img

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

        boxes = []
        keypoints = []
        
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

        im_size = (data.img_shape[1], data.img_shape[0])

        return boxes, keypoints, im_size
        

    def predict(self):
        start = time.time()
        timing = []

        img = self.img
        img_mmcv = mmcv.imread(self.img, channel_order='rgb')

        # STEP1: [mmpose] predict 2D sekeltal keypoints -> pose_est_result
        # + STEP3: [mmpose] 3D pose estimation -> pred_3d_instances
        pose_est_results_list = []
        pose_est_result, _, pred_3d_instances, _  = self.mmpose_predictor.process_one_image(self.mmpose_args, img_mmcv, pose_est_results_list=pose_est_results_list)
      
        # convert mmpose result to monoloco format 
        boxes, keypoints, im_size = self.preprocess_monoloco(pose_est_result)

        # STEP2: [monoloco] predict 3D depth
        kk = load_calibration(im_size, self.monoloco_args.focal_length)

        dic_out = self.monoloco_predictor.net.forward(keypoints, kk)
        dic_out = self.monoloco_predictor.net.post_process(
            dic_out, boxes, keypoints, kk)

        xyz_pred = dic_out["xyz_pred"] # 3D depth

        ##### STEP4: post-processing
       
        # 3D keypoints
        keypoints = pred_3d_instances.keypoints
        keypoints = np.stack([-keypoints[..., 1], keypoints[..., 0], keypoints[..., 2]], axis=-1)   

        fwd_time = (time.time()-start)*1000
        timing.append(fwd_time)  
        print(f"Forward time: {fwd_time:.0f} ms")

        xyzs = []
        # [mmpose] + [monoloco] All poses in 3D
        keypoints_3d = [] 
        for i, keypoint in enumerate(keypoints):
            xyz = xyz_pred[i]
            xyz = [xyz[2], -xyz[0], xyz[1] + 0.1]

            if self.args.robot == 'jackal':
                xyz = self.transform_points(xyz, [0., 0., 0.], -15)

            keypoint_3d = []
            translation_vector = xyz #- keypoint[0]

            for j, point in enumerate(keypoint): 
                kpt_3d = self.transform_points(point - keypoint[0], translation_vector)
                keypoint_3d.append(kpt_3d)
            keypoints_3d.append(keypoint_3d)
            xyzs.append(xyz)

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
        
        return keypoints_3d, xyzs, output_img


    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            bgr_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode as OpenCV BGR
            self.img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            # self.img_mmcv = mmcv.imread(self.img, channel_order='rgb')
        except Exception as e:
            rospy.logerr(f"Could not convert image: {e}")
            return
        
        if self.img is not None:
            if self.pose_cnt % 3 == 0:

                keypoints, xyz, output_img = self.predict()
                output_img = output_img.copy()
                ros_img_msg = self.numpy_to_rosimg(output_img, encoding="rgb8")
                self.image_pub.publish(ros_img_msg)

                # Publish the keypoints as RViz markers
                marker_array = MarkerArray()
                # First, clear all previous markers
                delete_marker = Marker()
                delete_marker.action = Marker.DELETEALL
                delete_marker.header.frame_id = "velodyne"
                delete_marker.header.stamp = rospy.Time.now()

                marker_array.markers.append(delete_marker)
                self.marker_pub.publish(marker_array)

                for i, keypoint in enumerate(keypoints):
                    for j, point in enumerate(keypoint):

                        marker = Marker()
                        marker.header.frame_id = "velodyne"
                        marker.header.stamp = rospy.Time.now()
                        marker.ns = "pose"
                        marker.id = i*100+j
                        marker.type = Marker.SPHERE
                        marker.action = Marker.ADD
                        marker.pose.position.x = point[0]
                        marker.pose.position.y = point[1]
                        marker.pose.position.z = point[2]
                        marker.pose.orientation.w = 1.0
                        marker.scale.x = 0.1
                        marker.scale.y = 0.1
                        marker.scale.z = 0.1
                        marker.color.a = 1.0
                        marker.color.r = self.mmpose_predictor.lift_dataset_link_color[j][0]/255
                        marker.color.g = self.mmpose_predictor.lift_dataset_link_color[j][1]/255
                        marker.color.b = self.mmpose_predictor.lift_dataset_link_color[j][2]/255
                        marker_array.markers.append(marker)

                    marker = Marker()
                    marker.header.frame_id = "velodyne"
                    marker.header.stamp = rospy.Time.now()
                    marker.ns = "pose"
                    marker.id = i
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.pose.position.x = xyz[i][0] #xyz[i][2]
                    marker.pose.position.y = xyz[i][1] #-xyz[i][0]
                    marker.pose.position.z = xyz[i][2] #xyz[i][1]
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = 0.3
                    marker.scale.y = 0.3
                    marker.scale.z = 0.3
                    marker.color.a = 1.0
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker_array.markers.append(marker)
                    # print(f"XYZ: {xyz[i][2]}, {-xyz[i][0]} {xyz[i][1]}")

                self.marker_pub.publish(marker_array)
                

            self.pose_cnt += 1 

if __name__ == '__main__':
    try:
        args = parse_arguments()
        node = PoseEstimatorNode(args)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
