#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage
from visualization_msgs.msg import Marker, MarkerArray
import cv2
import mmcv
import numpy as np
import time

from _mmpose import *
from _mmpose.configs.arg import parse_args


class PoseEstimatorNode:
    def __init__(self):
        rospy.init_node('pose_estimator_node')
        self.image_sub = rospy.Subscriber('/image_raw/compressed', CompressedImage, self.image_callback)
        # self.image_sub = rospy.Subscriber('/left/image_color/compressed', CompressedImage, self.image_callback)
        # self.image_sub = rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.image_callback)
        self.marker_pub = rospy.Publisher('/pose_markers', MarkerArray, queue_size=1)
        self.img = None

        ## Argument for MMPOSE
        self.pose_cnt = 0
        self.args = parse_args()

        self.mmpose_predictor = MMPosePredictor(self.args)
        rospy.loginfo("3D Pose Estimator Node Initialized")

    # def predict(self, image):
    #     # STEP1: predict


    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            bgr_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode as OpenCV BGR
            img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            self.frame = mmcv.imread(img, channel_order='rgb')
        except Exception as e:
            rospy.logerr(f"Could not convert image: {e}")
            return

        pose_est_results_list = []
        
        if self.frame is not None:
            if self.pose_cnt % 10 == 0:
                start = time.time()
                timing = []

                _, _, pred_3d_instances, _ = self.mmpose_predictor.process_one_image(
                    args=self.args,
                    frame=self.frame,
                    pose_est_results_list=pose_est_results_list)

                fwd_time = (time.time()-start)*1000
                timing.append(fwd_time)  
                print(f"Forward time: {fwd_time:.0f} ms")

                # First, clear all previous markers
                delete_marker = Marker()
                delete_marker.action = Marker.DELETEALL
                delete_marker.header.frame_id = "velodyne"
                delete_marker.header.stamp = rospy.Time.now()

                delete_array = MarkerArray()
                delete_array.markers.append(delete_marker)
                self.marker_pub.publish(delete_array)

                pred_3d_instances = pred_3d_instances.keypoints

                # Publish the keypoints as RViz markers
                marker_array = MarkerArray()
                for i, keypoint in enumerate(pred_3d_instances):
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
                        marker.color.r = self.mmpose_predictor.det_dataset_link_color[j][0]/255
                        marker.color.g = self.mmpose_predictor.det_dataset_link_color[j][1]/255
                        marker.color.b = self.mmpose_predictor.det_dataset_link_color[j][2]/255
                        marker_array.markers.append(marker)

                    # marker = Marker()
                    # marker.header.frame_id = "velodyne"
                    # marker.header.stamp = rospy.Time.now()
                    # marker.ns = "pose"
                    # marker.id = i
                    # marker.type = Marker.SPHERE
                    # marker.action = Marker.ADD
                    # marker.pose.position.x = xyz[i][2]
                    # marker.pose.position.y = -xyz[i][0]
                    # marker.pose.position.z = xyz[i][1] + 0.1
                    # marker.pose.orientation.w = 1.0
                    # marker.scale.x = 0.3
                    # marker.scale.y = 0.3
                    # marker.scale.z = 0.3
                    # marker.color.a = 1.0
                    # marker.color.r = 1.0
                    # marker.color.g = 0.0
                    # marker.color.b = 0.0
                    # marker_array.markers.append(marker)
                    # print(f"XYZ: {xyz[i][2]}, {-xyz[i][0]} {xyz[i][1]}")

                self.marker_pub.publish(marker_array)
                

            self.pose_cnt += 1 

if __name__ == '__main__':

    try:
        node = PoseEstimatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass