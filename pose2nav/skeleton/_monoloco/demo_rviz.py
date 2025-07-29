#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage
from visualization_msgs.msg import Marker, MarkerArray
import cv2
import PIL.Image
import numpy as np

import argparse

from _monoloco import *
from _monoloco.config.arg import cli


class PoseEstimatorNode:
    def __init__(self):
        rospy.init_node('pose_estimator_node')
        self.image_sub = rospy.Subscriber('/image_raw/compressed', CompressedImage, self.image_callback)
        # self.image_sub = rospy.Subscriber('/left/image_color/compressed', CompressedImage, self.image_callback)
        # self.image_sub = rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.image_callback)
        self.marker_pub = rospy.Publisher('/pose_markers', MarkerArray, queue_size=1)
        self.img = None

        ## Argument for MONOLOCO
        self.args = cli()
        self.cnt = 0
        self.pose_cnt = 0

        # for openpifpaf predictions
        self.predictor = MonoLocoPredictor(self.args)
        rospy.loginfo("3D Pose Estimator Node Initialized")


    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            bgr_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode as OpenCV BGR
            self.img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            # self.pil_image = PIL.Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        except Exception as e:
            rospy.logerr(f"Could not convert image: {e}")
            return


        if self.img is not None:
            if self.pose_cnt % 10 == 0:

                keypoints, xyz = self.predictor.predict(self.img)

                # First, clear all previous markers
                delete_marker = Marker()
                delete_marker.action = Marker.DELETEALL
                delete_marker.header.frame_id = "velodyne"
                delete_marker.header.stamp = rospy.Time.now()

                delete_array = MarkerArray()
                delete_array.markers.append(delete_marker)
                self.marker_pub.publish(delete_array)

                # Publish the keypoints as RViz markers
                marker_array = MarkerArray()
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
                        if j % 2 == 0: # right
                            marker.color.r = 0.0
                            marker.color.g = 0.0
                            marker.color.b = 1.0
                        else: # left
                            marker.color.r = 0.0
                            marker.color.g = 1.0
                            marker.color.b = 0.0    
                        marker_array.markers.append(marker)

                    marker = Marker()
                    marker.header.frame_id = "velodyne"
                    marker.header.stamp = rospy.Time.now()
                    marker.ns = "pose"
                    marker.id = i
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.pose.position.x = xyz[i][2]
                    marker.pose.position.y = -xyz[i][0]
                    marker.pose.position.z = xyz[i][1] + 0.1
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
        node = PoseEstimatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
