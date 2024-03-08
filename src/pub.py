#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
import cv2

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from Visual_Odom import Vis_Odometry
import cv2
import numpy as np


import math
import time


class Odometry_Publisher:
    def __init__(self):
        rospy.init_node('visual_odometory')
        rospy.Subscriber("/image",Image,self.image_call_back)
        self.odomPub = rospy.Publisher('vis_odom', Odometry, queue_size=10)
        self.odomFrameID='visual_odom'
        self.baseFrameID='base_link'
	self.br=CvBridge()
        self.current_image=None
        self.prev_image=None

        self.x=0
        self.y=0
        self.theta=0
        self.current_time=rospy.get_time()
        self.prev_time=rospy.get_time()
        self.dt=0.1
        self.dx=0
        self.dy=0
        self.d_theta=0

        self.vis_odom=Vis_Odometry()
        self.rate=rospy.Rate(10)



    def image_call_back(self,msg):


        if self.prev_image is None:
            self.current_image=self.br.imgmsg_to_cv2(msg)
            self.prev_image=self.br.imgmsg_to_cv2(msg)
            self.prev_time=self.current_time
            self.current_time=rospy.get_time()
            self.dt=self.current_time-self.prev_time

        else :
            self.current_image=self.br.imgmsg_to_cv2(msg)
            self.vis_odom.update_pos(self.prev_image,self.current_image)
            curr_x,curr_y,curr_theta=self.vis_odom.get_pose()

            self.dx=curr_x-self.x
            self.dy=curr_y-self.y
            self.d_theta=curr_theta-self.theta
            self.prev_time=self.current_time
            self.current_time=rospy.get_time()
            self.dt=self.current_time-self.prev_time

            self.x=curr_x
            self.y=curr_y
            self.theta=curr_theta

            self.prev_image=self.current_image

    def publisher(self):
            odom = Odometry()

            while not rospy.is_shutdown():
                q = quaternion_from_euler(0, 0, self.theta)
                odom.header.stamp = rospy.get_rostime()
                odom.header.frame_id = self.odomFrameID
                odom.child_frame_id = self.baseFrameID
                odom.pose.pose.position.x = self.x
                odom.pose.pose.position.y = self.y
                odom.pose.pose.orientation.x = q[0]
                odom.pose.pose.orientation.y = q[1]
                odom.pose.pose.orientation.z = q[2]
                odom.pose.pose.orientation.w = q[3]
                odom.twist.twist.linear.x = self.dx/self.dt
                odom.twist.twist.linear.y=self.dy/self.dt
                odom.twist.twist.angular.z = self.d_theta/self.dt

                odom.pose.covariance = [0.03, 0, 0, 0, 0, 0,
                                0, 0.03, 0, 0, 0, 0,
                                0, 0, 0.03, 0, 0, 0,
                                0, 0, 0, 0.03, 0, 0,
                                0, 0, 0, 0, 0.03, 0,
                                0, 0, 0, 0, 0, 0.03]
                self.odomPub.publish(odom)
                self.rate.sleep()


if __name__ == '__main__':
    odom=Odometry_Publisher()
    time.sleep(0.5)
    odom.publisher()

