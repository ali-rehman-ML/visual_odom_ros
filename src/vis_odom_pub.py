#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
import cv2

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from tf.transformations import quaternion_from_euler, euler_from_quaternion

#from Visual_Odom import Vis_Odometry
import cv2
import numpy as np


import math
import time

class Vis_Odometry:
 
    def __init__(self):
        self.R=np.identity(3,dtype=np.float32)
        self.t=np.zeros(shape=(3, 1))
        self.diag = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
        
        self.no_feature=0
        self.K = np.array([[630, 0, 637.44], [0, 630, 356.778], [0, 0, 1]], dtype=np.float32)
        self.detector=cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
        self.p1=None

        self.focal = 630
        self.pp = (637.44, 356.778)
        self.lk_params=dict(winSize  = (9,9),maxLevel = 4, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        self.x=0
        self.y=0
        self.theta=0

        self.x_scale=1.9047e-03
        self.y_sacle=1.49394e-03


        # self.current_image=None
        # self.prev_image=None


    def rotation_matrix_to_euler_angles(self,R):
        """
        Converts a rotation matrix to Euler angles (roll, pitch, yaw).

        Args:
            R (np.ndarray): Rotation matrix.

        Returns:
            np.ndarray: Euler angles in radians.
        """

        # Calculate the roll angle.

        roll = np.arctan2(R[2, 1], R[2, 2])

        # Calculate the pitch angle.

        pitch = np.arctan(-R[2, 0] / np.sqrt(R[2, 1]**2 + R[2, 2]**2))

        # Calculate the yaw angle.

        yaw = np.arctan2(R[1, 0], R[0, 0])
        # print(pitch.shape)
        # print(np.degrees(roll),np.degrees(pitch),np.degrees(yaw))

        return pitch
    
    def get_pose(self):
        return self.x,self.y,self.theta

    def update_pos(self,image_t,image_t_1):


        if self.no_feature<1000:
            self.p1=self.detector.detect(image_t.copy())
            self.p1=np.array([x.pt for x in self.p1], dtype=np.float32).reshape(-1, 1, 2)
        p2, st, err = cv2.calcOpticalFlowPyrLK(image_t, image_t_1, self.p1, None, **self.lk_params)

        good_old = self.p1[st == 1]
        good_new = p2[st == 1]



        # print("error ",np.linalg.norm(good_new-good_old))
        error=np.linalg.norm(good_new-good_old)
        E, mask=cv2.findEssentialMat(good_new, good_old, cameraMatrix=self.K, method=cv2.RANSAC, prob=0.89, threshold=1.0)

        _, R_n, t_n, _ = cv2.recoverPose(E, good_old, good_new,cameraMatrix=self.K,mask=None,R=self.R.copy(),t=self.t.copy())

        # pitch_angle=math.radians(yaw_data[k+1][0])#rotation_matrix_to_euler_angles(R_n.copy())
        # print(rotation_matrix_to_euler_angles(R_n.copy()),math.radians(yaw_data[k+5])-math.radians(yaw_data[k]))
        pitch_angle=self.rotation_matrix_to_euler_angles(R_n)
        # print("pitch ", pitch_angle)

        R_n=np.array([[np.cos(pitch_angle),0,np.sin(pitch_angle)],
                    [0,                1,          0],
                    [-np.sin(pitch_angle),0,np.cos(pitch_angle)]])
        t_n=t_n*(error/100)
        
        # if (error>450):

        self.R=R_n.dot(self.R)

        self.t=self.t+(self.R.dot(t_n))




        t_temp=np.matmul(self.diag,self.t.copy())


        self.x=self.x_scale*t_temp[2][0]
        self.y=self.y_sacle*t_temp[0][0]


        self.theta=self.rotation_matrix_to_euler_angles(self.R)



        # pred_pos.append([1.9047e-03*t_temp[2][0],1.49394e-03*t_temp[0][0],0])

        # pred_pos.append([9.99754901e-05*t_temp[2][0],9.99754901e-04*t_temp[0][0],0])

        self.no_feature=good_new.shape[0]

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

