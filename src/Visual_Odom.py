#!/usr/bin/env python
import cv2
import numpy as np
import os

import math

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











