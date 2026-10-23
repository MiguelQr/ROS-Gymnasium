#!/usr/bin/env python3

import numpy as np
import math
import os
import datetime
import logging
from pathlib import Path

def process_odom(goal_pose, odom):

    goal_distance = math.sqrt(
        (goal_pose[0]-odom[0])**2
        + (goal_pose[1]-odom[1])**2)

    path_theta = math.atan2(
        goal_pose[1]-odom[1],
        goal_pose[0]-odom[0])

    goal_angle = path_theta - odom[2]

    if goal_angle > math.pi:
        goal_angle -= 2 * math.pi

    elif goal_angle < -math.pi:
        goal_angle += 2 * math.pi

    goal_info = [goal_distance, goal_angle]
    robot_pose = [odom[0], odom[1], odom[2]]

    return goal_info, robot_pose

def create_logdir(policy, sensor, logdir):
    """
    """
    logdir_ = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')}_{sensor}_{policy}/"
    Path(os.path.join(logdir, logdir_)).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logdir, logdir_, 'screen_logger.log'), 
        level=logging.INFO)
    return str(Path(os.path.join(logdir, logdir_)))

def euler_from_quaternion(quat):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quat = [x, y, z, w]
    """
    x = quat.x
    y = quat.y
    z = quat.z
    w = quat.w

    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w*y - z*x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2 * (y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        return v, norm
        #norm=np.finfo(v.dtype).eps
    return v/norm, norm

def quat_to_euler(qz, qw):
    """
    """
    t1 = 2*(qw*qz)
    t2 = 1 - 2*(qz*qz)

    Wz = math.atan2(t1,t2)

    return Wz

def tf_decompose(robot_pose, goal_pose):
    """
    This method decomposes two consecutive reference frames.

    For example, it can return the pose of the goal, converted from 
    robot reference frame to fixed reference frame, with robot_pose given in 
    fixed reference frame and goal_pose given in robot reference frame.
    """ 
    xr, yr, zr = tuple(robot_pose)

    A = np.zeros([4,4], dtype=float)
    A[0,0] = math.cos(zr)
    A[1,0] = math.sin(zr)
    A[0,1] = -A[1,0]
    A[1,1] = A[0,0]
    A[0,3] = xr
    A[1,3] = yr
    A[2,3] = 0
    A[3,3] = 1
    A[2,2] = 1
    
    return np.matmul(A, goal_pose).tolist()
