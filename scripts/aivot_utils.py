#!/usr/bin/env python

from tf.transformations import euler_from_quaternion, quaternion_from_euler
from tf.transformations import concatenate_matrices, euler_from_matrix, translation_from_matrix
from tf.transformations import rotation_matrix, translation_matrix, quaternion_matrix, euler_matrix

from math import degrees as rad_to_deg
from math import radians as deg_to_rad

import numpy as np

# origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]

def get_rot_matrix(rot):
    # Rx = rotation_matrix(rot["x"], xaxis)
    # Ry = rotation_matrix(rot["y"], yaxis)
    # Rz = rotation_matrix(rot["z"], zaxis)
    # R = concatenate_matrices(Rx, Ry, Rz)
    
    # euler = euler_from_matrix(R, 'rxyz')
    # print(euler)

    R = euler_matrix(rot["x"], rot["y"], rot["z"], 'rzyx')
    return R


def get_matrix_from_pose_q(position, orientation):
    # M = quaternion_matrix(np.array([orientation["q_x"], orientation["q_y"], orientation["q_z"], orientation["q_w"]]))
    # M[0,3] = position["x"]
    # M[1,3] = position["y"]
    # M[2,3] = position["z"]
    # return M
    t_matrix = translation_matrix([position["x"], position["y"], position["z"]])
    r_matrix = quaternion_matrix([orientation["q_x"], orientation["q_y"], orientation["q_z"], orientation["q_w"]])
    return np.dot(t_matrix, r_matrix) 


def get_affine_from_slam_pose_and_cam_rot(slam_position, slam_orientation, cam_rot):
    slam_aff = get_matrix_from_pose_q(slam_position, slam_orientation)
    cam_rot_aff = get_rot_matrix(cam_rot)
    res_aff = np.dot(slam_aff, cam_rot_aff)
    return res_aff


def get_linear_and_euler_from_affine(aff_mat):
    linear = translation_from_matrix(aff_mat)
    euler = euler_from_matrix(aff_mat, 'rzyx')
    deg_euler = [rad_to_deg(x) for x in euler]
    return linear, euler, deg_euler

def get_linear_and_euler_from_slam_pose_and_cam_rot(slam_position, slam_orientation, cam_rot):
    res_aff = get_affine_from_slam_pose_and_cam_rot(slam_position, slam_orientation, cam_rot)
    linear, euler, deg_euler = get_linear_and_euler_from_affine(res_aff)
    return linear, euler, deg_euler