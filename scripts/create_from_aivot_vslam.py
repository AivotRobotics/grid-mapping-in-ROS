import numpy as np
# from math import degrees as rad_to_deg
# from math import radians as deg_to_rad
import matplotlib.pyplot as plt
import os

from time import perf_counter

import sys

from pathlib import Path
SCRIPTS_PATH = '/home/manish/Software/grid-mapping-in-ROS/scripts'
BASE_PATH = "/home/dev/tmp_manish/slam_data/test_x/"

MAP_DATA_PATH = BASE_PATH + "map_data"
SLAM_TRACK_DATA_PATH = BASE_PATH +  "keyframe_trajectory.txt"
MAPS_PATH = BASE_PATH + "map_generated" 

MAP_NAME = "aivot"
sys.path.insert(0, SCRIPTS_PATH)

from grid_map import *
# from utils import *
from aivot_utils import *

P_prior = 0.5	# Prior occupancy probability
P_occ = 0.9	    # Probability that cell is occupied with total confidence
P_free = 0.3	# Probability that cell is free with total confidence 

# Map config (needs to be consistent with the map data generator)
# TODO: read this from a yaml file
RESOLUTION = 0.05 # Grid resolution in [m]
MAP_LEN = 20.0

# # Note that these are in vslam frame
# CAMERA_ROT_X = -deg_to_rad(6)
# CAMERA_ROT_Y = +deg_to_rad(42)
# CAMERA_ROT_Z = deg_to_rad(0)

# CAMERA_ROT = {
#     "x": CAMERA_ROT_X,
#     "y": CAMERA_ROT_Y, 
#     "z": CAMERA_ROT_Z
# }

FLAG_UNKNOWN = 0
FLAG_FREE = 1
FLAG_OCCUPIED = 2

FRAME_GRID_DIM_LEN_METERS = 20.0
FRAME_GRID_DIM_LEN = int(FRAME_GRID_DIM_LEN_METERS / RESOLUTION)


def read_map_data():
    # Read map data
    slam_tracks = []
    map_data = list()
    # cam_rot_mat = get_rot_matrix(CAMERA_ROT)

    with open(SLAM_TRACK_DATA_PATH) as f:
        slam_track_data = list(f.readlines())
        print(len(slam_track_data))
        for line in slam_track_data:
            res = line.split()
            frame = res[0]
            slam_pos = {"x": float(res[1]), "y": float(res[2]), "z": float(res[3])}
            slam_orientation = {"q_x": float(res[4]), "q_y": float(res[5]), "q_z": float(res[6]), "q_w": float(res[7])}
            # slam_pos_mat = get_matrix_from_pose_q(pos, orientation)
            slam_tracks.append([frame, slam_pos, slam_orientation])
            print(frame, slam_pos, slam_orientation)
    for i, d in enumerate(slam_tracks):
        frame_id, slam_pos, slam_orientation = d
        map_file = Path(MAP_DATA_PATH) / (frame_id + ".txt")
        occ_file = Path(MAP_DATA_PATH) / (frame_id + "_occ.txt")
        free_file = Path(MAP_DATA_PATH) / (frame_id + "_free.txt")
        
        print(map_file)
        # linear, euler, deg_euler = get_linear_and_euler_from_slam_pose_and_cam_rot(slam_pos, slam_orientation, CAMERA_ROT)
        # print (i, frame_id, slam_pos, linear, euler)
        # print (i, frame_id, linear, deg_euler)

        metadata_dict = {}
        # metadata_dict["x_odom"] = linear[0]
        # metadata_dict["y_odom"] = linear[1]
        # metadata_dict["theta_odom"] = euler[2]
        # metadata_dict["map_file"] = map_file
        metadata_dict["free_file"] = free_file
        metadata_dict["occ_file"] = occ_file
        map_data.append(metadata_dict)

    return map_data


def build_map():
    gridMap = GridMap(X_lim = [-(MAP_LEN/2), (MAP_LEN/2)], 
            Y_lim = [-(MAP_LEN/2), (MAP_LEN/2)], 
            resolution = RESOLUTION, 
            p = P_prior)

    # Init time
    t_start = perf_counter()
    sim_time = 0
    step = 0

    map_metadata_list = read_map_data()

    # get robot traj
    robot_traj_file = Path(MAP_DATA_PATH) / "robot_trajectory.txt" 
    robot_traj = np.genfromtxt(robot_traj_file, dtype=np.float32)

    print("robot_traj: ", len(robot_traj), "map_metadata: ", len(map_metadata_list))

    rob_traj_index = 0
    for map_metadata in map_metadata_list:
        # map_data = np.genfromtxt(map_metadata["map_file"], dtype=np.int32)
        # print(len(map_data))

        # f = lambda x: 0.5 if x == FLAG_UNKNOWN else (2-x)
        # gray_image = np.array(list(map(f, map_data)))

        # # gray_image = f(gray_image)
        # gray_image = gray_image.reshape((FRAME_GRID_DIM_LEN, FRAME_GRID_DIM_LEN)) 
        # print(gray_image)

        # map_data = map_data.reshape((FRAME_GRID_DIM_LEN, FRAME_GRID_DIM_LEN))
        # print(map_data.shape)

        # resized_image = cv2.resize(src = gray_image, 
        #         dsize = (1000, 1000), 
        #         interpolation = cv2.INTER_AREA)
        # cv2.imshow("Grid map", resized_image)
        # cv2.waitKey(1000)

        map_data_free = np.genfromtxt(map_metadata["free_file"], dtype=np.float32)
        print("Free cells:", len(map_data_free))
        # print(map_data_free)

        map_data_occ = np.genfromtxt(map_metadata["occ_file"], dtype=np.float32)
        print("Occupied cells:", len(map_data_occ))
        # print(map_data_occ)

        ##################### Grid map update section #####################

        for (x_d, y_d) in map_data_free:
            x, y = gridMap.discretize(x_d, y_d)
            gridMap.update(x = x, y = y, p = P_free)

        for (x_d, y_d) in map_data_occ:
            x, y = gridMap.discretize(x_d, y_d)
            gridMap.update(x = x, y = y, p = P_occ)

        ##################### Image section #####################
        # converting grip map to BGR image
        bgr_image = gridMap.to_BGR_image()

        # marking robot position with blue pixel value
        x_r, y_r, z_r = robot_traj[rob_traj_index]
        rob_traj_index += 1
        xr, yr = gridMap.discretize(x_r, y_r)
        set_pixel_color(bgr_image, xr, yr, 'BLUE')
        
        # marking neighbouring pixels with blue pixel value 
        for (x, y) in gridMap.find_neighbours(xr, yr):
            set_pixel_color(bgr_image, x, y, 'BLUE')

        # marking occ hit spots with red value
        for (x_d, y_d) in map_data_occ:
            x, y = gridMap.discretize(x_d, y_d)
            set_pixel_color(bgr_image, x, y, 'RED')

        # marking occ free spots with green value
        for (x_d, y_d) in map_data_free:
            x, y = gridMap.discretize(x_d, y_d)
            set_pixel_color(bgr_image, x, y, 'GREEN')

        resized_image = cv2.resize(src = bgr_image, 
                        dsize = (1000, 1000), 
                        interpolation = cv2.INTER_AREA)

        rotated_image = cv2.rotate(src = resized_image, 
                        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imshow("Grid map", rotated_image)
        cv2.waitKey(3000)

        # Calculate step time in [s]
        t_step = perf_counter()
        step_time = t_step - t_start
        sim_time += step_time
        t_start = t_step
        step += 1 

        print('Step %d ==> %d [ms]' % (step, step_time * 1000))
        # break

    ##################### Final output section #####################

    # Terminal outputs
    print('\nSimulation time: %.2f [s]' % sim_time)
    print('Average step time: %d [ms]' % (sim_time * 1000 / step))
    print('Frames per second: %.1f' % (step / sim_time))

    if not os.path.exists(MAPS_PATH):
        os.mkdir(MAPS_PATH)

    # Saving Grid Map
    resized_image = cv2.resize(src = gridMap.to_BGR_image(), 
                                dsize = (500, 500), 
                                interpolation = cv2.INTER_AREA)

    rotated_image = cv2.rotate(src = resized_image, 
                    rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE)

    flag_1 = cv2.imwrite(img = rotated_image * 255.0, 
                    filename = MAPS_PATH + '/' + MAP_NAME + '_GRID_MAP.png')

    # Calculating Maximum likelihood estimate of the map
    gridMap.calc_MLE()

    # Saving MLE of the Grid Map
    resized_image_MLE = cv2.resize(src = gridMap.to_BGR_image(), 
                            dsize = (500, 500), 
                        interpolation = cv2.INTER_AREA)

    rotated_image_MLE = cv2.rotate(src = resized_image_MLE, 
                        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE)

    flag_2 = cv2.imwrite(img = rotated_image_MLE * 255.0, 
                    filename = MAPS_PATH + '/' + MAP_NAME + '_GRID_MAP_MLE.png')

    if flag_1 and flag_2:
        print('\nGrid map successfully saved!\n')

    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    build_map()


