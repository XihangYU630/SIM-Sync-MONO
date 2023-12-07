# Xihang Yu
# 12/07/2023

import os
import glob
import torch
import utils
import cv2
import argparse
import time
import os.path as osp
from scipy.spatial.transform import Rotation as R

import numpy as np

from imutils.video import VideoStream
from midas.model_loader import default_models, load_model
from collections import defaultdict

first_execution = True
def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    if "openvino" in model_type:
        if first_execution or not use_camera:
            print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                    "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                    "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction


def remove_nan_pairs(point_cloud_1, point_cloud_2):
    # Find indices of NaN values in either point cloud
    nan_indices_1 = np.isnan(point_cloud_1).any(axis=1)
    nan_indices_2 = np.isnan(point_cloud_2).any(axis=1)
    
    # Find indices of pairs to be removed
    remove_indices = np.logical_or(nan_indices_1, nan_indices_2)
    
    # Remove pairs from both point clouds
    filtered_point_cloud_1 = point_cloud_1[~remove_indices]
    filtered_point_cloud_2 = point_cloud_2[~remove_indices]

    return filtered_point_cloud_1, filtered_point_cloud_2

def scale_points(point_frame1, point_frame2, scale_factor):

    scaled_set1 = point_frame1 * scale_factor
    scaled_set2 = point_frame2 * scale_factor
    
    return scaled_set1, scaled_set2


def get_intrinsics(sequence_name = 'freiburg2_xyz'):
    if sequence_name == 'freiburg1_xyz' \
        or sequence_name == 'freiburg1_rpy'\
        or sequence_name == 'freiburg1_teddy'\
        or sequence_name == 'freiburg1_floor':
        # camera intrinsics
        fx = 517.3  # focal length x
        fy = 516.5  # focal length y
        cx = 318.6  # optical center x
        cy = 255.3  # optical center y
    elif sequence_name == 'freiburg2_xyz':
        # camera intrinsics
        fx = 520.9  # focal length x
        fy = 521.0  # focal length y
        cx = 325.1  # optical center x
        cy = 249.7  # optical center y
    return fx, fy, cx, cy

def Finetune_depth(weights, pose, edges, pointclouds, image_pair_correspondence, scale_factor):

    # midas param start #
    input_path = 'input'
    model_path = 'weights/dpt_beit_large_512.pt'
    model_type = 'dpt_beit_large_512'
    optimize = False
    output_path = 'output'
    height = None
    square = 'False'
    side = False
    grayscale = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)
    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)
    # midas param end #


    pred_path = "/media/xihang/Elements/solvers/realdata_experiments/SDP_realdata/rgbd_dataset_freiburg2_xyz_output/dpt_beit_large_512_60-260"
    depth_dir = pred_path + '/depth_dpt_beit_large_512/depth/'
    predict_files = sorted(os.listdir(depth_dir))
    rgb_dir = pred_path + '/color_full/'
    rgb_files = sorted(os.listdir(rgb_dir))
    number_files = int(len(predict_files)/2)

    scaled_cloud_camera_frame = defaultdict()
    for key, value in image_pair_correspondence.items():
        scaled_cloud_camera_frame[key] = defaultdict()
        frame1, frame2 = key
        x_frame1 = value[:, 0]
        y_frame1 = value[:, 1]
        x_frame2 = value[:, 2]
        y_frame2 = value[:, 3]


        ######### first frame ###########
        # predict the depth map
        rgb_file_input = osp.join(rgb_dir,rgb_files[frame1])
        if input_path is not None:
            if output_path is None:
                print("Warning: No output path specified. Images will be processed but not shown or stored anywhere.")


            # input
            original_image_rgb = utils.read_image(rgb_file_input)  # in [0, 1]
            image = transform({"image": original_image_rgb})["image"]

            # compute
            with torch.no_grad():
                prediction = process(device, model, model_type, image, (net_w, net_h), original_image_rgb.shape[1::-1],
                                    optimize, False)
            # output
            if output_path is not None:
                filename = os.path.join(
                    output_path, os.path.splitext(os.path.basename(rgb_file_input))[0] + '-' + model_type
                )
                utils.write_depth(filename, prediction, grayscale, bits=2)
                utils.write_pfm(filename + ".pfm", prediction.astype(np.float32))
            
            predicted_depth_inverse = prediction


        epsilon = 0.00001
        percentile_threshold = 90
        predicted_depth = 1.0 / (epsilon + predicted_depth_inverse)  # left predicted_depth is depth in real
        h, w = predicted_depth.shape
        # remove points out of range in prediction
        threshold = np.percentile(predicted_depth, percentile_threshold)
        predicted_depth = np.where(predicted_depth > threshold, np.nan, predicted_depth)
        predicted_depth = np.where(predicted_depth <= 0, np.nan, predicted_depth)


        # read intrinsics
        fx_frame1, fy_frame1, cx_frame1, cy_frame1 = get_intrinsics(sequence_name='freiburg2_xyz')
        target_width, target_height = 640,480

        points = np.zeros((y_frame1.shape[0], 3))
        y_frame1_ceil = np.ceil(y_frame1).astype(np.int32)
        y_frame1_floor = np.floor(y_frame1).astype(np.int32)
        x_frame1_ceil = np.ceil(x_frame1).astype(np.int32)
        x_frame1_floor = np.floor(x_frame1).astype(np.int32)
        y_frame1_ceil[y_frame1_ceil == target_height] = target_height - 1
        x_frame1_ceil[x_frame1_ceil == target_width] = target_width - 1
        depth1 = predicted_depth[y_frame1_ceil, x_frame1_ceil]
        depth2 = predicted_depth[y_frame1_ceil, x_frame1_floor]
        depth3 = predicted_depth[y_frame1_floor, x_frame1_ceil]
        depth4 = predicted_depth[y_frame1_floor, x_frame1_floor]
        predicted_depth = (depth1+depth2+depth3+depth4)/4
        points[:,2] = predicted_depth.reshape(-1,)
        points[:,0] = (x_frame1 - cx_frame1) * predicted_depth / fx_frame1
        points[:,1] = (y_frame1 - cy_frame1) * predicted_depth / fy_frame1

        scaled_cloud_camera_frame[key][frame1] = points










        ######### second frame ###########
        # predict the depth map
        rgb_file_input = osp.join(rgb_dir,rgb_files[frame2])
        if input_path is not None:
            if output_path is None:
                print("Warning: No output path specified. Images will be processed but not shown or stored anywhere.")


            # input
            original_image_rgb = utils.read_image(rgb_file_input)  # in [0, 1]
            image = transform({"image": original_image_rgb})["image"]

            # compute
            with torch.no_grad():
                prediction = process(device, model, model_type, image, (net_w, net_h), original_image_rgb.shape[1::-1],
                                    optimize, False)
            # output
            if output_path is not None:
                filename = os.path.join(
                    output_path, os.path.splitext(os.path.basename(rgb_file_input))[0] + '-' + model_type
                )
                utils.write_depth(filename, prediction, grayscale, bits=2)
                utils.write_pfm(filename + ".pfm", prediction.astype(np.float32))




        epsilon = 0.00001
        predicted_depth = 1.0 / (epsilon + predicted_depth_inverse)  # left predicted_depth is depth in real
        h, w = predicted_depth.shape
        # remove points out of range in prediction
        threshold = np.percentile(predicted_depth, percentile_threshold)
        predicted_depth = np.where(predicted_depth > threshold, np.nan, predicted_depth)
        predicted_depth = np.where(predicted_depth <= 0, np.nan, predicted_depth)

        fx_frame2, fy_frame2, cx_frame2, cy_frame2 = get_intrinsics('freiburg2_xyz')            

        # construct points
        points = np.zeros((y_frame2.shape[0], 3))
        y_frame2_ceil = np.ceil(y_frame2).astype(np.int32)
        y_frame2_floor = np.floor(y_frame2).astype(np.int32)
        x_frame2_ceil = np.ceil(x_frame2).astype(np.int32)
        x_frame2_floor = np.floor(x_frame2).astype(np.int32)
        y_frame2_ceil[y_frame2_ceil == target_height] = target_height - 1
        x_frame2_ceil[x_frame2_ceil == target_width] = target_width - 1
        depth1 = predicted_depth[y_frame2_ceil, x_frame2_ceil]
        depth2 = predicted_depth[y_frame2_ceil, x_frame2_floor]
        depth3 = predicted_depth[y_frame2_floor, x_frame2_ceil]
        depth4 = predicted_depth[y_frame2_floor, x_frame2_floor]
        predicted_depth = (depth1+depth2+depth3+depth4)/4
        points[:,2] = predicted_depth.reshape(-1,)
        points[:,0] = (x_frame2 - cx_frame2) * predicted_depth / fx_frame2
        points[:,1] = (y_frame2 - cy_frame2) * predicted_depth / fy_frame2

        scaled_cloud_camera_frame[key][frame2] = points


    for key, value in scaled_cloud_camera_frame.items():

        frame1, frame2 = key
        point_frame1 = scaled_cloud_camera_frame[key][frame1]
        point_frame2 = scaled_cloud_camera_frame[key][frame2]

        point_frame1, point_frame2 = remove_nan_pairs(point_frame1, point_frame2)

        point_frame1, point_frame2 = scale_points(point_frame1, point_frame2, scale_factor)
        scaled_cloud_camera_frame[key][frame1] = point_frame1
        scaled_cloud_camera_frame[key][frame2] = point_frame2
    tmp = 1


    translations = []
    rotations = []
    scales = []
    N = pose['t_est'].shape[1]
    for i in range(N):         
        translations.append(pose['t_est'][:, i])
        rotations.append(pose['R_est'][3*i:3*(i+1), :])
        scales.append(pose['s_est'][i])
    
    translations = np.array(translations)
    rotations = np.array(rotations)
    scales = np.array(scales)


    frame_pose_index_pair = {0:0, 108:1, 199:2}

    loss_function = 0 
    for key,value in scaled_cloud_camera_frame.items():

        frame1 = key[0]
        frame2 = key[1]
        pointcloud_frame1 = value[frame1]
        pointcloud_frame2 = value[frame2]

        rotation_frame1 = rotations[frame_pose_index_pair[frame1]]
        rotation_frame2 = rotations[frame_pose_index_pair[frame2]]
        translation_frame1 = translations[frame_pose_index_pair[frame1]]
        translation_frame2 = translations[frame_pose_index_pair[frame2]]
        scales_frame1 = scales[frame_pose_index_pair[frame1]]
        scales_frame2 = scales[frame_pose_index_pair[frame2]]

        loss_function += np.sum(np.linalg.norm(scales_frame1 * rotation_frame1 @ pointcloud_frame1.T + translation_frame1.reshape(translation_frame1.size,-1) - (scales_frame2 * rotation_frame2 @ pointcloud_frame2.T + translation_frame2.reshape(translation_frame2.size,-1)), axis=0))
        tmp = 1



