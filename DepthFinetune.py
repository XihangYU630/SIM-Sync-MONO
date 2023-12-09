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
from TEASER_SIM_Sync import TEASER_SimSync
import torch
import torch.optim as optim

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
        )

    return prediction


# def remove_nan_pairs(point_cloud_1, point_cloud_2):
#     # Find indices of NaN values in either point cloud
#     nan_indices_1 = np.isnan(point_cloud_1).any(axis=1)
#     nan_indices_2 = np.isnan(point_cloud_2).any(axis=1)
    
#     # Find indices of pairs to be removed
#     remove_indices = np.logical_or(nan_indices_1, nan_indices_2)
    
#     # Remove pairs from both point clouds
#     filtered_point_cloud_1 = point_cloud_1[~remove_indices]
#     filtered_point_cloud_2 = point_cloud_2[~remove_indices]

#     return filtered_point_cloud_1, filtered_point_cloud_2


def remove_nan_pairs(point_cloud_1, point_cloud_2):
    # Assuming point_cloud_1 and point_cloud_2 are PyTorch tensors

    # Find indices of NaN values in either point cloud
    nan_indices_1 = torch.isnan(point_cloud_1).any(dim=1)
    nan_indices_2 = torch.isnan(point_cloud_2).any(dim=1)

    # Find indices of pairs to be removed
    remove_indices = nan_indices_1 | nan_indices_2  # Logical OR operation

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

def Finetune_depth(weights, pose, edges, pointclouds, image_pair_correspondence, scale_factor, EssentialMap, model, transform, net_w, net_h):

    # # midas param start #
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
    # model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)

    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the last N layers
    for param in model.scratch.output_conv.parameters():
        param.requires_grad = True

    # for param in model.scratch.refinenet1.parameters():
    #     param.requires_grad = True

    # for param in model.scratch.refinenet2.parameters():
    #     param.requires_grad = True

    # for param in model.scratch.refinenet3.parameters():
    #     param.requires_grad = True

    # for param in model.scratch.refinenet4.parameters():
    #     param.requires_grad = True


    # Continue with optimizer and training
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    torch.cuda.empty_cache()
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
            # with torch.no_grad():
            prediction = process(device, model, model_type, image, (net_w, net_h), original_image_rgb.shape[1::-1],
                                optimize, False)
            # output
            # if output_path is not None:
            #     filename = os.path.join(
            #         output_path, os.path.splitext(os.path.basename(rgb_file_input))[0] + '-' + model_type
            #     )
                # utils.write_depth(filename, prediction, grayscale, bits=2)
                # utils.write_pfm(filename + ".pfm", prediction.astype(np.float32))
            
            predicted_depth_inverse = prediction


        epsilon = 0.00001
        percentile_threshold = 90
        predicted_depth = 1.0 / (epsilon + predicted_depth_inverse)  # left predicted_depth is depth in real
        h, w = predicted_depth.shape
        def torch_percentile(input_tensor, percentile):
            k = 1 + round(.01 * float(percentile) * (input_tensor.numel() - 1))
            return input_tensor.view(-1).kthvalue(int(k)).values.item()
        # remove points out of range in prediction
        threshold = torch_percentile(predicted_depth, percentile_threshold)
        predicted_depth = torch.where(predicted_depth > threshold, torch.tensor(float('nan')).to(predicted_depth.device), predicted_depth)
        predicted_depth = torch.where(predicted_depth <= 0, torch.tensor(float('nan')).to(predicted_depth.device), predicted_depth)



        # read intrinsics
        fx_frame1, fy_frame1, cx_frame1, cy_frame1 = get_intrinsics(sequence_name='freiburg2_xyz')
        target_width, target_height = 640,480

        points = torch.zeros((y_frame1.shape[0], 3))
        # Convert numpy arrays to PyTorch tensors
        y_frame1_tensor = torch.from_numpy(y_frame1).to(predicted_depth.device)
        x_frame1_tensor = torch.from_numpy(x_frame1).to(predicted_depth.device)

        # Assuming target_height and target_width are scalar values, convert them to tensors
        target_height_tensor = torch.tensor(target_height, device=predicted_depth.device)
        target_width_tensor = torch.tensor(target_width, device=predicted_depth.device)

        # Calculate ceil and floor values
        y_frame1_ceil = torch.ceil(y_frame1_tensor).to(torch.int64)
        y_frame1_floor = torch.floor(y_frame1_tensor).to(torch.int64)
        x_frame1_ceil = torch.ceil(x_frame1_tensor).to(torch.int64)
        x_frame1_floor = torch.floor(x_frame1_tensor).to(torch.int64)

        # Adjust the values at the boundaries
        y_frame1_ceil = torch.where(y_frame1_ceil == target_height_tensor, target_height_tensor - 1, y_frame1_ceil)
        x_frame1_ceil = torch.where(x_frame1_ceil == target_width_tensor, target_width_tensor - 1, x_frame1_ceil)

        # Perform depth calculations
        depth1 = predicted_depth[y_frame1_ceil.long(), x_frame1_ceil.long()]
        depth2 = predicted_depth[y_frame1_ceil.long(), x_frame1_floor.long()]
        depth3 = predicted_depth[y_frame1_floor.long(), x_frame1_ceil.long()]
        depth4 = predicted_depth[y_frame1_floor.long(), x_frame1_floor.long()]
        predicted_depth = (depth1 + depth2 + depth3 + depth4) / 4

        # Update points tensor
        # Ensure points is a PyTorch tensor and on the same device as predicted_depth
        points[:, 2] = predicted_depth.view(-1)
        points[:, 0] = (x_frame1_tensor - cx_frame1) * predicted_depth / fx_frame1
        points[:, 1] = (y_frame1_tensor - cy_frame1) * predicted_depth / fy_frame1

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
            # with torch.no_grad():
            prediction = process(device, model, model_type, image, (net_w, net_h), original_image_rgb.shape[1::-1],
                                optimize, False)
            # output
            # if output_path is not None:
            #     filename = os.path.join(
            #         output_path, os.path.splitext(os.path.basename(rgb_file_input))[0] + '-' + model_type
            #     )
            #     utils.write_depth(filename, prediction, grayscale, bits=2)
            #     utils.write_pfm(filename + ".pfm", prediction.astype(np.float32))
            predicted_depth_inverse = prediction


        epsilon = 0.00001
        predicted_depth = 1.0 / (epsilon + predicted_depth_inverse)  # left predicted_depth is depth in real
        h, w = predicted_depth.shape
        # remove points out of range in prediction
        threshold = torch_percentile(predicted_depth, percentile_threshold)
        predicted_depth = torch.where(predicted_depth > threshold, torch.tensor(float('nan')).to(predicted_depth.device), predicted_depth)
        predicted_depth = torch.where(predicted_depth <= 0, torch.tensor(float('nan')).to(predicted_depth.device), predicted_depth)

        fx_frame2, fy_frame2, cx_frame2, cy_frame2 = get_intrinsics('freiburg2_xyz')            

        # construct points
        points = torch.zeros((y_frame2.shape[0], 3))
        # Convert numpy arrays to PyTorch tensors
        y_frame2_tensor = torch.from_numpy(y_frame2).to(predicted_depth.device)
        x_frame2_tensor = torch.from_numpy(x_frame2).to(predicted_depth.device)

        # Assuming target_height and target_width are scalar values, convert them to tensors
        target_height_tensor = torch.tensor(target_height, device=predicted_depth.device)
        target_width_tensor = torch.tensor(target_width, device=predicted_depth.device)

        # Calculate ceil and floor values
        y_frame2_ceil = torch.ceil(y_frame2_tensor).to(torch.int64)
        y_frame2_floor = torch.floor(y_frame2_tensor).to(torch.int64)
        x_frame2_ceil = torch.ceil(x_frame2_tensor).to(torch.int64)
        x_frame2_floor = torch.floor(x_frame2_tensor).to(torch.int64)

        # Adjust the values at the boundaries
        y_frame2_ceil = torch.where(y_frame2_ceil == target_height_tensor, target_height_tensor - 1, y_frame2_ceil)
        x_frame2_ceil = torch.where(x_frame2_ceil == target_width_tensor, target_width_tensor - 1, x_frame2_ceil)

        # Perform depth calculations
        depth1 = predicted_depth[y_frame2_ceil.long(), x_frame2_ceil.long()]
        depth2 = predicted_depth[y_frame2_ceil.long(), x_frame2_floor.long()]
        depth3 = predicted_depth[y_frame2_floor.long(), x_frame2_ceil.long()]
        depth4 = predicted_depth[y_frame2_floor.long(), x_frame2_floor.long()]
        predicted_depth = (depth1 + depth2 + depth3 + depth4) / 4

        # Update points tensor
        # Ensure points is a PyTorch tensor and on the same device as predicted_depth
        points[:, 2] = predicted_depth.view(-1)
        points[:, 0] = (x_frame2_tensor - cx_frame2) * predicted_depth / fx_frame2
        points[:, 1] = (y_frame2_tensor - cy_frame2) * predicted_depth / fy_frame2

        scaled_cloud_camera_frame[key][frame2] = points


    for key, value in scaled_cloud_camera_frame.items():

        frame1, frame2 = key
        point_frame1 = scaled_cloud_camera_frame[key][frame1]
        point_frame2 = scaled_cloud_camera_frame[key][frame2]

        point_frame1, point_frame2 = remove_nan_pairs(point_frame1, point_frame2)

        point_frame1, point_frame2 = scale_points(point_frame1, point_frame2, scale_factor)
        scaled_cloud_camera_frame[key][frame1] = point_frame1
        scaled_cloud_camera_frame[key][frame2] = point_frame2


    translations = []
    rotations = []
    scales = []
    N = pose['t_est'].shape[1]
    for i in range(N):         
        translations.append(pose['t_est'][:, i])
        rotations.append(pose['R_est'][3*i:3*(i+1), :])
        scales.append(pose['s_est'][i])
    
    translations = torch.tensor(translations)
    rotations = torch.tensor(rotations)
    scales = torch.tensor(scales)





    edges = list(scaled_cloud_camera_frame.keys())
    for j in range(len(edges)):
        for i in range(N):
            if edges[j][0] == EssentialMap[i]:
                edges[j] = (i,edges[j][1])
            if edges[j][1] == EssentialMap[i]:
                edges[j] = (edges[j][0],i) 
    pointclouds = list(scaled_cloud_camera_frame.values())
    pointclouds = []
    for pointclouds_pair in list(scaled_cloud_camera_frame.values()):
        pointclouds_pair_list = list(pointclouds_pair.values())
        Pi = pointclouds_pair_list[0].T
        Pj = pointclouds_pair_list[1].T
        combined_array = torch.vstack((Pi, Pj))
        pointclouds.append(combined_array)
    

    pointclouds_teaser = [pc.detach().cpu().numpy() for pc in pointclouds]


    solution, weights = TEASER_SimSync(N, edges, pointclouds_teaser, reg_lambda=0)


    frame_pose_index_pair = {0:0, 108:1, 199:2}
    loss_function = 0 
    index_weights = 0
    for index,(key,value) in enumerate(scaled_cloud_camera_frame.items()):

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

        pointclouds_frame1_frame2 = pointclouds[index]
        weights_frame1_frame2 = torch.tensor(weights[index_weights:pointclouds_frame1_frame2.shape[1]+index_weights])
        index_weights = index_weights+pointclouds_frame1_frame2.shape[1]

        pointcloud_frame1 = pointcloud_frame1.double()
        pointcloud_frame2 = pointcloud_frame2.double()
        
        weighted_norms = weights_frame1_frame2 * (torch.norm(scales_frame1 * rotation_frame1 @ pointcloud_frame1.T + translation_frame1.reshape(translation_frame1.size()[0],1) - 
                                    (scales_frame2 * rotation_frame2 @ pointcloud_frame2.T + translation_frame2.reshape(translation_frame2.size()[0],1)), dim=0)**2)

        loss_function += torch.sum(weighted_norms)

    optimizer.zero_grad()
    loss_function.backward()
    optimizer.step()

    torch.cuda.empty_cache()

    print(f"Loss: {loss_function.item()}")

    # torch.save(model, 'weights/dpt_beit_large_512.pt')
    torch.save(model.state_dict(), 'weights/dpt_beit_large_512.pt')

    return loss_function.item()

