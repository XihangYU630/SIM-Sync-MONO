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

import numpy as np

from imutils.video import VideoStream
from midas.model_loader import default_models, load_model

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


def Finetune_depth(weights, pose, edges, pointclouds, image_pair_correspondence):

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
    N = number_files


    for key, value in image_pair_correspondence.items():

        frame1, frame2 = key
        x_frame1 = value[:, 0]
        y_frame1 = value[:, 1]
        x_frame2 = value[:, 2]
        y_frame2 = value[:, 3]
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
            predicted_depth_inverse = prediction
