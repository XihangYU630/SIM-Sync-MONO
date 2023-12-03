# Xihang Yu
# 06/01/2023

from os.path import join as pjoin
import os
import cv2
import numpy as np
import os.path as osp
import re
import pickle
from pose_optimization import PoseOptimizer
import time
from collections import defaultdict
from scipy.stats import chi2
import pandas as pd
from TEASER_SIM_Sync import TEASER_SimSync


class PoseOptimizerSintel(PoseOptimizer):
    def __init__(self, gt_path='/media/xihang/Elements', 
                        pred_path = '/media/xihang/Elements/solvers/realdata_experiments/SDP_realdata/mountain_1_output/dpt_beit_large_512_0-50'):

        super().__init__(gt_path, pred_path)

        self.use_gt_flow = False
        self.use_gt_depth = True
        self.pointCloudPairSeparateDistance = 0
        
        pattern = r"/(\w+)_output/"
        match = re.search(pattern, self.pred_path)
        self.sequence_name = match.group(1)

        # Extract the range
        match = re.search(r'(\d+-\d+)$', self.pred_path)
        range_value = match.group(1)
        self.start_frame_optimizer, self.end_frame_optimizer = map(int,range_value.split('-'))

        if self.sequence_name == 'mountain_1' or self.sequence_name == 'alley_2' or self.sequence_name == 'bamboo_1'\
            or self.sequence_name == 'bamboo_2' or self.sequence_name == 'alley_1' or self.sequence_name == 'temple_2'\
            or self.sequence_name == 'sleeping_1' or self.sequence_name == 'market_5' or self.sequence_name == 'sleeping_2'\
            or self.sequence_name == 'shaman_3':
            self.whole_sequence_len = 50
        self.target_width, self.target_height = 1024,436
        self.ground_truth_dir = self.gt_path + '/MPI-Sintel-depth-training-20150305/training/camdata_left/' \
                                                                        + self.sequence_name + "/"


    def depth_recad(self,filename):
        """ Read depth data from file, return as numpy array. """
        f = open(filename,'rb')
        check = np.fromfile(f,dtype=np.float32,count=1)[0]
        assert check == 202021.25, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '\
                                                                                                .format(202021.25,check)
        width = np.fromfile(f,dtype=np.int32,count=1)[0]
        height = np.fromfile(f,dtype=np.int32,count=1)[0]
        size = width*height
        assert width > 0 and height > 0 and size > 1 and size < 100000000,\
                            ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
        depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
        return depth

    def cam_read(self,filename):
        """ Read camera data, return (M,N) tuple.
        
        M is the intrinsic matrix, N is the extrinsic matrix, so that

        x = M*N*X,
        with x being a point in homogeneous image pixel coordinates, X being a
        point in homogeneous world coordinates.
        """
        f = open(filename,'rb')
        check = np.fromfile(f,dtype=np.float32,count=1)[0]
        assert check == 202021.25, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). \
                                                    Big-endian machine? '.format(202021.25,check)
        M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
        N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
        return M,N

    def read_flo_file(self, file_path):
        with open(file_path, "rb") as file:
            # Read the magic number
            magic = file.read(4)
            if magic != b'PIEH':
                raise ValueError("Invalid .flo file format")

            # Read the width and height of the flow map
            width = int.from_bytes(file.read(4), byteorder='little')
            height = int.from_bytes(file.read(4), byteorder='little')

            # Read the flow vectors
            flow = np.frombuffer(file.read(), dtype=np.float32)
            flow = flow.reshape((height, width, 2))

        return flow
    

    def get_gt_flow_constraints(self):

        flow_gt_dir = self.gt_path + "/MPI-Sintel-complete/training/flow/" + self.sequence_name
        flow_names = sorted(os.listdir(flow_gt_dir))
        image_pair_correspondence = {}
        for flow_name in flow_names:
            
            flow = self.read_flo_file(osp.join(flow_gt_dir, flow_name))

            # Extract image index
            pattern = r"\d+"
            match = re.search(pattern, flow_name)
            if match:
                index = int(match.group(0)) - 1 # -1 since we want to start at 0
            else:
                print("Number not found in the file name.")

            # Create a 2D matrix of pixel positions
            x_coords = np.arange(self.target_width)
            y_coords = np.arange(self.target_height)
            x_matrix, y_matrix = np.meshgrid(x_coords, y_coords)
            pixel_matrix_image1 = np.stack((y_matrix, x_matrix), axis=2)
            mapped_pixel_matrix_image1 = pixel_matrix_image1 + flow

            # mask out the corresponding pixel that has distance less than delta pixels
            ########################################
            ################ method1 ###############
            ########################################
            h = np.arange(self.target_height)[:, np.newaxis]
            w = np.arange(self.target_width)
            distance = np.sqrt((mapped_pixel_matrix_image1[h, w, 0] - h) ** 2 + (mapped_pixel_matrix_image1[h, w, 1] - w) ** 2)
            indices = distance < 0
            mapped_pixel_matrix_image1[indices, 0] = -1
            mapped_pixel_matrix_image1[indices, 1] = -1

            ########################################
            ################ method2 ###############
            ########################################
            # mapped_pixel_matrix_image1 = np.round(mapped_pixel_matrix_image1).astype(np.int32)

            # only keep those pixels that are in range of size of image2
            mask_lower_bound_y = mapped_pixel_matrix_image1[:,:,0] >= 0
            mask_upper_bound_y = mapped_pixel_matrix_image1[:,:,0] < self.target_height
            mask_lower_bound_x = mapped_pixel_matrix_image1[:,:,1] >= 0
            mask_upper_bound_x = mapped_pixel_matrix_image1[:,:,1] < self.target_width
            mask_range_y = np.logical_and(mask_lower_bound_y, mask_upper_bound_y)
            mask_range_x = np.logical_and(mask_lower_bound_x, mask_upper_bound_x)
            mask_range = np.logical_and(mask_range_y,mask_range_x)

            # mask out the dynamic objects
            mask_fmt = pjoin(self.pred_path, "dynamic_mask", "frame_{:06d}.png")
            dynamic_mask_file = mask_fmt.format(index)
            mask_image_dynamic_objects = cv2.imread(dynamic_mask_file, cv2.IMREAD_GRAYSCALE)
            mask_dynamic_objects = mask_image_dynamic_objects == 255


            mapped_pixel_matrix_image1[mask_range==False] = np.array((-1,-1))
            mapped_pixel_matrix_image1[mask_dynamic_objects==False] = np.array((-1,-1))

            filtered_mapped_pixel_matrix_image1 = mapped_pixel_matrix_image1 
            image_pair_correspondence[(index, index+1)] = filtered_mapped_pixel_matrix_image1
        
        return image_pair_correspondence
    
    def get_gt_depth_file_list(self):
        depth_gt = self.gt_path + '/MPI-Sintel-depth-training-20150305/training/depth/' + self.sequence_name + '/'
        gt_depth_file_list = []
        for i in range(self.start_frame_optimizer+1, self.end_frame_optimizer+1):
            gt_depth_map = f"frame_{str(i).zfill(4)}.dpt"
            gt_depth_file_list.append(gt_depth_map)
        
        return depth_gt, gt_depth_file_list
    
    def get_gt_depth(self, depth_gt, gt_depth_file_list, i):
        gt_file = depth_gt + gt_depth_file_list[i]
        gt_depth = self.depth_recad(gt_file)
        # remove points out of range in gt
        # remove points out of range in prediction
        threshold = np.percentile(gt_depth, self.percentile_threshold)
        gt_depth = np.where(gt_depth > threshold, np.nan, gt_depth)
        gt_depth = np.where(gt_depth <= 0, np.nan, gt_depth)
        predicted_depth = gt_depth


        # if i == 0:
        #     self.s_gt.append(1)
        #     print("scale: ", 1)
        # else:
        #     scale = np.random.uniform(1, 1)
        #     print("scale: ", scale)
        #     self.s_gt.append(scale)
        #     predicted_depth = predicted_depth / scale
        
        return predicted_depth
    
    def get_intrinsics(self, i):

        ground_truth_dir = self.ground_truth_dir
        ground_truth_files = sorted(os.listdir(ground_truth_dir))

        ground_truth_file = osp.join(ground_truth_dir,ground_truth_files[i])
        Intrinsics, _ = self.cam_read(ground_truth_file)
        fx = Intrinsics[0,0] 
        fy = Intrinsics[1,1] 
        cx = Intrinsics[0,2] 
        cy = Intrinsics[1,2]
        return fx, fy, cx, cy

    def get_gt_traj(self):

        # Extract gt translation and rotation
        ground_truth_dir = self.ground_truth_dir
        trans_list_ground_truth = []
        rotation_list_ground_truth = []
        ground_truth_files = sorted(os.listdir(ground_truth_dir))
        for i in range(len(ground_truth_files)):
            ground_truth_file = osp.join(ground_truth_dir,ground_truth_files[i])
            Intrinsics, Extrinsics = self.cam_read(ground_truth_file)
            trans_list_ground_truth.append(Extrinsics[0:3,3])
            rotation_list_ground_truth.append(Extrinsics[0:3,0:3])
        trans_ground_truth = np.array(trans_list_ground_truth)
        rotation_ground_truth = np.array(rotation_list_ground_truth)

        # Anchor the first timestamp
        rotation_matrix_reshaped = rotation_ground_truth.reshape(-1, 3)
        first_rotation = rotation_matrix_reshaped[:3, :]
        self.R_gt = np.zeros((3*len(trans_ground_truth), 3))
        self.t_gt =np.zeros((len(trans_ground_truth),3))
        for i in range(len(trans_ground_truth)):
            self.R_gt[3*i:3*(i+1), :] = first_rotation @ np.linalg.inv(rotation_matrix_reshaped[3*i:3*(i+1), :])
            self.t_gt[i,:] = -first_rotation @ np.linalg.inv(rotation_matrix_reshaped[3*i:3*(i+1), :]) \
                                                                @ trans_ground_truth[i] + trans_ground_truth[0]
    
    def get_relative_pose(self, cloud_pair):
        ground_truth_dir = self.ground_truth_dir
        ground_truth_files = sorted(os.listdir(ground_truth_dir))
        frame1 = cloud_pair[0]
        frame2 = cloud_pair[1]
        ground_truth_file1 = osp.join(ground_truth_dir,ground_truth_files[frame1])
        ground_truth_file2 = osp.join(ground_truth_dir,ground_truth_files[frame2])

        Intrinsics_src, Extrinsics_src = self.cam_read(ground_truth_file1)
        Intrinsics_dst, Extrinsics_dst = self.cam_read(ground_truth_file2)

        rotation_src = Extrinsics_src[0:3,0:3]
        translation_src = Extrinsics_src[0:3,3]
        rotation_dst = Extrinsics_dst[0:3,0:3]
        translation_dst = Extrinsics_dst[0:3,3]

        rel_rot = rotation_dst @ np.linalg.inv(rotation_src)
        rel_trans = -rotation_dst @ np.linalg.inv(rotation_src) @ translation_src + translation_dst

        rel_pose = np.zeros((4,4))
        rel_pose[0:3,0:3] = rel_rot
        rel_pose[0:3,3] = rel_trans
        rel_pose[3,3] = 1
        # rel_pose = Extrinsics_src @ np.linalg.inv(Extrinsics_dst)

        return rel_pose


    def get_relative_pose_given_frames(self, frame1, frame2):
        ground_truth_dir = self.ground_truth_dir
        ground_truth_files = sorted(os.listdir(ground_truth_dir))
        ground_truth_file1 = osp.join(ground_truth_dir,ground_truth_files[frame1])
        ground_truth_file2 = osp.join(ground_truth_dir,ground_truth_files[frame2])

        Intrinsics_src, Extrinsics_src = self.cam_read(ground_truth_file1)
        Intrinsics_dst, Extrinsics_dst = self.cam_read(ground_truth_file2)

        rotation_src = Extrinsics_src[0:3,0:3]
        translation_src = Extrinsics_src[0:3,3]
        rotation_dst = Extrinsics_dst[0:3,0:3]
        translation_dst = Extrinsics_dst[0:3,3]

        rel_rot = rotation_dst @ np.linalg.inv(rotation_src)
        rel_trans = -rotation_dst @ np.linalg.inv(rotation_src) @ translation_src + translation_dst

        rel_pose = np.zeros((4,4))
        rel_pose[0:3,0:3] = rel_rot
        rel_pose[0:3,3] = rel_trans
        rel_pose[3,3] = 1
        # rel_pose = Extrinsics_src @ np.linalg.inv(Extrinsics_dst)

        return rel_pose # convert a point in frame 1 to frame 2


def displayResults(results, inliers):
    err = []
    time = []
    itr = []
    alg_name = []
    field_names = results.keys()
    outliers = np.setdiff1d(np.arange(0, problem['NumberMeasurements']), inliers)
    for field_name in field_names:
        err.append(results[field_name]['f_val'])
        time.append(results[field_name]['time'])
        itr.append(results[field_name]['iterations'])
        alg_name.append(results[field_name]['algname'])
    
    T = pd.DataFrame({'Error': err, 'Itr': itr, 'Time': time, 'Outliers': len(outliers), 'Number Measurements':  problem['NumberMeasurements']}, index=alg_name)
    print(T)

if __name__ == "__main__":

    pose_optimizer = PoseOptimizerSintel()
    # image_pair_correspondence: (frame1, frame2) -> (h,w,2)

    print("Get flow constraints:")
    image_pair_correspondence = pose_optimizer.get_flow_constraints()
    print("Get depth maps:")
    scaled_cloud_camera_frame = pose_optimizer.get_depth_map()
    print("Match points in frame pairs:")
    pose_optimizer.match_frame_pair_points(image_pair_correspondence, scaled_cloud_camera_frame)
    print("Solve SDP:")
    # solution = pose_optimizer.bundle_adjustment_denseSOS_mosek()
    
    N = pose_optimizer.N
    edges = list(pose_optimizer.scaled_cloud_camera_frame_dict.keys())
    pointclouds = list(pose_optimizer.scaled_cloud_camera_frame_dict.values())
    pointclouds = []
    for pointclouds_pair in list(pose_optimizer.scaled_cloud_camera_frame_dict.values()):
        pointclouds_pair_list = list(pointclouds_pair.values())
        Pi = pointclouds_pair_list[0].T
        Pj = pointclouds_pair_list[1].T
        combined_array = np.vstack((Pi, Pj))
        pointclouds.append(combined_array)

    ########################################
    ############## SimSync-gnc #############
    ########################################

    # print("########################################")
    # print("############## SimSync-gnc #############")
    # print("########################################")
    # problem = defaultdict()
    # problem['dof'] = 3
    # problem["MeasurementNoiseStd"] = 1.8*1e-2
    # epsilon = chi2.ppf(0.99, problem['dof']) * ((2*problem['MeasurementNoiseStd']) ** 2)
    # problem['f'] = SimSync
    # problem["N"] = N
    # problem["edges"] = edges
    # problem["pointclouds"] = pointclouds
    # problem["NumberMeasurements"] = 0
    # problem["priors"] = []
    # start = 0
    # for idx in range(len(edges)):
    #     pointcloud = pointclouds[idx]
    #     problem["NumberMeasurements"] += pointcloud.shape[1]
    #     if edges[idx][0] == 0 or edges[idx][1] == 0:
    #         problem["priors"].append(np.arange(start, start+pointcloud.shape[1]))
        
    #     start += pointcloud.shape[1]
    # problem["priors"] = np.concatenate(problem["priors"])
    # # Solvers
    # results = {}

    # # Solve with GNC
    # print("Solving with:")
    # print("  - GNC")
    # start = time.time()
    # inliers, info, weights = gnc(problem, problem['f'], NoiseBound=epsilon)
    # end = time.time()

    # print("Time taken (s): ", end - start)

    # results['gnc'] = {}
    # results['gnc']['algname'] = 'GNC'
    # solutionGNC = SimSync(N, edges, pointclouds, Weights=weights)
    # results['gnc']['f_val'] = solutionGNC['f_val']
    # results['gnc']['iterations'] = info['Iterations']
    # results['gnc']['time'] = info['time']
    # print('Results:')
    # displayResults(results, inliers)

    # if pose_optimizer.use_gt_depth == True:
    #     solutionGNC["t_est"] = solutionGNC["t_est"] / pose_optimizer.scale_factor

    # # Save the defaultdict to a file cloud
    # with open('solutionGNC.pkl', 'wb') as file:
    #     pickle.dump(solutionGNC, file)

    # print("Get ground truth trajectory:")
    # pose_optimizer.get_gt_traj()
    # print("Visualize camera trajectory:")
    # pose_optimizer.visCameraTraj(solution_path = 'solutionGNC.pkl')


    print("########################################")
    print("################ SimSync ###############")
    print("########################################")

    start = time.time()
    solution = TEASER_SimSync(N, edges, pointclouds)
    # solution = SimSync2(N, edges, pointclouds)
    end = time.time()

    print("Time taken (s): ", end - start)

    if pose_optimizer.use_gt_depth == True:
        solution["t_est"] = solution["t_est"] / pose_optimizer.scale_factor

    # Save the defaultdict to a file cloud
    with open('solution.pkl', 'wb') as file:
        pickle.dump(solution, file)

    print("Get ground truth trajectory:")
    pose_optimizer.get_gt_traj()
    print("Visualize camera trajectory:")
    pose_optimizer.visCameraTraj(solution_path = 'solution.pkl')