
import numpy as np
import os,sys,time
import torch
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import time
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from jaxtyping import Float
from torch import Tensor
from typing import Any, Optional
import src.model.cameras.camera as camera
from src.model.cameras.align_trajectories import \
    (align_ate_c2b_use_a2b, align_translations, backtrack_from_aligning_and_scaling_to_first_cam, backtrack_from_aligning_the_trajectory)

from src.model.ray_diffusion.eval.utils import compute_angular_error_batch, compute_camera_center_error, n_to_np_rotations
from einops import rearrange

def initialize_noisy_poses(pose_gt_w2c: Float[Tensor, "batch view 3 4"], noise_level=0.05, gt_scene_scale=1.0):
    """ Define initial poses (to be optimized) """

    device = pose_gt_w2c.device
    pose_gt_w2c = rearrange(pose_gt_w2c, "batch view i j -> (batch view) i j")

    # get ground-truth (canonical) camera poses
    n_poses = pose_gt_w2c.shape[0]
    
    # fix the x first one to the GT poses and the ones to optimize to the GT ones 
    # + noise
    initial_poses_w2c = pose_gt_w2c.clone()
    n_first_fixed_poses = 0
    n_optimized_poses = n_poses - n_first_fixed_poses
    
    # same noise level in rotation and translation
    se3_noise = torch.randn(n_optimized_poses, 6, device=device)*noise_level
    pose_noise = camera.lie.se3_to_SE3(se3_noise)  # (n_optimized_poses, 3, 4)
    pose_noise = torch.cat((torch.eye(3, 4)[None, ...].repeat(n_first_fixed_poses, 1, 1).to(device), 
                            pose_noise), dim=0)
    
    initial_poses_w2c = camera.pose.compose([pose_noise, initial_poses_w2c])
    
    # R_pred_rel = n_to_np_rotations(n_poses, initial_poses_w2c[:, :3, :3]).clone().detach().cpu().numpy()
    # R_gt_rel = n_to_np_rotations(n_poses, pose_gt_w2c[:, :3, :3]).clone().detach().cpu().numpy()
    R_noisy_gt = initial_poses_w2c.clone().detach().cpu().numpy()[:, :3, :3]
    R_gt = pose_gt_w2c.clone().detach().cpu().numpy()[:, :3, :3]
    
    T_noisy_gt = initial_poses_w2c[:, :3, 3]
    T_gt = pose_gt_w2c[:, :3, 3]
    
    error_R = compute_angular_error_batch(rotation1=R_noisy_gt, rotation2=R_gt)
    error_T, _ = compute_camera_center_error(R_noisy_gt, T_noisy_gt, R_gt, T_gt, gt_scene_scale)
    
    # make it 4x4 by adding the last row
    initial_poses_w2c = torch.cat((initial_poses_w2c, torch.tensor([[[0., 0., 0., 1.]]], device=device).repeat(n_poses, 1, 1)), dim=1) 
    
    initial_poses_w2c = rearrange(initial_poses_w2c, "(batch view) i j -> batch view i j", view=n_poses) #* 3x4
    
    
    
            
    return initial_poses_w2c, error_R, error_T