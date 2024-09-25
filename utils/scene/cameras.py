#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

#
# Updated by Peizhi Yan
# Copyright (C) 2024
# 
# Updates:
#  1) Added PerspectiveCamera
# 

import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrix2


def prepare_camera(cam, device):
    # This is used by preparing camera for Gaussian renderer
    @dataclass
    class Cam:
        try:
            FoVx = float(np.radians(cam.FoVx))
            FoVy = float(np.radians(cam.FoVy))
        except:
            # for OrbitCamera
            FoVx = float(np.radians(cam.fovx))
            FoVy = float(np.radians(cam.fovy))            
        image_height = cam.image_height
        image_width = cam.image_width
        world_view_transform = torch.tensor(cam.world_view_transform).float().to(device)
        full_proj_transform = torch.tensor(cam.full_proj_transform).float().to(device)
        try:
            camera_center = cam.camera_center
        except:
            # for OrbitCamera
            camera_center = torch.tensor(cam.pose[:3, 3]).to(device)
    return Cam

class PerspectiveCamera(nn.Module):
    def __init__(self, Rt, fov, bg, image_width, image_height, znear=0.01, zfar=100.0):
        super(PerspectiveCamera, self).__init__()
        # Author: Peizhi Yan
        # - Rt: Pytorch tensor of the w2v matrix
        self.device = Rt.device
        self.FoVx = fov
        self.FoVy = fov
        self.bg = bg
        self.image_width = image_width
        self.image_height = image_height
        self.zfar = zfar
        self.znear = znear
        self.world_view_transform = Rt
        self.world_view_transform[:, [1,2]] *= -1 # opencv
        self.world_view_transform = self.world_view_transform.inverse()
        self.projection_matrix = getProjectionMatrix2(image_height, image_width, fov, znear, zfar, flip_y=False, z_sign=1)[0] # [4,4]
        self.projection_matrix = torch.from_numpy(self.projection_matrix).to(self.device).detach()
        self.full_proj_transform = self.projection_matrix @ self.world_view_transform
        self.camera_center = self.world_view_transform.T[3, :3]
        self.world_view_transform = self.world_view_transform.T
        self.full_proj_transform = self.full_proj_transform.T


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, bg, image_width, image_height, image_path, gt_alpha_mask,
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 timestep=None, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.bg = bg
        self.image_width = image_width
        self.image_height = image_height
        self.image_path = image_path
        self.image_name = image_name
        self.timestep = timestep

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)  #.cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)  #.cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, timestep):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.timestep = timestep

