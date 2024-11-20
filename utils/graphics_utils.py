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
#  1) Added LitePointCloud class
#  2) Added create_diff_world_to_view_matrix function
#  3) Added getProjectionMatrix2 function
#  4) Added verts_clip_to_ndc function
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)

def compute_face_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0)
    return face_normals

def compute_face_orientation(verts, faces, return_scale=False):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

    a0 = safe_normalize(v1 - v0)
    a1 = safe_normalize(torch.cross(a0, v2 - v0))
    a2 = -safe_normalize(torch.cross(a1, a0))  # will have artifacts without negation

    orientation = torch.cat([a0[..., None], a1[..., None], a2[..., None]], dim=-1)

    if return_scale:
        s0 = length(v1 - v0)
        s1 = dot(a2, (v2 - v0)).abs()
        scale = (s0 + s1) / 2
    return orientation, scale

def compute_vertex_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0) # the cross product of two vectors is perpendicular to them
    v_normals = torch.zeros_like(verts)
    N = verts.shape[0]
    v_normals.scatter_add_(1, i0[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i1[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i2[..., None].repeat(N, 1, 3), face_normals)

    v_normals = torch.where(dot(v_normals, v_normals) > 1e-20, v_normals, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
    v_normals = safe_normalize(v_normals)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_normals))
    return v_normals





# Copyright (c) 2024 DejaVu Authors. All rights reserved. 
#
# The University of British Columbia, its affiliates and licensors 
# retain all intellectual property and proprietary rights in and to 
# this material, related documentation and any modifications thereto. 

class LitePointCloud():
    ## Peizhi: the classic point cloud only has points and their colors
    def __init__(self, points: np.array, colors: np.array = None):
        self.points = points
        self.colors = colors

def getProjectionMatrix2(image_height, image_width, fovy, znear, zfar, flip_y: bool=False, z_sign=-1):
    # Peizhi
    # based on viewer_utils.py: projection_from_intrinsics
    w = image_width
    h = image_height
    focal = image_height / (2 * math.tan(np.radians(fovy) / 2))
    fx = fy = focal
    cx = image_width // 2
    cy = image_height // 2
    proj = np.zeros([1, 4, 4], dtype=np.float32)
    proj[:, 0, 0]  = fx * 2 / w 
    proj[:, 1, 1]  = fy * 2 / h
    proj[:, 0, 2]  = (w - 2 * cx) / w
    proj[:, 1, 2]  = (h - 2 * cy) / h
    proj[:, 2, 2]  = z_sign * (zfar+znear) / (zfar-znear)
    proj[:, 2, 3]  = -2*zfar*znear / (zfar-znear)
    proj[:, 3, 2]  = z_sign
    if flip_y:
        proj[:, 1, 1] *= -1
    return proj


def create_diff_world_to_view_matrix(cam_pose):
    # Author: Peizhi Yan
    # Date: Mar. 12, 2024
    # Description:
    #   Takes the camera pose tensor of shape [yaw, pitch, roll, dx, dy, dz] as input
    #   Computes the World2View matrix Rt: [4,4]
    #   This process is differentiable.
    #   Note that the camera pose are represented in radians
    device = cam_pose.device
    yaw, pitch, roll = cam_pose[:3]   # Unpack to yaw, pitch, roll

    # Rotation matrix for yaw (around the y-axis)
    Ry = torch.stack([
        torch.stack([torch.cos(yaw), torch.tensor(0., device=device), torch.sin(yaw)]),
        torch.stack([torch.tensor(0., device=device), torch.tensor(1., device=device), torch.tensor(0., device=device)]),
        torch.stack([-torch.sin(yaw), torch.tensor(0., device=device), torch.cos(yaw)])
    ]).to(device)

    # Rotation matrix for pitch (around the x-axis)
    Rx = torch.stack([
        torch.stack([torch.tensor(1., device=device), torch.tensor(0., device=device), torch.tensor(0., device=device)]),
        torch.stack([torch.tensor(0., device=device), torch.cos(pitch), -torch.sin(pitch)]),
        torch.stack([torch.tensor(0., device=device), torch.sin(pitch), torch.cos(pitch)])
    ]).to(device)

    # Rotation matrix for roll (around the z-axis)
    Rz = torch.stack([
        torch.stack([torch.cos(roll), -torch.sin(roll), torch.tensor(0., device=device)]),
        torch.stack([torch.sin(roll), torch.cos(roll), torch.tensor(0., device=device)]),
        torch.stack([torch.tensor(0., device=device), torch.tensor(0., device=device), torch.tensor(1., device=device)])
    ]).to(device)
    
    R = Rz @ Ry @ Rx    # Combine rotations
    
    # Construct the 4x4 world-to-view matrix Rt
    Rt = torch.eye(4).to(device)
    Rt[:3, :3] = R           # rotation
    Rt[:3, 3] = cam_pose[3:] # translation
    return Rt

def verts_clip_to_ndc(verts_clip : torch.tensor, image_size, out_dim=2):
    # Author: Peizhi Yan
    # transform the clipped vertices to NDC
    # this function is differentiable
    # inputs:
    #    - verts_clip: torch.tensor of shape [1, N, 3] where N is the number of vertices
    #    - image_size: int   it should match the rendered image size (assume height == width)
    #    - out_dim:    int   the output dimension be either 2 for 2D or 3 for 3D
    # returns:
    #    - verts_ndc: torch.tensor of shape [N, out_dim]
    verts_ndc = verts_clip[:, :, :3] / verts_clip[:, :, 3:]
    verts_ndc = verts_ndc[0, :, :out_dim]
    verts_ndc[:,1] *= -1 # flip y
    verts_ndc = (verts_ndc / 2.0 + 0.5) * image_size # de-normalize
    return verts_ndc

