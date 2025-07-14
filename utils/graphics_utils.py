#
# Peizhi Yan
# Copyright (C) 2025
#

import torch
import math
import numpy as np
import cv2

def compute_vertex_normals(verts, faces):
    # verts: [V, 3]
    # faces: [F, 3]
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0)  # [F, 3]
    face_normals = face_normals / (face_normals.norm(dim=-1, keepdim=True) + 1e-8)
    # Accumulate per-vertex normals
    normals = torch.zeros_like(verts)
    normals.index_add_(0, faces[:, 0], face_normals)
    normals.index_add_(0, faces[:, 1], face_normals)
    normals.index_add_(0, faces[:, 2], face_normals)
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
    return normals

def euler_to_matrix(yaw, pitch, roll):
    """Convert yaw, pitch, roll to a rotation matrix (B, 3, 3)"""
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    zeros = torch.zeros_like(cy)
    ones = torch.ones_like(cy)
    Rz = torch.stack([
        torch.stack([cr, -sr, zeros], dim=-1),
        torch.stack([sr, cr, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1)
    ], dim=-2)
    Ry = torch.stack([
        torch.stack([cy, zeros, sy], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-sy, zeros, cy], dim=-1)
    ], dim=-2)
    Rx = torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cp, -sp], dim=-1),
        torch.stack([zeros, sp, cp], dim=-1)
    ], dim=-2)
    return Rz @ Ry @ Rx

def build_view_matrix(camera_pose):
    """Build [B, 4, 4] world-to-view matrix from [B, 6] pose"""
    B = camera_pose.shape[0]
    yaw, pitch, roll = camera_pose[:, 0], camera_pose[:, 1], camera_pose[:, 2]
    tx, ty, tz = camera_pose[:, 3], camera_pose[:, 4], camera_pose[:, 5]
    R = euler_to_matrix(yaw, pitch, roll)
    #T = torch.stack([tx, ty, tz], dim=-1).unsqueeze(-1)  # [B, 3, 1]
    Rt = torch.eye(4, device=camera_pose.device, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)
    Rt[:, :3, :3] = R
    Rt[:, :3, 3] = torch.stack([tx, ty, tz], dim=-1)
    return Rt  # [B, 4, 4]

# def fov_to_focal(fov, sensor_size):
#     fov_rad = math.radians(fov)
#     focal_length = 0.5 * sensor_size / math.tan(0.5 * fov_rad)
#     return focal_length

def fov_to_focal(fov : torch.Tensor, sensor_size : int):
    """
    Args:
        fov : either an int/float value, or a tensor, or a batch of tensors [N]
        sensor_size : int  we assume the image has equal height and width
    Returns:
        focal_length : tensor [N]
    """
    if not torch.is_tensor(fov):
        # when fov is either an int/float value, convert it to tensor first
        fov = torch.as_tensor(fov, dtype=torch.float32)#[None]
    if len(fov.shape) == 0:
        # extend 0-d tensor to batch
        fov = fov[None]
    fov_rad = torch.deg2rad(fov)
    focal_length = 0.5 * sensor_size / torch.tan(0.5 * fov_rad)
    return focal_length

# def build_intrinsics(batch_size, image_size, focal_length, device='cuda'):
#     H = W = image_size
#     fx = fy = focal_length
#     cx = W / 2.0
#     cy = H / 2.0
#     K = torch.tensor([
#         [fx, 0, cx],
#         [0, fy, cy],
#         [0, 0, 1.0]
#     ], dtype=torch.float32, device=device)
#     K = K.unsqueeze(0).expand(batch_size, -1, -1).clone()
#     return K

def build_intrinsics(focal_length : torch.Tensor, image_size : int):
    """
    Args:
        focal_length : tensor [N]
        image_size : int (assumes square image)
    Returns:
        K : tensor [N, 3, 3]
    """
    batch_size = focal_length.shape[0]
    device = focal_length.device
    H = W = image_size
    cx = W / 2.0
    cy = H / 2.0
    # Each row as a [N, 3] tensor
    row0 = torch.stack([focal_length, torch.zeros(batch_size, device=device), torch.full((batch_size,), cx, device=device)], dim=1)
    row1 = torch.stack([torch.zeros(batch_size, device=device), focal_length, torch.full((batch_size,), cy, device=device)], dim=1)
    row2 = torch.tensor([0., 0., 1.], device=device).expand(batch_size, 3)
    # Stack rows into final K [N, 3, 3]
    K = torch.stack([row0, row1, row2], dim=1)
    return K

# def projection_from_intrinsics(K, image_size, z_near=0.1, z_far=10.0, z_sign=1):
#     B = K.shape[0]
#     h = w = image_size
#     fx = K[..., 0, 0]
#     fy = K[..., 1, 1]
#     cx = K[..., 0, 2]
#     cy = K[..., 1, 2]
#     proj = torch.zeros((B, 4, 4), device=K.device, dtype=K.dtype)
#     proj[:, 0, 0] = 2 * fx / w
#     proj[:, 1, 1] = 2 * fy / h
#     proj[:, 0, 2] = (w - 2 * cx) / w
#     proj[:, 1, 2] = (h - 2 * cy) / h
#     proj[:, 2, 2] = z_sign * (z_far + z_near) / (z_far - z_near)
#     proj[:, 2, 3] = -2 * z_far * z_near / (z_far - z_near)
#     proj[:, 3, 2] = z_sign
#     return proj

def projection_from_intrinsics(K, image_size, z_near=0.1, z_far=10.0, z_sign=1):
    """ Build projection matrices from camera intrinsic matrices """
    h = w = image_size
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    zeros = torch.zeros_like(fx)
    z_signs = torch.full_like(fx, float(z_sign))
    z22_value = float(z_sign * (z_far + z_near) / (z_far - z_near))
    z23_value = float(-2 * z_far * z_near / (z_far - z_near))
    z22 = torch.full_like(fx, z22_value)
    z23 = torch.full_like(fx, z23_value)
    row0 = torch.stack([2 * fx / w, zeros, (w - 2 * cx) / w, zeros], dim=-1)
    row1 = torch.stack([zeros, 2 * fy / h, (h - 2 * cy) / h, zeros], dim=-1)
    row2 = torch.stack([zeros, zeros, z22, z23], dim=-1)
    row3 = torch.stack([zeros, zeros, z_signs, zeros], dim=-1)
    proj = torch.stack([row0, row1, row2, row3], dim=1)  # [B, 4, 4]
    return proj

def batch_perspective_projection(verts, camera_pose, K, image_size, near=0.1, far=10.0):
    """
        - verts: [B, V, 3]
        - camera_pose: [B, 6]
        - K: [B, 3, 3] or [1, 3, 3]
    """
    B = verts.shape[0]
    device = verts.device

    # uild world-to-view matrix
    Rt = build_view_matrix(camera_pose)   # [B, 4, 4]
    Rt[:, :, [1,2]] *= -1                 # OpenCV: flip y, z
    world2view = torch.linalg.inv(Rt)     # [B, 4, 4]

    # Build projection matrix
    proj = projection_from_intrinsics(K, image_size, near, far, z_sign=1) # [B, 4, 4] or [1, 4, 4]

    if proj.shape[0] == 1 and world2view.shape[0] > 1:
        # Expand proj (when proj is [1, 4, 4] but batch size > 1) to match batch size
        proj = proj.expand(world2view.shape[0], -1, -1)
    
    # Full transform (proj @ view)
    full_proj = torch.bmm(proj, world2view)      # [B, 4, 4]

    # Flip y after projection
    full_proj[:, 1, :] *= -1

    # Project verts
    verts_h = torch.cat([verts, torch.ones(B, verts.shape[1], 1, device=device)], dim=-1)  # [B, V, 4]
    verts_clip = torch.bmm(verts_h, full_proj.transpose(-1, -2)) # [B, V, 4]
    return verts_clip

def batch_verts_clip_to_ndc(verts_clip):
    ndc = verts_clip[:, :, :3] / (verts_clip[:, :, 3:4] + 1e-8)
    ndc[:, :, 1] *= -1  # flip y
    return ndc  # [B, V, 3]

def batch_verts_ndc_to_screen(ndc, image_size):
    screen = (ndc / 2.0 + 0.5) * image_size
    return screen




# if __name__ == '__main__':
#     # unit test
#     fov = torch.tensor([12, 20, 25])
#     sensor_size = 512
#     focal_length = fov_to_focal(fov, sensor_size)

#     print(focal_length)

#     K = build_intrinsics(focal_length=focal_length, image_size=sensor_size)

#     print(K.shape)



