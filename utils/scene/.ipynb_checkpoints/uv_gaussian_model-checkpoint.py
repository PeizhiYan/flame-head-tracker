#
# Copyright (C) 2024, DejaVu Authors 
# The copyright applies to the modified code segments and new functions.
#

#
# Copyright (C) 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid
# from pytorch3d.transforms import quaternion_multiply
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.graphics_utils import LitePointCloud
from utils.uv_rasterizer import UV_Rasterizer



def compute_init_scales(xyz: np.array):
    """
    Compute the initial Gaussian scales
    Author: Peizhi Yan
    --------------------------------------------------------------------
    inputs:
        - xyz: [G, 3]  G is number of Gaussians
    outputs:
        - scales: [G, 3]
    """
    xyz = np.array(xyz, dtype=np.float32)
    xyz = torch.from_numpy(xyz).cuda()
    dist2 = torch.clamp_min(distCUDA2(xyz), 0.0000001) 
    scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3) # [G, 3]
    scales = scales.detach().cpu().numpy() # [G, 3]
    return scales



def mesh_2_uv(uv_rasterizer: UV_Rasterizer, vertices: np.array, 
              colors: np.array = None, compute_scales = False, mouth_interior=False):
    """
    Convert mesh to UV Gaussians
    Author: Peizhi Yan
    --------------------------------------------------------------------
    inputs:
        - uv_rasterizer: the UV_Rasterizer object
        - vertices: [V, 3]
        - scales: [V, 3]    scales derived from the template mesh
        - colros: [V, 3]    values in 0 ~ 1.0
    returns:
        - uv_map: np.array [256, 256, 13] the UV representation of 3D 
                  Gaussians
    ---------------------------------------------------------------------
    Format of UV Gaussians map:
        [:, :, 0:3]  --> mean locations
        [:, :, 3:6]  --> scales
        [:, :, 6:9]  --> rotations
        [:, :, 9:10] --> opacities
        [:, :, 10:]  --> SH features
    """

    uv_size = uv_rasterizer.uv_size
    sh_channels = 3 # 3 for SH L0; 4 for SH L1; 9 for SH L2; 16 for SH L3
    uv_map = np.zeros([uv_size, uv_size, 10 + sh_channels], dtype=np.float32)
    
    with torch.no_grad():
        
        if colors is None:
            colors = np.zeros(vertices.shape, dtype=np.float32)
            
        # convert RGB colors to SH colors
        fused_color = RGB2SH(torch.tensor(colors).float()).cpu().numpy()
            
        # rasterize the vertex locations
        uv_pos = uv_rasterizer.rasterize(vertices, mouth_interior) # [256, 256, 3]
        uv_map[:,:,:3] = uv_pos
    
        # rasterize the colors
        uv_tex = uv_rasterizer.rasterize(fused_color)  # [256, 256, 3]
        
        # scales
        if compute_scales:
            ## something is wrong, seems that the scales is not correct..
            scales = compute_init_scales(xyz = uv_pos[uv_rasterizer.valid_coords])
            scales_min = scales.min(axis=0)
            scales_max = scales.max(axis=0)
            scales = (scales - scales_min) / (scales_max - scales_min) # normalize for rasterization
            uv_scales = uv_rasterizer.rasterize(scales) # [256,256,3]
            uv_scales = uv_scales * (scales_max - scales_min) + scales_min # denormalize to restore
            uv_map[:,:,3:6] = uv_scales
        else:
            uv_map[:,:,3:6] = -8.3533 * np.ones([uv_size, uv_size, 3], dtype=np.float32)

        # opacities
        uv_map[:,:,9:10] = 0.1 * np.ones([uv_size, uv_size, 1], dtype=np.float32)

        # SH features
        uv_map[:,:,10:] = uv_tex

        return uv_map






class UVGaussianModel:
    """A UV-map version of 3D Gaussian model"""
    
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, uv_rasterizer: UV_Rasterizer, device: str):
        self.active_sh_degree = 0
        #self.max_sh_degree = sh_degree
        self.max_sh_degree = 3 # for simplicity we use SH level=0
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.uv_rasterizer = uv_rasterizer
        self.device = device   # computing device
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.binding,
            self.binding_counter,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def create_from_uv(self, uv_map: torch.tensor):
        """
        Crate 3D Gaussians from the UV map
            Format of UV Gaussians map:
            [:, :, 0:3]  --> mean locations
            [:, :, 3:6]  --> scales
            [:, :, 6:9]  --> rotations
            [:, :, 9:10] --> opacities
            [:, :, 10:]  --> SH features
        """
        gaussians_feature = uv_map[self.uv_rasterizer.valid_coords] # [V, 13]
        n_gaussians = len(gaussians_feature)

        fused_color = gaussians_feature[:, 10:] # [V, 3]
        features = fused_color.view(-1, 3, 1) # [V, 3, 1]

        self._xyz = gaussians_feature[:, 0:3]

        self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        self._features_rest = features[:,:,1:].transpose(1, 2).contiguous()

        self._scaling = gaussians_feature[:, 3:6]

        rots_ext = torch.ones((n_gaussians, 1), device=self.device)
        rots_xyz = gaussians_feature[:, 6:9]
        self._rotation = torch.cat((rots_ext, rots_xyz), dim=1)

        self._opacity = gaussians_feature[:, 9:10]

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

        


