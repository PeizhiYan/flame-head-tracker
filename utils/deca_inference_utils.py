#
# DECA Inference Utilities
# Author: Peizhi Yan
# Copyright (C) Peizhi Yan. 2025
#

# DECA code:
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import torch
from submodules.decalib.deca import DECA
from submodules.decalib.deca_utils.config import cfg as deca_cfg


def create_deca_model(device):
    deca_cfg.model.use_tex = True
    deca_cfg.rasterizer_type = 'pytorch3d' # using 'standard' causes compiling issue
    deca_cfg.model.extract_tex = True
    deca = DECA(config = deca_cfg, device = device)
    return deca


@torch.no_grad()
def get_flame_code_from_deca(deca, img, device):
    """
    input:
        deca: DECA model
        img: [H, W, 3] uint8 RGB-channels
    return:
        deca_dict: dict
    """
    image = deca.crop_image(img).to(device)
    deca_dict = deca.encode(image[None])
    return deca_dict


