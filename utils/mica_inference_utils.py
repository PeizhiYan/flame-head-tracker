#
# MICA Inference Utilities
# Author: Peizhi Yan
# Copyright (C) Peizhi Yan. 2025
#

#
# MICA code:
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de
#

import numpy as np
import cv2
import torch

from submodules.MICA.configs.config import get_cfg_defaults
from submodules.MICA.utils import util as mica_util
from submodules.MICA.micalib.models.mica import MICA
from utils.insightface_utils import norm_crop


def create_mica_model(device):
    cfg = get_cfg_defaults()
    cfg.model.testing = True
    mica = MICA(cfg, device)
    mica = mica.eval()
    return mica


def get_5_landmarks_from_68(landmarks):
    # input: [68, 2]
    # output: [5, 2]
    # Left eye: points 36-41
    left_eye = np.mean(landmarks[36:42], axis=0)
    # Right eye: points 42-47
    right_eye = np.mean(landmarks[42:48], axis=0)
    # Nose tip: point 30
    nose = landmarks[30]
    # Left mouth corner: point 48
    left_mouth = landmarks[48]
    # Right mouth corner: point 54
    right_mouth = landmarks[54]
    # Stack as (5, 2) array
    return np.vstack([left_eye, right_eye, nose, left_mouth, right_mouth])


def get_arcface_input(kps, img):
    input_mean = 127.5
    input_std = 127.5
    aimg = norm_crop(img, landmark=kps)
    blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)
    return blob[0], aimg


def mica_preprocess(img, lmks_68, image_size=224):
    # img: [H, W, 3] uint8 BGR
    # lmks_68: [68, 2] float32
    kps = get_5_landmarks_from_68(lmks_68) # extract 5 landmarks
    blob, aimg = get_arcface_input(kps, img)
    return norm_crop(img, landmark=kps, image_size=image_size), blob


@torch.no_grad()
def get_shape_code_from_mica(mica, img, lmks_68, device):
    """
    input:
        mica: MICA model
        img: [H, W, 3] uint8 BGR-channels
        lmks_68: [68, 2] float32
    return:
        shape_code: [1, 300] float32 numpy
    """
    img_aligned, arcface = mica_preprocess(img, lmks_68, image_size=224)
    img_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2RGB)
    img_aligned = np.array(img_aligned).astype(np.float32) / 255
    image = cv2.resize(img_aligned, (224, 224)).transpose(2, 0, 1)
    image = torch.tensor(image).to(device)[None]
    arcface = torch.tensor(arcface).to(device)[None]
    codedict = mica.encode(image, arcface)
    opdict = mica.decode(codedict)
    shape_code = opdict['pred_shape_code'].detach().cpu().numpy() # [1,300]
    return shape_code


