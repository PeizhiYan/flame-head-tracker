# Copyright (C) Peizhi Yan. 2025

# RobustVideoMatting is from https://github.com/PeterL1n/RobustVideoMatting

import torch
import numpy as np
from tqdm import tqdm

def load_matting_model(device='cuda'):
    video_matting_model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50").to(device)
    return video_matting_model.eval()

@torch.no_grad()
def matting_single_image(video_matting_model, img):
    """
    Perform matting on a single image using the video matting model.
    Args:
        video_matting_model: The RobustVideoMatting model.
        img: The input image as a numpy array (H, W, C) in RGB uint8 format.
    Returns:
        fg_img: The foreground image with alpha matte applied, in RGB uint8 format.
    """
    device = next(video_matting_model.parameters()).device
    downsample_ratio = 0.5
    rec = [None] * 4
    rgb = torch.from_numpy(img[None]).permute(0, 3, 1, 2).float().to(device) / 255
    fgr, pha, *rec = video_matting_model(rgb, *rec, downsample_ratio)

    alpha = pha[0][0].cpu().numpy() # [H, W] alpha matte
    # Set white background
    white_bg = np.ones_like(img, dtype=np.uint8) * 255
    fg_img = img * alpha[..., None] + white_bg * (1 - alpha[..., None])
    fg_img = np.clip(fg_img, 0, 255).astype(np.uint8)

    return fg_img

@torch.no_grad()
def matting_video_frames(video_matting_model, frames):
    """ Perform video matting on a list of frames using the provided model.
    Args:
        video_matting_model: The video matting model to use.
        frames (list): List of frames (numpy arrays in RGB uint8) to process.
    Returns:
        frames_with_matting (list): List of frames with applied matting.
    """
    device = next(video_matting_model.parameters()).device
    downsample_ratio = 0.5
    frames_with_matting = []
    rec = [None] * 4
    for i in tqdm(range(len(frames))):
        img = frames[i]
        rgb = torch.from_numpy(img[None]).permute(0, 3, 1, 2).float().to(device) / 255
        fgr, pha, *rec = video_matting_model(rgb, *rec, downsample_ratio)
        alpha = pha[0][0].cpu().numpy() # [H, W] alpha matte
        # Set white background
        white_bg = np.ones_like(img, dtype=np.uint8) * 255
        fg_img = img * alpha[..., None] + white_bg * (1 - alpha[..., None])
        fg_img = np.clip(fg_img, 0, 255).astype(np.uint8)
        frames_with_matting.append(fg_img)
    return frames_with_matting

