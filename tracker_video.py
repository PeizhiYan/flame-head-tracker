#
# FLAME Video Tracker
# Author: Peizhi Yan
# Copyright (C) Peizhi Yan. 2025
#

import os
import numpy as np
import cv2
from tqdm import tqdm
import pickle
import torch
import random
import copy

from utils.video_utils import video_to_images
from tracker_base import Tracker




def track_video(tracker_cfg):

    # load video frames to images
    frames = video_to_images(video_path    = tracker_cfg['video_path'], 
                             original_fps  = tracker_cfg['original_fps'], 
                             subsample_fps = tracker_cfg['subsample_fps'])

    video_path = tracker_cfg['video_path']
    save_path = tracker_cfg['save_path']
    photometric_fitting = tracker_cfg['photometric_fitting']
    realign = tracker_cfg['realign']

    video_base_name = os.path.basename(video_path)
    video_name = video_base_name.split('.')[0] # remove the name extension

    result_save_path = os.path.join(save_path, video_name)
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path) # create the output path if not exists

    ###########################
    ## Setup Flame Tracker    #     
    ###########################
    tracker = Tracker(tracker_cfg)
    tracker.set_landmark_detector('hybrid')

    ###############################
    ## Estimate Canonical Shape   #     
    ###############################
    print('>>> Estimating canonical shape code')
    MAX_SAMPLE_SIZE = 3
    frames_subset = frames[:MAX_SAMPLE_SIZE] # we take the first few frames to estimate the canonical shape code
    with torch.no_grad():
        mean_shape_code = np.zeros([1,tracker.NUM_SHAPE_COEFFICIENTS], dtype=np.float32)
        counter = 0
        for i in tqdm(range(len(frames_subset))):
            img = frames_subset[i]
            in_dict = tracker.prepare_intermediate_data_from_image(img, realign=realign) # run reconstruction models
            if in_dict is not None:
                mean_shape_code += in_dict['shape']
                counter += 1
        if counter == 0:
            mean_shape_code = None
        else:
            mean_shape_code /= counter # compute the average shape code

    #######################
    # Process All Frames  #
    #######################
    print(f'>>> Processing video: {video_path}')
    prev_ret_dict = None
    texture = None
    for fid in tqdm(range(len(frames))):
        if prev_ret_dict is None:
            # it's said mediapipe might fail on the first frame.. 
            _ = tracker.prepare_intermediate_data_from_image(img, realign=True)

        # # Skip processed files (optional)
        # if os.path.exists(os.path.join(result_save_path, f'{fid}_compare.jpg')):
        #     prev_ret_dict = None
        #     continue

        # fit on the current frame
        ret_dict = tracker.run(img=frames[fid], realign=realign, photometric_fitting=photometric_fitting, 
                               prev_ret_dict=prev_ret_dict, shape_code=mean_shape_code, texture=texture)
        if ret_dict is None:
            # cannot fit the current frame, skip it
            print(f'>>> Frame {fid} cannot be fitted, skipping...')
            prev_ret_dict = copy.deepcopy(prev_ret_dict)
            continue
        else:
            prev_ret_dict = copy.deepcopy(ret_dict)

        # save texture map
        if 'texture' in ret_dict.keys():
            texture_save_path = os.path.join(result_save_path, 'texture.png')
            if not os.path.exists(texture_save_path):
                img_texture = (np.transpose(ret_dict['texture'][0], (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(texture_save_path, cv2.cvtColor(img_texture, cv2.COLOR_RGB2BGR))
            texture = np.copy(ret_dict['texture']) # cache d_texture for the next frame
            # to save disk space
            del ret_dict['texture']
        
        # save other data
        save_file_path = os.path.join(result_save_path, f'{fid}.npz')
        np.savez_compressed(save_file_path, **ret_dict)

        # check result: reconstruct from saved parameters and save the visualization results
        with torch.no_grad():
            loaded_params = np.load(save_file_path)
            img = ret_dict['img'][0]

            result_img = np.zeros([256, 3*256, 3], dtype=np.uint8)

            # GT image            
            gt_img = cv2.resize(np.asarray(img), (256,256))
            gt_img = np.clip(np.array(gt_img, dtype=np.uint8), 0, 255) # uint8
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
            result_img[:,:256,:] = gt_img

            # rendered shape with landmarks on top of original image
            rendered = np.clip(cv2.resize(loaded_params['img_rendered'][0], (256,256)), 0, 255)
            rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            result_img[:,256:256*2,:] = rendered

            # rendered shape (w/ or w/o texture)
            rendered_mesh = np.clip(cv2.resize(loaded_params['mesh_rendered'][0], (256,256)), 0, 255)
            rendered_mesh = cv2.cvtColor(rendered_mesh, cv2.COLOR_RGB2BGR)
            result_img[:,256*2:256*3,:] = rendered_mesh

            cv2.imwrite(os.path.join(result_save_path, f'{fid}_compare.jpg'), result_img)




