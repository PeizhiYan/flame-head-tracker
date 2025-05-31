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

from utils.video_utils import video_to_images
from tracker_base import Tracker




def track_video(tracker_cfg):

    # load video frames to images
    frames = video_to_images(video_path    = tracker_cfg['video_path'], 
                             original_fps  = tracker_cfg['original_fps'], 
                             subsample_fps = tracker_cfg['subsample_fps'])

    video_path = tracker_cfg['video_path']
    save_path = tracker_cfg['save_path']

    video_base_name = os.path.basename(video_path)
    video_name = video_base_name.split('.')[0] # remove the name extension

    result_save_path = os.path.join(save_path, video_name)
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path) # create the output path if not exists

    ###########################
    ## Setup Flame Tracker    #     
    ###########################
    tracker = Tracker(tracker_cfg)
    tracker.set_landmark_detector('mediapipe')

    ##############################
    ## Estimate Canonical Shape  #     
    ##############################
    print('>>> Estimating canonical shape code')
    MAX_SAMPLE_SIZE = 10
    frames_subset = frames[:MAX_SAMPLE_SIZE] # we take the first few frames to estimate the canonical shape code
    with torch.no_grad():
        mean_shape_code = np.zeros([1,tracker.NUM_SHAPE_COEFFICIENTS], dtype=np.float32)
        counter = 0
        for i in tqdm(range(len(frames_subset))):
            img = frames_subset[i]
            in_dict = tracker.prepare_intermediate_data_from_image(img, realign=True) # run reconstruction models
            if in_dict is not None:
                mean_shape_code += in_dict['shape']
                counter += 1
        if counter == 0:
            mean_shape_code = None
        else:
            mean_shape_code /= counter # compute the average shape code

    #######################
    # process all frames  #
    #######################
    print(f'>>> Processing video: {video_path}')
    prev_ret_dict = None
    for fid in tqdm(range(len(frames))):
        if prev_ret_dict is None:
            # it's said mediapipe might fail on the first frame.. 
            _ = tracker.prepare_intermediate_data_from_image(img, realign=True)

        # # Skip processed files (optional)
        # if os.path.exists(os.path.join(result_save_path, f'{fid}_compare.jpg')):
        #     prev_ret_dict = None
        #     continue

        # fit on the current frame
        ret_dict = tracker.run(img=frames[fid], realign=True, prev_ret_dict=prev_ret_dict, shape_code=mean_shape_code)
        prev_ret_dict = ret_dict
        if ret_dict is None:
            continue

        # save
        save_file_path = os.path.join(result_save_path, f'{fid}.npy')
        with open(save_file_path, 'wb') as f:
            pickle.dump(ret_dict, f)

        # check result: reconstruct from saved parameters and save the visualization results
        with torch.no_grad():
            with open(save_file_path, 'rb') as f:
                loaded_params = pickle.load(f)
            img = ret_dict['img'][0]

            result_img = np.zeros([256, 2*256, 3], dtype=np.uint8)

            # GT image            
            gt_img = cv2.resize(np.asarray(img), (256,256))
            gt_img = np.clip(np.array(gt_img, dtype=np.uint8), 0, 255) # uint8
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
            result_img[:,:256,:] = gt_img

            # rendered with texture but canonical camera pose
            rendered = np.clip(cv2.resize(loaded_params['img_rendered'][0], (256,256)), 0, 255)
            rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            result_img[:,256:256*2,:] = rendered

            cv2.imwrite(os.path.join(result_save_path, f'{fid}_compare.jpg'), result_img)




