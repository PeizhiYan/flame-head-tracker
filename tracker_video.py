###########################################
## FLAME Video Tracker.                   #
## -------------------------------------- #
## Author: Peizhi Yan                     #
## Update: 09/30/2024                     #
###########################################

## Copyright (C) Peizhi Yan. 2024

import os
import numpy as np
import cv2
from tqdm import tqdm
import pickle
import torch

from utils.video_utils import video_to_images
from tracker_base import Tracker


def track_video_legacy(tracker_cfg):

    # load video frames to images
    frames = video_to_images(video_path = tracker_cfg['video_path'], 
                             original_fps = tracker_cfg['original_fps'], 
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

    #######################
    # process all frames  #
    #######################
    print(f'Processing video: {video_path}')
    prev_ret_dict = None
    for fid in tqdm(range(len(frames))):

        # # Skip processed files (optional)
        # if os.path.exists(os.path.join(result_save_path, f'{fid}_compare.jpg')):
        #     prev_ret_dict = None
        #     continue

        # fit on the current frame
        ret_dict = tracker.run(img=frames[fid], realign=True, prev_ret_dict=prev_ret_dict)
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
            img = ret_dict['img']

            result_img = np.zeros([256, 2*256, 3], dtype=np.uint8)

            # GT image            
            gt_img = cv2.resize(np.asarray(img), (256,256))
            gt_img = np.clip(np.array(gt_img, dtype=np.uint8), 0, 255) # uint8
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
            result_img[:,:256,:] = gt_img

            # rendered with texture but canonical camera pose
            rendered = np.clip(cv2.resize(loaded_params['img_rendered'], (256,256)), 0, 255)
            rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            result_img[:,256:256*2,:] = rendered

            cv2.imwrite(os.path.join(result_save_path, f'{fid}_compare.jpg'), result_img)



def track_video(tracker_cfg):

    # load video frames to images
    frames = video_to_images(video_path = tracker_cfg['video_path'], 
                             original_fps = tracker_cfg['original_fps'], 
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

    # frames = frames[:30] # for debugging only

    #######################
    # process all frames  #
    #######################
    print(f'Processing video: {video_path}')
    ret_dict_all = tracker.run_all_images(imgs=frames, realign=True)

    #################
    # save results  #
    #################
    print(f'Saving results: {video_path}')
    NUM_OF_RESULTS = len(ret_dict_all['shape'])
    for fid in tqdm(range(NUM_OF_RESULTS)):

        ret_dict = {}
        for key in ret_dict_all.keys():
            ret_dict[key] = ret_dict_all[key][fid]

        # save
        save_file_path = os.path.join(result_save_path, f'{fid}.npy')
        with open(save_file_path, 'wb') as f:
            pickle.dump(ret_dict, f)

        # check result: reconstruct from saved parameters and save the visualization results
        with torch.no_grad():
            with open(save_file_path, 'rb') as f:
                loaded_params = pickle.load(f)
            img = ret_dict['img']

            result_img = np.zeros([256, 2*256, 3], dtype=np.uint8)

            # GT image            
            gt_img = cv2.resize(np.asarray(img), (256,256))
            gt_img = np.clip(np.array(gt_img, dtype=np.uint8), 0, 255) # uint8
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
            result_img[:,:256,:] = gt_img

            # rendered with texture but canonical camera pose
            rendered = np.clip(cv2.resize(loaded_params['img_rendered'], (256,256)), 0, 255)
            rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            result_img[:,256:256*2,:] = rendered

            cv2.imwrite(os.path.join(result_save_path, f'{fid}_compare.jpg'), result_img)



