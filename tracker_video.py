#
# FLAME Video Tracker
# Author: Peizhi Yan
# Copyright (C) Peizhi Yan. 2025
#

import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
import copy

from tracker_base import Tracker

from utils.video_utils import video_to_images
from utils.matting_utils import matting_video_frames
from utils.general_utils import plot_landmarks


def track_video(tracker_cfg):

    if not os.path.exists(tracker_cfg['video_path']):
        video_path = tracker_cfg['video_path']
        print(f'ERROR: video file path does not exist: {video_path}')

    # load video frames to images
    frames = video_to_images(video_path    = tracker_cfg['video_path'], 
                             original_fps  = tracker_cfg['original_fps'], 
                             subsample_fps = tracker_cfg['subsample_fps'])

    video_path = tracker_cfg['video_path']
    save_path = tracker_cfg['save_path']
    photometric_fitting = tracker_cfg['photometric_fitting']
    realign = tracker_cfg['realign']
    batch_size = max(8, tracker_cfg['batch_size'])

    video_base_name = os.path.basename(video_path)
    video_name = video_base_name.split('.')[0] # remove the name extension

    result_save_path = os.path.join(save_path, video_name)
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path) # create the output path if not exists

    ###########################
    ## Setup Flame Tracker    #     
    ###########################
    tracker = Tracker(tracker_cfg)
    # tracker.set_landmark_detector('hybrid')
    tracker.set_landmark_detector('mediapipe')

    #########################
    ## Run Video Matting    #
    #########################
    if tracker.use_matting:
        print('>>> Running video background matting...')
        # run video matting on all frames
        frames = matting_video_frames(tracker.video_matting_model, frames)
 
    ############################################
    ## Estimate Canonical Shape (and Texture)  #     
    ############################################
    shape_code = None
    texture = None
    if not photometric_fitting:
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
        # canonical shape code
        shape_code = mean_shape_code
    else:
        print('>>> Estimating canonical shape code and texture')
        # NOTE: here, we uniformly sample batch_size of frames for the estimations
        # you can also predefine the frames used for the estimation
        frames_subset = [frames[i] for i in np.linspace(0, len(frames) - 1, batch_size, dtype=int)]
        # run estimation
        batch_ret_dict, batch_valid_indices = tracker.run(img=frames_subset, realign=realign, 
                                                          photometric_fitting=photometric_fitting, 
                                                          shape_code=None, texture=None, 
                                                          temporal_smoothing=False, estimate_canonical=True)
        # save texture map
        if 'texture' in batch_ret_dict.keys():
            texture_save_path = os.path.join(result_save_path, 'texture.png')
            if not os.path.exists(texture_save_path):
                img_texture = (np.transpose(batch_ret_dict['texture'][0], (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(texture_save_path, cv2.cvtColor(img_texture, cv2.COLOR_RGB2BGR))
            # canonical texture map
            texture = np.copy(batch_ret_dict['texture'][:1])
            # to save disk space
            del batch_ret_dict['texture']
        # canonical shape code
        shape_code = batch_ret_dict['shape'][:1]

    #######################
    # Track All Frames    #
    #######################
    print(f'>>> Tracking video: {video_path}')
    i = 0; total_steps = len(frames)
    pbar = tqdm(total=total_steps)
    while i < total_steps:

        batch_frames = frames[i : i+batch_size]

        # fit on the current frame
        batch_ret_dict, batch_valid_indices = tracker.run(img=batch_frames, realign=realign, photometric_fitting=photometric_fitting, 
                                                          shape_code=shape_code, texture=texture, temporal_smoothing=True)

        if 'texture' in batch_ret_dict.keys():
            # to save disk space
            del batch_ret_dict['texture']
        
        # save tracking results
        for j, idx in enumerate(batch_valid_indices):
            fid = i + idx

            ret_dict = {}
            for key in batch_ret_dict:
                ret_dict[key] = batch_ret_dict[key][j:j+1]

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

                # # add gt landmarks
                # lmks_68 = ret_dict['lmks_68'][0,:,:2]
                # if 'lmks_ears' in ret_dict:
                #     lmks_ears = ret_dict['lmks_ears'][0,:,:2]
                #     lmks_to_plot = (np.concatenate([lmks_68,lmks_ears], axis=0) * 0.5 + 0.5) * 256
                # else:
                #     lmks_to_plot = (lmks_68 * 0.5 + 0.5) * 256
                # rendered = plot_landmarks(rendered, lmks_to_plot, radius=1, thickness=0, color=(0,100,255)) # add gt landmarks
                result_img[:,256:256*2,:] = rendered


                # rendered shape (w/ or w/o texture)
                if 'mesh_rendered' in loaded_params:
                    rendered_mesh = np.clip(cv2.resize(loaded_params['mesh_rendered'][0], (256,256)), 0, 255)
                else:
                    rendered_mesh = np.clip(cv2.resize(loaded_params['shape_rendered'][0], (256,256)), 0, 255)
                rendered_mesh = cv2.cvtColor(rendered_mesh, cv2.COLOR_RGB2BGR)
                result_img[:,256*2:256*3,:] = rendered_mesh

                cv2.imwrite(os.path.join(result_save_path, f'{fid}_compare.jpg'), result_img)
    
        i += batch_size
        pbar.update(batch_size)

    pbar.close()



