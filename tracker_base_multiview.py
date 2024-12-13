###########################################
## FLAME Tracker Reconstruction Base for  #
## Multi-View Input Images                #
## -------------------------------------- #
## Author: Peizhi Yan                     #
## Update: 12/12/2024                     #
###########################################

## Copyright (C) Peizhi Yan. 2024

## Installed Packages
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np # version 1.23.4, higher version may cause problem
from tqdm import tqdm

# Mediapipe  (version 0.10.15)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# FAN (1.4.1)
import face_alignment

## FLAME
from utils.flame_lib.FLAME import FLAME, FLAMETex

## FLAME photometric fitting utilities
import utils.flame_fitting.fitting_util as util
from utils.flame_fitting.renderer import Renderer

## Face parsing model
from utils.face_parsing.FaceParsingUtil import FaceParsing

## Utility
import utils.o3d_utils as o3d_utils
from utils.image_utils import read_img, min_max, uint8_img, norm_img, image_align, display_landmarks_with_cv2, get_face_mask
from utils.graphics_utils import create_diff_world_to_view_matrix, verts_clip_to_ndc, fov_to_focal

## DECA
from utils.decalib.deca import DECA
from utils.decalib.deca_utils.config import cfg as deca_cfg

## Local
from utils.mesh_renderer import NVDiffRenderer
from utils.scene.cameras import PerspectiveCamera
from utils.mp2dlib import convert_landmarks_mediapipe_to_dlib
from utils.kalman_filter import initialize_kalman_matrix, kalman_filter_update_matrix
from utils.loss_utils import *


# Function to check for NaN in dictionary of NumPy arrays
def check_nan_in_dict(array_dict):
    nan_detected = {}
    flag = False # True: there exist at least a NaN value
    for key, array in array_dict.items():
        if np.isnan(array).any():
            nan_detected[key] = True
            flag = True
        else:
            nan_detected[key] = False
    return nan_detected, flag


class Tracker():

    def __init__(self, tracker_cfg):
        #######################
        ##   Load Modules    ##
        #######################

        flame_cfg = {
            'mediapipe_face_landmarker_v2_path': tracker_cfg['mediapipe_face_landmarker_v2_path'],
            'flame_model_path': tracker_cfg['flame_model_path'],
            'flame_lmk_embedding_path': tracker_cfg['flame_lmk_embedding_path'],
            'tex_space_path': tracker_cfg['tex_space_path'],
            'tex_type': 'BFM',
            'camera_params': 3,          # do not change it
            'shape_params': 100,
            'expression_params': 50,     # we use the first 50 FLAME expression coefficients
            'pose_params': 6,
            'tex_params': 50,            # we use the first 50 FLAME texture model coefficients
            'use_face_contour': False,   # we don't use the face countour landmarks
            'cropped_size': 256,         # the render size for rendering the mesh
            'batch_size': 1,             # do not change it
            'image_size': 224,           # used in DECA, do not change it
            'e_lr': 0.01,
            'e_wd': 0.0001,
            'savefolder': './test_results/',
            # weights of losses and reg terms
            'w_pho': 8,
            'w_lmks': 1,
            'w_shape_reg': 1e-4,
            'w_expr_reg': 1e-4,
            'w_pose_reg': 0,
        }
        self.flame_cfg = util.dict2obj(flame_cfg)
        self.device = tracker_cfg['device']
        self.img_size = tracker_cfg['result_img_size']
        if 'use_kalman_filter' in tracker_cfg:
            self.use_kalman_filter = tracker_cfg['use_kalman_filter']
            self.kf_measure_noise = tracker_cfg['kalman_filter_measurement_noise_factor']
            self.kf_process_noise = tracker_cfg['kalman_filter_process_noise_factor']
        else:
            self.use_kalman_filter = False
        self.set_landmark_detector('mediapipe')

        # Mediapipe face landmark detector
        base_options = python.BaseOptions(model_asset_path=tracker_cfg['mediapipe_face_landmarker_v2_path'])
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               num_faces=1)
        self.mediapipe_detector = vision.FaceLandmarker.create_from_options(options)

        # FAN face alignment predictor (68 landmarks)
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

        # Face parsing model
        self.face_parser = FaceParsing(model_path=tracker_cfg['face_parsing_model_path'])

        # FLAME model and FLAME texture model
        self.flame = FLAME(self.flame_cfg).to(self.device)
        self.flametex = FLAMETex(self.flame_cfg).to(self.device)

        # Eye Landmarks (mediapipe) and indices (FLAME mesh) 
        self.R_EYE_MP_LMKS = [468, 469, 470, 471, 472]      # right eye mediapipe landmarks
        self.L_EYE_MP_LMKS = [473, 474, 475, 476, 477]      # left eye mediapipe landmarks
        self.R_EYE_INDICES = [4597, 4543, 4511, 4479, 4575] # right eye FLAME mesh indices
        self.L_EYE_INDICES = [4051, 3997, 3965, 3933, 4020] # left eye FLAME mesh indices

        # Camera settings
        self.H = self.W = self.flame_cfg.cropped_size
        self.DEFAULT_FOV = 20.0  # default x&y-axis FOV in degrees
        self.DEFAULT_DISTANCE = 1.0 # default camera to 3D world coordinate center distance
        self.DEFAULT_FOCAL = fov_to_focal(fov = self.DEFAULT_FOV, sensor_size = self.H)
        self.update_fov(fov = self.DEFAULT_FOV) # initialize the camera focal length and to object distance
        self.bg_color = (1.0,1.0,1.0) # White
        self.znear = 0.01
        self.zfar  = 100.0

        # FLAME render (from DECA)
        self.flame_texture_render = Renderer(self.flame_cfg.cropped_size, 
                                     obj_filename=tracker_cfg['template_mesh_file_path']).to(self.device)

        # Nvidia differentiable mesh render (from GaussianAvatars)
        self.mesh_renderer = NVDiffRenderer().to(self.device)

        # Load the template FLAME triangle faces
        _, self.faces, self.uv_coords, _ = o3d_utils._read_obj_file(tracker_cfg['template_mesh_file_path'], uv=True)
        self.uv_coords = np.array(self.uv_coords, dtype=np.float32)
        self.mesh_faces = torch.from_numpy(self.faces).to(self.device).detach() # [F, 3]

        # Load DECA model
        deca_cfg.model.use_tex = True
        deca_cfg.rasterizer_type = 'pytorch3d' # using 'standard' causes compiling issue
        deca_cfg.model.extract_tex = True
        self.deca = DECA(config = deca_cfg, device=self.device)

        print('Flame Tracker ready.')


    def set_landmark_detector(self, landmark_detector='mediapipe'):
        """
        landmark_detector: choose either 'mediapipe' or 'FAN'
        """
        assert landmark_detector in ['mediapipe', 'FAN'], "landmark_detector need to be either mediapipe or FAN"
        self.landmark_detector = landmark_detector


    def update_fov(self, fov : float):
        """
        Update the camera FOV and adjust the camera to object distance
        correspondingly
        """
        # Assert that the FOV is within the reasonable range
        assert 20 <= fov <= 60, f"FOV must be between 20 and 60. Provided: {fov}"
        self.fov = fov # update the fov
        self.focal = fov_to_focal(fov = fov, sensor_size = self.H) # compute new focal length
        self.distance = self.DEFAULT_DISTANCE * (self.focal / self.DEFAULT_FOCAL) # compute new camera to object distance


    def mediapipe_face_detection(self, img):
        """
        Run Mediapipe face detector
        input:
            - img: image data  numpy
        output:
            - lmks_dense: landmarks numpy [478, 2], the locations are in image scale
            - blend_scores: facial blendshapes numpy [52]
        """
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img) # convert numpy image to Mediapipe Image

        # Detect face landmarks from the input image.
        detection_result = self.mediapipe_detector.detect(image)

        if len(detection_result.face_blendshapes) == 0:
            return None, None

        # Post-process mediapipe face blendshape scores
        blend_scores = detection_result.face_blendshapes[0]
        blend_scores = np.array(list(map(lambda l: l.score, blend_scores)), dtype=np.float32)

        # Post-process mediapipe dense face landmarks, re-scale to image space 
        lmks_dense = detection_result.face_landmarks[0] # the first detected face
        lmks_dense = np.array(list(map(lambda l: np.array([l.x, l.y]), lmks_dense)))
        lmks_dense[:, 0] = lmks_dense[:, 0] * img.shape[1]
        lmks_dense[:, 1] = lmks_dense[:, 1] * img.shape[0]

        return lmks_dense, blend_scores


    def fan_face_landmarks(self, img):
        """
        Run Mediapipe face detector
        input:
            - img: image data  numpy  uint8
        output:
            - lmks_dense: landmarks numpy [68, 2], the locations are in image scale
        """
        face_landmarks = self.fa.get_landmarks(img)
        if face_landmarks is None:
            # no face detected
            return None
        else:
            # return face landmarks of the first detected face
            return face_landmarks[0] # [68, 2]


    def load_images_and_run(self, 
                           img_paths, realign=True, photometric_fitting=False, shape_code=None):
        """
        Load image from given path, then run FLAME tracking
        input:
            -img_paths: list of image paths
            -realign: for FFHQ, use False. for in-the-wild images, use True
            -photometric_fitting: whether to use photometric fitting or landmarks only
            -shape_code: the pre-estimated global shape code
        output:
            -ret_dict: results dictionary
        """
        imgs = []
        for img_path in img_paths:
            img = read_img(img_path)
            imgs.append(img)
        return self.run(imgs, realign, photometric_fitting, shape_code)

    
    def run(self, imgs, realign=True, photometric_fitting=False, shape_code=None):
        """
        Run FLAME tracking on the given image
        input:
            -imgs: list of image data   [numpy] 
            -realign: for FFHQ, use False. for in-the-wild images, use True
            -photometric_fitting: whether to use photometric fitting or landmarks only
            -shape_code: the pre-estimated global shape code
        output:
            -ret_dict: results dictionary
        """
        
        # run Mediapipe face detector
        lmks_dense_list = []
        for img in imgs:
            lmks_dense, blend_scores = self.mediapipe_face_detection(img)
            if lmks_dense is None:
                # no face detected
                lmks_dense_list.append(None)
            else:
                lmks_dense_list.append(lmks_dense)

        # convert Mediapipe landmarks to 68 dlib landmarks
        face_landmarks_list = []
        for lmks_dense in lmks_dense_list:
            if lmks_dense is None:
                face_landmarks_list.append(None)
            else:
                face_landmarks = convert_landmarks_mediapipe_to_dlib(lmks_mp=lmks_dense)
                face_landmarks_list.append(face_landmarks)

        # re-align images (tracking standard), this image will be used in our network model
        imgs_aligned = []
        for img in imgs:
            img_aligned = image_align(img, face_landmarks, output_size=self.img_size, standard='tracking', 
                                    padding_mode='constant')
            imgs_aligned.append(img_aligned)

        if realign:
            # realign == True means that we will fit on realigned image
            imgs = imgs_aligned

        # run DECA reconstruction
        deca_dict_list = []
        for img in imgs:
            deca_dict = self.run_deca(img)
            deca_dict_list.append(deca_dict)

        # run face parsing
        parsing_mask_list = []
        parsing_mask_aligned_list = []
        for i in range(len(imgs)):
            img = imgs[i]
            img_aligned = imgs_aligned[i]
            parsing_mask = self.face_parser.run(img)
            parsing_mask_list.append(parsing_mask)
            if realign: parsing_mask_aligned = parsing_mask
            else: parsing_mask_aligned = self.face_parser.run(img_aligned)
            parsing_mask_aligned_list.append(parsing_mask_aligned)
        
        if photometric_fitting:
            # run photometric fitting
            # print('Photometric fitting not supported yet.') # I will add this in a future version
            # return None
            face_mask_list = []
            for i in range(len(imgs)):
                parsing_mask = parsing_mask_list[i]
                face_mask = get_face_mask(parsing=parsing_mask, keep_ears=False)
                face_mask_list.append(face_mask)
            ret_dict = self.run_fitting_multiview_photometric(imgs, face_mask_list, deca_dict_list, shape_code)
        else:
            # run facial landmark-based fitting
            ret_dict = self.run_fitting_multiview(imgs, deca_dict_list, shape_code)

        # check for NaNs, if there is any, return None
        _, nan_status = check_nan_in_dict(ret_dict)
        if nan_status:
            return None

        # add more data
        ret_dict['img'] = imgs
        ret_dict['img_aligned'] = imgs_aligned
        ret_dict['parsing'] = parsing_mask_list
        ret_dict['parsing_aligned'] = parsing_mask_aligned_list

        return ret_dict
    

    @torch.no_grad()
    def run_deca(self, img):
        """ Run DECA on a single input image """

        # DECA Encode
        image = self.deca.crop_image(img).to(self.device)
        deca_dict = self.deca.encode(image[None])
        
        return deca_dict
    

    # @torch.no_grad()
    # def run_deca_batch(self, imgs : list):
    #     """ Run Deca on a batch of images """

    #     # convert to batch tensor
    #     images = []
    #     for i in range(len(imgs)):
    #         image = self.deca.crop_image(imgs[i]).to(self.device)
    #         images.append(image[None])
    #     images = torch.cat(images) # [N,C,H,W]

    #     # DECA Encode
    #     deca_dict = self.deca.encode(images)

    #     return deca_dict

    
    def run_fitting_multiview(self, imgs, deca_dict_list, shape_code):
        """ Landmark-based Fitting        
            - Stage 1: rigid fitting on the camera pose (6DoF) based on detected landmarks
            - Stage 2: fine-tune the parameters including shape, tex, exp, pose, eye_pose, and light
        """

        # convert the parameters to numpy arrays
        params = {}
        for key in ['shape', 'tex', 'exp', 'pose', 'light']:
            if key == 'shape' and shape_code is not None:
                # use pre-estimated global shape code
                params[key] = shape_code.detach().cpu().numpy()
            else:
                count = 0
                temp = None
                for deca_dict in deca_dict_list:
                    if deca_dict is not None:
                        if temp is None:
                            temp = deca_dict[key].detach().cpu().numpy()
                        else:
                            temp += deca_dict[key].detach().cpu().numpy()
                        count += 1
                if count != 0:
                    temp = temp / count # compute the average
                    params[key] = temp
                else:
                    return None

        # resize for FLAME fitting
        imgs_resized = []
        for img in imgs:
            img_resized = cv2.resize(img, (self.flame_cfg.cropped_size, self.flame_cfg.cropped_size))
            imgs_resized.append(img_resized)

        # run Mediapipe face detector
        lmks_dense_list = []
        face_landmarks_list = []
        for img_resized in imgs_resized:
            lmks_dense, _ = self.mediapipe_face_detection(img_resized)
            if lmks_dense is None:
                # no face detected
                lmks_dense_list.append(None)
                face_landmarks_list.append(None)
                continue
            lmks_dense[:, :2] = lmks_dense[:, :2] / float(self.flame_cfg.cropped_size) * 2 - 1 # normalize landmarks
            if self.landmark_detector == 'mediapipe':
                face_landmarks = convert_landmarks_mediapipe_to_dlib(lmks_mp=lmks_dense)
            elif self.landmark_detector == 'FAN':
                # run FAN face landmarks predictor
                face_landmarks = self.fan_face_landmarks(img_resized)
                if face_landmarks is None:
                    # no face detected
                    lmks_dense_list.append(None)
                    face_landmarks_list.append(None)
                    continue
                face_landmarks = face_landmarks[:, :2] / float(self.flame_cfg.cropped_size) * 2 - 1 # normalize landmarks      
            lmks_dense_list.append(lmks_dense)
            face_landmarks_list.append(face_landmarks)

        # prepare ground-truth landmarks
        gt_landmark_list = []
        gt_eyes_landmark_list = []
        for face_landmarks in face_landmarks_list:
            # prepare target 68 landmarks
            gt_landmark = np.array(face_landmarks).astype(np.float32)
            gt_landmark_list.append(gt_landmark)
            # prepare target eyes landmarks
            gt_eyes_landmark = np.array(lmks_dense[self.R_EYE_MP_LMKS + self.L_EYE_MP_LMKS]).astype(np.float32)
            gt_eyes_landmark_list.append(gt_eyes_landmark)
        gt_landmark_tensor = np.array(gt_landmark_list).astype(np.float32)  # [N,68,2]
        gt_landmark_tensor = torch.from_numpy(gt_landmark_tensor).float().to(self.device)  # [N,68,2]
        gt_eyes_landmark_tensor = np.array(gt_eyes_landmark_list).astype(np.float32)  # [N,10,2]
        gt_eyes_landmark_tensor = torch.from_numpy(gt_eyes_landmark_tensor).float().to(self.device)  # [N,10,2]


        ############################################################
        ## Stage 1: rigid fitting (estimate the 6DoF camera pose)  #
        ############################################################

        # prepare 6DoF camera poses tensor
        camera_pose_list = [] # camera pose for each view
        for _ in range(len(imgs)):
            camera_pose = torch.tensor([0, 0, 0, 0, 0, self.distance], dtype=torch.float32).to(self.device) # [yaw, pitch, roll, x, y, z]
            camera_pose_list.append(camera_pose)

        # prepare camera pose offsets (to be optimized)
        d_camera_rotation_list = []
        d_camera_translation_list = []
        for _ in range(len(imgs)): 
            d_camera_rotation = nn.Parameter(torch.zeros(3).float().to(self.device))
            d_camera_translation = nn.Parameter(torch.zeros(3).float().to(self.device))
            d_camera_rotation_list.append(d_camera_rotation)
            d_camera_translation_list.append(d_camera_translation)
        camera_params = [
            {'params': d_camera_rotation_list, 'lr': 0.01}, {'params': d_camera_translation_list, 'lr': 0.05}
        ]

        # camera pose optimizer
        e_opt_rigid = torch.optim.Adam(
            camera_params,
            weight_decay=0.00001
        )
        
        # DECA's results
        shape = torch.from_numpy(params['shape']).to(self.device).detach()
        exp = torch.from_numpy(params['exp']).to(self.device).detach()
        pose = torch.from_numpy(params['pose']).to(self.device).detach()
        pose[0,:3] *= 0 # we clear FLAME's head pose (we use camera pose instead)

        # optimization loop
        total_iterations = 1000
        for iter in range(total_iterations):
            e_opt_rigid.zero_grad()

            # update learning rate
            if iter == 700:
                e_opt_rigid.param_groups[0]['lr'] = 0.005    # For translation
                e_opt_rigid.param_groups[1]['lr'] = 0.01     # For rotation
            # update loss term weights
            if iter <= 700: l_f = 100; l_c = 500 # more weights to contour
            else: l_f = 500; l_c = 100 # more weights to face

            # construct canonical shape
            vertices, _, _ = self.flame(shape_params=shape, expression_params=exp, pose_params=pose) # [1, N, 3]
            count = 0
            loss = 0
            for i in range(len(imgs)):
                if deca_dict_list[i] is None:
                    continue
                else:
                    count += 1

                # prep camera for the i-th view
                d_camera_rotation = d_camera_rotation_list[i]
                d_camera_translation = d_camera_translation_list[i]
                optimized_camera_pose = camera_pose + torch.cat((d_camera_rotation, d_camera_translation))
                Rt = create_diff_world_to_view_matrix(optimized_camera_pose)
                cam = PerspectiveCamera(Rt=Rt, fov=self.fov, bg=self.bg_color, 
                                        image_width=self.W, image_height=self.H, znear=self.znear, zfar=self.zfar)

                # project landmarks via NV diff renderer
                verts_clip = self.mesh_renderer.project_vertices_from_camera(vertices, cam)
                verts_ndc_3d = verts_clip_to_ndc(verts_clip, image_size=self.H, out_dim=3) # convert the clipped vertices to NDC, output [N, 3]
                landmarks3d = self.flame.seletec_3d68(verts_ndc_3d[None]) # [1, 68, 3]
                landmarks2d = landmarks3d[:,:,:2] / float(self.flame_cfg.cropped_size) * 2 - 1  # [1, 68, 2]

                # loss computation and optimization
                gt_landmark = gt_landmark_tensor[i][None]
                loss_facial = util.l2_distance(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]) * l_f
                loss_contour = util.l2_distance(landmarks2d[:, :17, :2], gt_landmark[:, :17, :2]) * l_c # contour loss
                loss = loss + loss_facial + loss_contour

            loss = loss / count # average across valid views
            loss.backward()
            e_opt_rigid.step()

        # the optimized camera pose from the Stage 1
        # prepare camera for Stage 2
        optimized_camera_pose_list = []
        cam_list = []
        for i in range(len(imgs)):
            # camera pose for the i-the view
            d_camera_rotation = d_camera_rotation_list[i]
            d_camera_translation = d_camera_translation_list[i]
            optimized_camera_pose = camera_pose + torch.cat((d_camera_rotation, d_camera_translation))
            optimized_camera_pose = optimized_camera_pose.detach()
            optimized_camera_pose_list.append(optimized_camera_pose)
            Rt = create_diff_world_to_view_matrix(optimized_camera_pose)
            cam = PerspectiveCamera(Rt=Rt, fov=self.fov, bg=self.bg_color, 
                                    image_width=self.W, image_height=self.H, znear=self.znear, zfar=self.zfar)
            cam_list.append(cam)

        ############################
        ## Stage 2: fine fitting   #
        ############################

        # prepare expression code offsets (to be optimized)
        d_exp = torch.zeros(params['exp'].shape)
        d_exp = nn.Parameter(d_exp.float().to(self.device))

        # prepare jaw pose offsets (to be optimized)
        d_jaw = torch.zeros(3)
        d_jaw = nn.Parameter(d_jaw.float().to(self.device))    
        
        # prepare eyes poses offsets (to be optimized)
        eye_pose = torch.zeros(1,6) # FLAME's default_eyeball_pose are zeros
        eye_pose = nn.Parameter(eye_pose.float().to(self.device))    

        expr_params = [
            {'params': [d_exp], 'lr': 0.01}, 
            {'params': [d_jaw], 'lr': 0.025},
            {'params': [eye_pose], 'lr': 0.03}
        ]

        # fine optimizer
        e_opt_fine = torch.optim.Adam(
            expr_params,
            weight_decay=0.0001
        )

        # optimization loop
        for iter in range(200):
            e_opt_fine.zero_grad()

            # update learning rate
            if iter == 100:
                e_opt_fine.param_groups[0]['lr'] = 0.005    
                e_opt_fine.param_groups[1]['lr'] = 0.01     
                e_opt_fine.param_groups[1]['lr'] = 0.01     

            # construct the canonical shape
            optimized_exp = exp + d_exp
            optimized_pose = torch.from_numpy(params['pose']).to(self.device).detach()
            optimized_pose[0,:3] *= 0 # we clear FLAME's head pose 
            optimized_pose[:,3:] = optimized_pose[:,3:] + d_jaw
            vertices, _, _ = self.flame(shape_params=shape, 
                                        expression_params=optimized_exp, 
                                        pose_params=optimized_pose, 
                                        eye_pose_params=eye_pose) # [1, N, 3]

            count = 0
            loss = 0
            for i in range(len(imgs)):
                if deca_dict_list[i] is None:
                    continue
                else:
                    count += 1

                # get the camera for the i-th view
                cam = cam_list[i]

                gt_landmark = gt_landmark_tensor[i][None]
                gt_eyes_landmark = gt_eyes_landmark_tensor[i][None]

                # project landmarks via NV diff renderer
                verts_clip = self.mesh_renderer.project_vertices_from_camera(vertices, cam)
                verts_ndc_3d = verts_clip_to_ndc(verts_clip, image_size=self.H, out_dim=3) # convert the clipped vertices to NDC, output [N, 3]
                landmarks3d = self.flame.seletec_3d68(verts_ndc_3d[None]) # [1, 68, 3]
                landmarks2d = landmarks3d[:,:,:2] / float(self.flame_cfg.cropped_size) * 2 - 1  # [1, 68, 2]

                # eyes landmarks
                eyes_landmarks3d = verts_ndc_3d[self.R_EYE_INDICES + self.L_EYE_INDICES][None]  # [1, 10, 3]
                eyes_landmarks2d = eyes_landmarks3d[:,:,:2] / float(self.flame_cfg.cropped_size) * 2 - 1  # [1, 10, 2]

                # loss computation and optimization
                loss_facial = util.l2_distance(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]) * 500
                loss_eyes = util.l2_distance(eyes_landmarks2d, gt_eyes_landmark) * 500
                loss = loss + loss_facial + loss_eyes
            loss = loss / count # average across valid views
            loss.backward()
            e_opt_fine.step()

        ##############################
        ## for displaying results    #
        ##############################
        with torch.no_grad():
            # construct canonical shape
            optimized_exp = exp + d_exp
            optimized_pose = torch.from_numpy(params['pose']).to(self.device).detach()
            optimized_pose[0,:3] *= 0 # we clear FLAME's head pose 
            optimized_pose[:,3:] = optimized_pose[:,3:] + d_jaw # clear head pose and set jaw pose
            vertices, _, _ = self.flame(shape_params=shape, 
                                        expression_params=optimized_exp, 
                                        pose_params=optimized_pose, 
                                        eye_pose_params=eye_pose)

            rendered_mesh_shape_img_list = []
            rendered_mesh_shape_list = []
            for i in range(len(imgs)):
                if deca_dict_list[i] is None:
                    rendered_mesh_shape_img_list.append(None)
                    rendered_mesh_shape_list.append(None)
                    continue

                cam = cam_list[i]
                img_resized = imgs_resized[i]

                # render landmarks via NV diff renderer
                new_mesh_renderer = NVDiffRenderer().to(self.device) # there seems to be a bug with the NVDiffRenderer, so I create this new
                                                                    # render everytime to render the image
                rendered = new_mesh_renderer.render_from_camera(vertices, self.mesh_faces, cam) # vertices should have the shape of [1, N, 3]
                verts_clip = rendered['verts_clip'] # [1, N, 3]
                verts_ndc_3d = verts_clip_to_ndc(verts_clip, image_size=self.H, out_dim=3) # convert the clipped vertices to NDC, output [N, 3]
                landmarks3d = self.flame.seletec_3d68(verts_ndc_3d[None]) # [1, 68, 3]
                landmarks2d = landmarks3d[:,:,:2].detach().cpu().numpy() # [1, 68, 2]
                eyes_landmarks3d = verts_ndc_3d[self.R_EYE_INDICES + self.L_EYE_INDICES][None]  # [1, 10, 3]
                eyes_landmarks2d = eyes_landmarks3d[:,:,:2].detach().cpu().numpy()  # [1, 10, 2]
                rendered_mesh_shape = rendered['rgba'][0,...,:3].detach().cpu().numpy()
                rendered_mesh_shape_img = (img_resized / 255. + rendered_mesh_shape) / 2
                rendered_mesh_shape_img = np.array(np.clip(rendered_mesh_shape_img * 255, 0, 255), dtype=np.uint8) # uint8
                rendered_mesh_shape = np.array(np.clip(rendered_mesh_shape * 255, 0, 255), dtype=np.uint8) # uint8

                # Draw 2D landmarks as green dots
                for coords in landmarks2d[0]:
                    coords = np.clip(coords, 0, self.H-1).astype(np.uint8)
                    #coords = np.clip((coords / 2 + 1) * self.H, 0, self.H).astype(np.uint8)
                    cv2.circle(rendered_mesh_shape, (coords[0], coords[1]), radius=1, color=(0, 255, 0), thickness=-1)  # Green color, filled circle

                # Optionally draw eye landmarks as red dots
                for coords in eyes_landmarks2d[0]:
                    coords = np.clip(coords, 0, self.H-1).astype(np.uint8)
                    #coords = np.clip((coords / 2 + 1) * self.H, 0, self.H).astype(np.uint8)
                    cv2.circle(rendered_mesh_shape, (coords[0], coords[1]), radius=1, color=(0, 0, 255), thickness=-1)  # Red color, filled circle

                rendered_mesh_shape_img_list.append(rendered_mesh_shape_img)
                rendered_mesh_shape_list.append(rendered_mesh_shape)

        ####################
        # Prepare results  #
        ####################
        for i in range(len(optimized_camera_pose_list)):
            optimized_camera_pose = optimized_camera_pose_list[i]
            optimized_camera_pose_list[i] = optimized_camera_pose.detach().cpu().numpy()
        
        ret_dict = {
            'vertices': vertices[0].detach().cpu().numpy(), # [N, 3] canonical shape (including expression)
            'shape': params['shape'],                       # canonical FLAME shape code
            'exp': optimized_exp.detach().cpu().numpy(),    # canonical FLAME expression code
            'pose': optimized_pose.detach().cpu().numpy(),  # canonical FLAME jaw pose code
            'eye_pose': eye_pose.detach().cpu().numpy(),    # canonical FLAME eye pose code
            'tex': params['tex'],                           # canonical FLAME texture code
            'light': params['light'],                       # canonical FLAME light coefficients
            'cam': optimized_camera_pose_list,              # [N][6] optimized camera poses for each view
            'img_rendered': rendered_mesh_shape_img_list,   # rendered result for each view
            'mesh_rendered': rendered_mesh_shape_list,      # rendered result for each view
        }
        
        return ret_dict



    def run_fitting_multiview_photometric(self, imgs, face_mask_list, deca_dict_list, shape_code):
        """ Photometric Fitting        
            - Stage 1: rigid fitting on the camera pose (6DoF) based on detected landmarks
            - Stage 2: fine-tune the parameters including shape, tex, exp, pose, eye_pose, and light
        """

        # convert the parameters to numpy arrays
        params = {}
        for key in ['shape', 'tex', 'exp', 'pose', 'light']:
            if key == 'shape' and shape_code is not None:
                # use pre-estimated global shape code
                params[key] = shape_code.detach().cpu().numpy()
            else:
                count = 0
                temp = None
                for deca_dict in deca_dict_list:
                    if deca_dict is not None:
                        if temp is None:
                            temp = deca_dict[key].detach().cpu().numpy()
                        else:
                            temp += deca_dict[key].detach().cpu().numpy()
                        count += 1
                if count != 0:
                    temp = temp / count # compute the average
                    params[key] = temp
                    if key in ['shape', 'exp']:
                        params[key] = deca_dict[key].detach().cpu().numpy() * 0.0
                else:
                    return None

        # resize for FLAME fitting
        imgs_resized = []
        gt_imgs = []
        gt_face_masks = []
        for i, img in enumerate(imgs):
            img_resized = cv2.resize(img, (self.flame_cfg.cropped_size, self.flame_cfg.cropped_size))
            imgs_resized.append(img_resized)

            gt_img = torch.from_numpy(np.array(img_resized,dtype=np.float32) / 255.).detach().to(self.device) # [H,W,C] float32
            gt_img = gt_img[None].permute(0,3,1,2) # [1,C,H,W]
            gt_imgs.append(gt_img)

            face_mask = face_mask_list[i]
            face_mask_resized = cv2.resize(face_mask, (self.flame_cfg.cropped_size, self.flame_cfg.cropped_size)) # [H,W]
            gt_face_mask = torch.from_numpy(face_mask_resized)[None].detach().to(self.device) # [1,H,W] boolean
            gt_face_masks.append(gt_face_mask)


        # # run Mediapipe face detector
        # lmks_dense_list = []
        # face_landmarks_list = []
        # for img_resized in imgs_resized:
        #     lmks_dense, _ = self.mediapipe_face_detection(img_resized)
        #     if lmks_dense is None:
        #         # no face detected
        #         lmks_dense_list.append(None)
        #         face_landmarks_list.append(None)
        #         continue
        #     lmks_dense[:, :2] = lmks_dense[:, :2] / float(self.flame_cfg.cropped_size) * 2 - 1 # normalize landmarks
        #     face_landmarks = convert_landmarks_mediapipe_to_dlib(lmks_mp=lmks_dense) # 68 dlib landmarks
        #     lmks_dense_list.append(lmks_dense)
        #     face_landmarks_list.append(face_landmarks)

        # run Mediapipe face detector
        lmks_dense_list = []
        face_landmarks_list = []
        for img_resized in imgs_resized:
            lmks_dense, _ = self.mediapipe_face_detection(img_resized)
            if lmks_dense is None:
                # no face detected
                lmks_dense_list.append(None)
                face_landmarks_list.append(None)
                continue
            lmks_dense[:, :2] = lmks_dense[:, :2] / float(self.flame_cfg.cropped_size) * 2 - 1 # normalize landmarks
            if self.landmark_detector == 'mediapipe':
                face_landmarks = convert_landmarks_mediapipe_to_dlib(lmks_mp=lmks_dense)
            elif self.landmark_detector == 'FAN':
                # run FAN face landmarks predictor
                face_landmarks = self.fan_face_landmarks(img_resized)
                if face_landmarks is None:
                    # no face detected
                    lmks_dense_list.append(None)
                    face_landmarks_list.append(None)
                    continue
                face_landmarks = face_landmarks[:, :2] / float(self.flame_cfg.cropped_size) * 2 - 1 # normalize landmarks
            lmks_dense_list.append(lmks_dense)
            face_landmarks_list.append(face_landmarks)

        # prepare ground-truth landmarks
        gt_landmark_list = []
        gt_eyes_landmark_list = []
        for face_landmarks in face_landmarks_list:
            if face_landmarks is None:
                gt_landmark = np.zeros([68,2], dtype=np.float32)
                gt_eyes_landmark = np.zeros([10,2], dtype=np.float32)
            else:
                # prepare target 68 landmarks
                gt_landmark = np.array(face_landmarks).astype(np.float32)
                # prepare target eyes landmarks
                gt_eyes_landmark = np.array(lmks_dense[self.R_EYE_MP_LMKS + self.L_EYE_MP_LMKS]).astype(np.float32)
            gt_landmark_list.append(gt_landmark)
            gt_eyes_landmark_list.append(gt_eyes_landmark)
        gt_landmark_tensor = np.array(gt_landmark_list).astype(np.float32)  # [N,68,2]
        gt_landmark_tensor = torch.from_numpy(gt_landmark_tensor).float().to(self.device)  # [N,68,2]
        gt_eyes_landmark_tensor = np.array(gt_eyes_landmark_list).astype(np.float32)  # [N,10,2]
        gt_eyes_landmark_tensor = torch.from_numpy(gt_eyes_landmark_tensor).float().to(self.device)  # [N,10,2]


        ############################################################
        ## Stage 1: rigid fitting (estimate the 6DoF camera pose)  #
        ############################################################

        # prepare 6DoF camera poses tensor
        camera_pose_list = [] # camera pose for each view
        for _ in range(len(imgs)):
            camera_pose = torch.tensor([0, 0, 0, 0, 0, self.distance], dtype=torch.float32).to(self.device) # [yaw, pitch, roll, x, y, z]
            camera_pose_list.append(camera_pose)

        # prepare camera pose offsets (to be optimized)
        d_camera_rotation_list = []
        d_camera_translation_list = []
        for _ in range(len(imgs)): 
            d_camera_rotation = nn.Parameter(torch.zeros(3).float().to(self.device))
            d_camera_translation = nn.Parameter(torch.zeros(3).float().to(self.device))
            d_camera_rotation_list.append(d_camera_rotation)
            d_camera_translation_list.append(d_camera_translation)
        camera_params = [
            {'params': d_camera_rotation_list, 'lr': 0.005}, {'params': d_camera_translation_list, 'lr': 0.005}
        ]

        # camera pose optimizer
        e_opt_rigid = torch.optim.Adam(
            camera_params,
            weight_decay=0.0001
        )
        
        # DECA's results
        shape = torch.from_numpy(params['shape']).to(self.device).detach()
        tex = torch.from_numpy(params['tex']).to(self.device).detach()
        exp = torch.from_numpy(params['exp']).to(self.device).detach()
        light = torch.from_numpy(params['light']).to(self.device).detach()
        pose = torch.from_numpy(params['pose']).to(self.device).detach()
        pose[0,:3] *= 0 # we clear FLAME's head pose (we use camera pose instead)

        # optimization loop
        total_iterations = 800
        for iter in range(total_iterations):
            e_opt_rigid.zero_grad()

            # # update learning rate
            # if iter == 400:
            #     e_opt_rigid.param_groups[0]['lr'] = 0.005    # For translation
            #     e_opt_rigid.param_groups[1]['lr'] = 0.01     # For rotation
            # # update loss term weights
            # if iter <= 400: l_f = 100; l_c = 500 # more weights to contour
            # else: l_f = 500; l_c = 100 # more weights to face

            # construct canonical shape
            vertices, _, _ = self.flame(shape_params=shape, expression_params=exp, pose_params=pose) # [1, N, 3]
            count = 0
            loss = 0
            for i in range(len(imgs)):
                if deca_dict_list[i] is None or lmks_dense_list[i] is None:
                    continue
                else:
                    count += 1

                # prep camera for the i-th view
                d_camera_rotation = d_camera_rotation_list[i]
                d_camera_translation = d_camera_translation_list[i]
                optimized_camera_pose = camera_pose + torch.cat((d_camera_rotation, d_camera_translation))
                Rt = create_diff_world_to_view_matrix(optimized_camera_pose)
                cam = PerspectiveCamera(Rt=Rt, fov=self.fov, bg=self.bg_color, 
                                        image_width=self.W, image_height=self.H, znear=self.znear, zfar=self.zfar)

                # project landmarks via NV diff renderer
                verts_clip = self.mesh_renderer.project_vertices_from_camera(vertices, cam)
                verts_ndc_3d = verts_clip_to_ndc(verts_clip, image_size=self.H, out_dim=3) # convert the clipped vertices to NDC, output [N, 3]
                landmarks3d = self.flame.seletec_3d68(verts_ndc_3d[None]) # [1, 68, 3]
                landmarks2d = landmarks3d[:,:,:2] / float(self.flame_cfg.cropped_size) * 2 - 1  # [1, 68, 2]

                # loss computation and optimization
                gt_landmark = gt_landmark_tensor[i][None]
                loss_facial = util.l2_distance(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]) #* l_f
                loss_contour = util.l2_distance(landmarks2d[:, :17, :2], gt_landmark[:, :17, :2]) #* l_c # contour loss
                loss = loss + loss_facial + loss_contour

            loss = loss / count # average across valid views
            loss.backward()
            e_opt_rigid.step()

        # the optimized camera pose from the Stage 1
        # prepare camera for Stage 2
        optimized_camera_pose_list = []
        cam_list = []
        for i in range(len(imgs)):
            # camera pose for the i-the view
            d_camera_rotation = d_camera_rotation_list[i]
            d_camera_translation = d_camera_translation_list[i]
            optimized_camera_pose = camera_pose + torch.cat((d_camera_rotation, d_camera_translation))
            optimized_camera_pose = optimized_camera_pose.detach()
            optimized_camera_pose_list.append(optimized_camera_pose)
            Rt = create_diff_world_to_view_matrix(optimized_camera_pose)
            cam = PerspectiveCamera(Rt=Rt, fov=self.fov, bg=self.bg_color, 
                                    image_width=self.W, image_height=self.H, znear=self.znear, zfar=self.zfar)
            cam_list.append(cam)

        ############################
        ## Stage 2: fine fitting   #
        ############################

        # prepare shape code offsets (to be optimized)
        d_shape = torch.zeros(params['shape'].shape)
        d_shape = nn.Parameter(d_shape.float().to(self.device))

        # prepare texture code offsets (to be optimized)
        d_tex = torch.zeros(params['tex'].shape)
        d_tex = nn.Parameter(d_tex.float().to(self.device))

        # prepare light code offsets (to be optimized)
        d_light = torch.zeros([1,9,3])
        d_light = nn.Parameter(d_light.float().to(self.device))

        # prepare expression code offsets (to be optimized)
        d_exp = torch.zeros(params['exp'].shape)
        d_exp = nn.Parameter(d_exp.float().to(self.device))

        # prepare jaw pose offsets (to be optimized)
        d_jaw = torch.zeros(3)
        d_jaw = nn.Parameter(d_jaw.float().to(self.device))    
        
        # prepare eyes poses offsets (to be optimized)
        eye_pose = torch.zeros(1,6) # FLAME's default_eyeball_pose are zeros
        eye_pose = nn.Parameter(eye_pose.float().to(self.device))    

        finetune_params = [
            {'params': [d_tex], 'lr': 0.005}, 
            {'params': [d_light], 'lr': 0.005}, 
            {'params': [d_exp], 'lr': 0.005}, 
            {'params': [d_jaw], 'lr': 0.005},
            {'params': [eye_pose], 'lr': 0.005},
            # {'params': [d_camera_translation], 'lr': 0.005}, 
            # {'params': [d_camera_rotation], 'lr': 0.005},
        ]
        if shape_code is None:
            finetune_params.append({'params': [d_shape], 'lr': 0.005})

        # fine optimizer
        e_opt_fine = torch.optim.Adam(
            finetune_params,
            weight_decay=0.0001
        )

        # optimization loop
        for iter in range(800):
            e_opt_fine.zero_grad()

            # flame shape model
            optimized_pose = torch.from_numpy(params['pose']).to(self.device).detach()
            optimized_pose[0,:3] *= 0 # we clear FLAME's head pose 
            optimized_pose[:,3:] = optimized_pose[:,3:] + d_jaw
            vertices, _, _ = self.flame(shape_params=shape+d_shape, 
                                        expression_params=exp+d_exp, 
                                        pose_params=optimized_pose, 
                                        eye_pose_params=eye_pose) # [1, N, 3]
            
            # flame texture model
            texture = self.flametex(tex + d_tex) # [N, 3, 256, 256]

            count = 0
            loss = 0
            for i in range(len(imgs)):
                if deca_dict_list[i] is None or lmks_dense_list[i] is None:
                    continue
                else:
                    count += 1

                # get the camera for the i-th view
                cam = cam_list[i]

                # retrive ground-truth data for the i-th view
                gt_landmark = gt_landmark_tensor[i][None]
                gt_eyes_landmark = gt_eyes_landmark_tensor[i][None]
                gt_face_mask = gt_face_masks[i]
                gt_img = gt_imgs[i]

                # project landmarks via NV diff renderer
                verts_clip = self.mesh_renderer.project_vertices_from_camera(vertices, cam)
                verts_ndc_3d = verts_clip_to_ndc(verts_clip, image_size=self.H, out_dim=3) # convert the clipped vertices to NDC, output [N, 3]
                landmarks3d = self.flame.seletec_3d68(verts_ndc_3d[None]) # [1, 68, 3]
                landmarks2d = landmarks3d[:,:,:2] / float(self.flame_cfg.cropped_size) * 2 - 1  # [1, 68, 2]

                # eyes landmarks
                eyes_landmarks3d = verts_ndc_3d[self.R_EYE_INDICES + self.L_EYE_INDICES][None]  # [1, 10, 3]
                eyes_landmarks2d = eyes_landmarks3d[:,:,:2] / float(self.flame_cfg.cropped_size) * 2 - 1  # [1, 10, 2]

                # render textured mesh (using PyTorch3D's renderer)
                verts_transformed = verts_clip[...,:3]
                verts_transformed = verts_transformed[..., :3] / (verts_clip[...,3:4] + 1e-8) # perspective division
                rendered_textured = self.flame_texture_render(vertices, verts_transformed, texture, light+d_light)
                rendered_textured = rendered_textured['images'].flip(2) # [1,C,H,W]
                rendered_textured = rendered_textured[:,:3,:,:] # [1,3,H,W] RGBA to RGB 

                # loss computation and optimization
                loss_photo = compute_batch_pixelwise_l1_loss(gt_img, rendered_textured, gt_face_mask) * 8  # photometric loss
                loss_facial = util.l2_distance(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]) * 1       # facial 51 landmarks loss
                loss_contour = util.l2_distance(landmarks2d[:, :17, :2], gt_landmark[:, :17, :2]) * 1      # contour 17 landmarks loss
                #loss_contour = 0
                loss_eyes = util.l2_distance(eyes_landmarks2d, gt_eyes_landmark) * 1
                loss_reg_shape = (torch.sum((shape + d_shape) ** 2) / 2) * 1e-4 # 1e-4
                loss_reg_exp = (torch.sum((exp + d_exp) ** 2) / 2) * 1e-4 # 1e-3
                #loss_reg_tex = (torch.sum((tex + d_tex) ** 2) / 2) * 1e-5 # 1e-4
                loss_reg = loss_reg_shape + loss_reg_exp #+ loss_reg_tex
                loss = loss_photo + loss_facial + loss_contour + loss_eyes + loss_reg
            loss = loss / count # average across valid views
            loss.backward()
            e_opt_fine.step()

        ##############################
        ## for displaying results    #
        ##############################
        with torch.no_grad():
            # construct canonical shape
            optimized_exp = exp + d_exp
            optimized_pose = torch.from_numpy(params['pose']).to(self.device).detach()
            optimized_pose[0,:3] *= 0 # we clear FLAME's head pose 
            optimized_pose[:,3:] = optimized_pose[:,3:] + d_jaw # clear head pose and set jaw pose
            vertices, _, _ = self.flame(shape_params=shape+d_shape, 
                                        expression_params=exp+d_exp, 
                                        pose_params=optimized_pose, 
                                        eye_pose_params=eye_pose)
            texture = self.flametex(tex + d_tex) # [N, 3, 256, 256]

            rendered_mesh_shape_img_list = []
            rendered_mesh_shape_list = []
            for i in range(len(imgs)):
                if deca_dict_list[i] is None:
                    rendered_mesh_shape_img_list.append(None)
                    rendered_mesh_shape_list.append(None)
                    continue

                cam = cam_list[i]
                img_resized = imgs_resized[i]

                # render landmarks via NV diff renderer
                new_mesh_renderer = NVDiffRenderer().to(self.device) # there seems to be a bug with the NVDiffRenderer, so I create this new
                                                                    # render everytime to render the image
                rendered = new_mesh_renderer.render_from_camera(vertices, self.mesh_faces, cam) # vertices should have the shape of [1, N, 3]
                verts_clip = rendered['verts_clip'] # [1, N, 3]
                verts_ndc_3d = verts_clip_to_ndc(verts_clip, image_size=self.H, out_dim=3) # convert the clipped vertices to NDC, output [N, 3]
                landmarks3d = self.flame.seletec_3d68(verts_ndc_3d[None]) # [1, 68, 3]
                landmarks2d = landmarks3d[:,:,:2].detach().cpu().numpy() # [1, 68, 2]
                eyes_landmarks3d = verts_ndc_3d[self.R_EYE_INDICES + self.L_EYE_INDICES][None]  # [1, 10, 3]
                eyes_landmarks2d = eyes_landmarks3d[:,:,:2].detach().cpu().numpy()  # [1, 10, 2]
                rendered_mesh_shape = rendered['rgba'][0,...,:3].detach().cpu().numpy()
                rendered_mesh_shape = np.array(np.clip(rendered_mesh_shape * 255, 0, 255), dtype=np.uint8) # uint8

                # render textured mesh (using PyTorch3D's renderer)
                verts_transformed = verts_clip[...,:3]
                verts_transformed = verts_transformed[..., :3] / (verts_clip[...,3:4] + 1e-8) # perspective division
                rendered_textured = self.flame_texture_render(vertices, verts_transformed, texture, light+d_light)
                rendered_textured = rendered_textured['images'].flip(2) # [1,C,H,W]
                rendered_textured = rendered_textured[:,:3,:,:] # [1,3,H,W] RGBA to RGB 
                rendered_textured = rendered_textured.permute(0,2,3,1)[0].detach().cpu().numpy()
                rendered_mesh_shape_img = rendered_textured
                # rendered_mesh_shape_img = (img_resized / 255. + rendered_textured) / 2
                rendered_mesh_shape_img = np.array(np.clip(rendered_mesh_shape_img * 255, 0, 255), dtype=np.uint8) # uint8

                # Draw 2D landmarks as green dots
                for coords in landmarks2d[0]:
                    coords = np.clip(coords, 0, self.H-1).astype(np.uint8)
                    #coords = np.clip((coords / 2 + 1) * self.H, 0, self.H).astype(np.uint8)
                    cv2.circle(rendered_mesh_shape, (coords[0], coords[1]), radius=1, color=(0, 255, 0), thickness=-1)  # Green color, filled circle

                # Optionally draw eye landmarks as red dots
                for coords in eyes_landmarks2d[0]:
                    coords = np.clip(coords, 0, self.H-1).astype(np.uint8)
                    #coords = np.clip((coords / 2 + 1) * self.H, 0, self.H).astype(np.uint8)
                    cv2.circle(rendered_mesh_shape, (coords[0], coords[1]), radius=1, color=(0, 0, 255), thickness=-1)  # Red color, filled circle

                rendered_mesh_shape_img_list.append(rendered_mesh_shape_img)
                rendered_mesh_shape_list.append(rendered_mesh_shape)

        ####################
        # Prepare results  #
        ####################
        for i in range(len(optimized_camera_pose_list)):
            optimized_camera_pose = optimized_camera_pose_list[i]
            optimized_camera_pose_list[i] = optimized_camera_pose.detach().cpu().numpy()
        
        ret_dict = {
            'vertices': vertices[0].detach().cpu().numpy(),      # [N, 3] canonical shape (including expression)
            'shape': (shape + d_shape).detach().cpu().numpy(),   # canonical FLAME shape code
            'exp': (exp + d_exp).detach().cpu().numpy(),         # canonical FLAME expression code
            'pose': optimized_pose.detach().cpu().numpy(),       # canonical FLAME jaw pose code
            'eye_pose': eye_pose.detach().cpu().numpy(),         # canonical FLAME eye pose code
            'tex': (tex + d_tex).detach().cpu().numpy(),         # canonical FLAME texture code
            'light': (light + d_light).detach().cpu().numpy(),   # canonical FLAME light coefficients
            'cam': optimized_camera_pose_list,                   # [N][6] optimized camera poses for each view
            'img_rendered': rendered_mesh_shape_img_list,        # rendered result for each view
            'mesh_rendered': rendered_mesh_shape_list,           # rendered result for each view
        }
        
        return ret_dict


