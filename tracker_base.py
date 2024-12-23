###########################################
## FLAME Tracker Reconstruction Base.     #
## -------------------------------------- #
## Author: Peizhi Yan                     #
## Update: 12/22/2024                     #
###########################################

## Copyright (C) Peizhi Yan. 2024

## Installed Packages
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np # version 1.23.4, higher version may cause problem
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

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
from utils.image_utils import read_img, min_max, uint8_img, norm_img, image_align, display_landmarks_with_cv2, get_face_mask, get_foreground_mask
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
        self.set_landmark_detector('mediapipe') # use Mediapipe as default face landmark detector

        if 'ear_landmarker_path' in tracker_cfg:
            # use ear landmarks during fitting
            self.use_ear_landmarks = True
            self.ear_landmarker = torch.load(tracker_cfg['ear_landmarker_path']).eval() # load the ONNX converted ear landmarker model
            self.ear_landmarker = self.ear_landmarker.to(self.device)
        else:
            self.use_ear_landmarks = False

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

        # Ear Indices (FLAME mesh), labeled following I-Bug Ear's landmarks from 0 to 19
        self.L_EAR_INDICES = [342, 341, 166, 514, 476, 185, 369, 29, 204, 641, 179, 178, 71, 68, 138, 141, 91, 40, 96, 184]
        self.R_EAR_INDICES = [1263, 844, 845, 2655, 870, 872, 1207, 523, 901, 1859, 860, 621, 618, 890, 981, 556, 554, 676, 1209, 868]

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

        # Kalman filters
        if self.use_kalman_filter:
            self.kf_R = initialize_kalman_matrix(m=1, n=3, 
                                measure_noise=self.kf_measure_noise, 
                                process_noise=self.kf_process_noise)  # Initialize Kalman filter for camera rotations
            self.kf_T = initialize_kalman_matrix(m=1, n=3,
                                measure_noise=self.kf_measure_noise, 
                                process_noise=self.kf_process_noise)  # Initialize Kalman filter for camera translations

        print('Flame Tracker ready.')


    def set_landmark_detector(self, landmark_detector='mediapipe'):
        """
        Set the face landmark detector
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


    def detect_ear_landmarks(self, img):
        """
        Run ear landmark detector
        Note that, the ear landmarks can be either from right or left or both ears.
        Thus, later when we use them we need to handle these potential scenarios.
        input:
            - img: image data   numpy  uint8
        output:
            - ear_lmks: 20 ear landmarks   numpy  [20, 2], the locations are in normalized scale (-1.0 ~ 1.0)
        """
        EAR_LMK_DETECTOR_INPUT_SIZE = 368
        input_size = (EAR_LMK_DETECTOR_INPUT_SIZE, EAR_LMK_DETECTOR_INPUT_SIZE)

        # run ear landmark detector model
        with torch.no_grad():
            input_image = cv2.resize(img, input_size) # resize the image to match the input size of the model
            input_image = input_image.astype(np.float32) / 255.0 # convert pixels from 0~255 to 0~1.0
            input_image = input_image[None] # extend to batch size == 1   [1, 368, 368, 3]
            input_image_tensor = torch.from_numpy(input_image).to(self.device) # [1, 368, 368, 3]
            heatmaps = self.ear_landmarker(input_image_tensor)  # [1, 46, 46, 55]
            heatmaps = heatmaps.detach().cpu().numpy()[0]  # [46, 46, 55]

        # resize headmap back to ear landmarker model's input image size
        heatmap = cv2.resize(heatmaps, input_size)  # [368, 368, 55] H == W
        blurred_heatmap = gaussian_filter(heatmap, sigma=2.5) # apply Gaussian blue on the heat map to make landmarks smooth

        # find the maximum indices
        temp = np.argmax(blurred_heatmap.reshape(-1, 55), axis=0)
        ear_landmarks = np.zeros([20, 2], dtype=np.float32) # only keep landmarks 0 ~ 19
        for i in range(20):
            idx = temp[i]
            x, y = idx % EAR_LMK_DETECTOR_INPUT_SIZE, idx // EAR_LMK_DETECTOR_INPUT_SIZE
            ear_landmarks[i, 0] = x
            ear_landmarks[i, 1] = y
        ear_landmarks = ear_landmarks / float(EAR_LMK_DETECTOR_INPUT_SIZE) * 2 - 1 # normalize landmarks to -1.0 ~ 1.0

        return ear_landmarks


    def load_image_and_run(self, 
                           img_path, realign=True, 
                           photometric_fitting=False,
                           prev_ret_dict=None, shape_code=None):
        """
        Load image from given path, then run FLAME tracking
        input:
            -img_path: image path
            -realign: for FFHQ, use False. for in-the-wild images, use True
            -photometric_fitting: whether to use photometric fitting or landmarks only
            -prev_ret_dict: the results dictionary from the previous frame
            -shape_code: the pre-estimated global shape code
        output:
            -ret_dict: results dictionary
        """
        img = read_img(img_path)
        return self.run(img, realign, photometric_fitting, prev_ret_dict, shape_code)

    
    def run(self, img, realign=True, photometric_fitting=False, prev_ret_dict=None, shape_code=None):
        """
        Run FLAME tracking on the given image
        input:
            -img: image data   numpy 
            -realign: for FFHQ, use False. for in-the-wild images, use True
            -photometric_fitting: whether to use photometric fitting or landmarks only
            -prev_ret_dict: the results dictionary from the previous frame
            -shape_code: the pre-estimated global shape code
        output:
            -ret_dict: results dictionary
        """

        # run Mediapipe face detector
        lmks_dense, blend_scores = self.mediapipe_face_detection(img)
        if lmks_dense is None:
            # no face detected
            return None
        
        if self.landmark_detector == 'mediapipe':
            face_landmarks = convert_landmarks_mediapipe_to_dlib(lmks_mp=lmks_dense)
        elif self.landmark_detector == 'FAN':
            # run FAN face landmarks predictor
            face_landmarks = self.fan_face_landmarks(img)
            if face_landmarks is None:
                # no face detected
                return None
        else:
            return None
        
        # re-align image (tracking standard), this image will be used in our network model
        img_aligned = image_align(img, face_landmarks, output_size=self.img_size, standard='tracking', 
                                  padding_mode='constant')

        if realign:
            # realign == True means that we will fit on realigned image
            img = img_aligned

        # run DECA reconstruction
        deca_dict = self.run_deca(img)

        # run face parsing
        parsing_mask = self.face_parser.run(img)
        if realign: parsing_mask_aligned = parsing_mask
        else: parsing_mask_aligned = self.face_parser.run(img_aligned)
        
        if photometric_fitting:
            # run photometric fitting
            face_mask = get_face_mask(parsing=parsing_mask, keep_ears=False)
            ret_dict = self.run_fitting_photometric(img, face_mask, deca_dict, prev_ret_dict, shape_code)
        else:
            # run facial landmark-based fitting
            ret_dict = self.run_fitting(img, deca_dict, prev_ret_dict, shape_code)

        # check for NaNs, if there is any, return None
        _, nan_status = check_nan_in_dict(ret_dict)
        if nan_status:
            return None

        # add more data
        ret_dict['img'] = img
        ret_dict['img_aligned'] = img_aligned
        ret_dict['parsing'] = parsing_mask
        ret_dict['parsing_aligned'] = parsing_mask_aligned
        ret_dict['lmks_dense'] = lmks_dense
        ret_dict['lmks_68'] = face_landmarks
        ret_dict['blendshape_scores'] = blend_scores

        return ret_dict
    

    def run_all_images(self, imgs : list, realign=True):
        """
        Run FLAME tracking (landmark-based) on all loaded images
        input:
            -imgs: list of image data, [numpy] 
            -realign: for FFHQ, use False. for in-the-wild images, use True
        output:
            -ret_dict: results dictionary
        """
        
        NUM_OF_IMGS = len(imgs)
        DECA_BATCH_SIZE = 16

        # initialize preprocessing dictionary
        ret_dict_all = {}
        for key in ['img', 'img_aligned', 'parsing', 'parsing_aligned', 'lmks_dense', 'lmks_68', 'blendshape_scores', 
                    'vertices', 'shape', 'exp', 'pose', 'eye_pose', 'tex', 'light', 'cam', 'img_rendered']:
            ret_dict_all[key] = []

        # run Mediapipe face detector and align images
        print('Running face detector, face parser, and aligning images')
        for i in tqdm(range(NUM_OF_IMGS)):
            img = imgs[i]

            # run Mediapipe face detector
            lmks_dense, blend_scores = self.mediapipe_face_detection(img)
            if lmks_dense is None:
                # no face detected
                continue
            face_landmarks = convert_landmarks_mediapipe_to_dlib(lmks_mp=lmks_dense)

            # re-align image (tracking standard), this image will be used in our network model
            img_aligned = image_align(img, face_landmarks, output_size=self.img_size, standard='tracking', 
                                    padding_mode='constant')

            if realign:
                # realign == True means that we will fit on realigned image
                img = img_aligned

            # run face parsing
            parsing_mask = self.face_parser.run(img)
            if realign: parsing_mask_aligned = parsing_mask
            else: parsing_mask_aligned = self.face_parser.run(img_aligned)

            ret_dict_all['img'].append(img)
            ret_dict_all['img_aligned'].append(img_aligned)
            ret_dict_all['parsing'].append(parsing_mask)
            ret_dict_all['parsing_aligned'].append(parsing_mask_aligned)
            ret_dict_all['lmks_dense'].append(lmks_dense)
            ret_dict_all['lmks_68'].append(face_landmarks)
            ret_dict_all['blendshape_scores'].append(blend_scores)


        # run DECA reconstruction
        print("Running DECA")
        with torch.no_grad():
            deca_dict_all = None
            if NUM_OF_IMGS % DECA_BATCH_SIZE == 0: NUM_OF_BATCHES = NUM_OF_IMGS // DECA_BATCH_SIZE
            else: NUM_OF_BATCHES = NUM_OF_IMGS // DECA_BATCH_SIZE + 1
            for i in tqdm(range(NUM_OF_BATCHES)):
                imgs_batch = ret_dict_all['img'][i*DECA_BATCH_SIZE : (i+1)*DECA_BATCH_SIZE]
                deca_dict_batch = self.run_deca_batch(imgs_batch)
                if deca_dict_all is None:
                    deca_dict_all = deca_dict_batch
                else:
                    for key in deca_dict_batch:
                        deca_dict_all[key] = torch.cat([deca_dict_all[key], deca_dict_batch[key]])
        
        # compute the mean shape code
        mean_shape_code = torch.mean(deca_dict_all['shape'], dim=0)[None] # [1, 100]

        # run facial landmark-based fitting
        print("Fitting")
        output_dict_all = {}
        for key in ['img', 'img_aligned', 'parsing', 'parsing_aligned', 'lmks_dense', 'lmks_68', 'blendshape_scores', 
                    'vertices', 'shape', 'exp', 'pose', 'eye_pose', 'tex', 'light', 'cam', 'img_rendered']:
            output_dict_all[key] = []
        prev_ret_dict = None
        for i in tqdm(range(NUM_OF_IMGS)):
            img = ret_dict_all['img'][i]
            deca_dict = {}
            for key in deca_dict_all.keys():
                if key == 'shape':
                    deca_dict[key] = mean_shape_code
                else:
                    deca_dict[key] = deca_dict_all[key][i:i+1]
            ret_dict = self.run_fitting(img, deca_dict, prev_ret_dict)
            prev_ret_dict = ret_dict
            if ret_dict is None:
                continue
            for key in ret_dict.keys():
                output_dict_all[key].append(ret_dict[key])
            for key in ret_dict_all.keys():
                output_dict_all[key].append(ret_dict[key])

        return output_dict_all
    

    @torch.no_grad()
    def run_deca(self, img):
        """ Run DECA on a single input image """

        # DECA Encode
        image = self.deca.crop_image(img).to(self.device)
        deca_dict = self.deca.encode(image[None])
        
        return deca_dict
    

    @torch.no_grad()
    def run_deca_batch(self, imgs : list):
        """ Run Deca on a batch of images """

        # convert to batch tensor
        images = []
        for i in range(len(imgs)):
            image = self.deca.crop_image(imgs[i]).to(self.device)
            images.append(image[None])
        images = torch.cat(images) # [N,C,H,W]

        # DECA Encode
        deca_dict = self.deca.encode(images)

        return deca_dict

    
    def run_fitting(self, img, deca_dict, prev_ret_dict, shape_code):
        """ Landmark-based Fitting        
            - Stage 1: rigid fitting on the camera pose (6DoF) based on detected landmarks
            - Stage 2: fine-tune the parameters including shape, tex, exp, pose, eye_pose, and light
        """

        if prev_ret_dict is not None:
            continue_fit = True
        else:
            continue_fit = False

        # convert the parameters to numpy arrays
        params = {}
        for key in ['shape', 'tex', 'exp', 'pose', 'light']:
            if key == 'shape' and shape_code is not None:
                # use pre-estimated global shape code
                params[key] = shape_code.detach().cpu().numpy()
            else:
                params[key] = deca_dict[key].detach().cpu().numpy()

        # resize for FLAME fitting
        img_resized = cv2.resize(img, (self.flame_cfg.cropped_size, self.flame_cfg.cropped_size))

        # run Mediapipe face detector
        lmks_dense, _ = self.mediapipe_face_detection(img_resized)
        if lmks_dense is None:
            # no face detected
            return None
        lmks_dense[:, :2] = lmks_dense[:, :2] / float(self.flame_cfg.cropped_size) * 2 - 1 # normalize landmarks
        
        # gt_landmark can be either mediapipe's or FAN's landmarks
        if self.landmark_detector == 'mediapipe':
            face_landmarks = convert_landmarks_mediapipe_to_dlib(lmks_mp=lmks_dense)
        elif self.landmark_detector == 'FAN':
            # run FAN face landmarks predictor
            face_landmarks = self.fan_face_landmarks(img_resized)
            if face_landmarks is None:
                # no face detected
                return None
            face_landmarks = face_landmarks[:, :2] / float(self.flame_cfg.cropped_size) * 2 - 1 # normalize landmarks

        # prepare target 68 landmarks
        gt_landmark = np.array(face_landmarks).astype(np.float32)
        gt_landmark = torch.from_numpy(gt_landmark[None]).float().to(self.device)

        # prepare target eyes landmarks
        gt_eyes_landmark = np.array(lmks_dense[self.R_EYE_MP_LMKS + self.L_EYE_MP_LMKS]).astype(np.float32)
        gt_eyes_landmark = torch.from_numpy(gt_eyes_landmark[None]).float().to(self.device)

        # prepare ear 20 landmarks
        if self.use_ear_landmarks:
            ear_landmarks = self.detect_ear_landmarks(img_resized) # [20, 2] normalized ear landmarks
            gt_ear_landmark = torch.from_numpy(ear_landmarks[None]).float().to(self.device) # [1, 20, 2]

        ############################################################
        ## Stage 1: rigid fitting (estimate the 6DoF camera pose)  #
        ############################################################

        # prepare 6DoF camera pose tensor
        if continue_fit == False:
            camera_pose = torch.tensor([0, 0, 0, 0, 0, self.distance], dtype=torch.float32).to(self.device) # [yaw, pitch, roll, x, y, z]
        else:
            # use previous frame's estimation to initialize
            camera_pose = torch.tensor(prev_ret_dict['cam'], dtype=torch.float32).to(self.device)

        # prepare camera pose offsets (to be optimized)
        d_camera_rotation = nn.Parameter(torch.zeros(3).float().to(self.device))
        d_camera_translation = nn.Parameter(torch.zeros(3).float().to(self.device))
        camera_params = [
            {'params': [d_camera_translation], 'lr': 0.01}, {'params': [d_camera_rotation], 'lr': 0.05}
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
        if continue_fit:
            # continue to fit on the next video frame
            total_iterations = 200
        else:
            # initial fitting, take longer time
            total_iterations = 1000
        for iter in range(total_iterations):

            if continue_fit == False:
                ## initial fitting
                # update learning rate
                if iter == 700:
                    e_opt_rigid.param_groups[0]['lr'] = 0.005    # For translation
                    e_opt_rigid.param_groups[1]['lr'] = 0.01     # For rotation
                # update loss term weights
                if iter <= 700: l_f = 100; l_c = 500 # more weights to contour
                else: l_f = 500; l_c = 100 # more weights to face
            else:
                ## continue fitting
                # update learning rate
                e_opt_rigid.param_groups[0]['lr'] = 0.005    # For translation
                e_opt_rigid.param_groups[1]['lr'] = 0.01     # For rotation
                # update loss term weights
                if iter <= 100: l_f = 100; l_c = 500 # more weights to contour
                else: l_f = 500; l_c = 100 # more weights to face

            vertices, _, _ = self.flame(shape_params=shape, expression_params=exp, pose_params=pose) # [1, N, 3]

            # prep camera
            optimized_camera_pose = camera_pose + torch.cat((d_camera_rotation, d_camera_translation))
            Rt = create_diff_world_to_view_matrix(optimized_camera_pose)
            cam = PerspectiveCamera(Rt=Rt, fov=self.fov, bg=self.bg_color, 
                                    image_width=self.W, image_height=self.H, znear=self.znear, zfar=self.zfar)

            # project landmarks via NV diff renderer
            verts_clip = self.mesh_renderer.project_vertices_from_camera(vertices, cam)
            verts_ndc_3d = verts_clip_to_ndc(verts_clip, image_size=self.H, out_dim=3) # convert the clipped vertices to NDC, output [N, 3]
            landmarks3d = self.flame.seletec_3d68(verts_ndc_3d[None]) # [1, 68, 3]
            landmarks2d = landmarks3d[:,:,:2] / float(self.flame_cfg.cropped_size) * 2 - 1  # [1, 68, 2]

            # ears landmarks
            if self.use_ear_landmarks:
                left_ear_landmarks3d = verts_ndc_3d[self.L_EAR_INDICES][None]  # [1, 20, 3]
                left_ear_landmarks2d = left_ear_landmarks3d[:,:,:2] / float(self.flame_cfg.cropped_size) * 2 - 1  # [1, 20, 2]
                right_ear_landmarks3d = verts_ndc_3d[self.R_EAR_INDICES][None]  # [1, 20, 3]
                right_ear_landmarks2d = right_ear_landmarks3d[:,:,:2] / float(self.flame_cfg.cropped_size) * 2 - 1  # [1, 20, 2]

            EAR_LOSS_THRESHOLD = 0.2
            loss_ear = 0
            if self.use_ear_landmarks:
                loss_l_ear = util.l2_distance(left_ear_landmarks2d, gt_ear_landmark)
                loss_r_ear = util.l2_distance(right_ear_landmarks2d, gt_ear_landmark)
                #loss_ear = torch.min(loss_l_ear, loss_r_ear) # select the one with the smallest loss
                if loss_l_ear < EAR_LOSS_THRESHOLD:
                    loss_ear = loss_ear + loss_l_ear
                if loss_r_ear < EAR_LOSS_THRESHOLD:
                    loss_ear = loss_ear + loss_r_ear
                #print(loss_l_ear, loss_r_ear)
            loss_ear = loss_ear * 100

            # loss computation and optimization
            loss_facial = util.l2_distance(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]) * l_f
            loss_contour = util.l2_distance(landmarks2d[:, :17, :2], gt_landmark[:, :17, :2]) * l_c # contour loss
            loss = loss_facial + loss_contour + loss_ear
            e_opt_rigid.zero_grad()
            loss.backward()
            e_opt_rigid.step()

        # the optimized camera pose from the Stage 1
        optimized_camera_pose = camera_pose + torch.cat((d_camera_rotation, d_camera_translation))
        optimized_camera_pose = optimized_camera_pose.detach()

        if self.use_kalman_filter == True:
            # apply Kalman filter on camera pose
            optimized_camera_pose_np = optimized_camera_pose.cpu().numpy()[None] # [1,6]
            R = optimized_camera_pose_np[:,:3]    # [1, 3]
            T = optimized_camera_pose_np[:,3:]    # [1, 3]
            # print(R, T)
            R = kalman_filter_update_matrix(self.kf_R, R)
            T = kalman_filter_update_matrix(self.kf_T, T)
            # print(R, T)
            # print('-----------------')
            optimized_camera_pose = torch.from_numpy(np.concatenate([R, T], axis=1)[0]).to(self.device).detach()

        # prepare camera for Stage 2
        Rt = create_diff_world_to_view_matrix(optimized_camera_pose)
        cam = PerspectiveCamera(Rt=Rt, fov=self.fov, bg=self.bg_color, 
                                image_width=self.W, image_height=self.H, znear=self.znear, zfar=self.zfar)

        ############################
        ## Stage 2: fine fitting   #
        ############################

        # prepare shape code offsets (to be optimized)
        d_shape = torch.zeros(params['shape'].shape)
        d_shape = nn.Parameter(d_shape.float().to(self.device))

        # prepare expression code offsets (to be optimized)
        d_exp = torch.zeros(params['exp'].shape)
        d_exp = nn.Parameter(d_exp.float().to(self.device))

        # prepare jaw pose offsets (to be optimized)
        d_jaw = torch.zeros(3)
        d_jaw = nn.Parameter(d_jaw.float().to(self.device))    
        
        # prepare eyes poses offsets (to be optimized)
        eye_pose = torch.zeros(1,6) # FLAME's default_eyeball_pose are zeros
        eye_pose = nn.Parameter(eye_pose.float().to(self.device))    

        fine_params = [
            {'params': [d_exp], 'lr': 0.01}, 
            {'params': [d_jaw], 'lr': 0.025},
            {'params': [eye_pose], 'lr': 0.03}
        ]
        # if shape_code is None:
        #     fine_params.append({'params': [d_shape], 'lr': 0.01})

        # fine optimizer
        e_opt_fine = torch.optim.Adam(
            fine_params,
            weight_decay=0.0001
        )

        # optimization loop
        for iter in range(200):

            # update learning rate
            if iter == 100:
                e_opt_fine.param_groups[0]['lr'] = 0.005    
                e_opt_fine.param_groups[1]['lr'] = 0.01     
                e_opt_fine.param_groups[2]['lr'] = 0.01     

            optimized_exp = exp + d_exp
            optimized_pose = torch.from_numpy(params['pose']).to(self.device).detach()
            optimized_pose[0,:3] *= 0 # we clear FLAME's head pose 
            optimized_pose[:,3:] = optimized_pose[:,3:] + d_jaw
            vertices, _, _ = self.flame(shape_params=shape+d_shape, 
                                        expression_params=optimized_exp, 
                                        pose_params=optimized_pose, 
                                        eye_pose_params=eye_pose) # [1, N, 3]

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
            loss = loss_facial + loss_eyes
            e_opt_fine.zero_grad()
            loss.backward()
            e_opt_fine.step()

        ##############################
        ## for displaying results    #
        ##############################
        with torch.no_grad():
            optimized_shape = shape + d_shape
            optimized_exp = exp + d_exp
            optimized_pose = torch.from_numpy(params['pose']).to(self.device).detach()
            optimized_pose[0,:3] *= 0 # we clear FLAME's head pose 
            optimized_pose[:,3:] = optimized_pose[:,3:] + d_jaw # clear head pose and set jaw pose
            vertices, _, _ = self.flame(shape_params=optimized_shape, 
                                        expression_params=optimized_exp, 
                                        pose_params=optimized_pose, 
                                        eye_pose_params=eye_pose)

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
            ears_landmarks3d = verts_ndc_3d[self.R_EAR_INDICES + self.L_EAR_INDICES][None]  # [1, 40, 3]
            ears_landmarks2d = ears_landmarks3d[:,:,:2].detach().cpu().numpy()  # [1, 40, 2]
            rendered_mesh_shape = rendered['rgba'][0,...,:3].detach().cpu().numpy()
            rendered_mesh_shape_img = (img_resized / 255. + rendered_mesh_shape) / 2
            rendered_mesh_shape_img = np.array(np.clip(rendered_mesh_shape_img * 255, 0, 255), dtype=np.uint8) # uint8
            rendered_mesh_shape = np.array(np.clip(rendered_mesh_shape * 255, 0, 255), dtype=np.uint8) # uint8

            # Draw 2D landmarks as green dots
            for coords in landmarks2d[0]:
                coords = np.clip(coords, 0, self.H-1).astype(np.uint8)
                #coords = np.clip((coords / 2 + 1) * self.H, 0, self.H).astype(np.uint8)
                cv2.circle(rendered_mesh_shape_img, (coords[0], coords[1]), radius=1, color=(0, 255, 0), thickness=-1)  # Green color, filled circle

            # Optionally draw eye landmarks as red dots
            for coords in eyes_landmarks2d[0]:
                coords = np.clip(coords, 0, self.H-1).astype(np.uint8)
                #coords = np.clip((coords / 2 + 1) * self.H, 0, self.H).astype(np.uint8)
                cv2.circle(rendered_mesh_shape_img, (coords[0], coords[1]), radius=1, color=(0, 0, 255), thickness=-1)  # Red color, filled circle

            # Optionally draw ear landmarks as pink dots
            for coords in ears_landmarks2d[0]:
                coords = np.clip(coords, 0, self.H-1).astype(np.uint8)
                #coords = np.clip((coords / 2 + 1) * self.H, 0, self.H).astype(np.uint8)
                cv2.circle(rendered_mesh_shape_img, (coords[0], coords[1]), radius=1, color=(255, 0, 255), thickness=-1)

        ####################
        # Prepare results  #
        ####################
        ret_dict = {
            'vertices': vertices[0].detach().cpu().numpy(),  # [N, 3]
            'shape': optimized_shape.detach().cpu().numpy(),    # [1,100]
            'exp': optimized_exp.detach().cpu().numpy(),
            'pose': optimized_pose.detach().cpu().numpy(),
            'eye_pose': eye_pose.detach().cpu().numpy(),
            'tex': params['tex'],
            'light': params['light'],
            'cam': optimized_camera_pose.detach().cpu().numpy(), # [6]
            'img_rendered': rendered_mesh_shape_img,
            'mesh_rendered': rendered_mesh_shape,
        }
        
        return ret_dict


    def run_fitting_photometric(self, img, face_mask, deca_dict, prev_ret_dict, shape_code):
        """ Landmark and Photometric Fitting        
            - Stage 1: rigid fitting on the camera pose (6DoF)
            - Stage 2: fine-tune the expression parameters, the jaw pose (3DoF), and the eyes poses (3DoF + 3DoF)
        """

        if prev_ret_dict is not None:
            continue_fit = True
        else:
            continue_fit = False

        # convert the parameters to numpy arrays
        params = {}
        for key in ['shape', 'tex', 'exp', 'pose', 'light']:
            if key == 'shape' and shape_code is not None:
                # use pre-estimated global shape code
                params[key] = shape_code.detach().cpu().numpy()
            elif key in ['shape', 'exp']:
                params[key] = deca_dict[key].detach().cpu().numpy() * 0.0 # clear DECA's code
            else:
                params[key] = deca_dict[key].detach().cpu().numpy()

        # resize for FLAME fitting
        img_resized = cv2.resize(img, (self.flame_cfg.cropped_size, self.flame_cfg.cropped_size))
        gt_img = torch.from_numpy(np.array(img_resized,dtype=np.float32) / 255.).detach().to(self.device) # [H,W,C] float32
        gt_img = gt_img[None].permute(0,3,1,2) # [1,C,H,W]
        face_mask_resized = cv2.resize(face_mask, (self.flame_cfg.cropped_size, self.flame_cfg.cropped_size)) # [H,W]
        gt_face_mask = torch.from_numpy(face_mask_resized)[None].detach().to(self.device) # [1,H,W] boolean

        # run Mediapipe face detector
        lmks_dense, _ = self.mediapipe_face_detection(img_resized)
        lmks_dense[:, :2] = lmks_dense[:, :2] / float(self.flame_cfg.cropped_size) * 2 - 1 # normalize landmarks
        ##face_landmarks = convert_landmarks_mediapipe_to_dlib(lmks_mp=lmks_dense) # 68 dlib landmarks

        # gt_landmark can be either mediapipe's or FAN's landmarks
        if self.landmark_detector == 'mediapipe':
            face_landmarks = convert_landmarks_mediapipe_to_dlib(lmks_mp=lmks_dense)
        elif self.landmark_detector == 'FAN':
            # run FAN face landmarks predictor
            face_landmarks = self.fan_face_landmarks(img_resized)
            if face_landmarks is None:
                # no face detected
                return None
            face_landmarks = face_landmarks[:, :2] / float(self.flame_cfg.cropped_size) * 2 - 1 # normalize landmarks

        # prepare target 68 landmarks
        gt_landmark = np.array(face_landmarks).astype(np.float32)
        gt_landmark = torch.from_numpy(gt_landmark[None]).float().to(self.device)

        # prepare target eyes landmarks
        gt_eyes_landmark = np.array(lmks_dense[self.R_EYE_MP_LMKS + self.L_EYE_MP_LMKS]).astype(np.float32)
        gt_eyes_landmark = torch.from_numpy(gt_eyes_landmark[None]).float().to(self.device)

        # prepare ear 20 landmarks
        if self.use_ear_landmarks:
            ear_landmarks = self.detect_ear_landmarks(img_resized) # [20, 2] normalized ear landmarks
            gt_ear_landmark = torch.from_numpy(ear_landmarks[None]).float().to(self.device) # [1, 20, 2]

        ############################################################
        ## Stage 1: rigid fitting (estimate the 6DoF camera pose)  #
        ############################################################

        # prepare 6DoF camera pose tensor
        if continue_fit == False:
            camera_pose = torch.tensor([0, 0, 0, 0, 0, self.distance], dtype=torch.float32).to(self.device) # [yaw, pitch, roll, x, y, z]
        else:
            # use previous frame's estimation to initialize
            camera_pose = torch.tensor(prev_ret_dict['cam'], dtype=torch.float32).to(self.device)

        # prepare camera pose offsets (to be optimized)
        d_camera_rotation = nn.Parameter(torch.zeros(3).float().to(self.device))
        d_camera_translation = nn.Parameter(torch.zeros(3).float().to(self.device))
        camera_params = [
            #{'params': [d_camera_translation], 'lr': 0.01}, {'params': [d_camera_rotation], 'lr': 0.05}
            {'params': [d_camera_translation], 'lr': 0.005}, {'params': [d_camera_rotation], 'lr': 0.005}, 
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
        if continue_fit:
            # continue to fit on the next video frame
            total_iterations = 200
        else:
            # initial fitting, take longer time
            total_iterations = 200 #500
        for iter in range(total_iterations):

            vertices, _, _ = self.flame(shape_params=shape, expression_params=exp, pose_params=pose) # [1, N, 3]

            # prep camera
            optimized_camera_pose = camera_pose + torch.cat((d_camera_rotation, d_camera_translation))
            Rt = create_diff_world_to_view_matrix(optimized_camera_pose)
            cam = PerspectiveCamera(Rt=Rt, fov=self.fov, bg=self.bg_color, 
                                    image_width=self.W, image_height=self.H, znear=self.znear, zfar=self.zfar)

            # project landmarks via NV diff renderer
            verts_clip = self.mesh_renderer.project_vertices_from_camera(vertices, cam)
            verts_ndc_3d = verts_clip_to_ndc(verts_clip, image_size=self.H, out_dim=3) # convert the clipped vertices to NDC, output [N, 3]
            landmarks3d = self.flame.seletec_3d68(verts_ndc_3d[None]) # [1, 68, 3]
            landmarks2d = landmarks3d[:,:,:2] / float(self.flame_cfg.cropped_size) * 2 - 1  # [1, 68, 2]

            # ears landmarks
            if self.use_ear_landmarks:
                left_ear_landmarks3d = verts_ndc_3d[self.L_EAR_INDICES][None]  # [1, 20, 3]
                left_ear_landmarks2d = left_ear_landmarks3d[:,:,:2] / float(self.flame_cfg.cropped_size) * 2 - 1  # [1, 20, 2]
                right_ear_landmarks3d = verts_ndc_3d[self.R_EAR_INDICES][None]  # [1, 20, 3]
                right_ear_landmarks2d = right_ear_landmarks3d[:,:,:2] / float(self.flame_cfg.cropped_size) * 2 - 1  # [1, 20, 2]

            EAR_LOSS_THRESHOLD = 0.2
            loss_ear = 0
            if self.use_ear_landmarks:
                loss_l_ear = util.l2_distance(left_ear_landmarks2d, gt_ear_landmark)
                loss_r_ear = util.l2_distance(right_ear_landmarks2d, gt_ear_landmark)
                #loss_ear = torch.min(loss_l_ear, loss_r_ear) # select the one with the smallest loss
                if loss_l_ear < EAR_LOSS_THRESHOLD:
                    loss_ear = loss_ear + loss_l_ear
                if loss_r_ear < EAR_LOSS_THRESHOLD:
                    loss_ear = loss_ear + loss_r_ear
                #print(loss_l_ear, loss_r_ear)
            loss_ear = loss_ear * 1.0

            # loss computation and optimization
            loss_facial = util.l2_distance(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]) #* l_f
            loss_contour = util.l2_distance(landmarks2d[:, :17, :2], gt_landmark[:, :17, :2]) #* l_c # contour loss
            loss = loss_facial + loss_contour + loss_ear
            e_opt_rigid.zero_grad()
            loss.backward()
            e_opt_rigid.step()

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
            {'params': [d_camera_translation], 'lr': 0.005}, 
            {'params': [d_camera_rotation], 'lr': 0.005},
        ]
        if shape_code is None:
            finetune_params.append({'params': [d_shape], 'lr': 0.005})

        # fine optimizer
        e_opt_fine = torch.optim.Adam(
            finetune_params,
            weight_decay=0.0001
        )

        # optimization loop
        if continue_fit:
            # continue to fit on the next video frame
            total_iterations = 300
        else:
            # initial fitting, take longer time
            total_iterations = 800      
        for iter in range(total_iterations):
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

            # prepare camera            
            optimized_camera_pose = camera_pose + torch.cat((d_camera_rotation, d_camera_translation))
            Rt = create_diff_world_to_view_matrix(optimized_camera_pose)
            cam = PerspectiveCamera(Rt=Rt, fov=self.fov, bg=self.bg_color, 
                                    image_width=self.W, image_height=self.H, znear=self.znear, zfar=self.zfar)

            # project landmarks via NVdiffrast
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

            # ears landmarks
            if self.use_ear_landmarks:
                left_ear_landmarks3d = verts_ndc_3d[self.L_EAR_INDICES][None]  # [1, 20, 3]
                left_ear_landmarks2d = left_ear_landmarks3d[:,:,:2] / float(self.flame_cfg.cropped_size) * 2 - 1  # [1, 20, 2]
                right_ear_landmarks3d = verts_ndc_3d[self.R_EAR_INDICES][None]  # [1, 20, 3]
                right_ear_landmarks2d = right_ear_landmarks3d[:,:,:2] / float(self.flame_cfg.cropped_size) * 2 - 1  # [1, 20, 2]

            EAR_LOSS_THRESHOLD = 0.2
            loss_ear = 0
            if self.use_ear_landmarks:
                loss_l_ear = util.l2_distance(left_ear_landmarks2d, gt_ear_landmark)
                loss_r_ear = util.l2_distance(right_ear_landmarks2d, gt_ear_landmark)
                #loss_ear = torch.min(loss_l_ear, loss_r_ear) # select the one with the smallest loss
                if loss_l_ear < EAR_LOSS_THRESHOLD:
                    loss_ear = loss_ear + loss_l_ear
                if loss_r_ear < EAR_LOSS_THRESHOLD:
                    loss_ear = loss_ear + loss_r_ear
                #print(loss_l_ear, loss_r_ear)
            loss_ear = loss_ear * 1.0

            # # Exclude nose wing landmarks
            # exclude_indices = [31, 32, 34, 35]
            # lmks_mask = torch.ones(landmarks2d[:, :, :2].shape[1], dtype=torch.bool)
            # lmks_mask[exclude_indices] = False
            # filtered_landmarks2d = landmarks2d[:, :, :2][:, lmks_mask]
            # filtered_gt_landmark = gt_landmark[:, :, :2][:, lmks_mask]

            # loss computation and optimization
            loss_photo = compute_batch_pixelwise_l1_loss(gt_img, rendered_textured, gt_face_mask) * 8  # photometric loss
            # loss_facial = util.l2_distance(filtered_landmarks2d[:, 17:, :2], filtered_gt_landmark[:, 17:, :2]) * 10   # facial 51-4 landmarks loss
            loss_facial = util.l2_distance(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]) * 1       # facial 51-4 landmarks loss
            loss_contour = util.l2_distance(landmarks2d[:, :17, :2], gt_landmark[:, :17, :2]) * 1 #0.2    # contour 17 landmarks loss
            #loss_contour = 0
            loss_eyes = util.l2_distance(eyes_landmarks2d, gt_eyes_landmark) * 1
            loss_reg_shape = (torch.sum((shape + d_shape) ** 2) / 2) * 1e-4 # 1e-4
            loss_reg_exp = (torch.sum((exp + d_exp) ** 2) / 2) * 1e-4 # 1e-3
            #loss_reg_tex = (torch.sum((tex + d_tex) ** 2) / 2) * 1e-5 # 1e-4
            loss_reg = loss_reg_shape + loss_reg_exp # + loss_reg_tex
            loss = loss_photo + loss_facial + loss_contour + loss_eyes + loss_ear + loss_reg
            # if iter % 50 == 0: 
            #     print(loss_photo.item(), loss_facial.item(), loss_reg.item())
            #     import matplotlib.pyplot as plt
            #     plt.figure(figsize=(3,3))
            #     temp = np.array(np.clip(rendered_textured.permute(0,2,3,1).detach().cpu().numpy()[0] * 255, 0, 255), dtype=np.uint8)
            #     plt.imshow(temp); plt.axis('off')
            #     plt.show()
            loss.backward()
            e_opt_fine.step()

        #####################
        ## final results    #
        #####################
        with torch.no_grad():
            optimized_pose = torch.from_numpy(params['pose']).to(self.device).detach()
            optimized_pose[0,:3] *= 0 # we clear FLAME's head pose 
            optimized_pose[:,3:] = optimized_pose[:,3:] + d_jaw # clear head pose and set jaw pose
            vertices, _, _ = self.flame(shape_params=shape+d_shape, 
                                        expression_params=exp+d_exp, 
                                        pose_params=optimized_pose, 
                                        eye_pose_params=eye_pose)
            
            # flame texture model
            texture = self.flametex(tex + d_tex) # [N, 3, 256, 256]
            
            # render via NVdiffrast
            new_mesh_renderer = NVDiffRenderer().to(self.device) # there seems to be a bug with the NVDiffRenderer, so I create this new
                                                                 # render everytime to render the image
            rendered = new_mesh_renderer.render_from_camera(vertices, self.mesh_faces, cam) # vertices should have the shape of [1, N, 3]
            verts_clip = rendered['verts_clip'] # [1, N, 3]
            verts_ndc_3d = verts_clip_to_ndc(verts_clip, image_size=self.H, out_dim=3) # convert the clipped vertices to NDC, output [N, 3]
            landmarks3d = self.flame.seletec_3d68(verts_ndc_3d[None]) # [1, 68, 3]
            landmarks2d = landmarks3d[:,:,:2].detach().cpu().numpy() # [1, 68, 2]

            # eyes landmarks
            eyes_landmarks3d = verts_ndc_3d[self.R_EYE_INDICES + self.L_EYE_INDICES][None]  # [1, 10, 3]
            eyes_landmarks2d = eyes_landmarks3d[:,:,:2].detach().cpu().numpy()  # [1, 10, 2]

            # ears landmarks
            ears_landmarks3d = verts_ndc_3d[self.R_EAR_INDICES + self.L_EAR_INDICES][None]  # [1, 40, 3]
            ears_landmarks2d = ears_landmarks3d[:,:,:2].detach().cpu().numpy()  # [1, 40, 2]

            # render textured mesh (using PyTorch3D's renderer)
            verts_transformed = verts_clip[...,:3]
            verts_transformed = verts_transformed[..., :3] / (verts_clip[...,3:4] + 1e-8) # perspective projection
            rendered_textured = self.flame_texture_render(vertices, verts_transformed, texture, light+d_light)
            rendered_textured = rendered_textured['images'].flip(2) # [1,C,H,W]
            rendered_textured = rendered_textured[:,:3,:,:] # [1,3,H,W] RGBA to RGB 

            # for visualization only
            rendered_mesh_shape_no_texture = rendered['rgba'][0,...,:3].detach().cpu().numpy()
            rendered_mesh_shape_img = (img_resized / 255. + rendered_mesh_shape_no_texture) / 2
            rendered_mesh_shape = rendered_textured.permute(0,2,3,1).detach().cpu().numpy()[0] # [H,W,3]
            #rendered_mesh_shape_img = 0.4 * (img_resized / 255.) + 0.6 * rendered_mesh_shape
            rendered_mesh_shape_img = np.array(np.clip(rendered_mesh_shape_img * 255, 0, 255), dtype=np.uint8) # uint8
            rendered_mesh_shape = np.array(np.clip(rendered_mesh_shape * 255, 0, 255), dtype=np.uint8) # uint8
            rendered_mesh_shape_img = cv2.cvtColor(rendered_mesh_shape_img, cv2.COLOR_RGB2BGR) # convert to BGR to draw landmarks

            # Draw 2D landmarks as green dots
            for coords in landmarks2d[0]:
                coords = np.clip(coords, 0, self.H-1).astype(np.uint8)
                #coords = np.clip((coords / 2 + 1) * self.H, 0, self.H).astype(np.uint8)
                cv2.circle(rendered_mesh_shape_img, (coords[0], coords[1]), radius=1, color=(0, 255, 0), thickness=-1)  # Green color, filled circle

            # Optionally draw eye landmarks as blue dots
            for coords in eyes_landmarks2d[0]:
                coords = np.clip(coords, 0, self.H-1).astype(np.uint8)
                #coords = np.clip((coords / 2 + 1) * self.H, 0, self.H).astype(np.uint8)
                cv2.circle(rendered_mesh_shape_img, (coords[0], coords[1]), radius=1, color=(255, 0, 0), thickness=-1)  # Blue color, filled circle

            # Optionally draw ear landmarks as pink dots
            for coords in ears_landmarks2d[0]:
                coords = np.clip(coords, 0, self.H-1).astype(np.uint8)
                #coords = np.clip((coords / 2 + 1) * self.H, 0, self.H).astype(np.uint8)
                cv2.circle(rendered_mesh_shape_img, (coords[0], coords[1]), radius=1, color=(255, 0, 255), thickness=-1)

            rendered_mesh_shape_img = cv2.cvtColor(rendered_mesh_shape_img, cv2.COLOR_BGR2RGB) # convert back to RGB

        ####################
        # Prepare results  #
        ####################
        ret_dict = {
            'vertices': vertices[0].detach().cpu().numpy(),  # [N, 3]
            'shape': (shape + d_shape).detach().cpu().numpy(),
            'exp': (exp + d_exp).detach().cpu().numpy(),
            'pose': optimized_pose.detach().cpu().numpy(),
            'eye_pose': eye_pose.detach().cpu().numpy(),
            'tex': (tex + d_tex).detach().cpu().numpy(),
            'light': (light + d_light).detach().cpu().numpy(),
            'cam': optimized_camera_pose.detach().cpu().numpy(), # [6]
            'img_rendered': rendered_mesh_shape_img,
            'mesh_rendered': rendered_mesh_shape,
        }
        
        return ret_dict


