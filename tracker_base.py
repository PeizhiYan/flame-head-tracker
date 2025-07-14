#
# FLAME Tracker Base
# Author: Peizhi Yan
# Copyright (C) Peizhi Yan. 2025
#

# Installed Packages
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

# FLAME
from submodules.flame_lib.FLAME import FLAME, FLAMETex

# FLAME photometric fitting utilities
import submodules.flame_fitting.fitting_util as fitting_util
from submodules.flame_fitting.renderer import Renderer

# Face parsing model
from submodules.face_parsing.FaceParsingUtil import FaceParsing

# DECA
from utils.deca_inference_utils import create_deca_model, get_flame_code_from_deca

# MICA
from utils.mica_inference_utils import create_mica_model, get_shape_code_from_mica

# Utility
from utils.mp2dlib import convert_landmarks_mediapipe_to_dlib
from utils.loss_utils import *
from utils.general_utils import check_nan_in_dict, prepare_batch_visualized_results
import utils.o3d_utils as o3d_utils
from utils.image_utils import read_img, image_align, get_face_mask
from utils.graphics_utils import fov_to_focal, build_intrinsics, batch_perspective_projection, \
                                 batch_verts_clip_to_ndc, batch_verts_ndc_to_screen
from utils.matting_utils import load_matting_model, matting_single_image


class Tracker():

    def __init__(self, tracker_cfg):
        self.VERSION = '4.0'
        flame_cfg = {
            'mediapipe_face_landmarker_v2_path': tracker_cfg['mediapipe_face_landmarker_v2_path'],
            'flame_model_path': tracker_cfg['flame_model_path'],
            'flame_lmk_embedding_path': tracker_cfg['flame_lmk_embedding_path'],
            'tex_space_path': tracker_cfg['tex_space_path'],
            'tex_type': 'BFM',
            'camera_params': 3,          # do not change it
            'shape_params': 300,
            'expression_params': 100,    # by default, we use 100 FLAME expression coefficients (should be >= 50 and <= 100)
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
        self.device = tracker_cfg['device']
        self.flame_cfg = fitting_util.dict2obj(flame_cfg)
        self.IMG_SIZE = tracker_cfg['result_img_size']
        self.NUM_SHAPE_COEFFICIENTS = flame_cfg['shape_params']     # number of shape coefficients 
        self.NUM_EXPR_COEFFICIENTS = flame_cfg['expression_params'] # number of expression coefficients
        self.NUM_TEX_COEFFICIENTS = flame_cfg['tex_params']         # number of texture coefficients
        
        self.set_landmark_detector('hybrid') # use hybrid as default face landmark detector

        self.num_expr_coeffs = flame_cfg['expression_params'] # number of expression coefficients

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
                                               output_facial_transformation_matrixes=False,
                                               num_faces=1)
        self.mediapipe_detector = vision.FaceLandmarker.create_from_options(options)

        # FAN face alignment predictor (68 landmarks)
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_HALF_D, flip_input=True, face_detector='sfd')

        # Face parsing model
        self.face_parser = FaceParsing(model_path=tracker_cfg['face_parsing_model_path'])

        # Matting model
        if 'use_matting' in tracker_cfg:
            self.use_matting = tracker_cfg['use_matting']
            self.video_matting_model = load_matting_model(device=self.device)
        else:
            self.use_matting = False

        # FLAME model and FLAME texture model
        self.flame = FLAME(self.flame_cfg).to(self.device)
        # self.flame.v_template[3931:5023, 2] -= 0.005 # move the eyeballs backward a little bit
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
        self.RENDER_SIZE = self.H = self.W = self.flame_cfg.cropped_size
        self.DEFAULT_FOV = 20.0  # default x&y-axis FOV in degrees
        self.DEFAULT_DISTANCE = 1.0 # default camera to 3D world coordinate center distance
        self.DEFAULT_FOCAL = fov_to_focal(fov = self.DEFAULT_FOV, sensor_size = self.H)
        self.update_init_fov(fov = self.DEFAULT_FOV) # initialize the camera focal length and to object distance
        self.bg_color = (1.0,1.0,1.0) # White
        self.znear = 0.01
        self.zfar  = 100.0
        if 'optimize_fov' in tracker_cfg:
            self.optimize_fov = tracker_cfg['optimize_fov']
        else:
            self.optimize_fov = False

        # FLAME render (from DECA)
        self.flame_texture_render = Renderer(self.flame_cfg.cropped_size, 
                                     obj_filename=tracker_cfg['template_mesh_file_path']).to(self.device)

        # Load the template FLAME triangle faces
        _, self.faces, self.uv_coords, _ = o3d_utils._read_obj_file(tracker_cfg['template_mesh_file_path'], uv=True)
        self.uv_coords = np.array(self.uv_coords, dtype=np.float32)
        self.mesh_faces = torch.from_numpy(self.faces).to(self.device).detach() # [F, 3]

        # Load DECA model
        self.deca = create_deca_model(self.device)

        # Load MICA model
        self.mica = create_mica_model(self.device)

        print(f'\n>>> Flame Head Tracker v{self.VERSION} ready.')


    def set_landmark_detector(self, landmark_detector='hybrid'):
        """
        Set the face landmark detector
        landmark_detector: choose one of 'mediapipe' or 'FAN' or 'hybrid'
        """
        assert landmark_detector in ['mediapipe', 'FAN', 'hybrid'], "ERROR: landmark_detector need to be one of [mediapipe, FAN, hybrid]"
        self.landmark_detector = landmark_detector
    

    def update_init_fov(self, fov : float):
        """
        Update the initial camera FOV and adjust the camera to object distance correspondingly
        """
        # Assert that the FOV is within the reasonable range
        assert 10 <= fov <= 60, f"FOV must be between 10 and 60. Provided: {fov}"
        self.fov = fov # update the fov
        self.focal = fov_to_focal(fov = fov, sensor_size = self.H) # compute new focal length
        self.distance = self.DEFAULT_DISTANCE * (self.focal / self.DEFAULT_FOCAL) # compute new camera to object distance
        # # Initialize the camera intrinsic matrix
        # self.K = build_intrinsics(batch_size=1, image_size=self.H, focal_length=self.focal, device=self.device) # K is [1, 3, 3]


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
            return face_landmarks[0][:,:2] # [68, 2]


    def detect_face_landmarks(self, img):
        """
        input:
            - img: image data  numpy  uint8
        output:
            - dictionary
            {
              lmks_dense: landmarks numpy [478, 2], the locations are in image scale
              lmks_68:    landmarks numpy [68, 2], the locations are in image scale
              lmks_eyes:  eye landmarks numpy [10, 2], the locations are in image scale
              blend_scores: facial blendshapes numpy [52]
            }
        """
        # run Mediapipe face detector
        lmks_dense, blend_scores = self.mediapipe_face_detection(img)
        if lmks_dense is None:
            # no face detected by Mediapipe
            return None
        
        # get 68 face landmarks (NOT REDUNDANT, used in both the output and re-aligning the image)
        if self.landmark_detector == 'mediapipe':
            lmks_68 = convert_landmarks_mediapipe_to_dlib(lmks_mp=lmks_dense)
        elif self.landmark_detector == 'FAN':
            # run FAN face landmarks predictor
            lmks_68 = self.fan_face_landmarks(img)
            if lmks_68 is None:
                # no face detected by FAN
                return None
        elif self.landmark_detector == 'hybrid':
            # use results from both Mediapipe and FAN to improve the alignment robustness
            # run FAN face landmarks predictor
            lmks_68_fa = self.fan_face_landmarks(img)
            if lmks_68_fa is None:
                # no face detected by FAN
                return None
            lmks_68_mp = convert_landmarks_mediapipe_to_dlib(lmks_mp=lmks_dense)
            lmks_68 = np.copy(lmks_68_mp)
            lmks_68[:4] = (lmks_68_fa[:4] + lmks_68_mp[:4]) / 2           # contour
            lmks_68[13:17] = (lmks_68_fa[13:17] + lmks_68_mp[13:17]) / 2  # contour
            lmks_68[30] = (lmks_68_fa[30] + lmks_68_mp[30]) / 2           # nose tip
            lmks_68[31:35] = (lmks_68_fa[31:35] + lmks_68_mp[31:35]) / 2  # nose base
        else:
            return None

        # get eye landmarks
        lmks_eyes = lmks_dense[self.R_EYE_MP_LMKS + self.L_EYE_MP_LMKS]

        return {
            'lmks_dense': lmks_dense.astype(np.float32),
            'lmks_68': lmks_68.astype(np.float32),
            'lmks_eyes': lmks_eyes.astype(np.float32),
            'blend_scores': blend_scores.astype(np.float32)
        }
    

    def unpack_face_landmarks_result(self, ret_dict_lmks):
        return ret_dict_lmks['lmks_dense'], ret_dict_lmks['lmks_68'], \
               ret_dict_lmks['lmks_eyes'], ret_dict_lmks['blend_scores']


    @torch.no_grad()
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


    @torch.no_grad()
    def run_reconstruction_models(self, img, lmks_68):
        """
        Run DECA and MICA to get the initial FLAME coefficients
        input:
            - img: image data   numpy  uint8 RGB-channels
            - lmks_68: landmarks numpy [68, 2], the locations are in image scale
        output:
            - recon_dict: a dictionary containing the initial FLAME coefficients
            - shape: shape coefficients [1, D_shape] from MICA
            - exp:   expression coefficients [1, D_exp] from DECA
            - head_pose: head pose [1, 3] from DECA
            - jaw_pose:  jaw pose [1, 3] from DECA
            - tex:  texture coefficients [1, D_tex] from DECA
            - light: light coefficients [1, 9, 3] from DECA
        """
        # run DECA and MICA models
        deca_dict = get_flame_code_from_deca(self.deca, img, self.device)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # convert RGB to BGR
        shape_code = get_shape_code_from_mica(self.mica, img_bgr, lmks_68, self.device) # [1, 300]
        recon_dict = {}
        # shape coefficients
        recon_dict['shape'] = shape_code[:, :self.NUM_SHAPE_COEFFICIENTS]
        # expression coefficients
        recon_dict['exp'] = np.zeros([1, self.NUM_EXPR_COEFFICIENTS], dtype=np.float32)
        exp_code = deca_dict['exp'].detach().cpu().numpy()[:,:min(50, self.NUM_EXPR_COEFFICIENTS)]
        recon_dict['exp'][:, :min(50, self.NUM_EXPR_COEFFICIENTS)] = exp_code
        # pose coefficients
        pose = deca_dict['pose'].detach().cpu().numpy()[:,:self.NUM_TEX_COEFFICIENTS] # [1, 6] head + jaw pose
        recon_dict['head_pose'] = pose[:,:3] # [1, 3] head pose
        recon_dict['jaw_pose'] =  pose[:,3:] # [1, 3] jaw pose
        # texture coefficients
        recon_dict['tex'] = np.zeros([1, self.NUM_TEX_COEFFICIENTS], dtype=np.float32)
        tex_code = deca_dict['tex'].detach().cpu().numpy()[:,:min(50, self.NUM_TEX_COEFFICIENTS)]
        recon_dict['tex'][:, :min(50, self.NUM_TEX_COEFFICIENTS)] = tex_code
        # SH light coefficients
        recon_dict['light'] = deca_dict['light'].detach().cpu().numpy() # [1, 9, 3]
        return recon_dict


    @torch.no_grad()
    def prepare_intermediate_data_from_image(self, img, realign=True):
        """ Prepare the data needed for tracking
        input:
            - img: image is a numpy array [H, W, 3] uint8 RGB-channels
            - realign: for FFHQ, use False. for in-the-wild images, use True
        output:
            - in_dict or None: intermediate data dictionary, contains the following keys:
        """

        # realign image
        lmks_dense, blend_scores = self.mediapipe_face_detection(img)
        if lmks_dense is None:
            # no face detected
            return None
        lmks_68 = convert_landmarks_mediapipe_to_dlib(lmks_mp=lmks_dense)
        img_aligned = image_align(img, face_landmarks=lmks_68, output_size=self.IMG_SIZE, 
                                    standard='tracking', padding_mode='constant')
        
        if realign:
            # if realign == True
            # update img to the aligned image, which will be used in fitting
            img = img_aligned
            
        # run face parsing
        parsing_mask = self.face_parser.run(img)
        if realign: parsing_mask_aligned = parsing_mask
        else: parsing_mask_aligned = self.face_parser.run(img_aligned)

        # detect landmarks (to output)
        ret_dict_lmks = self.detect_face_landmarks(img)
        if ret_dict_lmks is None: return None
        else: lmks_dense, lmks_68, lmks_eyes, blend_scores = self.unpack_face_landmarks_result(ret_dict_lmks)

        # run neural network FLAME reconstruction models
        recon_dict = self.run_reconstruction_models(img = np.copy(img), lmks_68 = np.copy(lmks_68))

        # resize image to match the mesh renderer's output size
        img_resized = cv2.resize(img, (self.RENDER_SIZE, self.RENDER_SIZE))

        # run landmark detector again on resized image
        ret_dict_lmks_resized = self.detect_face_landmarks(img_resized)
        if ret_dict_lmks_resized is None: return None
        else: _, lmks_68_resized, lmks_eyes_resized, _ = self.unpack_face_landmarks_result(ret_dict_lmks_resized)

        # normalize the face landmarks to -1.0 ~ 1.0 (only normalize along x and y axes)
        lmks_68_resized[:, :2] = lmks_68_resized[:, :2] / float(self.RENDER_SIZE) * 2 - 1
        lmks_eyes_resized[:, :2] = lmks_eyes_resized[:, :2] / float(self.RENDER_SIZE) * 2 - 1

        # prepare ear 20 landmarks
        if self.use_ear_landmarks: 
            ear_landmarks = self.detect_ear_landmarks(img_resized)  # [20, 2] normalized ear landmarks
            ear_landmarks = ear_landmarks[None] # [1, 20, 2]

        # prepare the input dictionary to optimizer
        in_dict = {
            'img':  np.array([img], dtype=np.uint8),                  # [1, H, W, 3]
            'parsing': parsing_mask[None],                            # [1, 512, 512]
            'img_resized':  np.array([img_resized], dtype=np.uint8),  # [1, 256, 256, 3]
            'img_aligned':  np.array([img_aligned], dtype=np.uint8),  # [1, H, W, 3]
            'parsing_aligned':  parsing_mask_aligned[None],           # [1, 512, 512]
            'shape': recon_dict['shape'],                             # [1, D_shape]
            'exp':   recon_dict['exp'],                               # [1, D_exp]
            'head_pose':  recon_dict['head_pose'],                    # [1, 3]
            'jaw_pose':  recon_dict['jaw_pose'],                      # [1, 3]
            'tex':   recon_dict['tex'],                               # [1, D_tex]
            'light': recon_dict['light'],                             # [1, 9, 3]
            'blendshape_scores': blend_scores[None],                  # [1, 52]
            'gt_landmarks':     lmks_68_resized[None],                # [1, 68, 3]
            'gt_eye_landmarks': lmks_eyes_resized[None],              # [1, 10, 3]
        }
        if self.use_ear_landmarks:
            in_dict['gt_ear_landmarks'] = ear_landmarks               # [1, 20, 3]

        return in_dict


    @torch.no_grad()
    def prepare_batch_intermediate_data_from_images(self, imgs : list, realign = True):        
        list_in_dicts = []
        batch_valid_indices = [] # indicate which images are valid
        for i, img in enumerate(imgs):
            in_dict = self.prepare_intermediate_data_from_image(img = img, realign = realign)
            if in_dict is not None:
                list_in_dicts.append(in_dict)
                batch_valid_indices.append(i)

        if len(batch_valid_indices) == 0: return None
        batch_valid_indices = np.array(batch_valid_indices)

        batch_in_dicts = {}
        keys = list_in_dicts[0].keys()
        for key in keys:
            vals = [d[key] for d in list_in_dicts]
            batch_in_dicts[key] = np.concatenate(vals, axis=0)

        return batch_in_dicts, batch_valid_indices


    def load_image_and_run(self, 
                           img_path, realign=True, 
                           photometric_fitting=False,
                           shape_code=None, texture=None):
        """
        Load image from given path, then run FLAME tracking
        input:
            -img_path: image path
            -realign: for FFHQ, use False. for in-the-wild images, use True
            -photometric_fitting: whether to use photometric fitting or landmarks only
            -shape_code: the pre-estimated global shape code
            -texture: texture map
        output:
            -ret_dict: results dictionary
        """
        img = read_img(img_path)
        if self.use_matting:
            img = matting_single_image(self.video_matting_model, img)
        return self.run(img, realign, photometric_fitting, shape_code, texture)
    

    def load_batch_images_and_run(self, 
                                  img_paths, realign=True, 
                                  photometric_fitting=False,
                                  shape_code=None, texture=None):
        imgs = []
        for img_path in img_paths:
            img = read_img(img_path)
            if img is None:
                raise ValueError(f"Cannot read image from path: {img_path}")
            if self.use_matting:
                img = matting_single_image(self.video_matting_model, img)
            imgs.append(img)
        return self.run(imgs, realign, photometric_fitting, shape_code, texture)

    
    def run(self, img, realign=True, photometric_fitting=False, shape_code=None, texture=None, 
            temporal_smoothing=False, estimate_canonical=False):
        """
        Run FLAME tracking on a given image or list of images
        input:
            -img: image data   numpy.ndarray uint8  or list of numpy.ndarray 
            -realign: for FFHQ, use False. for in-the-wild images, use True
            -photometric_fitting: whether to use photometric fitting or landmarks only
            -shape_code: (optional) the pre-estimated global shape code [1,300]
            -texture: (optional, only used in photometric video tracking) texture map [1,3,256,256]
            -temporal_smoothing: (optional, only used in video tracking), True for smoothing expression codes
            -estimate_canonical: (optional, only used in photometric video tracking), True for estimating canonical shape and texture
        output:
            -ret_dict: results dictionary
            -batch_valid_indices: (only return it when input img is a list of images)
        """
        if not isinstance(img, list):
            in_dict = self.prepare_intermediate_data_from_image(img = img, realign = realign)
            batch_processing = False
            if in_dict is None: return None
        else:
            in_dict, batch_valid_indices = \
                self.prepare_batch_intermediate_data_from_images(imgs = img, realign = realign)
            batch_processing = True
            if in_dict is None: return None, None

        if photometric_fitting:
            if texture is not None:
                in_dict['texture'] = texture # [1,3,256,256]
            face_mask_resized = []
            for i in range(len(in_dict['parsing'])):
                parsing_mask = in_dict['parsing'][i] # [512,512]
                _face_mask_ = get_face_mask(parsing=parsing_mask, 
                                        keep_mouth = False, 
                                        keep_ears = False,
                                        keep_neck = False) # [512,512]
                _face_mask_resized_ = cv2.resize(_face_mask_, (self.RENDER_SIZE, self.RENDER_SIZE)) # [256,256]
                face_mask_resized.append(_face_mask_resized_)
            in_dict['face_mask_resized'] = np.array(face_mask_resized) # [N,256,256]
            # run photometric fitting
            ret_dict = self.run_fitting_photometric(in_dict=in_dict, shape_code=shape_code, 
                                                    temporal_smoothing=temporal_smoothing, estimate_canonical=estimate_canonical)
        else:
            # run facial landmark-based fitting
            ret_dict = self.run_fitting(in_dict=in_dict, shape_code=shape_code, temporal_smoothing=temporal_smoothing)

        if ret_dict is None:
            return None

        # check for NaNs, if there is any, return None
        _, nan_status = check_nan_in_dict(ret_dict)
        if nan_status:
            return None

        # add more data
        ret_dict['img'] = in_dict['img']
        ret_dict['img_aligned'] = in_dict['img_aligned']
        ret_dict['parsing'] = in_dict['parsing']
        ret_dict['parsing_aligned'] = in_dict['parsing_aligned']
        ret_dict['lmks_68'] = in_dict['gt_landmarks']
        if self.use_ear_landmarks:
            ret_dict['lmks_ears'] = in_dict['gt_ear_landmarks']
        ret_dict['lmks_eyes'] = in_dict['gt_eye_landmarks']
        ret_dict['blendshape_scores'] = in_dict['blendshape_scores']

        if batch_processing:
            return ret_dict, batch_valid_indices
        else:
            return ret_dict
    

    def run_fitting(self, in_dict, shape_code : np.array = None, temporal_smoothing : bool = False):
        """ Landmark-Based Fitting
            - Stage 1: rigid fitting on the camera pose (6DoF) based on detected landmarks
            - Stage 2: fine-tune the parameters including shape, tex, exp, pose, eye_pose, and light
        Note:
            - shape_code (optional) is the canonical shape code [1, 300], if it is provided, it means
              we are optimizing on the monocular video frames. Otherwise, the batch can be images of
              random face identities.
        """
        batch_size = in_dict['img'].shape[0]
        if shape_code is None: canonical_shape = False
        else: canonical_shape = True

        # convert the parameters to numpy arrays
        params = {}
        for key in ['shape', 'exp', 'head_pose', 'jaw_pose', 'tex', 'light']:
            if key == 'shape' and shape_code is not None:
                # use pre-estimated global shape code
                params[key] = np.repeat(shape_code, batch_size, axis=0) # [N, 300]
            else:
                params[key] = in_dict[key] # [N, ...]

        # prepare ground truth landmarks
        gt_landmarks = torch.from_numpy(in_dict['gt_landmarks']).to(self.device).detach() # [N, 68, 2]
        if self.use_ear_landmarks:
            gt_ear_landmarks = torch.from_numpy(in_dict['gt_ear_landmarks']).to(self.device).detach() # [N, 20, 2]
        gt_eye_landmarks = torch.from_numpy(in_dict['gt_eye_landmarks']).to(self.device).detach() # [N, 10, 2]

        # prepare FLAME coefficients in pytorch tensors
        shape = torch.from_numpy(params['shape']).to(self.device).detach()           # [N, D_shape]
        exp = torch.from_numpy(params['exp']).to(self.device).detach()               # [N, D_exp]
        head_pose = torch.from_numpy(params['head_pose']).to(self.device).detach()   # [N, 3]
        jaw_pose = torch.from_numpy(params['jaw_pose']).to(self.device).detach()     # [N, 3]
        head_pose *= 0  # clear FLAME's head pose (estimate the camera pose instead)

        ############################################################
        ## Stage 1: rigid fitting (estimate the 6DoF camera pose)  #
        ############################################################

        # FLAME reconstruction from coefficients (only do it once it rigid optimization)
        with torch.no_grad():
            vertices, _, _ = self.flame(shape_params=shape, expression_params=exp, 
                                        head_pose_params=head_pose, jaw_pose_params=jaw_pose) # [N, V, 3]
            face_68_vertices = self.flame.seletec_3d68(vertices)    # [N, 68, 3]
            left_ear_vertices = vertices[:, self.L_EAR_INDICES, :]  # [N, 20, 3]
            right_ear_vertices = vertices[:, self.R_EAR_INDICES, :] # [N, 20, 3]
            concat_vertices = torch.cat([face_68_vertices, left_ear_vertices, right_ear_vertices], dim=1) # [N, 108, 3]

        # prepare 6DoF camera pose tensor and FOV tensor
        fov = np.full((batch_size,), self.fov, dtype=np.float32)  # [N]
        fov = torch.tensor(fov, dtype=torch.float32).to(self.device).detach()
        camera_pose = np.zeros([batch_size, 6], dtype=np.float32) # (yaw, pitch, roll, x, y, z)
        camera_pose[:, -1] = self.distance # set the camera to origin distance
        camera_pose = torch.tensor(camera_pose, dtype=torch.float32).to(self.device).detach()

        # prepare camera pose and fov offsets (to be optimized)
        d_camera_rotation = nn.Parameter(torch.zeros([batch_size, 3], dtype=torch.float32, device=self.device))
        d_camera_translation = nn.Parameter(torch.zeros([batch_size, 3], dtype=torch.float32, device=self.device))
        d_fov = nn.Parameter(torch.zeros([batch_size], dtype=torch.float32, device=self.device))
        camera_params = [
            {'params': [d_camera_translation], 'lr': 0.05}, 
            {'params': [d_camera_rotation], 'lr': 0.05},
        ]
        if self.optimize_fov:
            camera_params.append({'params': [d_fov], 'lr': 0.1})

        # camera pose optimizer
        e_opt_rigid = torch.optim.Adam(
            camera_params,
            weight_decay=0.00001
        )
        
        # optimization loop
        total_iterations = 1200
        for iter in range(total_iterations):

            # update learning rate
            if iter == 700:
                e_opt_rigid.param_groups[0]['lr'] = 0.005    # translation
                e_opt_rigid.param_groups[1]['lr'] = 0.005    # rotation
                if self.optimize_fov:
                    e_opt_rigid.param_groups[2]['lr'] = 0.01     # fov

            # compute camera intrinsics
            optimized_fov = torch.clamp(fov + d_fov, min=10.0, max=50.0)                    # [N]
            optimized_focal_length = fov_to_focal(fov=optimized_fov, sensor_size=self.H)    # [N]
            Ks = build_intrinsics(focal_length=optimized_focal_length, image_size=self.H)   # [N,3,3]

            # project the vertices to 2D
            optimized_camera_pose = camera_pose + torch.cat([d_camera_rotation, d_camera_translation], dim=-1) # [N, 6]
            concat_verts_clip = batch_perspective_projection(verts=concat_vertices, camera_pose=optimized_camera_pose, 
                                                             K=Ks, image_size=self.H, near=self.znear, far=self.zfar) # [N, 108, 3]
            concat_verts_ndc_3d = batch_verts_clip_to_ndc(concat_verts_clip) # output [N, 108, 3] normalized to -1.0 ~ 1.0
            concat_verts_ndc_2d = concat_verts_ndc_3d[:,:,:2]

            # face 68 landmarks loss
            landmarks2d = concat_verts_ndc_2d[:,:68,:] # [N, 68, 3] normalized to -1.0 ~ 1.0
            loss_facial = compute_l2_distance_per_sample(landmarks2d[:, 17:, :2], gt_landmarks[:, 17:, :2]).sum() * 300  # face 51 landmarks
            loss_jawline = compute_l2_distance_per_sample(landmarks2d[:, :17, :2], gt_landmarks[:, :17, :2]).sum() * 200 # jawline loss

            # ear landmarks loss
            EAR_LOSS_THRESHOLD = 0.20 # sometimes the detected ear landmarks are not accurate
            loss_ear = 0
            if self.use_ear_landmarks:
                left_ear_landmarks2d = concat_verts_ndc_2d[:,68:88,:2]    # [N, 20, 2]
                right_ear_landmarks2d = concat_verts_ndc_2d[:,88:108,:2]  # [N, 20, 2]
                
                loss_l_ear = compute_l2_distance_per_sample(left_ear_landmarks2d, gt_ear_landmarks)
                mask_l_ear = (loss_l_ear < EAR_LOSS_THRESHOLD).float()
                loss_l_ear = loss_l_ear * mask_l_ear  # values above threshold become 0

                loss_r_ear = compute_l2_distance_per_sample(right_ear_landmarks2d, gt_ear_landmarks)
                mask_r_ear = (loss_r_ear < EAR_LOSS_THRESHOLD).float()
                loss_r_ear = loss_r_ear * mask_r_ear  # values above threshold become 0
                
                abs_yaw_angles = torch.abs(optimized_camera_pose[:,0].detach()) # [N]
                loss_ear = loss_l_ear + loss_r_ear # [N]
                loss_ear = loss_ear * 150 * abs_yaw_angles
                loss_ear = loss_ear.sum()

            loss = loss_facial + loss_jawline + loss_ear
            e_opt_rigid.zero_grad()
            loss.backward()
            e_opt_rigid.step()

        # the optimized camera pose and intrinsics from the Stage 1
        optimized_camera_pose = camera_pose + torch.cat([d_camera_rotation, d_camera_translation], dim=-1)        
        optimized_camera_pose = optimized_camera_pose.detach() # [1, 6]
        optimized_fov = torch.clamp(fov + d_fov, min=10.0, max=50.0)                            # [N]
        optimized_focal_length = fov_to_focal(fov=optimized_fov, sensor_size=self.H)            # [N]
        Ks = build_intrinsics(focal_length=optimized_focal_length, image_size=self.H).detach()  # [N,3,3]

        ####################################
        ## Stage 2: coefficients fitting   #
        ####################################

        # prepare expression code offsets (to be optimized)
        d_exp = torch.zeros(params['exp'].shape)
        d_exp = nn.Parameter(d_exp.float().to(self.device))

        # prepare jaw pose offsets (to be optimized)
        d_jaw = torch.zeros([batch_size, 3])
        d_jaw = nn.Parameter(d_jaw.float().to(self.device))    

        # prepare eyes poses offsets (to be optimized)
        eye_pose = torch.zeros([batch_size,6]) # FLAME's default_eyeball_pose are zeros
        eye_pose = nn.Parameter(eye_pose.float().to(self.device))    

        fine_params = [
            {'params': [d_exp], 'lr': 0.01}, 
            {'params': [d_jaw], 'lr': 0.025},
            {'params': [eye_pose], 'lr': 0.03}
        ]

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

            optimized_shape = shape #+ d_shape
            optimized_exp = exp + d_exp
            optimized_head_pose = head_pose
            optimized_jaw_pose = jaw_pose + d_jaw
            vertices, _, _ = self.flame(shape_params=optimized_shape, 
                                        expression_params=optimized_exp, 
                                        head_pose_params=optimized_head_pose, 
                                        jaw_pose_params=optimized_jaw_pose, 
                                        eye_pose_params=eye_pose) # [1, V, 3]

            # project the vertices to 2D
            verts_clip = batch_perspective_projection(verts=vertices, camera_pose=optimized_camera_pose, 
                                                      K=Ks, image_size=self.H, near=self.znear, far=self.zfar) # [N, V, 3]
            verts_ndc_3d = batch_verts_clip_to_ndc(verts_clip) # output [N, V, 3] normalized to -1.0 ~ 1.0

            # 68 face landmarks
            landmarks3d = self.flame.seletec_3d68(verts_ndc_3d) # [N, 68, 3]
            landmarks2d = landmarks3d[:,:,:2] #/ float(self.H) * 2 - 1  # [N, 68, 2] normalized to -1.0 ~ 1.0

            # eyes landmarks
            eyes_landmarks2d = verts_ndc_3d[:,self.R_EYE_INDICES + self.L_EYE_INDICES,:2]    # [N, 10, 2]

            # loss computation and optimization
            loss_facial = compute_l2_distance_per_sample(landmarks2d[:, 17:, :2], gt_landmarks[:, 17:, :2]).sum() * 500
            loss_eyes = compute_l2_distance_per_sample(eyes_landmarks2d, gt_eye_landmarks).sum() * 500
            loss_reg_exp = torch.sum(optimized_exp**2) * 0.025 # regularization on expression coefficients
            if temporal_smoothing and batch_size >= 2:
                # when optimizing on monocular video, we smooth the expression codes temporally
                loss_reg_exp_smooth = (torch.sum((optimized_exp[1:] - optimized_exp[:-1]) ** 2) / 2) * 1e-4
            else:
                loss_reg_exp_smooth = 0
            
            loss = loss_facial + loss_eyes + loss_reg_exp + loss_reg_exp_smooth
            e_opt_fine.zero_grad()
            loss.backward()
            e_opt_fine.step()

        ##############################
        ## for displaying results    #
        ##############################
        with torch.no_grad():
            optimized_shape = shape #+ d_shape
            optimized_exp = exp + d_exp
            optimized_head_pose = head_pose
            optimized_jaw_pose = jaw_pose + d_jaw
            vertices, _, _ = self.flame(shape_params=optimized_shape, 
                                        expression_params=optimized_exp, 
                                        head_pose_params=optimized_head_pose, 
                                        jaw_pose_params=optimized_jaw_pose, 
                                        eye_pose_params=eye_pose) # [N, V, 3]

            verts_clip = batch_perspective_projection(verts=vertices, camera_pose=optimized_camera_pose, 
                                        K=Ks, image_size=self.H, near=self.znear, far=self.zfar) # [N, V, 3]
            verts_ndc_3d = batch_verts_clip_to_ndc(verts_clip)    # convert the clipped vertices to NDC, output [N, V, 3]
            verts_screen_3d = batch_verts_ndc_to_screen(verts_ndc_3d, image_size=self.H)
            landmarks_3d_screen = self.flame.seletec_3d68(verts_screen_3d).detach().cpu().numpy()   # [N, 68, 3]
            landmarks_2d_screen = landmarks_3d_screen[:,:,:2]                                       # [N, 68, 2] 
            verts_screen_2d = verts_screen_3d[:,:,:2]                                               # [N, V, 2] 
            verts_screen_2d = verts_screen_3d.detach().cpu().numpy()                                # [N, V, 2]
            eye_landmarks2d_screen = verts_screen_2d[:, self.R_EYE_INDICES + self.L_EYE_INDICES, :] # [N, 10, 2]
            ear_landmarks2d_screen = verts_screen_2d[:, self.R_EAR_INDICES + self.L_EAR_INDICES, :] # [N, 40, 2]   

            rendered_mesh_shape_img, rendered_mesh_shape = \
                  prepare_batch_visualized_results(vertices, self.faces, in_dict, verts_ndc_3d, self.RENDER_SIZE, 
                                                   landmarks_2d_screen, eye_landmarks2d_screen, ear_landmarks2d_screen)

        ####################
        # Prepare results  #
        ####################
        ret_dict = {
            'shape': optimized_shape.detach().cpu().numpy(),         # [N,300]
            'exp': optimized_exp.detach().cpu().numpy(),             # [N,100]
            'head_pose': optimized_head_pose.detach().cpu().numpy(), # [N,3]
            'jaw_pose': optimized_jaw_pose.detach().cpu().numpy(),   # [N,3]
            'eye_pose': eye_pose.detach().cpu().numpy(),             # [N,6]
            'tex': params['tex'],                                    # [N,50]
            'light': params['light'],                                # [N,9,3]
            'cam': optimized_camera_pose.detach().cpu().numpy(),     # [N,6]
            'fov': optimized_fov.detach().cpu().numpy(),             # [N]
            'K': Ks.detach().cpu().numpy(),                          # [N,3,3]
            'img_rendered': rendered_mesh_shape_img,                 # [N,256,256,3]
            'shape_rendered': rendered_mesh_shape,                   # [N,256,256,3]
        }
        
        return ret_dict


    def run_fitting_photometric(self, in_dict, shape_code : np.array = None, 
                                temporal_smoothing : bool = False, estimate_canonical : bool = False):
        """ Landmark + Photometric Fitting        
            - Stage 1: rigid fitting on the camera pose (6DoF)
            - Stage 2: fine-tune the parameters
        Note:
            - shape_code (optional) is the canonical shape code [1,300], if it is provided, it means
              we are optimizing on the monocular video frames. Otherwise, the batch can be images of
              random face identities.
        """
        batch_size = in_dict['img'].shape[0]
        optimize_shape_and_tex = 'texture' not in in_dict   # True for optimizing the canonical shape and texture
        canonical_shape = shape_code is not None            # True for using canonical shape code

        # convert the parameters to numpy arrays
        params = {}
        for key in ['shape', 'exp', 'head_pose', 'jaw_pose', 'tex', 'light']:
            if key == 'shape' and canonical_shape and not estimate_canonical:
                params[key] = np.repeat(shape_code, batch_size, axis=0)         # [N, 300] canonical shape code
            elif key == 'shape' and not canonical_shape and estimate_canonical:
                shape_code = np.mean(in_dict['shape'], axis=0, keepdims=True)   # [1, 300] average shape code
                params[key] = np.repeat(shape_code, batch_size, axis=0)         # [N, 300] canonical shape code
            elif key == 'tex' and (canonical_shape or estimate_canonical):
                tex_code = np.mean(in_dict['tex'], axis=0, keepdims=True)       # [1, 50] average texture code
                params[key] = np.repeat(tex_code, batch_size, axis=0)           # [N, 50] canonical tex code
            else:
                params[key] = in_dict[key] # [N, ...]

        # prepare ground truth face mask
        face_mask_resized = in_dict['face_mask_resized']                                    # [N,256,256]
        gt_face_mask = torch.from_numpy(face_mask_resized).detach().to(self.device)         # [N,256,256] boolean

        # prepare ground truth image
        img_resized = np.array(in_dict['img_resized'], dtype=np.float32) / 255.             # [N,256,256,3]
        gt_img = torch.from_numpy(img_resized).detach().to(self.device)                     # [N,256,256,3]
        gt_img = gt_img.permute(0,3,1,2)                                                    # [N,3,256,256]

        # prepare ground truth landmarks
        gt_landmarks = torch.from_numpy(in_dict['gt_landmarks']).to(self.device).detach()   # [N, 68, 2]
        if self.use_ear_landmarks:
            gt_ear_landmarks = torch.from_numpy(in_dict['gt_ear_landmarks']).to(self.device).detach()   # [N, 20, 2]
        gt_eye_landmarks = torch.from_numpy(in_dict['gt_eye_landmarks']).to(self.device).detach()       # [N, 10, 2]

        # prepare FLAME coefficients in pytorch tensors
        shape = torch.from_numpy(params['shape']).to(self.device).detach()                  # [N,300]
        exp = torch.from_numpy(params['exp']).to(self.device).detach()                      # [N,100]
        tex = torch.from_numpy(params['tex']).to(self.device).detach()                      # [N,50]
        light = torch.from_numpy(params['light']).to(self.device).detach()                  # [N,9,3]
        head_pose = torch.from_numpy(params['head_pose']).to(self.device).detach()          # [N,3]
        jaw_pose = torch.from_numpy(params['jaw_pose']).to(self.device).detach()            # [N,3]
        head_pose *= 0 # clear FLAME's head pose (use camera pose instead)

        ############################################################
        ## Stage 1: rigid fitting (estimate the 6DoF camera pose)  #
        ############################################################
        
        # FLAME reconstruction from coefficients (only do it once it rigid optimization)
        with torch.no_grad():
            vertices, _, _ = self.flame(shape_params=shape, expression_params=exp, 
                                        head_pose_params=head_pose, jaw_pose_params=jaw_pose) # [N, V, 3]
            face_68_vertices = self.flame.seletec_3d68(vertices)    # [N, 68, 3]
            left_ear_vertices = vertices[:, self.L_EAR_INDICES, :]  # [N, 20, 3]
            right_ear_vertices = vertices[:, self.R_EAR_INDICES, :] # [N, 20, 3]
            concat_vertices = torch.cat([face_68_vertices, left_ear_vertices, right_ear_vertices], dim=1) # [N, 108, 3]

        # prepare 6DoF camera pose tensor and FOV tensor
        fov = np.full((batch_size,), self.fov, dtype=np.float32)  # [N]
        fov = torch.tensor(fov, dtype=torch.float32).to(self.device).detach()
        camera_pose = np.zeros([batch_size, 6], dtype=np.float32) # (yaw, pitch, roll, x, y, z)
        camera_pose[:, -1] = self.distance # set the camera to origin distance
        camera_pose = torch.tensor(camera_pose, dtype=torch.float32).to(self.device).detach()

        # prepare camera pose offsets (to be optimized)
        d_camera_rotation = nn.Parameter(torch.zeros([batch_size, 3], dtype=torch.float32, device=self.device))
        d_camera_translation = nn.Parameter(torch.zeros([batch_size, 3], dtype=torch.float32, device=self.device))
        d_fov = nn.Parameter(torch.zeros([batch_size], dtype=torch.float32, device=self.device))
        camera_params = [
            {'params': [d_camera_translation], 'lr': 0.005}, 
            {'params': [d_camera_rotation], 'lr': 0.005}, 
        ]
        if self.optimize_fov:
            camera_params.append({'params': [d_fov], 'lr': 0.05})

        # camera pose optimizer
        e_opt_rigid = torch.optim.Adam(
            camera_params,
            weight_decay=0.00001
        )
        
        # optimization loop
        max_iterations = 1000
        for iter in range(max_iterations):

            # compute camera intrinsics
            optimized_fov = torch.clamp(fov + d_fov, min=10.0, max=50.0)                    # [N]
            optimized_focal_length = fov_to_focal(fov=optimized_fov, sensor_size=self.H)    # [N]
            Ks = build_intrinsics(focal_length=optimized_focal_length, image_size=self.H)   # [N,3,3]

            # project the vertices to 2D
            optimized_camera_pose = camera_pose + torch.cat([d_camera_rotation, d_camera_translation], dim=-1) # [N, 6]
            concat_verts_clip = batch_perspective_projection(verts=concat_vertices, camera_pose=optimized_camera_pose, 
                                                             K=Ks, image_size=self.H, near=self.znear, far=self.zfar) # [N, 108, 3]
            concat_verts_ndc_3d = batch_verts_clip_to_ndc(concat_verts_clip) # output [N, 108, 3] normalized to -1.0 ~ 1.0
            concat_verts_ndc_2d = concat_verts_ndc_3d[:,:,:2]

            # face 68 landmarks loss
            landmarks2d = concat_verts_ndc_2d[:,:68,:] # [N, 68, 3] normalized to -1.0 ~ 1.0
            loss_facial = compute_l2_distance_per_sample(landmarks2d[:, 17:, :2], gt_landmarks[:, 17:, :2]).sum() * 1.0  # face 51 landmarks
            loss_jawline = compute_l2_distance_per_sample(landmarks2d[:, :17, :2], gt_landmarks[:, :17, :2]).sum() * 0.5 # jawline loss

            # ear landmarks loss
            EAR_LOSS_THRESHOLD = 0.2 # sometimes the detected ear landmarks are not accurate
            loss_ear = 0
            if self.use_ear_landmarks:
                left_ear_landmarks2d = concat_verts_ndc_2d[:,68:88,:2]    # [N, 20, 2]
                right_ear_landmarks2d = concat_verts_ndc_2d[:,88:108,:2]  # [N, 20, 2]
                
                loss_l_ear = compute_l2_distance_per_sample(left_ear_landmarks2d, gt_ear_landmarks)
                mask_l_ear = (loss_l_ear < EAR_LOSS_THRESHOLD).float()
                loss_l_ear = loss_l_ear * mask_l_ear  # values above threshold become 0

                loss_r_ear = compute_l2_distance_per_sample(right_ear_landmarks2d, gt_ear_landmarks)
                mask_r_ear = (loss_r_ear < EAR_LOSS_THRESHOLD).float()
                loss_r_ear = loss_r_ear * mask_r_ear  # values above threshold become 0
                
                abs_yaw_angles = torch.abs(optimized_camera_pose[:,0].detach()) # [N]
                loss_ear = loss_l_ear + loss_r_ear # [N]
                loss_ear = loss_ear * 0.5 * abs_yaw_angles
                loss_ear = loss_ear.sum()

            # loss computation
            loss = loss_facial + loss_jawline + loss_ear

            # optimization step
            e_opt_rigid.zero_grad()
            loss.backward()
            e_opt_rigid.step()

        ############################
        ## Stage 2: fine fitting   #
        ############################
        e_opt_rigid.zero_grad() # clear gradients from stage 1

        # prepare shape code offsets (to be optimized)
        if canonical_shape:
            d_shape = torch.zeros([1, params['shape'].shape[1]]) # [1,300], will be broadcasted later
        else:
            d_shape = torch.zeros(params['shape'].shape)         # [N,300]
        d_shape = nn.Parameter(d_shape.float().to(self.device))

        # prepare expression code offsets (to be optimized)
        d_exp = torch.zeros(params['exp'].shape)
        d_exp = nn.Parameter(d_exp.float().to(self.device))

        # prepare jaw pose offsets (to be optimized)
        d_jaw = torch.zeros([batch_size, 3])
        d_jaw = nn.Parameter(d_jaw.float().to(self.device))

        # prepare neck pose offsets (to be optimized)
        d_neck = torch.zeros([batch_size, 3])
        d_neck = nn.Parameter(d_neck.float().to(self.device))
        
        # prepare eyes poses offsets (to be optimized)
        eye_pose = torch.zeros([batch_size,6]) # FLAME's default_eyeball_pose are zeros
        eye_pose = nn.Parameter(eye_pose.float().to(self.device))    

        # prepare texture code offsets and texture map offsets (to be optimized)
        if canonical_shape or estimate_canonical:
            d_tex = torch.zeros([1, params['tex'].shape[1]])    # [1,50], will be broadcasted later
            d_texture = torch.zeros([1, 3, 256, 256])           # [1,3,256,256], will be broadcasted later
        else:
            d_tex = torch.zeros(params['tex'].shape)            # [N,50]
            d_texture = torch.zeros([batch_size, 3, 256, 256])
        d_tex = nn.Parameter(d_tex.float().to(self.device))
        d_texture = nn.Parameter(d_texture.float().to(self.device))

        # prepare light code offsets (to be optimized)
        d_light = torch.zeros(params['light'].shape)
        d_light = nn.Parameter(d_light.float().to(self.device))

        finetune_params = [
            {'params': [d_exp], 'lr': 0.005}, 
            {'params': [d_jaw], 'lr': 0.005},
            {'params': [eye_pose], 'lr': 0.005},
            {'params': [d_camera_translation], 'lr': 0.0001}, 
            {'params': [d_camera_rotation], 'lr': 0.0001},
            {'params': [d_light], 'lr': 0.005},
        ]

        if optimize_shape_and_tex:
            finetune_params.append({'params': [d_shape], 'lr': 0.001})
            finetune_params.append({'params': [d_texture], 'lr': 0.001})
            finetune_params.append({'params': [d_tex], 'lr': 0.001})
        
        # if self.optimize_fov:
        #     finetune_params.append({'params': [d_fov], 'lr': 0.001})

        # fine optimizer
        e_opt_fine = torch.optim.Adam(
            finetune_params,
            weight_decay=0.0001
        )

        # initialize the texture map
        if 'texture' in in_dict:
            texture = np.repeat(in_dict['texture'], batch_size, axis=0)     # [N, 3, 256, 256]
            texture = torch.from_numpy(texture).to(self.device).detach()    # [N, 3, 256, 256]
            use_canonical_texture = True
        else:
            texture = torch.clamp(self.flametex(tex + d_tex) + d_texture, 0.0, 1.0).detach()   # [N, 3, 256, 256]
            use_canonical_texture = False

        # compute camera intrinsics
        optimized_fov = torch.clamp(fov + d_fov, min=10.0, max=50.0).detach()                    # [N]
        optimized_focal_length = fov_to_focal(fov=optimized_fov, sensor_size=self.H).detach()    # [N]
        Ks = build_intrinsics(focal_length=optimized_focal_length, image_size=self.H).detach()   # [N,3,3]

        # optimization loop
        max_iterations = 400
        for iter in range(max_iterations):

            optimized_shape = shape + d_shape
            optimized_exp = exp + d_exp
            optimized_head_pose = head_pose
            optimized_jaw_pose = jaw_pose + d_jaw
            optimized_neck_pose = d_neck
            vertices, _, _ = self.flame(shape_params=optimized_shape, 
                                        expression_params=optimized_exp, 
                                        head_pose_params=optimized_head_pose,
                                        jaw_pose_params=optimized_jaw_pose,
                                        neck_pose_params=optimized_neck_pose,
                                        eye_pose_params=eye_pose) # [1, V, 3]
            
            if use_canonical_texture == False:
                texture = torch.clamp(self.flametex(tex + d_tex) + d_texture, 0.0, 1.0)     # [N, 3, 256, 256]

            # project the vertices to 2D
            verts_clip = batch_perspective_projection(verts=vertices, camera_pose=optimized_camera_pose, 
                                                      K=Ks, image_size=self.H, near=self.znear, far=self.zfar) # [N, V, 3]
            verts_ndc_3d = batch_verts_clip_to_ndc(verts_clip) # output [N, V, 3] normalized to -1.0 ~ 1.0

            # face landmarks
            landmarks3d = self.flame.seletec_3d68(verts_ndc_3d) # [N, 68, 3]
            landmarks2d = landmarks3d[:,:,:2] #/ float(self.H) * 2 - 1  # [N, 68, 2] normalized to -1.0 ~ 1.0

            # eyes landmarks
            eyes_landmarks2d = verts_ndc_3d[:,self.R_EYE_INDICES + self.L_EYE_INDICES,:2]    # [N, 10, 2]

            # ears landmarks
            if self.use_ear_landmarks:
                left_ear_landmarks2d = verts_ndc_3d[:,self.L_EAR_INDICES,:2]   # [N, 20, 2]
                right_ear_landmarks2d = verts_ndc_3d[:,self.R_EAR_INDICES,:2]  # [N, 20, 2]

            # render textured mesh (using PyTorch3D's renderer)
            rendered_output = self.flame_texture_render(vertices, verts_ndc_3d, texture, light+d_light)
            rendered_textured = rendered_output['images'][:,:3,:,:]             # [N,3,H,W] RGBA to RGB
            photo_loss_mask = gt_face_mask

            # face 68 landmarks loss
            loss_facial = compute_l2_distance_per_sample(landmarks2d[:, 17:, :2], gt_landmarks[:, 17:, :2]).sum() * 1.0  # face 51 landmarks
            loss_jawline = compute_l2_distance_per_sample(landmarks2d[:, :17, :2], gt_landmarks[:, :17, :2]).sum() * 0.5 # jawline loss

            # ear landmarks loss
            EAR_LOSS_THRESHOLD = 0.2 # sometimes the detected ear landmarks are not accurate
            loss_ear = 0
            if self.use_ear_landmarks:
                loss_l_ear = compute_l2_distance_per_sample(left_ear_landmarks2d, gt_ear_landmarks)
                mask_l_ear = (loss_l_ear < EAR_LOSS_THRESHOLD).float()
                loss_l_ear = loss_l_ear * mask_l_ear  # values above threshold become 0

                loss_r_ear = compute_l2_distance_per_sample(right_ear_landmarks2d, gt_ear_landmarks)
                mask_r_ear = (loss_r_ear < EAR_LOSS_THRESHOLD).float()
                loss_r_ear = loss_r_ear * mask_r_ear  # values above threshold become 0
                
                abs_yaw_angles = torch.abs(optimized_camera_pose[:,0].detach()) # [N]
                loss_ear = loss_l_ear + loss_r_ear # [N]
                loss_ear = loss_ear * 0.5 * abs_yaw_angles
                loss_ear = loss_ear.sum()

            # loss computation and optimization
            loss_photo = compute_batch_pixelwise_l1_loss(gt_img, rendered_textured, photo_loss_mask) * 1.5     # photometric loss
            loss_eyes = compute_l2_distance_per_sample(eyes_landmarks2d, gt_eye_landmarks).sum() * 2
            loss_reg_shape = (torch.sum(d_shape ** 2) / 2) * 0.1 # 1e-4
            loss_reg_exp = (torch.sum(optimized_exp ** 2) / 2) * 1e-4 # 1e-3
            if temporal_smoothing and batch_size >= 2:
                # when optimizing on monocular video, we smooth the expression codes temporally
                loss_reg_exp_smooth = (torch.sum((optimized_exp[1:] - optimized_exp[:-1]) ** 2) / 2) * 1e-5
            else:
                loss_reg_exp_smooth = 0
            loss_reg = loss_reg_exp + loss_reg_exp_smooth + loss_reg_shape
            loss = loss_photo + loss_facial + loss_jawline + loss_eyes + loss_ear + loss_reg

            # optimization step
            e_opt_fine.zero_grad()
            loss.backward()
            e_opt_fine.step()

        #####################
        ## final results    #
        #####################
        with torch.no_grad():
            optimized_shape = shape + d_shape
            optimized_exp = exp + d_exp
            optimized_head_pose = head_pose
            optimized_jaw_pose = jaw_pose + d_jaw
            optimized_neck_pose = d_neck
            vertices, _, _ = self.flame(shape_params=optimized_shape, 
                                        expression_params=optimized_exp, 
                                        head_pose_params=optimized_head_pose,
                                        jaw_pose_params=optimized_jaw_pose,
                                        neck_pose_params=optimized_neck_pose,
                                        eye_pose_params=eye_pose) # [1, V, 3]

            if optimize_shape_and_tex:
                if iter < 100:
                    texture = torch.clamp(self.flametex(tex + d_tex), 0.0, 1.0) # [N, 3, 256, 256]
                else:
                    texture = torch.clamp(self.flametex(tex + d_tex) + d_texture, 0.0, 1.0) # [N, 3, 256, 256]

            # compute camera intrinsics
            optimized_fov = torch.clamp(fov + d_fov, min=10.0, max=50.0)                    # [N]
            optimized_focal_length = fov_to_focal(fov=optimized_fov, sensor_size=self.H)    # [N]
            Ks = build_intrinsics(focal_length=optimized_focal_length, image_size=self.H)   # [N,3,3]

            # project the vertices to 2D
            verts_clip = batch_perspective_projection(verts=vertices, camera_pose=optimized_camera_pose, 
                                                      K=Ks, image_size=self.H, near=self.znear, far=self.zfar) # [N, V, 3]
            verts_ndc_3d = batch_verts_clip_to_ndc(verts_clip) # output [N, V, 3] normalized to -1.0 ~ 1.0
            verts_screen_3d = batch_verts_ndc_to_screen(verts_ndc_3d, image_size=self.H)
            landmarks_3d_screen = self.flame.seletec_3d68(verts_screen_3d).detach().cpu().numpy()   # [N, 68, 3]
            landmarks_2d_screen = landmarks_3d_screen[:,:,:2]                                       # [N, 68, 2] 
            verts_screen_2d = verts_screen_3d[:,:,:2]                                               # [N, V, 2] 
            verts_screen_2d = verts_screen_3d.detach().cpu().numpy()                                # [N, V, 2]
            eye_landmarks2d_screen = verts_screen_2d[:, self.R_EYE_INDICES + self.L_EYE_INDICES, :] # [N, 10, 2]
            ear_landmarks2d_screen = verts_screen_2d[:, self.R_EAR_INDICES + self.L_EAR_INDICES, :] # [N, 40, 2]   

            rendered_mesh_shape_img, rendered_mesh_shape = \
                  prepare_batch_visualized_results(vertices, self.faces, in_dict, verts_ndc_3d, self.RENDER_SIZE, 
                                                   landmarks_2d_screen, eye_landmarks2d_screen, ear_landmarks2d_screen)

            # render textured mesh (using PyTorch3D's renderer)
            rendered_textured = self.flame_texture_render(vertices, verts_ndc_3d, texture, light+d_light)
            rendered_textured = rendered_textured['images'][:,:3,:,:] # [N,3,H,W] RGBA to RGB 
            rendered_textured = rendered_textured.permute(0,2,3,1).detach().cpu().numpy() # [N,H,W,3]
            rendered_textured = np.array(np.clip(rendered_textured * 255, 0, 255), dtype=np.uint8) # uint8

        ####################
        # Prepare results  #
        ####################
        ret_dict = {
            'shape': optimized_shape.detach().cpu().numpy(),          # [N,300]
            'exp': optimized_exp.detach().cpu().numpy(),              # [N,100]
            'head_pose': optimized_head_pose.detach().cpu().numpy(),  # [N,3]
            'jaw_pose': optimized_jaw_pose.detach().cpu().numpy(),    # [N,3]
            'neck_pose': optimized_neck_pose.detach().cpu().numpy(),  # [N,3]
            'eye_pose': eye_pose.detach().cpu().numpy(),              # [N,6]
            'tex': (tex + d_tex).detach().cpu().numpy(),              # [N,50]
            'texture': texture.detach().cpu().numpy(),                # [N,3,256,256]
            'light': (light + d_light).detach().cpu().numpy(),        # [N,9,3]
            'cam': optimized_camera_pose.detach().cpu().numpy(),      # [N,6]
            'fov': optimized_fov.detach().cpu().numpy(),              # [N]
            'K': Ks.detach().cpu().numpy(),                           # [N,3,3]
            'img_rendered': rendered_mesh_shape_img,                  # [N,256,256,3]
            'shape_rendered': rendered_mesh_shape,                    # [N,256,256,3]
            'mesh_rendered': rendered_textured,                       # [N,256,256,3]
        }
        
        return ret_dict


