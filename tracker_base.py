###########################################
## FLAME Tracker Reconstruction Base.     #
## -------------------------------------- #
## Author: Peizhi Yan                     #
## Update: 09/24/2024                     #
###########################################

## Copyright (C) Peizhi Yan. 2024

## Installed Packages
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np # version 1.23.4, higher version may cause problem

# Mediapipe  (version 0.10.15)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

## FLAME
from utils.flame_lib.FLAME import FLAME, FLAMETex

## FLAME fitting
import utils.flame_fitting.fitting_util as util
from utils.flame_fitting.renderer import Renderer

## Face parsing model
from utils.face_parsing.FaceParsingUtil import FaceParsing

## Utility
import utils.o3d_utils as o3d_utils
from utils.image_utils import read_img, min_max, uint8_img, norm_img, image_align, display_landmarks_with_cv2
from utils.graphics_utils import create_diff_world_to_view_matrix, verts_clip_to_ndc

## DECA
from utils.decalib.deca import DECA
from utils.decalib.deca_utils.config import cfg as deca_cfg

## Local
from utils.mesh_renderer import NVDiffRenderer
from utils.scene.cameras import PerspectiveCamera
from utils.mp2dlib import convert_landmarks_mediapipe_to_dlib
from utils.kalman_filter import initialize_kalman_matrix, kalman_filter_update_matrix


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
            'camera_params': 3,
            'shape_params': 100,
            'expression_params': 50,
            'pose_params': 6,
            'tex_params': 50,
            'use_face_contour': False,   # we don't use the face countour landmarks
            'cropped_size': 256,
            'batch_size': 1,
            'image_size': 224,
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

        # Mediapipe face landmark detector
        base_options = python.BaseOptions(model_asset_path=tracker_cfg['mediapipe_face_landmarker_v2_path'])
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               num_faces=1)
        self.mediapipe_detector = vision.FaceLandmarker.create_from_options(options)

        # Face parsing model
        self.face_parser = FaceParsing(model_path=tracker_cfg['face_parsing_model_path'])

        # FLAME model
        self.flame = FLAME(self.flame_cfg).to(self.device)
        self.flametex = FLAMETex(self.flame_cfg).to(self.device)

        # Eye Landmarks (mediapipe) and indices (FLAME mesh) 
        self.R_EYE_MP_LMKS = [468, 469, 470, 471, 472]      # right eye mediapipe landmarks
        self.L_EYE_MP_LMKS = [473, 474, 475, 476, 477]      # left eye mediapipe landmarks
        self.R_EYE_INDICES = [4597, 4543, 4511, 4479, 4575] # right eye FLAME mesh indices
        self.L_EYE_INDICES = [4051, 3997, 3965, 3933, 4020] # left eye FLAME mesh indices

        # FLAME render
        self.flame_render = Renderer(self.flame_cfg.cropped_size, obj_filename=tracker_cfg['template_mesh_file_path']).to(self.device)

        # Camera settings
        self.H = self.W = self.flame_cfg.cropped_size
        self.fov = 20.0  # x&y-axis FOV
        self.bg_color = (1.0,1.0,1.0) # White
        self.znear = 0.01
        self.zfar  = 100.0
        
        # Nvidia differentiable mesh render
        self.mesh_renderer = NVDiffRenderer().to(self.device)

        # Load the template FLAME triangle faces
        _, self.faces, _, _ = o3d_utils._read_obj_file(tracker_cfg['template_mesh_file_path'], uv=True)
        self.mesh_faces = torch.from_numpy(self.faces).to(self.device).detach() # [F, 3]

        # Load DECA model
        deca_cfg.model.use_tex = True
        deca_cfg.rasterizer_type = 'pytorch3d' # using 'standard' causes compiling issue
        deca_cfg.model.extract_tex = True
        self.deca = DECA(config = deca_cfg, device=self.device)

        # Kalman filter
        self.kf = initialize_kalman_matrix(m=1, n=6)  # Initialize Kalman filter for 6DoF camera pose

        print('Flame Tracker ready.')


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

        # Post-process mediapipe face blendshape scores
        blend_scores = detection_result.face_blendshapes[0]
        blend_scores = np.array(list(map(lambda l: l.score, blend_scores)), dtype=np.float32)

        # Post-process mediapipe dense face landmarks, re-scale to image space 
        lmks_dense = detection_result.face_landmarks[0] # the first detected face
        lmks_dense = np.array(list(map(lambda l: np.array([l.x, l.y]), lmks_dense)))
        lmks_dense[:, 0] = lmks_dense[:, 0] * img.shape[1]
        lmks_dense[:, 1] = lmks_dense[:, 1] * img.shape[0]

        return lmks_dense, blend_scores


    def load_image_and_run(self, img_path, realign=True, prev_ret_dict=None, kalman_filter=False):
        """
        Load image from given path, then run FLAME tracking
        input:
            -img_path: image path
            -realign: for FFHQ, use False. for in-the-wild images, use True
            -prev_ret_dict: the results dictionary from the previous frame
            -kalman_filter: only use it when track on video frames
        output:
            -ret_dict: results dictionary
        """
        img = read_img(img_path)
        return self.run(img, realign, prev_ret_dict, kalman_filter)

    
    def run(self, img, realign=True, prev_ret_dict=None, kalman_filter=False):
        """
        Run FLAME tracking on the given image
        input:
            -img: image data   numpy 
            -realign: for FFHQ, use False. for in-the-wild images, use True
            -prev_ret_dict: the results dictionary from the previous frame
            -kalman_filter: only use it when track on video frames
        output:
            -ret_dict: results dictionary
        """
        
        # run Mediapipe face detector
        lmks_dense, blend_scores = self.mediapipe_face_detection(img)
        face_landmarks = convert_landmarks_mediapipe_to_dlib(lmks_mp=lmks_dense)

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
        if realign:
            parsing_mask_aligned = parsing_mask
        else:
            parsing_mask_aligned = self.face_parser.run(img_aligned)
        
        # run facial landmark-based fitting
        ret_dict = self.run_fitting(img, deca_dict, prev_ret_dict, kalman_filter)

        # add more
        ret_dict['img'] = img
        ret_dict['img_aligned'] = img_aligned
        ret_dict['parsing'] = parsing_mask
        ret_dict['parsing_aligned'] = parsing_mask_aligned
        ret_dict['lmks_dense'] = lmks_dense
        ret_dict['lmks_68'] = face_landmarks
        ret_dict['blendshape_scores'] = blend_scores

        return ret_dict
    
    
    def run_deca(self, img):
        image = self.deca.crop_image(img).to(self.device)

        # DECA Encode
        with torch.no_grad():
            codedict = self.deca.encode(image[None])

        # Reconstruct
        opdict, visdict = self.deca.decode(codedict) #tensor
        
        deca_dict = {}
        for key in codedict:
            deca_dict[key] = codedict[key]
        for key in opdict:
            deca_dict[key] = opdict[key]            
        
        return deca_dict
    
    
    def run_fitting(self, img, deca_dict, prev_ret_dict, kalman_filter):
        ## Stage 1: rigid fitting on the camera pose (6DoF)
        ## Stage 2: fine-tune the expression parameters, the jaw pose (3DoF), and the eyes poses (6DoF)

        if prev_ret_dict is not None:
            continue_fit = True
        else:
            continue_fit = False

        # convert the parameters to numpy arrays
        params = {}
        for key in ['shape', 'tex', 'exp', 'pose', 'light']:
            params[key] = deca_dict[key].detach().cpu().numpy()

        # resize for FLAME fitting
        img_resized = cv2.resize(img, (self.flame_cfg.cropped_size, self.flame_cfg.cropped_size))

        # run Mediapipe face detector
        lmks_dense, _ = self.mediapipe_face_detection(img_resized)
        lmks_dense[:, :2] = lmks_dense[:, :2] / float(self.flame_cfg.cropped_size) * 2 - 1 # normalize landmarks
        face_landmarks = convert_landmarks_mediapipe_to_dlib(lmks_mp=lmks_dense) # 68 dlib landmarks

        # prepare target 68 landmarks
        gt_landmark = np.array(face_landmarks).astype(np.float32)
        gt_landmark = torch.from_numpy(gt_landmark[None]).float().to(self.device)

        # prepare target eyes landmarks
        gt_eyes_landmark = np.array(lmks_dense[self.R_EYE_MP_LMKS + self.L_EYE_MP_LMKS]).astype(np.float32)
        gt_eyes_landmark = torch.from_numpy(gt_eyes_landmark[None]).float().to(self.device)


        ############################################################
        ## Stage 1: rigid fitting (estimate the 6DoF camera pose)  #
        ############################################################

        # prepare 6DoF camera pose tensor
        if continue_fit == False:
            camera_pose = torch.tensor([0, 0, 0, 0, 0, 1.0], dtype=torch.float32).to(self.device) # [yaw, pitch, roll, x, y, z]
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

            # render landmarks via NV diff renderer
            rendered = self.mesh_renderer.render_from_camera(vertices, self.mesh_faces, cam) # vertices should have the shape of [1, N, 3]
            verts_clip = rendered['verts_clip'] # [1, N, 3]
            verts_ndc_3d = verts_clip_to_ndc(verts_clip, image_size=self.H, out_dim=3) # convert the clipped vertices to NDC, output [N, 3]
            landmarks3d = self.flame.seletec_3d68(verts_ndc_3d[None]) # [1, 68, 3]
            landmarks2d = landmarks3d[:,:,:2] / float(self.flame_cfg.cropped_size) * 2 - 1  # [1, 68, 2]

            # loss computation and optimization
            loss_facial = util.l2_distance(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]) * l_f
            loss_contour = util.l2_distance(landmarks2d[:, :17, :2], gt_landmark[:, :17, :2]) * l_c # contour loss
            loss = loss_facial + loss_contour
            e_opt_rigid.zero_grad()
            loss.backward()
            e_opt_rigid.step()

            # # for debugging
            # if iter % 100 == 0:
            #     rendered_mesh_shape = rendered['rgba'][0,...,:3].detach().cpu().numpy()
            #     temp = (img_resized / 255. + rendered_mesh_shape) / 2
            #     temp = np.array(np.clip(temp * 255, 0, 255), dtype=np.uint8) # uint8
            #     plt.imshow(temp)
            #     plt.title(f'iter {iter} - loss {loss.item()}')
            #     plt.show()

        # the optimized camera pose from the Stage 1
        optimized_camera_pose = camera_pose + torch.cat((d_camera_rotation, d_camera_translation))
        optimized_camera_pose = optimized_camera_pose.detach()

        if kalman_filter:
            # apply Kalman filter on camera pose
            optimized_camera_pose_np = optimized_camera_pose.cpu().numpy()[None] # [1,6]
            optimized_camera_pose_np = kalman_filter_update_matrix(self.kf, optimized_camera_pose_np)
            optimized_camera_pose = torch.from_numpy(optimized_camera_pose_np[0]).to(self.device).detach()

        # prepare camera for Stage 2
        Rt = create_diff_world_to_view_matrix(optimized_camera_pose)
        cam = PerspectiveCamera(Rt=Rt, fov=self.fov, bg=self.bg_color, 
                                image_width=self.W, image_height=self.H, znear=self.znear, zfar=self.zfar)


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
            {'params': [eye_pose], 'lr': 0.02}
        ]

        # fine optimizer
        e_opt_fine = torch.optim.Adam(
            expr_params,
            weight_decay=0.0001
        )

        # optimization loop
        for iter in range(200):

            # update learning rate
            if iter == 100:
                e_opt_fine.param_groups[0]['lr'] = 0.005    # For translation
                e_opt_fine.param_groups[1]['lr'] = 0.01     # For rotation
                e_opt_fine.param_groups[1]['lr'] = 0.01     # For rotation

            optimized_exp = exp + d_exp
            optimized_pose = torch.from_numpy(params['pose']).to(self.device).detach()
            optimized_pose[0,:3] *= 0 # we clear FLAME's head pose 
            optimized_pose[:,3:] = optimized_pose[:,3:] + d_jaw
            vertices, _, _ = self.flame(shape_params=shape, 
                                        expression_params=optimized_exp, 
                                        pose_params=optimized_pose, 
                                        eye_pose_params=eye_pose) # [1, N, 3]

            # render landmarks via NV diff renderer
            rendered = self.mesh_renderer.render_from_camera(vertices, self.mesh_faces, cam) # vertices should have the shape of [1, N, 3]
            verts_clip = rendered['verts_clip'] # [1, N, 3]
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
            optimized_exp = exp + d_exp
            optimized_pose = torch.from_numpy(params['pose']).to(self.device).detach()
            optimized_pose[0,:3] *= 0 # we clear FLAME's head pose 
            optimized_pose[:,3:] = optimized_pose[:,3:] + d_jaw # clear head pose and set jaw pose
            vertices, _, _ = self.flame(shape_params=shape, 
                                        expression_params=optimized_exp, 
                                        pose_params=optimized_pose, 
                                        eye_pose_params=eye_pose)

            # render landmarks via NV diff renderer
            rendered = self.mesh_renderer.render_from_camera(vertices, self.mesh_faces, cam) # vertices should have the shape of [1, N, 3]
            verts_clip = rendered['verts_clip'] # [1, N, 3]
            verts_ndc_3d = verts_clip_to_ndc(verts_clip, image_size=self.H, out_dim=3) # convert the clipped vertices to NDC, output [N, 3]
            landmarks3d = self.flame.seletec_3d68(verts_ndc_3d[None]) # [1, 68, 3]
            landmarks2d = landmarks3d[:,:,:2] # [1, 68, 2]
            rendered_mesh_shape = rendered['rgba'][0,...,:3].detach().cpu().numpy()
            rendered_mesh_shape = (img_resized / 255. + rendered_mesh_shape) / 2
            rendered_mesh_shape = np.array(np.clip(rendered_mesh_shape * 255, 0, 255), dtype=np.uint8) # uint8

        uv_texture = deca_dict['uv_texture_gt'].detach().cpu().permute(0,2,3,1)[0].numpy()
        uv_texture = np.array(np.clip(uv_texture,0,1) * 255, dtype=np.uint8) # uint8 format
        
        ####################
        # Prepare results  #
        ####################
        ret_dict = {
            'vertices': vertices[0].detach().cpu().numpy(),  # [N, 3]
            'shape': params['shape'],
            'exp': optimized_exp.detach().cpu().numpy(),
            'pose': optimized_pose.detach().cpu().numpy(),
            'eye_pose': eye_pose.detach().cpu().numpy(),
            'tex': params['tex'],
            'light': params['light'],
            'cam': optimized_camera_pose.detach().cpu().numpy(), # [6]
            'uv_texture': uv_texture,
            'img_rendered': rendered_mesh_shape
        }
        
        return ret_dict

