######################################
## Utility functions for Images.     #
## Author: Peizhi Yan                #
##   Date: 02/27/2024                #
## Update: 11/19/2024                #
######################################

import numpy as np
import cv2
import PIL
import scipy


def get_foreground_mask(parsing : np.array):
    """
    Given parsing mask get the foreground region mask.
    {
     0: 'background'
     1: 'skin', 
     2: 'l_brow', 3: 'r_brow', 4: 'l_eye', 5: 'r_eye', 6: 'eye_g', 
     7: 'l_ear', 8: 'r_ear', 9: 'ear_r', 
     10: 'nose', 
     11: 'mouth', 12: 'u_lip', 13: 'l_lip', 
     14: 'neck', 15: 'neck_l', 
     16: 'cloth', 17: 'hair', 18: 'hat'
    }
    inputs:
        - parsing: [N, 512, 512] or [512, 512]   np.uint8
    returns:
        - face_mask: [N, 512, 512, 1] or [512, 512, 1]    np.float32
    """
    # Expand parsing mask dimensions to match imgs for broadcasting
    parsing_expanded = np.expand_dims(parsing, -1)
    
    # Create a mask where parsing == 0 (background), then invert it to target the foreground
    foreground_mask = parsing_expanded != 0 # remove background first
    return foreground_mask.astype(np.float32)


def get_face_mask(parsing : np.array, keep_ears : bool = False):
    """
    Given parsing mask get the face region mask.
    {
     0: 'background'
     1: 'skin', 
     2: 'l_brow', 3: 'r_brow', 4: 'l_eye', 5: 'r_eye', 6: 'eye_g', 
     7: 'l_ear', 8: 'r_ear', 9: 'ear_r', 
     10: 'nose', 
     11: 'mouth', 12: 'u_lip', 13: 'l_lip', 
     14: 'neck', 15: 'neck_l', 
     16: 'cloth', 17: 'hair', 18: 'hat'
    }
    inputs:
        - parsing: [N, 512, 512] or [512, 512]   np.uint8
    returns:
        - face_mask: [N, 512, 512, 1] or [512, 512, 1]    np.float32
    """
    # Expand parsing mask dimensions to match imgs for broadcasting
    parsing_expanded = np.expand_dims(parsing, -1)
    
    # Create a mask where parsing == 0 (background), then invert it to target the foreground
    face_mask = parsing_expanded != 0 # remove background first
    non_face_mask = parsing_expanded >= 14 # non-facial region
    face_mask = (face_mask ^ non_face_mask)
    if keep_ears == False:
        ear_mask = (parsing_expanded >= 7) & (parsing_expanded <=9) # ears
        face_mask ^ ear_mask
    
    return face_mask.astype(np.float32)


def resize_image_proportionally(img, max_length=3000):
    """
    Resize an image proportionally so that its longest dimension does not exceed max_length.

    Parameters:
    - img_path (str): The path to the image file to resize.
    - max_length (int): The maximum allowed size for the longest dimension of the image.

    Returns:
    - resized_img (np.array): The resized image array.
    """
    # Get the current dimensions of the image
    height, width = img.shape[:2]

    # Determine the scaling factor
    scaling_factor = max_length / max(height, width) if max(height, width) > max_length else 1

    # Calculate new dimensions
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_img

def read_img(img_path, resize=None):
    # returns uint8 3-channel image
    img = PIL.Image.open(img_path)
    img = np.asarray(img)
    img = np.array(img, dtype=np.uint8)
    if resize is not None:
        img = cv2.resize(img, (resize, resize))
    #img = np.array(img, dtype=np.float32) / 255. # 0 ~ 1
    return img

def min_max(img):
    # min max standardize the image to 0 ~ 1
    img_min = img.min(axis=(0, 1), keepdims=True)
    img_max = img.max(axis=(0, 1), keepdims=True)
    img_normalized = (img - img_min) / (img_max - img_min)
    return img_normalized

def uint8_img(img):
    # convert an image to uint8
    img = np.array(img*255, dtype=np.uint8)
    return np.clip(img, 0, 255)

def norm_img(img):
    # normalize image to 0 ~ 1.0
    return np.array(img, dtype=np.float32) / 255. # 0 ~ 1

def image_align(img, face_landmarks, output_size=256, transform_size=1024, 
                enable_padding=True, standard='FFHQ', padding_mode='reflect'):
    # Align function from FFHQ dataset pre-processing step
    # Modified by Peizhi
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
    # input:
    #    - img: [H,W,3]   numpy.array  uint8
    #    - face_landmarks: [68, 3]    numpy.array
    #    - standard: string 
    #          + 'FFHQ': The FFHQ standard will rotate and crop
    #          + 'tracking': The tracking standard is for our FLAME tracking 
    # return:
    #    - img: [S,S,3] aligned image     numpy.array    uint8

    if standard == 'tracking':
        scale_factor = 1.2
    else:
        scale_factor = 1.0 # FFHQ
        
    lm = np.array(face_landmarks)
    lm_chin          = lm[0  : 17, :2]  # left-right
    lm_eyebrow_left  = lm[17 : 22, :2]  # left-right
    lm_eyebrow_right = lm[22 : 27, :2]  # left-right
    lm_nose          = lm[27 : 31, :2]  # top-down
    lm_nostrils      = lm[31 : 36, :2]  # top-down
    lm_eye_left      = lm[36 : 42, :2]  # left-clockwise
    lm_eye_right     = lm[42 : 48, :2]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60, :2]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68, :2]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8) * scale_factor
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    """
    # Load in-the-wild image.
    if not os.path.isfile(src_file):
        print('\nCannot find source image. Please run "--wilds" before "--align".')
        return
    img = PIL.Image.open(src_file)
    """
    # Convert numpy image to PIL Image
    img = PIL.Image.fromarray(img)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        if padding_mode == 'reflect':
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        else:
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'constant', constant_values=255)
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
    
    return np.asarray(img)


def display_landmarks_with_cv2(image, landmarks, color=(0, 0, 255), radius=2):
    """
    Displays an image with landmarks using OpenCV.
    
    Parameters:
    - image: A numpy array representing the image.
    - landmarks: A numpy array of shape (N, 2), where N is the number of landmarks,
                 and each landmark is represented by (x, y) coordinates.
    - show: Boolean, if True, displays the image with landmarks. If False, returns the image.
    - color: Tuple, the color of the landmarks in BGR format (default is red).
    - radius: Integer, the radius of the circle to draw for each landmark.
    """
    # Make a copy of the image to draw on
    image_with_landmarks = np.copy(image)
    
    # Draw each landmark
    for (x, y, z) in landmarks:
        cv2.circle(image_with_landmarks, (int(x), int(y)), radius, color, -1)  # Filled circle
    
    return image_with_landmarks



