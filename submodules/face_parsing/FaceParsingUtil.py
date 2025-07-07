"""
Peizhi Yan
2024
"""

import gc
import torch
from torchvision.transforms import transforms
import numpy as np
import cv2

from submodules.face_parsing.model import BiSeNet


def class_remapping(parsing_mask):
    """
    Old Classes:
        atts = {0: 'background'
                1: 'skin', 
                2: 'l_brow', 3: 'r_brow', 4: 'l_eye', 5: 'r_eye', 6: 'eye_g', 
                7: 'l_ear', 8: 'r_ear', 9: 'ear_r', 
                10: 'nose', 
                11: 'mouth', 12: 'u_lip', 13: 'l_lip', 
                14: 'neck', 15: 'neck_l', 
                16: 'cloth', 17: 'hair', 18: 'hat'}
    New Classes (11 classes):
            0: background
            1: face skin
            2: eye brows
            3: eyes
            4: nose
            5: upper lip
            6: lower lip
            7: ears
            8: hair
            9: hat
            10: eyeglasses
            11: mouth
    """
    #new_mask = np.copy(silhouette).astype(np.int)
    new_mask = np.zeros(parsing_mask.shape).astype(np.int)
    def process(parsing_mask, old_class, new_class):
        one_mask = np.where(parsing_mask == old_class, 1, 0)
        return one_mask, new_class*one_mask
    mapping = { 1: 1,
                2: 2, 
                3: 2, 
                4: 3, 
                5: 3, 
                10: 4, 
                12: 5, 
                13: 6,
                7: 7,
                8: 7,
                17: 8,
                18: 9,
                6: 10,
                11: 11} # format  old_class: new_class
    for old_class in mapping.keys():
        one_mask, class_mask = process(parsing_mask, old_class=old_class, new_class=mapping[old_class])
        new_mask = new_mask * (1 - one_mask) + class_mask    
    return new_mask



class FaceParsing():
    
    def __init__(self, model_path='./face_parsing/79999_iter.pth'):

        # Face Parsing Model
        self.parsing_net = BiSeNet(n_classes=19).cuda().eval()
        self.parsing_net.load_state_dict(torch.load(model_path))

    def run(self, img):
        # input image should be standardized to 0 ~ 1
        
        n_classes = 19
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        with torch.no_grad():
            img = cv2.resize(img, (512,512)) # Face Parsing Net only correctly works on 512x512 !!!
            img = to_tensor(img)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = self.parsing_net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

        ## Release CUDA
        gc.collect() # colloct memory
        torch.cuda.empty_cache() # empty cuda
        
        return parsing
    
    @torch.no_grad()
    def run_batch(self, imgs):
        # imgs: list of images, each should be standardized to 0~1 (numpy arrays)
        n_classes = 19
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        imgs_tensor = []
        for img in imgs:
            img = cv2.resize(img, (512, 512))  # Ensure 512x512
            img = to_tensor(img)
            imgs_tensor.append(img)
        imgs_tensor = torch.stack(imgs_tensor, dim=0).cuda()  # Create a batch and move to GPU

        with torch.no_grad():
            outs = self.parsing_net(imgs_tensor)[0]
            # outs: [batch, n_classes, H, W]
            parsings = outs.argmax(1).cpu().numpy()  # Shape: (batch, H, W)
        
        # Release CUDA
        gc.collect()
        torch.cuda.empty_cache()
        
        return parsings  # (batch_size, 512, 512)

    def binary_mask(self, mask, keep=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]):
        # input:
        #   - mask: [H,W]  the face parsing mask  (numpy array)
        #   - keep: a list of classes to be kept
        # return:
        #   - mask: [H,W]  the binary mask of the selected classes to be kept
        #                  all the other classes are converted to background (class 0)
        binary_mask = np.isin(mask, keep).astype(int)
        return binary_mask
    
    
    