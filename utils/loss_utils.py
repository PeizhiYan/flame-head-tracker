#
## Copyright (C) Peizhi Yan. 2024
#

import torch


def compute_batch_pixelwise_l1_loss(gt_imgs, pred_imgs, gt_face_masks):
    """
    Compute the pixel-wise mean L1 loss between ground truth and rendered images within the face mask region for a batch.
    
    Args:
        gt_imgs (torch.Tensor): Ground truth images, shape [N, C, H, W], values in [0, 1].
        pred_imgs (torch.Tensor): Rendered images, shape [N, C, H, W], values in [0, 1].
        gt_face_masks (torch.Tensor): Face masks, shape [N, H, W, 1], values in [0, 1].

    Returns:
        torch.Tensor: Scalar tensor representing the mean L1 loss across the batch.
    """
    # Ensure face masks are boolean
    gt_face_masks = gt_face_masks.bool()  # [N, H, W, 1]

    # Broadcast face masks to match [N, 1, H, W]
    gt_face_masks = gt_face_masks[:, None, :, :]  # [N, 1, H, W]

    # Compute pixel-wise L1 loss
    l1_loss = torch.abs(gt_imgs - pred_imgs)  # [N, C, H, W]
    l1_loss = l1_loss * gt_face_masks # [N, C, H, W]

    # Return the average loss over the batch
    return l1_loss.mean()




