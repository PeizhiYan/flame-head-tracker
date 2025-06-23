#
## Copyright (C) Peizhi Yan. 2025
#

import torch
import numpy as np


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


def compute_ear_landmarks_loss(left_ear_landmarks2d, right_ear_landmarks2d, gt_ear_landmarks, yaw_angle):
    """
    Computes selective ear landmark loss based on a distance threshold.

    Args:
        left_ear_landmarks2d (torch.Tensor): [1, 20, 2]
        right_ear_landmarks2d (torch.Tensor): [1, 20, 2]
        gt_ear_landmarks (torch.Tensor): [1, 20, 2]
        yaw_angle (float): camera's yaw angle

    Returns:
        torch.Tensor: scalar loss
    """
    THRESHOLD = 0.3

    # Per-landmark L2 distances
    dist_l = torch.norm(left_ear_landmarks2d - gt_ear_landmarks, dim=2)  # [1, 20]
    dist_r = torch.norm(right_ear_landmarks2d - gt_ear_landmarks, dim=2) # [1, 20]

    # Mask: keep only distances <= THRESHOLD
    mask_l = dist_l <= THRESHOLD
    mask_r = dist_r <= THRESHOLD

    # Apply mask
    valid_l = dist_l * mask_l
    valid_r = dist_r * mask_r

    # Avoid division by zero
    if mask_l.sum() > 0:
        valid_l_mean = valid_l.sum() / (mask_l.sum() + 1e-6)
    else:
        valid_l_mean = valid_l.sum()
    if mask_r.sum() > 0:
        valid_r_mean = valid_r.sum() / (mask_r.sum() + 1e-6)
    else:
        valid_r_mean = valid_r.sum()

    # Final loss
    if yaw_angle < -0.1:
        # assume only left ear is visible
        loss = valid_l_mean
    elif yaw_angle > -0.1:
        # assume only right ear is visible
        loss = valid_r_mean
    else:
        # assume both ears are visible
        loss = valid_l_mean + valid_r_mean

    print(valid_l_mean.item(), valid_r_mean.item(), yaw_angle, loss.item())
    return loss


class EarlyStopping:
    def __init__(self, window_size=10, slope_threshold=-1e-4, flat_patience=3, verbose=False):
        """
        Early stopping based on loss curve slope.

        Args:
            window_size (int): Number of recent losses to consider for trend estimation.
            slope_threshold (float): Minimum slope (negative) to qualify as improving.
            flat_patience (int): Number of consecutive flat/slightly increasing slopes before stopping.
            verbose (bool): If True, prints status each check.
        """
        self.window_size = window_size
        self.slope_threshold = slope_threshold
        self.flat_patience = flat_patience
        self.verbose = verbose

        self.loss_history = []
        self.flat_count = 0
        self.early_stop = False
        self._last_slope = None

    def __call__(self, current_loss):
        self.loss_history.append(current_loss)

        if len(self.loss_history) < self.window_size:
            return  # Not enough data to evaluate

        # Calculate slope over the last window
        y = np.array(self.loss_history[-self.window_size:])
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        self._last_slope = slope

        if self.verbose:
            print(f"[EarlyStopping] Slope: {slope:.6f}, Flat count: {self.flat_count}")

        if slope > self.slope_threshold:
            self.flat_count += 1
            if self.flat_count >= self.flat_patience:
                self.early_stop = True
                if self.verbose:
                    print("[EarlyStopping] Triggered early stop.")
        else:
            self.flat_count = 0  # reset if improving

    def reset(self):
        """Resets the early stopping state (if reusing the object)."""
        self.loss_history.clear()
        self.flat_count = 0
        self.early_stop = False
        self._last_slope = None

    def last_slope(self):
        """Returns the most recent slope of the loss curve."""
        return self._last_slope
    

