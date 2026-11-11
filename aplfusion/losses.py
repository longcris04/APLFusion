"""The base fusion loss L_base, to which training adds the intensity term.

    L_base = 10 L_grad + 5 L_max + 4 L_ssim

    L_grad = L1( |grad I_f|,  max(|grad I_A|, |grad I_B|) )       |grad x| = |Sobel_x x| + |Sobel_y x|
    L_max  = L1( I_f,  max(I_A, I_B) )
    L_ssim = ( (1 - SSIM(I_A, I_f)) / 2 + (1 - SSIM(I_B, I_f)) / 2 ) / 2      11 x 11 window

L1 is the mean absolute difference over pixels and batch, max the pixel-wise maximum. This is
the fusion loss of MixFuse (``fusion_loss_vif``) with its weights. The Sobel filters (not
normalised) and the SSIM loss ((1 - SSIM) / 2) are kornia's.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import SpatialGradient
from kornia.losses import SSIMLoss


class BaseLoss(nn.Module):
    """``forward(img_a, img_b, fused)`` -> (L_base, {"grad", "max", "ssim"}: its weighted terms)."""

    def __init__(self, w_grad=10, w_max=5, w_ssim=4):
        super().__init__()
        self.w_grad, self.w_max, self.w_ssim = w_grad, w_max, w_ssim
        self.sobel = SpatialGradient("sobel", normalized=False)
        self.ssim = SSIMLoss(window_size=11, reduction="mean")

    def gradient(self, x):
        s = self.sobel(x)                       # (B, C, 2, H, W)
        return torch.abs(s[:, :, 0]) + torch.abs(s[:, :, 1])

    def forward(self, img_a, img_b, fused):
        loss_max = self.w_max * F.l1_loss(fused, torch.max(img_a, img_b))
        grad_a = self.gradient(img_a)
        grad_b = self.gradient(img_b)
        grad_f = self.gradient(fused)
        loss_grad = self.w_grad * F.l1_loss(grad_f, torch.max(grad_a, grad_b))
        loss_ssim = self.w_ssim * (0.5 * self.ssim(img_a, fused) + 0.5 * self.ssim(img_b, fused))
        total = loss_grad + loss_max + loss_ssim
        return total, {"grad": loss_grad, "max": loss_max, "ssim": loss_ssim}
