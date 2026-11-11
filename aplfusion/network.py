"""The fusion network of APLFusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionNet(nn.Module):
    """Dual-branch CNN that fuses two registered single-channel images.

    Each source has its own encoder of two 3x3 convolutions with ReLU, E_A and E_B; the two
    feature maps are added, F_AB = F_A + F_B, and the decoder D (a 3x3 convolution with ReLU,
    then a 3x3 convolution with a sigmoid) maps F_AB to the fused image I_f in [0, 1]:

        E_A: conv11 (1 -> 32), conv21 (32 -> 32)     I_A: CT, or the Y channel of PET / SPECT
        E_B: conv12 (1 -> 32), conv22 (32 -> 32)     I_B: MRI
        D:   conv3 (32 -> 32), conv4 (32 -> 1)

    28,673 parameters. The layers are created in the order E_A, E_B, D, so a given seed always
    gives the same initial weights.
    """

    def __init__(self, channels=32):
        super().__init__()
        self.conv11 = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

    def forward(self, img_a, img_b):
        """(B,1,H,W) sources in [0, 1] -> (B,1,H,W) fused image in [0, 1]."""
        f_a = F.relu(self.conv11(img_a))
        f_b = F.relu(self.conv12(img_b))
        f_a = F.relu(self.conv21(f_a))    # F_A = E_A(I_A)
        f_b = F.relu(self.conv22(f_b))    # F_B = E_B(I_B)
        x = F.relu(self.conv3(f_a + f_b))
        return torch.sigmoid(self.conv4(x))
