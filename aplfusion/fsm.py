"""Feature Similarity Module (FSM): the pseudo-label I_PL of Stage 2.

A frozen DINOv2 ViT-S/14 embeds the two sources I_A, I_B and the current fused image I_f.
Each image (one channel, in [0, 1]) is replicated to three channels and resized bilinearly
to 224 x 224, which gives a 16 x 16 grid of patch embeddings e_ij (DINOv2's normalised patch
tokens, 384-d; the input is not mean/std normalised). For every patch (i, j):

    d1_ij = ||e^F_ij - e^A_ij||_2           d2_ij = ||e^F_ij - e^B_ij||_2
    w1_ij = 1 - d1_ij / (d1_ij + d2_ij)     w2_ij = 1 - d2_ij / (d1_ij + d2_ij)

so the source that the fused image already resembles more, in DINOv2's feature space, gets
the larger weight. The weight maps are upsampled to the image size (nearest neighbour) and
mix the sources into the pseudo-label

    I_PL = w1 * I_A + w2 * I_B.

I_PL is a regression target, not part of the network: it is computed without gradients from
the detached fused image, at every training step of Stage 2. The embeddings of the sources do
not change during training, so they can be computed once (``embed``) and passed in; then only
I_f goes through DINOv2 at each step. The FSM is not used at test time.
"""

import math
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

DINO_REPO = "facebookresearch/dinov2"
# Pinned upstream commit: the last one before PEP 604 annotations (``float | None``) were
# introduced, so it also imports under Python 3.9. Same ViT code and pretrained weights.
DINO_COMMIT = "4d83ffd4f4b39df112cef7ed7833de7ebf3a202e"
DINO_ARCH = "dinov2_vits14"
DINO_INPUT = 224    # 224 / 14 = 16 patches per side


def load_dinov2(arch=DINO_ARCH):
    """Frozen DINOv2 backbone from torch.hub: downloaded on first use, then read from the hub cache."""
    warnings.filterwarnings("ignore", message="xFormers is not available")
    local = os.path.join(torch.hub.get_dir(), f"{DINO_REPO.replace('/', '_')}_{DINO_COMMIT}")
    if os.path.isdir(local):
        model = torch.hub.load(local, arch, source="local")
    else:
        model = torch.hub.load(f"{DINO_REPO}:{DINO_COMMIT}", arch, trust_repo=True, skip_validation=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


class FeatureSimilarityModule(nn.Module):
    """Builds the pseudo-label I_PL from the current fused image (see the module docstring)."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.dino = load_dinov2()
        self.eps = eps

    def train(self, mode=True):
        # a frozen feature extractor: it always stays in eval mode
        return super().train(False)

    @torch.no_grad()
    def embed(self, x):
        """(B,1,H,W) images in [0, 1] -> (B, 256, 384) DINOv2 patch embeddings."""
        x = F.interpolate(x.repeat(1, 3, 1, 1), size=(DINO_INPUT, DINO_INPUT), mode="bilinear",
                          align_corners=False)
        return self.dino.forward_features(x)["x_norm_patchtokens"]

    @torch.no_grad()
    def weights(self, e_a, e_b, e_f):
        """Patch weights (w1, w2) of I_A and I_B, each (B,1,16,16), from (B,N,D) embeddings."""
        batch = e_a.shape[0]
        d1 = torch.norm(e_f - e_a, p=2, dim=-1)    # (B, N)
        d2 = torch.norm(e_f - e_b, p=2, dim=-1)
        sum_d = d1 + d2 + self.eps
        w1 = 1.0 - d1 / sum_d
        w2 = 1.0 - d2 / sum_d
        grid = int(math.sqrt(w1.shape[1]))
        return w1.view(batch, 1, grid, grid), w2.view(batch, 1, grid, grid)

    @torch.no_grad()
    def forward(self, img_a, img_b, fused, e_a=None, e_b=None):
        """Pseudo-label I_PL (B,1,H,W) and the weight map of I_A, upsampled to (B,1,H,W).

        ``e_a``, ``e_b``: the sources' embeddings (``embed``), computed here when not given.
        """
        if e_a is None:
            e_a = self.embed(img_a)
        if e_b is None:
            e_b = self.embed(img_b)
        e_f = self.embed(fused.detach())
        w1, w2 = self.weights(e_a.to(e_f.dtype), e_b.to(e_f.dtype), e_f)
        size = img_a.shape[-2:]
        w1 = F.interpolate(w1, size=size, mode="nearest")
        w2 = F.interpolate(w2, size=size, mode="nearest")
        return w1 * img_a + w2 * img_b, w1
