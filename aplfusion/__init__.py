"""APLFusion: adaptive pseudo-labeling for multimodal medical image fusion.

    aplfusion.network   FusionNet, the dual-branch CNN that fuses an image pair
    aplfusion.fsm       FeatureSimilarityModule, the pseudo-label of Stage 2 (DINOv2)
    aplfusion.losses    BaseLoss, the base fusion loss L_base
    aplfusion.data      image pairs, colour handling and the 4-fold splits
"""
