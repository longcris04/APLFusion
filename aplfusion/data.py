"""Image pairs, colour handling and the 4-fold splits of the paper.

Data layout (``--data-root``, ``data`` by default)::

    <data_root>/CT-MRI/CT/<name>.bmp          <data_root>/CT-MRI/MRI/<name>.bmp
    <data_root>/PET-MRI/PET/<name>.bmp        <data_root>/PET-MRI/MRI/<name>.bmp
    <data_root>/SPECT-MRI/SPECT/<name>.bmp    <data_root>/SPECT-MRI/MRI/<name>.bmp

The two images of a pair share their file stem (any common image format). ``splits/<PAIR>.json``
lists the stems of the training and test pairs of each fold. Images are used at their native
size (256 x 256 in the paper), with no crop, resize or augmentation.

Colour: PET and SPECT are colour images. APLFusion fuses their luminance Y with the MRI; the
colour result is the fused Y put back together with the functional image's own Cb / Cr. The
conversion is OpenCV's full-range BT.601 YCrCb on 8-bit images:

    Y = 0.299 R + 0.587 G + 0.114 B,    Cr = 0.713 (R - Y) + 128,    Cb = 0.564 (B - Y) + 128

CT and MRI are grey; their Y is the grey level.
"""

import json
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

#: task -> (folder of the source paired with MRI, whether that source is a colour image)
PAIRS = OrderedDict([
    ("CT-MRI", ("CT", False)),
    ("PET-MRI", ("PET", True)),
    ("SPECT-MRI", ("SPECT", True)),
])
MRI = "MRI"
N_FOLDS = 4
IMG_EXT = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
SPLIT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "splits")


def rgb_to_ycbcr(rgb):
    """uint8 (H,W,3) RGB -> uint8 Y, Cb, Cr, each (H,W)."""
    ycrcb = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2YCrCb)
    return ycrcb[..., 0], ycrcb[..., 2], ycrcb[..., 1]


def ycbcr_to_rgb(y, cb, cr):
    """uint8 Y, Cb, Cr (H,W) -> uint8 (H,W,3) RGB."""
    ycrcb = np.stack([y, cr, cb], axis=-1).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def to_uint8(x):
    """Float image in [0, 1] -> uint8, rounded to the nearest grey level and clipped."""
    return np.clip(np.rint(np.asarray(x, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)


def read_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def list_images(folder):
    """{stem: path} of the images in ``folder``."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"no such folder: {folder}")
    return {os.path.splitext(name)[0]: os.path.join(folder, name) for name in sorted(os.listdir(folder))
            if name.lower().endswith(IMG_EXT)}


def split_stems(pair, fold, split, split_dir=SPLIT_DIR):
    """File stems of the ``train`` or ``test`` pairs of ``fold`` (1-4)."""
    with open(os.path.join(split_dir, f"{pair}.json")) as handle:
        folds = json.load(handle)["folds"]
    return list(folds[str(int(fold))][split])


class PairImages:
    """Registered image pairs, decoded once into uint8 (H,W) arrays.

    ``func_y`` / ``func_cb`` / ``func_cr``: the source paired with MRI (CT, PET or SPECT) as Y, Cb,
    Cr; ``mri``: the MRI; ``color``: whether that source is a colour image; ``names``: file stems.
    """

    def __init__(self, func_paths, mri_paths, names):
        self.names = list(names)
        self.func_y, self.func_cb, self.func_cr, self.mri, self.color = [], [], [], [], []
        for func_path, mri_path, name in zip(func_paths, mri_paths, self.names):
            rgb = read_rgb(func_path)
            y, cb, cr = rgb_to_ycbcr(rgb)
            mri = rgb_to_ycbcr(read_rgb(mri_path))[0]
            if y.shape != mri.shape:
                raise ValueError(f"{name}: the two images differ in size, {y.shape} vs {mri.shape}")
            self.func_y.append(np.ascontiguousarray(y))
            self.func_cb.append(np.ascontiguousarray(cb))
            self.func_cr.append(np.ascontiguousarray(cr))
            self.mri.append(np.ascontiguousarray(mri))
            self.color.append(not (np.array_equal(rgb[..., 0], rgb[..., 1])
                                   and np.array_equal(rgb[..., 1], rgb[..., 2])))

    def __len__(self):
        return len(self.names)

    @classmethod
    def from_folders(cls, func_dir, mri_dir, stems=None):
        """The pairs ``<func_dir>/<stem>.*`` + ``<mri_dir>/<stem>.*``: ``stems``, or every stem of the folders."""
        func, mri = list_images(func_dir), list_images(mri_dir)
        if stems is None:
            unpaired = sorted(set(func) ^ set(mri))
            if unpaired:
                raise FileNotFoundError(f"{len(unpaired)} images without a partner of the same name in "
                                        f"{func_dir} / {mri_dir}, e.g. {unpaired[:5]}")
            stems = sorted(func)
        missing = [s for s in stems if s not in func or s not in mri]
        if missing:
            raise FileNotFoundError(f"{len(missing)} pairs missing from {func_dir} / {mri_dir}, e.g. {missing[:5]}")
        return cls([func[s] for s in stems], [mri[s] for s in stems], stems)

    @classmethod
    def from_split(cls, data_root, pair, fold, split):
        """The ``train`` or ``test`` pairs of ``fold`` of ``pair`` (``splits/<PAIR>.json``)."""
        modality = PAIRS[pair][0]
        return cls.from_folders(os.path.join(data_root, pair, modality), os.path.join(data_root, pair, MRI),
                                split_stems(pair, fold, split))


class FusionDataset(Dataset):
    """I_A (Y of CT / PET / SPECT) and I_B (MRI) of each pair as (1,H,W) float tensors in [0, 1]."""

    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        d = self.images
        return {
            "img_a": torch.from_numpy(d.func_y[i]).float().div_(255.0).unsqueeze(0),
            "img_b": torch.from_numpy(d.mri[i]).float().div_(255.0).unsqueeze(0),
            "index": i,
            "name": d.names[i],
        }
