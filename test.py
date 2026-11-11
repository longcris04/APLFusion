"""Fuse image pairs with a trained APLFusion network.

The test split of one fold (the paper's protocol):

    python test.py --checkpoint runs/PET-MRI/fold1/checkpoint.pth --pair PET-MRI --fold 1 --data-root data

Any two folders of registered pairs, matched by file name:

    python test.py --checkpoint runs/PET-MRI/fold1/checkpoint.pth \\
        --func-dir my_data/PET --mri-dir my_data/MRI --out results/my_data

Writes <out>/gray/<name>.png, the fused luminance, and for colour sources (PET / SPECT)
<out>/color/<name>.png, the fused Y with the source's own Cb / Cr. Only the fusion network
runs at test time; the FSM is used for training only.
"""

import argparse
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader

from aplfusion.data import N_FOLDS, PAIRS, FusionDataset, PairImages, to_uint8, ycbcr_to_rgb
from aplfusion.network import FusionNet


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="checkpoint.pth written by train.py")
    p.add_argument("--pair", choices=list(PAIRS), help="task of the test split to fuse")
    p.add_argument("--fold", type=int, choices=range(1, N_FOLDS + 1), help="fold of the test split")
    p.add_argument("--data-root", default="data", help="folder holding <PAIR>/<MODALITY>/ (default: data)")
    p.add_argument("--func-dir", help="instead of a split: folder of the CT / PET / SPECT images")
    p.add_argument("--mri-dir", help="instead of a split: folder of the MRI images (same file names)")
    p.add_argument("--out", default=None, help="output folder (default for a split: results/<PAIR>/fold<k>)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args(argv)
    if args.func_dir or args.mri_dir:
        if not (args.func_dir and args.mri_dir and args.out):
            p.error("--func-dir needs --mri-dir and --out")
    elif args.pair is None or args.fold is None:
        p.error("give --pair and --fold (a test split), or --func-dir and --mri-dir")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.func_dir:
        images = PairImages.from_folders(args.func_dir, args.mri_dir)
        color = images.color
        out = args.out
    else:
        images = PairImages.from_split(args.data_root, args.pair, args.fold, "test")
        color = [PAIRS[args.pair][1]] * len(images)
        out = args.out or os.path.join("results", args.pair, f"fold{args.fold}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = FusionNet()
    model.load_state_dict(blob["model"] if "model" in blob else blob)
    model.to(device).eval()

    same_size = len({y.shape for y in images.func_y}) == 1
    loader = DataLoader(FusionDataset(images), batch_size=args.batch_size if same_size else 1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    gray_dir, color_dir = os.path.join(out, "gray"), os.path.join(out, "color")
    os.makedirs(gray_dir, exist_ok=True)
    if any(color):
        os.makedirs(color_dir, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            fused = model(batch["img_a"].to(device, non_blocking=True), batch["img_b"].to(device, non_blocking=True))
            fused = torch.nan_to_num(fused.float(), nan=0.0).clamp_(0, 1).cpu().numpy()[:, 0]
            for k, i in enumerate(batch["index"].tolist()):
                y = to_uint8(fused[k])
                name = images.names[i]
                Image.fromarray(y).save(os.path.join(gray_dir, name + ".png"))
                if color[i]:
                    rgb = ycbcr_to_rgb(y, images.func_cb[i], images.func_cr[i])
                    Image.fromarray(rgb).save(os.path.join(color_dir, name + ".png"))
    print(f"fused {len(images)} pairs -> {out}")


if __name__ == "__main__":
    main()
