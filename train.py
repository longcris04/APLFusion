"""Train APLFusion on one fold of one task.

    python train.py --pair CT-MRI --fold 1 --data-root data

With the fused image I_f = FusionNet(I_A, I_B), training has two stages:

    Stage 1, epochs 1 .. N:     L = L_base + lambda * MSE(I_f, (I_A + I_B) / 2)
    Stage 2, epochs N+1 .. T:   L = L_base + lambda * MSE(I_f, I_PL)

I_PL is the pseudo-label of the Feature Similarity Module, rebuilt at every step of Stage 2
from the current I_f. The defaults are the paper's settings: T = 20, N = 9, lambda = 1, Adam
with lr 5e-3, batch 64, seed 0, whole 256 x 256 images, no augmentation.

Writes to --out (default runs/<PAIR>/fold<k>):

    checkpoint.pth   the fusion network after the last epoch (read by test.py)
    config.json      the settings of the run
    train_log.csv    the mean loss terms of every epoch
"""

import argparse
import csv
import json
import os
import platform
import random
import time

import kornia
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aplfusion.data import N_FOLDS, PAIRS, FusionDataset, PairImages
from aplfusion.fsm import DINO_ARCH, DINO_COMMIT, FeatureSimilarityModule
from aplfusion.losses import BaseLoss
from aplfusion.network import FusionNet

LOG_FIELDS = ["epoch", "stage", "steps", "loss", "base", "grad", "max", "ssim", "intensity", "w_a", "seconds"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", required=True, choices=list(PAIRS))
    p.add_argument("--fold", type=int, required=True, choices=range(1, N_FOLDS + 1))
    p.add_argument("--data-root", default="data", help="folder holding <PAIR>/<MODALITY>/ (default: data)")
    p.add_argument("--out", default=None, help="output folder (default: runs/<PAIR>/fold<k>)")
    p.add_argument("--epochs", type=int, default=20, help="T, the number of epochs (default: 20)")
    p.add_argument("--stage1-epochs", type=int, default=9,
                   help="N, the epochs of Stage 1; the pseudo-label is used from epoch N+1 (default: 9)")
    p.add_argument("--lam", type=float, default=1.0, help="lambda, the weight of the intensity term (default: 1)")
    p.add_argument("--lr", type=float, default=5e-3, help="Adam learning rate (default: 5e-3)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4, help="data-loader workers")
    return p.parse_args(argv)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def embed_sources(fsm, dataset, device, batch_size=64):
    """DINOv2 embeddings of both sources of every training pair, in dataset order, on ``device``.

    They do not depend on the network, so they are computed once instead of at every step.
    """
    emb_a, emb_b = [], []
    for start in range(0, len(dataset), batch_size):
        samples = [dataset[i] for i in range(start, min(len(dataset), start + batch_size))]
        emb_a.append(fsm.embed(torch.stack([s["img_a"] for s in samples]).to(device)))
        emb_b.append(fsm.embed(torch.stack([s["img_b"] for s in samples]).to(device)))
    return torch.cat(emb_a), torch.cat(emb_b)


def train_step(model, optimizer, criterion, fsm, batch, stage, lam, emb_a, emb_b, device):
    """One optimiser step of Stage 1 or 2; returns the loss terms as floats."""
    img_a = batch["img_a"].to(device, non_blocking=True)
    img_b = batch["img_b"].to(device, non_blocking=True)
    fused = model(img_a, img_b)
    if stage == 1:     # target: the global average of the two sources
        target, w_a = (img_a + img_b) / 2.0, 0.5
    else:              # target: the pseudo-label of the FSM
        index = batch["index"].to(device)
        target, w_map = fsm(img_a, img_b, fused, emb_a[index], emb_b[index])
        w_a = w_map.mean().item()
    loss_base, terms = criterion(img_a, img_b, fused)
    loss_int = F.mse_loss(fused, target)
    loss = loss_base + lam * loss_int

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    values = {"loss": loss.item(), "base": loss_base.item(), "intensity": loss_int.item(), "w_a": w_a}
    values.update({key: term.item() for key, term in terms.items()})
    return values


def main(argv=None):
    args = parse_args(argv)
    out = args.out or os.path.join("runs", args.pair, f"fold{args.fold}")
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage1_epochs >= args.epochs:
        print(f"note: --stage1-epochs {args.stage1_epochs} >= --epochs {args.epochs}, Stage 2 never starts")

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    images = PairImages.from_split(args.data_root, args.pair, args.fold, "train")
    train_set = FusionDataset(images)

    # The network is built first, so its initial weights depend on the seed only
    # (building DINOv2 also draws from the random number generator).
    model = FusionNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = BaseLoss().to(device)
    fsm = FeatureSimilarityModule().to(device)
    emb_a, emb_b = embed_sources(fsm, train_set, device)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        drop_last=len(train_set) % args.batch_size == 1,    # never a batch of one
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        generator=generator, persistent_workers=args.num_workers > 0)

    n_params = sum(p.numel() for p in model.parameters())
    config = {
        "method": "APLFusion",
        "settings": vars(args),
        "n_train": len(train_set),
        "network": f"FusionNet, {n_params} parameters",
        "fsm": f"{DINO_ARCH} @ {DINO_COMMIT}, frozen",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "kornia": kornia.__version__,
                        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
    }
    with open(os.path.join(out, "config.json"), "w") as handle:
        json.dump(config, handle, indent=1)
    print(f"APLFusion | {args.pair} fold {args.fold} | {len(train_set)} training pairs | "
          f"{n_params} parameters | {device} | -> {out}", flush=True)

    with open(os.path.join(out, "train_log.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            stage = 1 if epoch <= args.stage1_epochs else 2
            model.train()
            sums, steps, started = {}, 0, time.time()
            for batch in loader:
                values = train_step(model, optimizer, criterion, fsm, batch, stage, args.lam, emb_a, emb_b, device)
                steps += 1
                for key, value in values.items():
                    sums[key] = sums.get(key, 0.0) + value
            means = {key: value / steps for key, value in sums.items()}
            seconds = time.time() - started
            writer.writerow({"epoch": epoch, "stage": stage, "steps": steps, "seconds": round(seconds, 2),
                             **{key: f"{value:.6g}" for key, value in means.items()}})
            handle.flush()
            print(f"epoch {epoch:2d}/{args.epochs} stage {stage} | loss {means['loss']:.4f} = L_base "
                  f"{means['base']:.4f} + {args.lam:g} x L_int {means['intensity']:.4f} | "
                  f"mean weight of I_A {means['w_a']:.3f} | {steps} steps, {seconds:.1f} s", flush=True)

    path = os.path.join(out, "checkpoint.pth")
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "args": vars(args)}, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
