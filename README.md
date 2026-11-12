# APLFusion

PyTorch implementation of **APLFusion**, an adaptive pseudo-labeling framework for multimodal
medical image fusion (CT-MRI, PET-MRI and SPECT-MRI).

Medical image fusion has no ground-truth fused images to train on. APLFusion trains a small
fusion network against a pseudo-label that its Feature Similarity Module (FSM) rebuilds at every
training step from the network's current output. A frozen DINOv2 compares the fused image with
each source patch by patch, and each patch of the pseudo-label takes more of the source that the
fused image already resembles. The FSM is used for training only: at test time, fusing a pair is
one forward pass of a CNN with 28.7 K parameters.

## Method

**Fusion network** ([`aplfusion/network.py`](aplfusion/network.py)). Two encoders $E_A$ and $E_B$
(two 3×3 convolutions with ReLU, 32 channels) process the sources $I_A$ (CT, or the Y channel of
PET / SPECT) and $I_B$ (MRI) separately. Their features are added, $F_{AB} = F_A + F_B$, and a
decoder $D$ (a 3×3 convolution with ReLU, then a 3×3 convolution with a sigmoid) outputs the fused
image $I_f$.

**Feature Similarity Module** ([`aplfusion/fsm.py`](aplfusion/fsm.py)). A frozen DINOv2 ViT-S/14
embeds $I_A$, $I_B$ and $I_f$ (resized to 224×224) into a 16×16 grid of patch embeddings $e_{ij}$.
For each patch,

$$d^1_{ij} = \lVert e^F_{ij} - e^A_{ij} \rVert_2, \qquad d^2_{ij} = \lVert e^F_{ij} - e^B_{ij} \rVert_2, \qquad
w^1_{ij} = 1 - \frac{d^1_{ij}}{d^1_{ij} + d^2_{ij}}, \qquad w^2_{ij} = 1 - \frac{d^2_{ij}}{d^1_{ij} + d^2_{ij}},$$

and the weight maps, upsampled to the image size, give the pseudo-label
$I_{PL} = w^1 \odot I_A + w^2 \odot I_B$. $I_{PL}$ is a target: it is computed without gradients
from the detached $I_f$.

**Training** ([`train.py`](train.py)) has two stages:

| stage | epochs | loss |
|---|---|---|
| 1 | $1, \dots, N$ | $\mathcal{L}_{base} + \lambda \cdot \mathrm{MSE}\left(I_f, (I_A + I_B)/2\right)$ |
| 2 | $N+1, \dots, T$ | $\mathcal{L}_{base} + \lambda \cdot \mathrm{MSE}(I_f, I_{PL})$ |

$\mathcal{L}_{base}$ ([`aplfusion/losses.py`](aplfusion/losses.py)) combines a gradient, a
max-intensity and an SSIM term with weights 10, 5 and 4 (the fusion loss of
[MixFuse](https://github.com/Bitlijinfu/MixFuse)). In the paper, $N = 9$, $T = 20$ and $\lambda = 1$.

## Repository

```
aplfusion/
  network.py     FusionNet, the fusion network
  fsm.py         FeatureSimilarityModule: DINOv2 patch embeddings -> pseudo-label
  losses.py      BaseLoss, L_base
  data.py        image pairs, YCbCr conversion, splits
splits/          the 4-fold splits of the paper (file names of the training and test pairs)
train.py         training (Stage 1 and Stage 2)
test.py          fusion of a test split, or of any two folders of image pairs
evaluation/      the eight metrics of the paper (MATLAB / GNU Octave) and a Python driver
data/            (empty) the image pairs go here, see Data
checkpoints/     (empty) the trained models of the paper go here, see Pretrained models
runs/            (empty) checkpoints and training logs written by train.py
results/         (empty) fused images and scores written by test.py and evaluation/evaluate.py
```

## Installation

```bash
conda create -n aplfusion python=3.11 -y
conda activate aplfusion
pip install -r requirements.txt
```

Tested with PyTorch 2.6.0 and CUDA 12.4 on an NVIDIA A100. DINOv2 is loaded with `torch.hub`
from a pinned commit of [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2): its
code and the ViT-S/14 weights (84 MB) are downloaded to `~/.cache/torch/hub` on the first training
run, and no network access is needed afterwards. Testing does not use DINOv2. The evaluation needs
GNU Octave with the image package,
or MATLAB (see [`evaluation/`](evaluation/)).

## Data

The experiments use CT-MRI, PET-MRI and SPECT-MRI images of the
[Harvard Whole Brain Atlas](https://www.med.harvard.edu/AANLIB/): 187, 288 and 370 registered
pairs of 256×256 images. PET and SPECT are colour images; CT and MRI are grey.

**Download.** The 845 pairs of the paper, already in the layout below:
[`APLFusion_data.zip`](https://drive.google.com/file/d/1rgL2uj34P1Kt51joLEjkotuyWQEzvDOp/view) (Google Drive, 34 MB).
Unzip it in the root of this repository, so that the images end up in `data/`. From the command
line:

```bash
pip install gdown
gdown 1rgL2uj34P1Kt51joLEjkotuyWQEzvDOp   # downloads APLFusion_data.zip
unzip APLFusion_data.zip                  # creates data/CT-MRI, data/PET-MRI and data/SPECT-MRI
```

The scripts read the images from `data/` (`--data-root` to use another folder), in this layout,
where the two images of a pair have the same file name:

```
data/
├── CT-MRI/
│   ├── CT/      2004.bmp  2005.bmp  ...
│   └── MRI/     2004.bmp  2005.bmp  ...
├── PET-MRI/
│   ├── PET/
│   └── MRI/
└── SPECT-MRI/
    ├── SPECT/
    └── MRI/
```

`splits/<PAIR>.json` lists the training and test pairs of each of the 4 folds by file name. Each
fold tests on 25 % of the pairs, and every pair is a test pair in exactly one fold (test pairs per
fold: CT-MRI 47 / 47 / 47 / 46, PET-MRI 72, SPECT-MRI 93 / 93 / 92 / 92). The split is per slice
and stratified by case: a file name is a case id followed by a 3-digit slice number (`16003` is
slice 3 of case 16; the 3-digit names form one more group), and the slices of each case are dealt
to the four folds in turn (seed 42).

Colour images are fused on their luminance: the Y channel of PET / SPECT (OpenCV's full-range
BT.601 YCbCr) is fused with the MRI, and the fused Y is put back together with the source's Cb and
Cr. The images are used whole, at 256×256, without augmentation.

## Pretrained models

The 12 models of the paper (3 tasks × 4 folds, after epoch 20), which give the table of
[Results](#results):
[`APLFusion_checkpoints.zip`](https://drive.google.com/file/d/1PpK4FcBqW0MYnV6PP-2eDnDXsJaOpqh8/view) (Google Drive,
1.3 MB). Unzip it in the root of this repository:

```bash
gdown 1PpK4FcBqW0MYnV6PP-2eDnDXsJaOpqh8   # downloads APLFusion_checkpoints.zip
unzip APLFusion_checkpoints.zip           # creates checkpoints/<PAIR>/fold1.pth ... fold4.pth
```

`checkpoints/<PAIR>/fold<k>.pth` was trained on the training pairs of fold k, so it is scored on
the test pairs of fold k only. To reproduce the table without training, fuse the test pairs of
each fold with the model of that fold (the four test splits of a task cover every pair once), then
score them:

```bash
for pair in CT-MRI PET-MRI SPECT-MRI; do
  for fold in 1 2 3 4; do
    python test.py --checkpoint checkpoints/$pair/fold$fold.pth --pair $pair --fold $fold \
        --out results/$pair/all
  done
  # ${pair%-MRI} is CT, PET or SPECT
  python evaluation/evaluate.py --func-dir data/$pair/${pair%-MRI} --mri-dir data/$pair/MRI \
      --fused-dir results/$pair/all/gray
done
```

The fused images of a task go to `results/<PAIR>/all/gray/` (and `color/` for PET / SPECT), the
scores of every image to `results/<PAIR>/all/scores.csv`, and `evaluate.py` prints the means over
all pairs of the task: one row of the table. On a GPU, the fused images are identical to the
paper's. On a CPU, floating-point rounding changes about 0.4 % of the pixels by one grey level.

## Training

```bash
python train.py --pair CT-MRI --fold 1 --data-root data
```

trains one model with the paper's settings and writes `runs/CT-MRI/fold1/checkpoint.pth`, with
`config.json` and `train_log.csv` (the loss terms of every epoch). The 12 models of the paper
(3 tasks × 4 folds):

```bash
for pair in CT-MRI PET-MRI SPECT-MRI; do
  for fold in 1 2 3 4; do
    python train.py --pair $pair --fold $fold
  done
done
```

| setting | paper | option |
|---|---|---|
| epochs $T$ | 20 | `--epochs` |
| Stage-1 epochs $N$ | 9 (pseudo-label from epoch 10) | `--stage1-epochs` |
| weight $\lambda$ | 1 | `--lam` |
| optimiser | Adam, lr 5e-3, default betas, no weight decay | `--lr` |
| batch size | 64 | `--batch-size` |
| seed | 0 | `--seed` |

A run takes 25–35 s on an A100, including the one-off DINOv2 embedding of the training sources
(Stage 2 then only embeds $I_f$ at each step).

## Testing

```bash
python test.py --checkpoint runs/PET-MRI/fold1/checkpoint.pth --pair PET-MRI --fold 1
```

fuses the test pairs of the fold into `results/PET-MRI/fold1/gray/` (the fused luminance) and
`results/PET-MRI/fold1/color/` (with the source's colour). `--checkpoint` takes a model trained
with `train.py` or one of the [pretrained models](#pretrained-models), e.g.
`checkpoints/PET-MRI/fold1.pth`. Any two folders of registered pairs, matched by file name:

```bash
python test.py --checkpoint runs/PET-MRI/fold1/checkpoint.pth \
    --func-dir my_data/PET --mri-dir my_data/MRI --out results/my_data
```

## Evaluation

```bash
python evaluation/evaluate.py --func-dir data/PET-MRI/PET --mri-dir data/PET-MRI/MRI \
    --fused-dir results/PET-MRI/fold1/gray
```

scores every fused image with the eight metrics of the paper (SD, $Q^{AB/F}$, FMI, $Q_S$, $Q_W$,
$Q_E$, SSIM and $Q_{CB}$) in parallel GNU Octave processes, writes
`results/PET-MRI/fold1/scores.csv` and prints the means. See
[`evaluation/README.md`](evaluation/README.md) for the metric code and its sources.

## Results

Means over the four folds (every pair scored once, as a test pair, with the model of its fold
after epoch 20). These models are the [pretrained models](#pretrained-models), which reproduce the
table without training:

| task | SD | $Q^{AB/F}$ | FMI | $Q_S$ | $Q_W$ | $Q_E$ | SSIM | $Q_{CB}$ |
|---|---|---|---|---|---|---|---|---|
| CT-MRI | 0.3242 | 0.4833 | 0.8739 | 0.7929 | 0.7641 | 0.6477 | 0.7520 | 0.6396 |
| PET-MRI | 0.3079 | 0.7603 | 0.8512 | 0.8953 | 0.9225 | 0.8957 | 0.7172 | 0.6091 |
| SPECT-MRI | 0.2215 | 0.7292 | 0.8713 | 0.8934 | 0.8988 | 0.8603 | 0.7471 | 0.6525 |

Training on a GPU is not bit-for-bit deterministic (cuDNN picks convolution algorithms by
timing, and training at lr 5e-3 amplifies small numerical differences), so a retrained model is
not identical to the paper's. Retraining the 12 models with this code on an A100 gave means within
0.006 of the table above for every task and metric, except $Q_{CB}$ on SPECT-MRI (+0.009).

## Complexity

The fusion network has 28,673 parameters and needs 3.74 GFLOPs per 256×256 pair (0.56 ms on an
A100, batch 1). The DINOv2 ViT-S/14 of the FSM (22.06 M parameters) is used during training only.


