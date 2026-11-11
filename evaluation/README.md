# Evaluation

The eight metrics of the paper, computed per image by [`aplfusion_metrics.m`](aplfusion_metrics.m)
(MATLAB or GNU Octave). [`evaluate.py`](evaluate.py) builds the list of images, runs the
function in parallel Octave (or MATLAB) processes, writes one CSV row per image and prints
the means.

| column | metric | code |
|---|---|---|
| `SD` | standard deviation of the fused image, on [0, 1] | in `aplfusion_metrics.m` |
| `Qabf` | $Q^{AB/F}$, gradient-based fusion metric [1] | `metrics/Qabf.m` |
| `FMI` | feature mutual information of the edge maps, 3×3 windows [2] | `metrics/fmi.m`, `fmi(A, B, F, 'edge', 3)` |
| `Qs` | $Q_S$, Piella's fusion quality index [3, 4] | `metrics/metricPeilla.m`, mode 1 |
| `Qw` | $Q_W$, weighted fusion quality index [3, 4] | `metrics/metricPeilla.m`, mode 2 |
| `Qe` | $Q_E$, edge-dependent fusion quality index [3, 4] | `metrics/metricPeilla.m`, mode 3 |
| `SSIM` | $(\mathrm{SSIM}(A, F) + \mathrm{SSIM}(B, F))/2$, single scale [5] | `metrics/MS_SSIM.m`, `metrics/ssim_index.m` |
| `Qcb` | $Q_{CB}$, Chen–Blum metric [6] | `metrics/metricChenBlum.m`, `metrics/normalize1.m` |

Per image, source A (CT, or the colour PET / SPECT image as stored) is converted with
`rgb2gray`, the MRI with `ConvertRGBtoYUV` (Y channel), and the three images are passed to
every metric as uint8, in the order (A, MRI, fused). The `Qe` of this implementation is $Q_W$
computed on the Prewitt edge images, squared.

## Running

```bash
python evaluation/evaluate.py --func-dir data/CT-MRI/CT --mri-dir data/CT-MRI/MRI \
    --fused-dir results/CT-MRI/fold1/gray --workers 8
```

Every `<name>.png` of `--fused-dir` is scored against `<name>.*` of the two source folders.
Options: `--out` (default: `scores.csv` next to `--fused-dir`), `--engine matlab`, and
`--binary` for the path of `octave-cli` / `matlab`. In MATLAB or Octave, the function can also
be called directly on a tab-separated list `name, source A, MRI, fused` per line:

```matlab
addpath('evaluation');
aplfusion_metrics('list.tsv', 'scores.csv');
```

GNU Octave needs the Octave Forge image package: `pkg install -forge image`. The paper's numbers
were computed with GNU Octave 10.3 and this code; other Octave or MATLAB versions may give
slightly different values for some metrics.

## Files

* `metrics/`: the metric implementations, unmodified copies of the files used for the paper.
  `metricPeilla.m`, `metricChenBlum.m` and `normalize1.m` come from Z. Liu's image fusion metrics
  toolbox [4] (provided "as is"); `ssim_index.m` is Z. Wang's SSIM implementation (for
  educational and research use, see its header); `fmi.m` is M. Haghighat's (see its header for
  the papers to cite); `Qabf.m` implements [1].
* `octave/`: added to the path under GNU Octave only. `freqspace.m` stands in for MATLAB's
  `freqspace`, which Octave lacks (used by `metricChenBlum`). `Qabf.m` and `fmi.m` (with
  `private/` and the lookup table `fmi_edge3_table.mat`, rebuilt by `fmi_build_table.m`) are
  faster versions of the files in `metrics/` that give the same values bit for bit: under
  Octave the original `fmi.m` and `Qabf.m` take about 70 s per 256×256 image, these well under
  a second.

## References

1. C. S. Xydeas and V. Petrović, "Objective image fusion performance measure," *Electronics Letters*, 36(4), 2000.
2. M. B. A. Haghighat, A. Aghagolzadeh and H. Seyedarabi, "A non-reference image fusion metric based on mutual information of image features," *Computers & Electrical Engineering*, 37(5), 2011.
3. G. Piella and H. Heijmans, "A new quality metric for image fusion," *ICIP*, 2003.
4. Z. Liu, E. Blasch, Z. Xue, J. Zhao, R. Laganière and W. Wu, "Objective assessment of multiresolution image fusion algorithms for context enhancement in night vision: a comparative study," *IEEE TPAMI*, 34(1), 2012.
5. Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," *IEEE TIP*, 13(4), 2004.
6. Y. Chen and R. S. Blum, "A new automated quality assessment algorithm for image fusion," *Image and Vision Computing*, 27, pp. 1421–1432, 2009.
