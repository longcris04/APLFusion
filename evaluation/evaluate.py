"""Score fused images with the eight metrics of the paper, under GNU Octave (or MATLAB).

    python evaluation/evaluate.py --func-dir data/PET-MRI/PET --mri-dir data/PET-MRI/MRI \\
        --fused-dir results/PET-MRI/fold1/gray

Every fused grey image <name>.png of --fused-dir is scored against its sources <name>.* in
--func-dir (CT, or the colour PET / SPECT as stored) and --mri-dir, by aplfusion_metrics.m.
Writes one row per image to --out (default: scores.csv next to --fused-dir) and prints the
mean of every metric: SD, Qabf, FMI, Qs, Qw, Qe, SSIM, Qcb. The images are split into chunks
that run in parallel processes (--workers).
"""

import argparse
import csv
import math
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ["SD", "Qabf", "FMI", "Qs", "Qw", "Qe", "SSIM", "Qcb"]
IMG_EXT = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def list_images(folder):
    """{stem: absolute path} of the images in ``folder``."""
    return {os.path.splitext(name)[0]: os.path.abspath(os.path.join(folder, name))
            for name in sorted(os.listdir(folder)) if name.lower().endswith(IMG_EXT)}


def quoted(path):
    return "'" + path.replace("'", "''") + "'"


def score_chunk(engine, binary, rows, work):
    """Score ``rows`` (name, source A, MRI, fused) in one Octave / MATLAB process."""
    list_file, out_csv = os.path.join(work, "list.tsv"), os.path.join(work, "scores.csv")
    with open(list_file, "w") as handle:
        for row in rows:
            handle.write("\t".join(row) + "\n")
    expr = f"addpath({quoted(HERE)}); aplfusion_metrics({quoted(list_file)}, {quoted(out_csv)});"
    if engine == "octave":
        cmd = [binary, "--no-gui", "--norc", "--quiet", "--no-window-system", "--eval", expr]
    else:
        cmd = [binary, "-batch", expr]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=work)
    if proc.returncode != 0 or not os.path.exists(out_csv):
        raise RuntimeError(f"{engine} exited with code {proc.returncode}:\n{(proc.stdout + proc.stderr)[-2000:]}")
    if proc.stderr.strip():
        print(proc.stderr.strip())
    with open(out_csv, newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--func-dir", required=True, help="source A as stored: CT, or the colour PET / SPECT")
    p.add_argument("--mri-dir", required=True, help="the MRI sources")
    p.add_argument("--fused-dir", required=True, help="the fused grey images, <name>.png")
    p.add_argument("--out", default=None, help="per-image CSV (default: scores.csv next to --fused-dir)")
    p.add_argument("--engine", choices=["octave", "matlab"], default="octave")
    p.add_argument("--binary", default=None, help="the Octave / MATLAB executable (default: octave-cli / matlab)")
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = p.parse_args(argv)

    binary = args.binary or shutil.which("octave-cli" if args.engine == "octave" else "matlab")
    if not binary:
        raise SystemExit(f"{args.engine} not found: pass --binary")
    fused, func, mri = list_images(args.fused_dir), list_images(args.func_dir), list_images(args.mri_dir)
    missing = [s for s in fused if s not in func or s not in mri]
    if missing:
        raise SystemExit(f"{len(missing)} fused images have no sources, e.g. {missing[:5]}")
    if not fused:
        raise SystemExit(f"no images in {args.fused_dir}")
    rows = [(stem, func[stem], mri[stem], path) for stem, path in fused.items()]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.fused_dir)), "scores.csv")

    size = math.ceil(len(rows) / max(1, args.workers))
    chunks = [rows[i:i + size] for i in range(0, len(rows), size)]
    print(f"scoring {len(rows)} images with {len(chunks)} {args.engine} processes")
    with tempfile.TemporaryDirectory(prefix="aplfusion_eval_") as tmp, ThreadPoolExecutor(len(chunks)) as pool:
        works = [tempfile.mkdtemp(dir=tmp) for _ in chunks]
        results = list(pool.map(lambda job: score_chunk(args.engine, binary, *job), zip(chunks, works)))
    scores = [row for part in results for row in part]

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Name"] + METRICS)
        writer.writeheader()
        writer.writerows(scores)
    means = {}
    for metric in METRICS:
        values = [float(r[metric]) for r in scores if math.isfinite(float(r[metric]))]
        means[metric] = sum(values) / len(values) if values else float("nan")
    print(f"{len(scores)} images -> {out}")
    print("  ".join(f"{m} {v:.4f}" for m, v in means.items()))


if __name__ == "__main__":
    main()
