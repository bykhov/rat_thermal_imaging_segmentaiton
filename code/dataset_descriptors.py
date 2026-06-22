"""
Compute the dataset descriptors reported in the data descriptor paper.

Two groups of statistics are produced directly from ``data_full/``:

  (i)  Class-composition: per-class ground-truth pixel fraction and the median
       per-class area per frame (and the net animal footprint).
  (ii) Per-class radiometric summary: median (and inter-quartile range) surface
       temperature for each class, overall and split by cohort
       (ethanol = Rat1-15, ketamine = Rat16-25), plus the per-frame
       temperature range.

Central tendency is reported as the median rather than the mean, so the figures
are robust to the skew and outliers typical of radiometric pixel distributions.
Per-class temperatures are summarized from fine-binned histograms accumulated
over every pixel, which yields an exact median (to the bin width) without having
to hold all ~5e8 pixel values in memory.

Every number is recomputed from the raw radiometric CSVs and the uint8 masks, so
the values reported in the manuscript are reproducible. Run from anywhere:

    python code/dataset_descriptors.py                 # uses ./data_full
    python code/dataset_descriptors.py /path/to/data   # explicit data root
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CLASS_NAMES = ['Background', 'Head', 'Body', 'Tail']
ETHANOL = {f'Rat{i}' for i in range(1, 16)}
KETAMINE = {f'Rat{i}' for i in range(16, 26)}

# Histogram grid for the per-class temperature medians (degrees Celsius). The
# range comfortably brackets laboratory surface temperatures; the 0.01 C bin
# width sets the resolution of the reported median/quartiles.
T_MIN, T_MAX, BIN_WIDTH = -10.0, 70.0, 0.01
BIN_EDGES = np.arange(T_MIN, T_MAX + BIN_WIDTH, BIN_WIDTH)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2.0
NBINS = BIN_CENTERS.size


def _data_dir(explicit=None):
    """Locate the dataset root, preferring an explicit path argument.

    A directory is accepted only if it actually contains RatN subdirectories, so
    an unpopulated placeholder data_full/ (README only) is skipped automatically.
    """
    here = Path(__file__).resolve().parent
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [here.parent / 'data_full', here.parent.parent / 'data_full']
    for cand in candidates:
        if cand.is_dir() and any(cand.glob('Rat*/CSV')):
            return cand
    raise SystemExit('[ERROR] could not find a populated data_full/ (no Rat*/CSV); '
                     'pass the dataset root as an argument')


def _load_frame(csv_path, mask_path):
    """Return (temperature_matrix, mask) for one frame, or None on mismatch.

    The CSV carries a two-line header and a leading row-label column; the mask is
    a single-channel uint8 PNG with class values {0,1,2,3}. Masks stored in the
    opposite orientation are transposed to match the temperature grid.
    """
    df = pd.read_csv(csv_path, skiprows=2, header=None)
    temp = df.iloc[:, 1:].to_numpy(dtype=float)
    mask = np.array(Image.open(mask_path))
    if mask.shape != temp.shape:
        if mask.T.shape == temp.shape:
            mask = mask.T
        else:
            return None
    return temp, mask


def _blank_histograms():
    # per class: an integer temperature histogram over BIN_EDGES
    return {k: np.zeros(NBINS, dtype=np.int64) for k in range(len(CLASS_NAMES))}


def _accumulate(hists, temp, mask):
    for k in range(len(CLASS_NAMES)):
        vals = temp[mask == k]
        if vals.size:
            hists[k] += np.histogram(vals, bins=BIN_EDGES)[0]


def _hist_quantile(hist, q):
    """Quantile q in [0,1] of a value distribution given as a histogram."""
    total = hist.sum()
    if total == 0:
        return float('nan')
    idx = int(np.searchsorted(np.cumsum(hist), q * total, side='left'))
    return float(BIN_CENTERS[min(idx, NBINS - 1)])


def _median_iqr(hist):
    return (_hist_quantile(hist, 0.50),
            _hist_quantile(hist, 0.25),
            _hist_quantile(hist, 0.75))


def _process_rat(rat_path):
    """Accumulate one rat's per-class temperature histograms (worker process).

    Returning the rat-level histograms (a few arrays) rather than per-frame data
    keeps the inter-process payload small.
    """
    rat = Path(rat_path)
    hist = _blank_histograms()
    counts, medians = [], []
    n_frames = n_skipped = 0
    for csv_path in sorted((rat / 'CSV').glob('Thermal_*_CSV.csv')):
        idx = csv_path.stem.split('_')[1]
        mask_path = rat / 'Mask' / f'Thermal_{idx}.png'
        if not mask_path.exists():
            n_skipped += 1
            continue
        loaded = _load_frame(csv_path, mask_path)
        if loaded is None:
            n_skipped += 1
            continue
        temp, mask = loaded
        n_frames += 1
        counts.append(np.bincount(mask.ravel(), minlength=len(CLASS_NAMES)))
        medians.append(float(np.median(temp)))
        _accumulate(hist, temp, mask)
    return rat.name, hist, counts, medians, n_frames, n_skipped


def compute(data_dir, jobs=0):
    """Scan the dataset (one process per rat) and aggregate the descriptors.

    ``jobs`` is the worker-process count; 0 picks a sensible default, 1 forces a
    serial scan (handy for debugging).
    """
    from multiprocessing import Pool, cpu_count

    rat_dirs = sorted(data_dir.glob('Rat*'), key=lambda p: int(p.name[3:]))
    paths = [str(p) for p in rat_dirs]
    workers = jobs if jobs > 0 else min(cpu_count(), len(paths)) or 1

    if workers == 1:
        results = map(_process_rat, paths)
    else:
        pool = Pool(processes=workers)
        results = pool.imap_unordered(_process_rat, paths)

    overall = _blank_histograms()
    cohort = {'ethanol': _blank_histograms(), 'ketamine': _blank_histograms()}
    frame_counts, frame_medians = [], []
    n_frames = n_skipped = 0
    try:
        for name, hist, counts, medians, nf, ns in results:
            ck = 'ethanol' if name in ETHANOL else 'ketamine'
            for k in range(len(CLASS_NAMES)):
                overall[k] += hist[k]
                cohort[ck][k] += hist[k]
            frame_counts.extend(counts)
            frame_medians.extend(medians)
            n_frames += nf
            n_skipped += ns
    finally:
        if workers != 1:
            pool.close()
            pool.join()

    return (overall, cohort, np.array(frame_counts),
            np.array(frame_medians), n_frames, n_skipped)


def _print_composition(overall, frame_counts):
    counts = np.array([int(overall[k].sum()) for k in range(len(CLASS_NAMES))])
    total = counts.sum()
    print('\n=== (i) Class composition ===')
    print(f'{"Class":12s} {"pixels":>14s} {"fraction":>9s} {"median px/frame":>16s}')
    for k, name in enumerate(CLASS_NAMES):
        med = np.median(frame_counts[:, k])
        print(f'{name:12s} {counts[k]:14d} {counts[k] / total:8.4%} {med:16.0f}')
    animal_px = int(counts[1:].sum())
    animal_med = np.median(frame_counts[:, 1:].sum(axis=1))
    print(f'{"Animal":12s} {animal_px:14d} {animal_px / total:8.4%} {animal_med:16.0f}')


def _print_temperatures(overall, cohort, frame_medians):
    print('\n=== (ii) Per-class surface temperature (deg C, median) ===')
    print(f'per-frame median T: min {frame_medians.min():.1f}  '
          f'max {frame_medians.max():.1f}  median {np.median(frame_medians):.1f}')
    print(f'{"Class":12s} {"Overall (IQR)":>22s} {"Ethanol":>10s} {"Ketamine":>10s}')
    for k, name in enumerate(CLASS_NAMES):
        med, q1, q3 = _median_iqr(overall[k])
        me = _hist_quantile(cohort['ethanol'][k], 0.50)
        mk = _hist_quantile(cohort['ketamine'][k], 0.50)
        overall_str = f'{med:5.2f} [{q1:5.2f}, {q3:5.2f}]'
        print(f'{name:12s} {overall_str:>22s} {me:10.2f} {mk:10.2f}')


def save_distribution_plot(cohort, out_path):
    """Box-and-whisker plot of the per-class temperature distribution by cohort.

    Boxes are drawn from the histogram quantiles (IQR with the median line) and
    whiskers from the 5th/95th percentiles, so the figure is reproduced from the
    same accumulators as the table without holding the raw pixels.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cohorts = [('Ethanol', 'ethanol', '#d1495b'),
               ('Ketamine', 'ketamine', '#30638e')]
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4))
    for ci, (label, key, color) in enumerate(cohorts):
        stats, positions = [], []
        for k in range(len(CLASS_NAMES)):
            h = cohort[key][k]
            stats.append(dict(
                med    = _hist_quantile(h, 0.50),
                q1     = _hist_quantile(h, 0.25),
                q3     = _hist_quantile(h, 0.75),
                whislo = _hist_quantile(h, 0.05),
                whishi = _hist_quantile(h, 0.95),
                fliers = [],
            ))
            positions.append(k + (ci - 0.5) * width * 1.05)
        bp = ax.bxp(stats, positions=positions, widths=width,
                    patch_artist=True, showfliers=False)
        for box in bp['boxes']:
            box.set(facecolor=color, alpha=0.65, edgecolor='black', linewidth=0.8)
        for med in bp['medians']:
            med.set(color='black', linewidth=1.4)
        ax.plot([], [], 's', color=color, alpha=0.65, label=label)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_xlabel('Anatomical class')
    ax.set_ylabel(r'Surface temperature ($^\circ$C)')
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.legend(title='Cohort', frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[INFO] wrote plot {out_path}')


def save_roi_size_plot(frame_counts, out_path):
    """Box-and-whisker plot of per-frame ROI area (% of frame) by class.

    Background is omitted (it would dominate the axis); the anatomical ROIs are
    drawn from the exact per-frame pixel counts, so these are true box plots.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    roi_idx = [1, 2, 3]  # Head, Body, Tail (skip Background)
    roi_names = [CLASS_NAMES[k] for k in roi_idx]
    frac = 100.0 * frame_counts / frame_counts.sum(axis=1, keepdims=True)
    data = [frac[:, k] for k in roi_idx]

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    bp = ax.boxplot(data, widths=0.55, patch_artist=True,
                    showfliers=False, whis=(5, 95))
    for box in bp['boxes']:
        box.set(facecolor='#30638e', alpha=0.65, edgecolor='black', linewidth=0.8)
    for med in bp['medians']:
        med.set(color='black', linewidth=1.4)

    ax.set_xticklabels(roi_names)
    ax.set_xlabel('Anatomical class')
    ax.set_ylabel(r'ROI area (\% of frame)' if plt.rcParams['text.usetex']
                  else 'ROI area (% of frame)')
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[INFO] wrote plot {out_path}')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('data_dir', nargs='?', default=None,
                        help='dataset root (defaults to ./data_full)')
    parser.add_argument('--plot', metavar='PDF', default=None,
                        help='also save the per-class temperature distribution plot here')
    parser.add_argument('--roi-plot', metavar='PDF', default=None,
                        help='also save the per-class ROI size (%% of frame) distribution plot here')
    parser.add_argument('--jobs', type=int, default=4,
                        help='worker processes (default 4; 0 = auto, 1 = serial)')
    args = parser.parse_args()

    data_dir = _data_dir(args.data_dir)
    print(f'[INFO] scanning {data_dir}  (jobs={args.jobs or "auto"})')
    overall, cohort, frame_counts, frame_medians, n_frames, n_skipped = compute(data_dir, args.jobs)
    print(f'[INFO] frames used = {n_frames}  skipped = {n_skipped}')
    if not n_frames:
        return 1
    _print_composition(overall, frame_counts)
    _print_temperatures(overall, cohort, frame_medians)
    if args.plot:
        save_distribution_plot(cohort, args.plot)
    if args.roi_plot:
        save_roi_size_plot(frame_counts, args.roi_plot)
    return 0


if __name__ == '__main__':
    sys.exit(main())
