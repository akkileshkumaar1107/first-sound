"""
FIRST SOUND - Basic Image Processing of a Fetal Ultrasound Image
================================================================
Case: fetal head biometry (HC18 dataset).

What it does (all basic, textbook image processing):
  1. Noise (speckle) removal : Mean filter, Gaussian filter, Median filter
  2. Contrast enhancement    : Contrast stretching, Gamma correction,
                               Histogram equalization, CLAHE (adaptive HE)
  3. Sharpening              : Unsharp masking
  4. Final pipeline          : Median -> Contrast stretching -> Unsharp masking
  5. Quality measures        : Speckle index, SNR, CNR, Entropy, PSNR, SSIM,
                               Edge preservation index (EPI)
It saves report-ready figures (PNG, 200 dpi) and a metrics table (CSV).

HOW TO RUN (Mac terminal, inside the project folder):
    source .venv/bin/activate
    pip install matplotlib            # once
    python3 image_processing/first_sound_ip.py  path/to/010_HC.png  --ann path/to/010_HC_Annotation.png --px 0.1386

    --ann  (optional) HC18 annotation image -> used to place the measurement regions
    --px   (optional) pixel size in mm from training_set_pixel_size_and_HC.csv
    --out  (optional) output folder (default: ip_results)
Without arguments it runs on a built-in SYNTHETIC test image, only to check that it works.
"""
import argparse
import csv
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FOREST, SAND, CLAY, SAGE = '#1f302b', '#e8dcc4', '#a5603a', '#7f9c86'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
                     'axes.titleweight': 'bold', 'figure.facecolor': 'white'})


# ----------------------------------------------------------------- loading
def load_gray(path):
    """Read any image as 8-bit grayscale, returned as float in [0, 1]."""
    return np.asarray(Image.open(path).convert('L'), dtype=np.float64) / 255.0


def synthetic_test_image(h=540, w=800, seed=0):
    """SYNTHETIC stand-in (only for testing the script): speckled background,
    dark fluid, bright elliptical skull, mid-grey brain, multiplicative speckle."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / 210) ** 2 + ((yy - h / 2) / 160) ** 2)
    tissue = np.full((h, w), 0.45)
    tissue[r < 1.25] = 0.08                       # amniotic fluid around the head
    tissue[r < 1.0] = 0.35                        # brain
    tissue += 0.9 * np.exp(-((r - 1.0) * 160 / 6) ** 2)   # skull ring
    speckle = rng.rayleigh(1 / np.sqrt(np.pi / 2), (h, w))  # mean 1
    img = ndi.gaussian_filter(tissue * speckle, 0.8) + rng.normal(0, 0.02, (h, w))
    ann = (np.abs(r - 1.0) * 160 < 1.0).astype(np.uint8) * 255
    return np.clip(img / img.max(), 0, 1), ann


# ----------------------------------------------------------------- 1. noise removal
def mean_filter(img, k=5):
    """Each pixel = average of its k x k neighbourhood (box filter)."""
    return ndi.uniform_filter(img, size=k)


def gaussian_filter(img, sigma=1.5):
    """Weighted average; weights follow a 2-D Gaussian (closer pixels count more)."""
    return ndi.gaussian_filter(img, sigma=sigma)


def median_filter(img, k=5):
    """Each pixel = median of its k x k neighbourhood. Removes speckle 'spikes'
    while keeping edges sharper than averaging filters."""
    return ndi.median_filter(img, size=k)


# ----------------------------------------------------------------- 2. contrast
def contrast_stretch(img, low_pct=2, high_pct=98):
    """Linear stretch: the p2..p98 intensity range is mapped onto 0..1."""
    lo, hi = np.percentile(img, [low_pct, high_pct])
    return np.clip((img - lo) / (hi - lo + 1e-12), 0, 1)


def gamma_correction(img, gamma=0.8):
    """s = r^gamma. gamma < 1 brightens dark regions, gamma > 1 darkens."""
    return np.clip(img, 0, 1) ** gamma


def histogram_equalization(img, mask=None):
    """Global HE: maps intensities through the cumulative histogram (CDF) so the
    output histogram is roughly flat. With a mask, the histogram is built only
    from pixels inside the ultrasound sector, and the black border stays black."""
    if mask is None:
        return exposure.equalize_hist(img)
    out = np.zeros_like(img)
    out[mask] = exposure.equalize_hist(img[mask])
    return out


def clahe(img, clip=0.01):
    """Contrast Limited Adaptive HE: HE done in small tiles, with a clip limit so
    noise is not over-amplified."""
    return exposure.equalize_adapthist(np.clip(img, 0, 1), clip_limit=clip)


# ----------------------------------------------------------------- 3. sharpening
def unsharp_mask(img, sigma=2.0, amount=0.8):
    """sharpened = original + amount * (original - blurred)."""
    return np.clip(img + amount * (img - ndi.gaussian_filter(img, sigma)), 0, 1)


# ----------------------------------------------------------------- 5. measures
def speckle_index(img, k=7):
    """Average of (local std / local mean). Lower = less speckle."""
    m = ndi.uniform_filter(img, k)
    s = np.sqrt(np.maximum(ndi.uniform_filter(img ** 2, k) - m ** 2, 0))
    valid = m > 0.05
    return float(np.mean(s[valid] / m[valid]))


def snr_roi(img, roi):
    v = img[roi]
    return float(v.mean() / (v.std() + 1e-12))


def cnr(img, roi_a, roi_b):
    a, b = img[roi_a], img[roi_b]
    return float(abs(a.mean() - b.mean()) / np.sqrt(a.var() + b.var() + 1e-12))


def entropy(img):
    p, _ = np.histogram(np.clip(img, 0, 1), 256, (0, 1))
    p = p[p > 0] / p.sum()
    return float(-np.sum(p * np.log2(p)))


def edge_preservation_index(orig, proc, roi):
    """Edge preservation: correlation between the edge-strength (Sobel gradient
    magnitude) maps of the original and processed image, inside the skull band.
    The original is lightly smoothed first so its speckle does not count as edges.
    1 = edges kept exactly where they were."""
    o = ndi.gaussian_filter(orig, 2.0)
    a = np.hypot(ndi.sobel(o, 0), ndi.sobel(o, 1))[roi]
    b = np.hypot(ndi.sobel(proc, 0), ndi.sobel(proc, 1))[roi]
    a = a - a.mean(); b = b - b.mean()
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b) + 1e-12))


def all_measures(orig, img, roi_t, roi_b, roi_e):
    return {'Speckle index (lower better)': speckle_index(img),
            'SNR in brain (higher better)': snr_roi(img, roi_t),
            'CNR brain vs fluid (higher better)': cnr(img, roi_t, roi_b),
            'Entropy (bits)': entropy(img),
            'PSNR vs original (dB)': (float(peak_signal_noise_ratio(orig, np.clip(img, 0, 1), data_range=1))
                                      if not np.array_equal(orig, img) else float('nan')),
            'SSIM vs original': float(structural_similarity(orig, np.clip(img, 0, 1), data_range=1)),
            'Edge preservation index (EPI)': edge_preservation_index(orig, img, roi_e)}


# ----------------------------------------------------------------- regions of interest
def fit_ellipse_from_annotation(ann):
    """Least-squares conic fit to the annotated head outline -> (xc, yc, a, b, theta)."""
    yy, xx = np.nonzero(ann > 0)
    x = xx.astype(float); y = yy.astype(float)
    D = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
    _, _, V = np.linalg.svd(D, full_matrices=False)
    A, B, C, Dd, E, F = V[-1]
    den = B * B - 4 * A * C
    xc = (2 * C * Dd - B * E) / den; yc = (2 * A * E - B * Dd) / den
    th = 0.5 * np.arctan2(B, A - C)
    # axis lengths from the centred conic
    Fc = A * xc ** 2 + B * xc * yc + C * yc ** 2 + Dd * xc + E * yc + F
    M = np.array([[A, B / 2], [B / 2, C]])
    ev = np.linalg.eigvalsh(M)
    ax = np.sqrt(-Fc / ev)
    return xc, yc, float(ax.max()), float(ax.min()), th


def make_rois(shape, ann=None):
    """Target = inside the skull (brain), background = ring just outside the head
    (amniotic fluid / soft tissue), edge = band around the skull."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    if ann is not None and np.count_nonzero(ann) > 20:
        xc, yc, a, b, th = fit_ellipse_from_annotation(ann)
        c, s = np.cos(th), np.sin(th)
        u = (xx - xc) * c + (yy - yc) * s; v = -(xx - xc) * s + (yy - yc) * c
        # the fitted axis order may not match u/v; compute both and take the valid one
        r1 = np.sqrt((u / a) ** 2 + (v / b) ** 2); r2 = np.sqrt((u / b) ** 2 + (v / a) ** 2)
        on1 = np.mean(np.abs(r1[ann > 0] - 1)); on2 = np.mean(np.abs(r2[ann > 0] - 1))
        r = r1 if on1 < on2 else r2
        return (r < 0.6), (r > 1.12) & (r < 1.3), (r > 0.8) & (r < 1.2), (xc, yc, a, b, th)
    r = np.sqrt(((xx - w / 2) / (0.3 * w)) ** 2 + ((yy - h / 2) / (0.3 * h)) ** 2)
    return (r < 0.5), (r > 1.15) & (r < 1.35), (r > 0.8) & (r < 1.2), None


def ellipse_circumference(a, b):
    h = ((a - b) / (a + b)) ** 2
    return np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))


# ----------------------------------------------------------------- figures
def save_grid(images, titles, path, cols=3, cmap='gray', suptitle=None):
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(np.clip(im, 0, 1), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(t, color=FOREST); ax.axis('off')
    for ax in axes[len(images):]:
        ax.axis('off')
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight='bold', color=FOREST)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='FIRST SOUND basic image processing')
    ap.add_argument('image', nargs='?', help='input ultrasound image (e.g. HC18 010_HC.png)')
    ap.add_argument('--ann', help='HC18 annotation image (optional)')
    ap.add_argument('--px', type=float, help='pixel size in mm (optional)')
    ap.add_argument('--out', default='ip_results')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.image:
        img = load_gray(args.image)
        ann = np.asarray(Image.open(args.ann).convert('L')) if args.ann else None
        src = os.path.basename(args.image)
    else:
        img, ann = synthetic_test_image()
        src = 'SYNTHETIC test image (not patient data)'
        print('No image given -> using a SYNTHETIC test image. Pass your HC18 image for real results.')

    sector = ndi.uniform_filter(img, 15) > 0.03 if args.image else np.ones(img.shape, bool)
    roi_t, roi_b, roi_e, ell = make_rois(img.shape, ann)
    # ignore the black area outside the ultrasound sector (and the black text box)
    roi_t &= sector; roi_b &= sector; roi_e &= sector

    # ---- processing
    res = {
        'Original': img,
        'Mean 5x5': mean_filter(img, 5),
        'Gaussian σ=1.5': gaussian_filter(img, 1.5),
        'Median 5x5': median_filter(img, 5),
        'Contrast stretching': contrast_stretch(img),
        'Gamma 0.8': gamma_correction(img, 0.8),
        'Histogram equalization': histogram_equalization(img, sector),
        'CLAHE': clahe(img),
        'Unsharp masking': unsharp_mask(img),
    }
    final = unsharp_mask(contrast_stretch(median_filter(img, 5)), 2.0, 0.6)
    res['Final pipeline'] = final

    # ---- figures
    fig_n = 1
    def name(t):
        nonlocal fig_n
        p = os.path.join(args.out, f'fig_{fig_n:02d}_{t}.png'); fig_n += 1; return p

    # 1. original (+ annotation overlay)
    fig, axes = plt.subplots(1, 2 if ann is not None else 1, figsize=(10 if ann is not None else 5.5, 4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=1); axes[0].set_title('Original image'); axes[0].axis('off')
    if ann is not None:
        rgb = np.dstack([img] * 3).copy(); rgb[ann > 0] = [0.9, 0.55, 0.25]
        axes[1].imshow(rgb); axes[1].set_title('Expert head outline (annotation)'); axes[1].axis('off')
    fig.tight_layout(); fig.savefig(name('original'), dpi=200, bbox_inches='tight'); plt.close(fig)

    # 2. regions used for measurement
    rgb = np.dstack([img] * 3).copy()
    rgb[roi_t] = 0.6 * rgb[roi_t] + 0.4 * np.array([0.12, 0.19, 0.17])
    rgb[roi_b] = 0.6 * rgb[roi_b] + 0.4 * np.array([0.65, 0.38, 0.23])
    fig, ax = plt.subplots(figsize=(5.5, 4)); ax.imshow(rgb); ax.axis('off')
    ax.set_title('Measurement regions: brain (green) and surrounding (clay)')
    fig.tight_layout(); fig.savefig(name('regions'), dpi=200, bbox_inches='tight'); plt.close(fig)

    keys = ['Original', 'Mean 5x5', 'Gaussian σ=1.5', 'Median 5x5']
    save_grid([res[k] for k in keys], keys, name('noise_removal'), cols=2, suptitle='Speckle noise removal')
    keys = ['Original', 'Contrast stretching', 'Gamma 0.8', 'Histogram equalization', 'CLAHE']
    save_grid([res[k] for k in keys], keys, name('contrast'), cols=3, suptitle='Contrast enhancement')
    save_grid([img, res['Unsharp masking'], final], ['Original', 'Unsharp masking', 'Final pipeline'],
              name('sharpening_final'), cols=3, suptitle='Sharpening and final pipeline')

    # histograms
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, k in zip(axes.ravel(), ['Original', 'Median 5x5', 'Contrast stretching', 'Histogram equalization', 'CLAHE', 'Final pipeline']):
        ax.hist(np.clip(res[k], 0, 1).ravel(), 128, range=(0, 1), color=FOREST)
        ax.set_title(k); ax.set_xlabel('intensity'); ax.set_ylabel('pixels'); ax.set_yticks([])
    fig.suptitle('Histograms before and after', fontsize=13, fontweight='bold', color=FOREST)
    fig.tight_layout(); fig.savefig(name('histograms'), dpi=200, bbox_inches='tight'); plt.close(fig)

    # zoom on the skull edge
    ys, xs = np.nonzero(roi_e)
    cy, cx = int(np.median(ys)), int(xs.max() - 20)   # right-hand skull edge
    hz = 60
    sl = (slice(max(cy - hz, 0), cy + hz), slice(max(cx - hz, 0), cx + hz))
    keys = ['Original', 'Mean 5x5', 'Gaussian σ=1.5', 'Median 5x5', 'Unsharp masking', 'Final pipeline']
    save_grid([res[k][sl] for k in keys], keys, name('zoom_edges'), cols=3, suptitle='Zoom on the skull edge')

    # line profile across the skull
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for k, col in [('Original', '#9aa5a0'), ('Median 5x5', SAGE), ('Final pipeline', CLAY)]:
        ax.plot(res[k][cy, max(cx - 150, 0):cx + 60], color=col, lw=1.4 if k != 'Original' else 0.9, label=k)
    ax.set_xlabel('pixel along the line'); ax.set_ylabel('intensity'); ax.legend(frameon=False)
    ax.set_title('Intensity profile across the skull (speckle is smoothed, the skull peak stays)')
    fig.tight_layout(); fig.savefig(name('line_profile'), dpi=200, bbox_inches='tight'); plt.close(fig)

    # ---- measures
    rows = []
    for k, v in res.items():
        m = all_measures(img, v, roi_t, roi_b, roi_e)
        rows.append({'Method': k, **{kk: round(vv, 4) for kk, vv in m.items()}})
    with open(os.path.join(args.out, 'metrics.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    # bar chart
    names = [r['Method'] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, key, col in zip(axes, ['Speckle index (lower better)', 'CNR brain vs fluid (higher better)',
                                   'Edge preservation index (EPI)'], [CLAY, FOREST, SAGE]):
        ax.barh(names, [r[key] for r in rows], color=col); ax.set_title(key); ax.invert_yaxis()
    fig.suptitle('Quality measures for each method', fontsize=13, fontweight='bold', color=FOREST)
    fig.tight_layout(); fig.savefig(name('metrics_chart'), dpi=200, bbox_inches='tight'); plt.close(fig)

    # head circumference from the annotation (optional)
    if ell is not None and args.px:
        hc = ellipse_circumference(ell[2], ell[3]) * args.px
        with open(os.path.join(args.out, 'head_circumference.txt'), 'w') as f:
            f.write(f'Head circumference from annotation ellipse: {hc:.1f} mm\n'
                    f'BPD (minor axis): {2 * ell[3] * args.px:.1f} mm, OFD (major axis): {2 * ell[2] * args.px:.1f} mm\n')
        print(f'HC from annotation: {hc:.1f} mm')

    # final images individually (full resolution, for the report)
    for k in ['Original', 'Median 5x5', 'Histogram equalization', 'CLAHE', 'Final pipeline']:
        Image.fromarray(np.uint8(np.clip(res[k], 0, 1) * 255)).save(
            os.path.join(args.out, 'img_' + k.lower().replace(' ', '_').replace('σ=', 's').replace('.', '') + '.png'))

    print(f'Source: {src}')
    print(f'Saved {fig_n - 1} figures + metrics.csv to {os.path.abspath(args.out)}')
    print('\nMethod'.ljust(26) + 'Speckle idx  CNR    EPI    PSNR')
    for r in rows:
        print(f"{r['Method']:<25} {r['Speckle index (lower better)']:<12.3f} {r['CNR brain vs fluid (higher better)']:<6.2f} "
              f"{r['Edge preservation index (EPI)']:<6.2f} {r['PSNR vs original (dB)']:.1f}")


if __name__ == '__main__':
    main()
