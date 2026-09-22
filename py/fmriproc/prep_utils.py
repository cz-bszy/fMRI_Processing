"""Small numeric helpers of stage 03 (func_prep), one sub-command each.

Every sub-command prints a single value (or one line) on stdout so that the bash
stage can capture it; diagnostics go to stderr.

    nss               non-steady-state volumes at the start of the ORIGINAL series
    min-outlier       index of the minimum of a 1D file (motion reference volume)
    max-index         "index value" of the maximum of a 1D file
    mean-1d           mean of a 1D file
    despike-fraction  fraction of in-brain samples changed by 3dDespike
    scale-factor      target / median(in-mask mean image)
    dice              Dice coefficient of two masks (optionally inside a third)
    masked-corr       Pearson r between two 3D images inside a mask
    obliquity         largest angle (deg) between a voxel axis and its cardinal axis
    make-grid         geometry-only reference grid around a mask (T1w-space BOLD grid)
    write-info        <RUN>_desc-prep_info.json from key=value pairs
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np

from fmriproc.utils import load_img, read_1d, read_json, same_grid, save_like, write_json

NSS_WINDOW = 50       # volumes inspected at the start of the series
NSS_CAP = 10          # never report more than this many
NSS_ROBUST_SDS = 3.0
DESPIKE_MAX_VOLS = 40
MAD_TO_SD = 1.4826

# keys of DESIGN.md section 10 plus the self-check numbers of stage 03
PREP_INFO_KEYS: tuple[str, ...] = (
    "source", "run_label", "tr", "n_volumes_raw", "n_dropped", "n_volumes", "nss_detected",
    "despike", "despike_fraction", "stc_applied", "stc_reason", "stc_interp",
    "slice_timing_source", "slice_timing_evidence", "tzero", "hmc_reference",
    "coreg_method", "bbr_cost", "bbr_vs_init_mm", "bbr_rejected", "epi_mask_method",
    "scale_factor", "func_t1w_res", "mni_res", "template", "voxel_size", "obliquity_deg",
    "wm_mask_voxels", "csf_mask_voxels", "gm_mask_voxels", "tissue_erosion_relaxed",
    "coreg_dice", "hmc_consistency_r", "tool_versions",
)


# ----------------------------------------------------------------------------
# image access
# ----------------------------------------------------------------------------

def _read_volumes(img: nib.spatialimages.SpatialImage, time_slice: slice) -> np.ndarray:
    """(X, Y, Z, t) float32 block of a 4D image; only the requested volumes are read."""
    if len(img.shape) != 4:
        raise ValueError(f"expected a 4D image, got shape {img.shape}")
    return np.asarray(img.dataobj[..., time_slice], dtype=np.float32)


def _read_3d(path: str | Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    img = load_img(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"expected a 3D image: {path} has shape {img.shape}")
    return img, data


def _read_mask(path: str | Path, like: nib.spatialimages.SpatialImage | None = None) -> np.ndarray:
    img, data = _read_3d(path)
    if like is not None and not same_grid(img, like):
        raise ValueError(f"grid of {path} differs from the grid of the first image")
    return np.nan_to_num(data) > 0


def intensity_mask(mean_image: np.ndarray) -> np.ndarray:
    """Crude head mask: voxels brighter than a quarter of the robust maximum."""
    finite = np.isfinite(mean_image)
    positive = mean_image[finite & (mean_image > 0)]
    if positive.size == 0:
        return np.zeros(mean_image.shape, dtype=bool)
    return finite & (mean_image > 0.25 * np.percentile(positive, 98))


# ----------------------------------------------------------------------------
# non-steady-state detection
# ----------------------------------------------------------------------------

def global_signal(bold: nib.spatialimages.SpatialImage, n_volumes: int = NSS_WINDOW) -> np.ndarray:
    """Mean in-head signal of the first `n_volumes` volumes."""
    block = _read_volumes(bold, slice(0, n_volumes))
    mask = intensity_mask(block.mean(axis=3))
    if not mask.any():
        return np.zeros(block.shape[3], dtype=np.float64)
    return block[mask].mean(axis=0, dtype=np.float64)


def detect_nss(signal: Sequence[float], cap: int = NSS_CAP, n_sd: float = NSS_ROBUST_SDS) -> int:
    """Number of leading volumes whose global signal is still above steady state.

    Steady state is described by the later volumes of the window (everything after
    the first `cap` volumes, or the second half of a short series): a leading
    volume counts while it exceeds median + n_sd * robust SD of those volumes.
    Only T1 saturation is looked for, so low outliers are ignored.
    """
    values = np.asarray(signal, dtype=np.float64).ravel()
    if values.size < 4:
        return 0
    first_later = cap if values.size >= 2 * cap else values.size // 2
    later = values[first_later:]
    later = later[np.isfinite(later)]
    if later.size < 2:
        return 0
    centre = float(np.median(later))
    robust_sd = MAD_TO_SD * float(np.median(np.abs(later - centre)))
    # a perfectly flat reference would flag numerical noise
    robust_sd = max(robust_sd, 1e-6 * max(abs(centre), 1.0))
    threshold = centre + n_sd * robust_sd
    count = 0
    for value in values[:cap]:
        if not np.isfinite(value) or value <= threshold:
            break
        count += 1
    return count


# ----------------------------------------------------------------------------
# 1D files
# ----------------------------------------------------------------------------

def read_vector(path: str | Path) -> np.ndarray:
    """First column of a 1D file as a flat vector (a single row is taken as the vector)."""
    data = read_1d(path)
    if data.shape[0] == 1 and data.shape[1] > 1:
        return data[0]
    return data[:, 0]


def min_outlier_index(values: Sequence[float]) -> int:
    """Index of the smallest finite value; ties go to the volume nearest the middle
    of the run (smallest average displacement from the other volumes)."""
    vector = np.asarray(values, dtype=np.float64).ravel()
    finite = np.isfinite(vector)
    if not finite.any():
        raise ValueError("no finite value in the outlier series")
    lowest = vector[finite].min()
    candidates = np.flatnonzero(finite & (vector == lowest))
    middle = (vector.size - 1) / 2.0
    return int(candidates[np.argmin(np.abs(candidates - middle))])


def max_index(values: Sequence[float]) -> tuple[int, float]:
    vector = np.asarray(values, dtype=np.float64).ravel()
    if not np.isfinite(vector).any():
        raise ValueError("no finite value in the series")
    index = int(np.nanargmax(vector))
    return index, float(vector[index])


# ----------------------------------------------------------------------------
# despiking, scaling, overlap, correlation, geometry
# ----------------------------------------------------------------------------

def despike_fraction(
    before: nib.spatialimages.SpatialImage,
    after: nib.spatialimages.SpatialImage,
    max_vols: int = DESPIKE_MAX_VOLS,
) -> float:
    """Fraction of in-head samples changed by despiking, from an evenly spaced
    subset of at most `max_vols` volumes (one sequential pass through each file)."""
    if tuple(before.shape) != tuple(after.shape):
        raise ValueError(f"shapes differ: {before.shape} vs {after.shape}")
    n_t = before.shape[3]
    step = max(1, math.ceil(n_t / max_vols))
    picked = slice(0, None, step)
    raw = _read_volumes(before, picked)
    mask = intensity_mask(raw.mean(axis=3))
    if not mask.any():
        return float("nan")
    raw_in = raw[mask]
    del raw
    clean_in = _read_volumes(after, picked)[mask]
    tolerance = 1e-6 * np.maximum(np.abs(raw_in), 1.0)
    changed = np.abs(clean_in - raw_in) > tolerance
    return float(changed.mean())


def scale_factor(mean_image: np.ndarray, mask: np.ndarray, target: float) -> float:
    """target / median of the mean image inside the mask."""
    values = mean_image[mask & np.isfinite(mean_image)]
    if values.size == 0:
        raise ValueError("empty mask: cannot derive a scale factor")
    median = float(np.median(values))
    if not median > 0:
        raise ValueError(f"in-mask median of the mean image is {median:g}: cannot scale")
    return float(target) / median


def dice(a: np.ndarray, b: np.ndarray, within: np.ndarray | None = None) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("masks have different shapes")
    if within is not None:
        within = np.asarray(within, dtype=bool)
        a = a & within
        b = b & within
    total = int(a.sum()) + int(b.sum())
    if total == 0:
        return float("nan")
    return 2.0 * int((a & b).sum()) / total


def masked_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if a.shape != b.shape or a.shape != mask.shape:
        raise ValueError("images and mask have different shapes")
    keep = mask & np.isfinite(a) & np.isfinite(b)
    if int(keep.sum()) < 3:
        return float("nan")
    x = a[keep].astype(np.float64)
    y = b[keep].astype(np.float64)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def obliquity_deg(affine: np.ndarray) -> float:
    """Largest angle between a voxel axis and the cardinal axis it is closest to."""
    return float(np.degrees(np.max(np.abs(nib.affines.obliquity(np.asarray(affine, dtype=np.float64))))))


def voxel_sizes(affine: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in nib.affines.voxel_sizes(np.asarray(affine, dtype=np.float64))]


def make_grid(mask_img: nib.spatialimages.SpatialImage, resolution: float, pad_mm: float = 10.0) -> nib.Nifti1Image:
    """Isotropic, world-axis-aligned grid (LAS storage, like the FSL templates)
    enclosing the non-zero voxels of `mask_img` plus `pad_mm` on every side.

    Only the geometry matters (reference of antsApplyTransforms); voxels are 1.
    Built from world coordinates, so an oblique T1w needs no special handling.
    """
    if not resolution > 0:
        raise ValueError(f"resolution must be positive: {resolution}")
    data = np.nan_to_num(np.asarray(mask_img.dataobj, dtype=np.float32))
    if data.ndim == 4:
        data = data[..., 0]
    index = np.argwhere(data > 0)
    if index.size == 0:
        raise ValueError("mask is empty: cannot build a grid around it")
    low = index.min(axis=0) - 0.5
    high = index.max(axis=0) + 0.5
    corners = np.array([[x, y, z, 1.0] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    world = (np.asarray(mask_img.affine, dtype=np.float64) @ corners.T)[:3].T
    start = world.min(axis=0) - pad_mm
    stop = world.max(axis=0) + pad_mm
    shape = np.maximum(np.ceil((stop - start) / resolution - 1e-6).astype(int), 1)
    centre = (start + stop) / 2.0
    extent = shape * resolution
    affine = np.diag([-resolution, resolution, resolution, 1.0])
    affine[0, 3] = centre[0] + extent[0] / 2.0 - resolution / 2.0   # first voxel is the rightmost one
    affine[1, 3] = centre[1] - extent[1] / 2.0 + resolution / 2.0
    affine[2, 3] = centre[2] - extent[2] / 2.0 + resolution / 2.0
    grid = nib.Nifti1Image(np.ones(tuple(int(n) for n in shape), dtype=np.uint8), affine)
    grid.set_qform(affine, code=1)
    grid.set_sform(affine, code=1)
    grid.header.set_xyzt_units("mm", "sec")
    return grid


# ----------------------------------------------------------------------------
# prep_info.json
# ----------------------------------------------------------------------------

_NULL_WORDS = {"", "nan", "n/a", "na", "none", "null"}


def infer_value(text: str) -> Any:
    """bash string -> JSON value: true/false/yes/no, numbers, JSON lists/objects,
    null for empty/nan/n/a; everything else stays a string."""
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered in _NULL_WORDS:
        return None
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    try:
        value = json.loads(stripped)
    except ValueError:
        return text
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_pairs(pairs: Sequence[str], as_string: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"expected key=value, got '{pair}'")
        out[key] = value if as_string else infer_value(value)
    return out


def build_info(
    values: dict[str, Any],
    versions: dict[str, Any] | None = None,
    geometry_from: nib.spatialimages.SpatialImage | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Ordered prep_info dictionary and the list of contract keys left empty."""
    info = dict(values)
    if geometry_from is not None:
        info.setdefault("voxel_size", voxel_sizes(geometry_from.affine))
        info.setdefault("obliquity_deg", round(obliquity_deg(geometry_from.affine), 3))
    info["tool_versions"] = versions if versions is not None else info.get("tool_versions", {})
    missing = [key for key in PREP_INFO_KEYS if key not in info]
    ordered: dict[str, Any] = {key: info.get(key) for key in PREP_INFO_KEYS}
    ordered.update({key: value for key, value in info.items() if key not in ordered})
    return ordered, missing


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _fmt(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.8g}"


def _cmd_nss(args: argparse.Namespace) -> int:
    signal = global_signal(load_img(args.bold), args.window)
    print(detect_nss(signal, cap=args.cap))
    return 0


def _cmd_min_outlier(args: argparse.Namespace) -> int:
    print(min_outlier_index(read_vector(args.file)))
    return 0


def _cmd_max_index(args: argparse.Namespace) -> int:
    index, value = max_index(read_vector(args.file))
    print(f"{index} {_fmt(value)}")
    return 0


def _cmd_mean_1d(args: argparse.Namespace) -> int:
    vector = read_vector(args.file)
    vector = vector[np.isfinite(vector)]
    print(_fmt(float(vector.mean()) if vector.size else float("nan")))
    return 0


def _cmd_despike_fraction(args: argparse.Namespace) -> int:
    print(_fmt(despike_fraction(load_img(args.before), load_img(args.after), args.max_vols)))
    return 0


def _cmd_scale_factor(args: argparse.Namespace) -> int:
    img, mean_image = _read_3d(args.mean)
    print(_fmt(scale_factor(mean_image, _read_mask(args.mask, img), args.target)))
    return 0


def _cmd_dice(args: argparse.Namespace) -> int:
    img, a = _read_3d(args.a)
    within = _read_mask(args.within, img) if args.within else None
    print(_fmt(dice(np.nan_to_num(a) > 0, _read_mask(args.b, img), within)))
    return 0


def _cmd_masked_corr(args: argparse.Namespace) -> int:
    img, a = _read_3d(args.a)
    other, b = _read_3d(args.b)
    if not same_grid(img, other):
        raise ValueError(f"grid of {args.b} differs from the grid of {args.a}")
    print(_fmt(masked_corr(a, b, _read_mask(args.mask, img))))
    return 0


def _cmd_obliquity(args: argparse.Namespace) -> int:
    print(f"{obliquity_deg(load_img(args.image).affine):.3f}")
    return 0


def _cmd_make_grid(args: argparse.Namespace) -> int:
    grid = make_grid(load_img(args.mask), args.res, args.pad_mm)
    save_like(np.asarray(grid.dataobj), grid, args.out, dtype=np.uint8)
    print(" ".join(str(n) for n in grid.shape))
    return 0


def _cmd_write_info(args: argparse.Namespace) -> int:
    values = parse_pairs(args.set or [])
    values.update(parse_pairs(args.set_str or [], as_string=True))
    versions = read_json(args.versions_json, default={}) if args.versions_json else None
    geometry = load_img(args.geometry_from) if args.geometry_from else None
    info, missing = build_info(values, versions, geometry)
    if missing:
        print(f"WARNING: prep_info keys without a value: {', '.join(missing)}", file=sys.stderr)
    write_json(args.out, info)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fmriproc.prep_utils", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("nss", help="number of non-steady-state volumes at the start of a series")
    p.add_argument("--bold", type=Path, required=True, help="ORIGINAL 4D series (before dropping volumes)")
    p.add_argument("--window", type=int, default=NSS_WINDOW)
    p.add_argument("--cap", type=int, default=NSS_CAP)
    p.set_defaults(func=_cmd_nss)

    p = sub.add_parser("min-outlier", help="index (0-based) of the minimum of a 1D file")
    p.add_argument("--file", type=Path, required=True)
    p.set_defaults(func=_cmd_min_outlier)

    p = sub.add_parser("max-index", help="'index value' of the maximum of a 1D file")
    p.add_argument("--file", type=Path, required=True)
    p.set_defaults(func=_cmd_max_index)

    p = sub.add_parser("mean-1d", help="mean of the first column of a 1D file")
    p.add_argument("--file", type=Path, required=True)
    p.set_defaults(func=_cmd_mean_1d)

    p = sub.add_parser("despike-fraction", help="fraction of in-head samples changed by despiking")
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--max-vols", type=int, default=DESPIKE_MAX_VOLS)
    p.set_defaults(func=_cmd_despike_fraction)

    p = sub.add_parser("scale-factor", help="target / median of the mean image inside the mask")
    p.add_argument("--mean", type=Path, required=True, help="temporal mean image (3D)")
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--target", type=float, required=True)
    p.set_defaults(func=_cmd_scale_factor)

    p = sub.add_parser("dice", help="Dice coefficient of two masks on one grid")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--within", type=Path, default=None, help="restrict both masks to this mask (e.g. the EPI field of view)")
    p.set_defaults(func=_cmd_dice)

    p = sub.add_parser("masked-corr", help="Pearson r of two 3D images inside a mask")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--mask", type=Path, required=True)
    p.set_defaults(func=_cmd_masked_corr)

    p = sub.add_parser("obliquity", help="obliquity of an image in degrees")
    p.add_argument("--image", type=Path, required=True)
    p.set_defaults(func=_cmd_obliquity)

    p = sub.add_parser("make-grid", help="geometry-only isotropic grid around a mask")
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--res", type=float, required=True, help="isotropic voxel size in mm")
    p.add_argument("--pad-mm", type=float, default=10.0)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_cmd_make_grid)

    p = sub.add_parser("write-info", help="write <RUN>_desc-prep_info.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="value with type inference (repeatable)")
    p.add_argument("--set-str", action="append", metavar="KEY=VALUE", help="value kept as a string (repeatable)")
    p.add_argument("--versions-json", type=Path, default=None, help="JSON written by fp_tool_versions")
    p.add_argument("--geometry-from", type=Path, default=None, help="image providing voxel_size and obliquity_deg")
    p.set_defaults(func=_cmd_write_info)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError) as err:
        print(f"ERROR: {args.command}: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
