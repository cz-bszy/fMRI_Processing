"""Stage 04: confound time series, FD, DVARS, aCompCor and the censor vector.

Input is the unsmoothed, globally scaled T1w-space BOLD of one run. Only voxels
inside the brain / WM / CSF masks are ever held in memory.

Column names follow fMRIPrep. Derivatives are backward differences whose first
row is 0 (not n/a), because the columns are handed to 3dTproject unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from fmriproc.utils import _atomic_write, load_img, longest_true_run, read_1d, same_grid, write_json, write_tsv

MOTION_COLUMNS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
SIGNAL_COLUMNS = ["global_signal", "csf", "white_matter"]
EXPANSIONS = ["", "_derivative1", "_power2", "_derivative1_power2"]
FD_RADIUS_MM = 50.0
CHUNK_BYTES = 64 * 1024 * 1024


def _warn(message: str) -> None:
    print(f"[confounds] WARNING: {message}", file=sys.stderr)


# ----------------------------------------------------------------------------
# small numeric building blocks
# ----------------------------------------------------------------------------

def backward_difference(values: np.ndarray) -> np.ndarray:
    """x[t] - x[t-1] along axis 0; the first row is 0."""
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values)
    out[1:] = np.diff(values, axis=0)
    return out


def expand_signal(name: str, values: np.ndarray) -> dict[str, np.ndarray]:
    """name, name_derivative1, name_power2, name_derivative1_power2."""
    values = np.asarray(values, dtype=np.float64)
    derivative = backward_difference(values)
    return {
        name: values,
        f"{name}_derivative1": derivative,
        f"{name}_power2": values ** 2,
        f"{name}_derivative1_power2": derivative ** 2,
    }


def read_motion_par(path: str | Path) -> np.ndarray:
    """mcflirt .par (rx ry rz [rad], tx ty tz [mm]) -> (T, 6) ordered as MOTION_COLUMNS."""
    par = read_1d(path)
    if par.shape[1] != 6:
        raise ValueError(f"{path}: expected 6 columns (mcflirt .par), found {par.shape[1]}")
    return np.column_stack([par[:, 3:6], par[:, 0:3]])


def framewise_displacement(motion: np.ndarray, radius: float = FD_RADIUS_MM) -> np.ndarray:
    """Power FD from (T, 6) [trans mm x3, rot rad x3]; first value 0."""
    delta = np.abs(backward_difference(motion))
    return delta[:, :3].sum(axis=1) + radius * delta[:, 3:6].sum(axis=1)


def pad_relrms(relrms: np.ndarray, n_volumes: int) -> np.ndarray:
    """mcflirt _rel.rms has one value per volume pair (T-1 rows)."""
    relrms = np.asarray(relrms, dtype=np.float64).ravel()
    if relrms.size == n_volumes - 1:
        return np.concatenate([[0.0], relrms])
    if relrms.size == n_volumes:
        _warn("relative RMS file has one row per volume; used without padding")
        return relrms
    raise ValueError(f"relative RMS has {relrms.size} rows, expected {n_volumes - 1}")


def dct_basis(n_volumes: int, tr: float, period_cut: float) -> np.ndarray:
    """Discrete cosine drift basis (SPM / nilearn convention) without the constant.

    Column k-1 is sqrt(2/T) * cos(pi/T * (t + 0.5) * k), k = 1..K with
    K = floor(2 * T * TR / period_cut): every cosine slower than the cut-off.
    """
    if period_cut <= 0:
        return np.zeros((n_volumes, 0))
    order = min(n_volumes - 1, int(np.floor(2.0 * n_volumes * tr / period_cut)))
    if order < 1:
        return np.zeros((n_volumes, 0))
    times = np.arange(n_volumes, dtype=np.float64) + 0.5
    k = np.arange(1, order + 1, dtype=np.float64)
    return np.sqrt(2.0 / n_volumes) * np.cos(np.pi / n_volumes * np.outer(times, k))


def acompcor(series: np.ndarray, basis: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, int]:
    """PCA of the voxel time series of one tissue mask.

    series: (T, V). The DCT basis and a constant are regressed out, voxels are
    z-scored, and the first left singular vectors are returned.
    Returns (components (T, n), variance explained by every singular vector,
    number of voxels that entered the SVD). n can be smaller than requested.
    """
    n_t = series.shape[0]
    data = np.asarray(series, dtype=np.float64)
    if data.shape[1]:
        data = data[:, np.ptp(data, axis=0) > 0]
    if n_components < 1 or data.shape[1] == 0:
        return np.zeros((n_t, 0)), np.zeros(0), int(data.shape[1])
    design = np.column_stack([basis, np.ones(n_t)])
    beta, *_ = np.linalg.lstsq(design, data, rcond=None)
    data = data - design @ beta
    sd = data.std(axis=0)
    varying = sd > 1e-10 * max(float(sd.max()), np.finfo(float).tiny)
    data = data[:, varying] / sd[varying]
    if data.shape[1] == 0:
        return np.zeros((n_t, 0)), np.zeros(0), 0
    u, s, _ = np.linalg.svd(data, full_matrices=False)
    variance = s ** 2 / np.sum(s ** 2)
    rank = int(np.sum(s > s[0] * 1e-8))
    n_keep = min(n_components, rank)
    components = u[:, :n_keep]
    # deterministic sign: the largest-magnitude sample of every component is positive
    peak = np.abs(components).argmax(axis=0)
    components = components * np.sign(components[peak, np.arange(n_keep)])
    return components, variance, int(data.shape[1])


def compute_dvars(series: np.ndarray, variance_tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """DVARS and standardised DVARS (Nichols 2013, as nipype compute_dvars).

    series: (T, V) brain voxels of the scaled data. The robust SD of every voxel
    is IQR / 1.349; the expected SD of the temporal difference of an AR(1)
    series is sqrt(2 * (1 - rho)) * SD; std_dvars = DVARS / mean expected SD.
    The first value of both outputs is 0.
    """
    n_t = series.shape[0]
    dvars = np.zeros(n_t)
    std_dvars = np.zeros(n_t)
    if n_t < 3 or series.shape[1] == 0:
        std_dvars[1:] = np.nan
        return dvars, std_dvars
    q25, q75 = np.percentile(series, [25, 75], axis=0, method="lower")
    robust_sd = (q75.astype(np.float64) - q25.astype(np.float64)) / 1.349
    good = robust_sd > variance_tol
    if not good.any():
        std_dvars[1:] = np.nan
        return dvars, std_dvars
    data = np.asarray(series[:, good], dtype=np.float32)   # boolean indexing already made a copy
    robust_sd = robust_sd[good]
    diff = np.diff(data, axis=0)
    dvars[1:] = np.sqrt(np.einsum("tv,tv->t", diff, diff, dtype=np.float64) / data.shape[1])
    del diff
    data -= data.mean(axis=0, dtype=np.float64).astype(np.float32)
    r0 = np.einsum("tv,tv->v", data, data, dtype=np.float64)
    r1 = np.einsum("tv,tv->v", data[1:], data[:-1], dtype=np.float64)
    ar1 = np.clip(r1 / r0, -1.0, 1.0)
    expected_sd = float(np.mean(np.sqrt(2.0 * (1.0 - ar1)) * robust_sd))
    if expected_sd > 0:
        std_dvars[1:] = dvars[1:] / expected_sd
    else:
        std_dvars[1:] = np.nan
    return dvars, std_dvars


def short_segments(keep: np.ndarray, min_length: int) -> np.ndarray:
    """True for kept frames inside a stretch of fewer than min_length consecutive kept frames."""
    keep = np.asarray(keep, dtype=bool)
    out = np.zeros(keep.size, dtype=bool)
    start = None
    for index, kept in enumerate(np.r_[keep, False]):
        if kept and start is None:
            start = index
        elif not kept and start is not None:
            if index - start < min_length:
                out[start:index] = True
            start = None
    return out


def censor_vector(
    fd: np.ndarray,
    std_dvars: np.ndarray,
    fd_threshold: float,
    censor_prev: bool,
    dvars_threshold: float,
    censor_next: int = 0,
    min_segment: int = 0,
) -> np.ndarray:
    """1 = keep, 0 = censored. A threshold <= 0 switches that criterion off.

    censor_prev: also the frame before a high-FD frame; censor_next: also that many
    frames after it (spin history; Power et al. 2014 used 2). Both extend the FD
    criterion only. min_segment: kept stretches shorter than this many frames are
    censored as well (Power et al. 2014 used 5); applied last, and only when
    something is censored at all.
    """
    fd = np.asarray(fd, dtype=np.float64)
    if censor_next < 0 or min_segment < 0:
        raise ValueError(f"censor_next and min_segment must not be negative: {censor_next}, {min_segment}")
    bad = np.zeros(fd.size, dtype=bool)
    if fd_threshold > 0:
        bad_fd = fd > fd_threshold
        bad |= bad_fd
        if censor_prev:
            # FD[t] is the movement between t-1 and t, so t-1 is affected as well
            bad[:-1] |= bad_fd[1:]
        for shift in range(1, min(int(censor_next), fd.size - 1) + 1):
            bad[shift:] |= bad_fd[:-shift]
    if dvars_threshold > 0:
        with np.errstate(invalid="ignore"):
            bad |= np.asarray(std_dvars, dtype=np.float64) > dvars_threshold
    if min_segment > 1 and bad.any():
        bad |= short_segments(~bad, int(min_segment))
    return (~bad).astype(np.int64)


# ----------------------------------------------------------------------------
# image access
# ----------------------------------------------------------------------------

def load_mask(path: str | Path, like: nib.spatialimages.SpatialImage, name: str) -> np.ndarray:
    img = load_img(path)
    if not same_grid(img, like):
        raise ValueError(f"{name} mask is not on the BOLD grid: {path}")
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    return np.nan_to_num(data) > 0


def load_masked_series(bold: nib.spatialimages.SpatialImage, mask: np.ndarray) -> np.ndarray:
    """(T, V) float32 of the voxels in `mask`, read in blocks of volumes so the
    full 4D array never exists in memory."""
    n_t = bold.shape[3]
    out = np.empty((n_t, int(mask.sum())), dtype=np.float32)
    bytes_per_volume = int(np.prod(bold.shape[:3])) * 8
    step = max(1, CHUNK_BYTES // max(bytes_per_volume, 1))
    for start in range(0, n_t, step):
        stop = min(start + step, n_t)
        block = np.asanyarray(bold.dataobj[..., start:stop])
        out[start:stop] = block[mask].T
    return out


# ----------------------------------------------------------------------------
# assembling the table
# ----------------------------------------------------------------------------

def _tissue_mean(series: np.ndarray, columns: np.ndarray, n_volumes: int) -> np.ndarray:
    if not columns.any():
        return np.full(n_volumes, np.nan)
    if columns.all():
        return series.mean(axis=1, dtype=np.float64)
    return series[:, columns].mean(axis=1, dtype=np.float64)


def build_confounds(
    series: np.ndarray,
    in_brain: np.ndarray,
    in_wm: np.ndarray,
    in_csf: np.ndarray,
    motion: np.ndarray,
    relrms: np.ndarray,
    outliers: np.ndarray,
    tr: float,
    acompcor_n: int,
    highpass_sec: float,
    censor_fd: float,
    censor_prev: bool,
    censor_dvars: float,
    censor_next: int = 0,
    censor_min_segment: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """series: (T, V) over the union of the three masks; in_*: boolean columns selectors."""
    n_t = series.shape[0]
    columns: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}

    signals = {
        "global_signal": (in_brain, "brain mask"),
        "csf": (in_csf, "eroded CSF mask"),
        "white_matter": (in_wm, "eroded white-matter mask"),
    }
    for name, (selector, where) in signals.items():
        expanded = expand_signal(name, _tissue_mean(series, selector, n_t))
        columns.update(expanded)
        for key in expanded:
            meta[key] = {"Description": f"mean signal in the {where}" + _expansion_text(key, name), "Units": "scaled BOLD"}

    basis = dct_basis(n_t, tr, highpass_sec)
    acompcor_meta: dict[str, Any] = {
        "n_requested": acompcor_n,
        "highpass_sec": highpass_sec,
        "n_cosines": int(basis.shape[1]),
    }
    for prefix, tissue, selector in (("w", "wm", in_wm), ("c", "csf", in_csf)):
        components, variance, n_used = acompcor(series[:, selector], basis, acompcor_n)
        n_found = components.shape[1]
        if n_found < acompcor_n:
            _warn(f"aCompCor {tissue}: {n_found} of {acompcor_n} components ({int(selector.sum())} voxels in mask)")
        cumulative = np.cumsum(variance)
        for i in range(n_found):
            key = f"{prefix}_comp_cor_{i:02d}"
            columns[key] = components[:, i]
            meta[key] = {
                "Method": "aCompCor",
                "Mask": tissue.upper(),
                "VarianceExplained": float(variance[i]),
                "CumulativeVarianceExplained": float(cumulative[i]),
                "Retained": True,
            }
        acompcor_meta[f"n_{tissue}_voxels"] = int(selector.sum())
        acompcor_meta[f"n_{tissue}_voxels_used"] = n_used
        acompcor_meta[f"n_components_{tissue}"] = n_found
        acompcor_meta[f"variance_explained_{tissue}"] = [float(v) for v in variance[:n_found]]

    for i in range(basis.shape[1]):
        key = f"cosine_{i:02d}"
        columns[key] = basis[:, i]
        meta[key] = {
            "Description": "discrete cosine drift regressor removed before the aCompCor PCA",
            "FrequencyHz": float((i + 1) / (2.0 * n_t * tr)),
        }

    brain_series = series if in_brain.all() else series[:, in_brain]
    dvars, std_dvars = compute_dvars(brain_series)
    del brain_series
    fd = framewise_displacement(motion)
    columns["dvars"] = dvars
    columns["std_dvars"] = std_dvars
    columns["framewise_displacement"] = fd
    columns["fd_jenkinson"] = pad_relrms(relrms, n_t)
    columns["outlier_fraction"] = np.asarray(outliers, dtype=np.float64)
    meta["dvars"] = {"Description": "RMS over brain voxels of the backward temporal difference; first value 0", "Units": "scaled BOLD"}
    meta["std_dvars"] = {"Description": "DVARS divided by its expectation under an AR(1) null (robust IQR-based SD; Nichols 2013)"}
    meta["framewise_displacement"] = {"Description": f"Power FD, rotations on a {FD_RADIUS_MM:g} mm sphere; first value 0", "Units": "mm"}
    meta["fd_jenkinson"] = {"Description": "mcflirt relative RMS displacement (Jenkinson); first value 0", "Units": "mm"}
    meta["outlier_fraction"] = {"Description": "3dToutcount fraction of outlier voxels, before despiking"}

    for i, name in enumerate(MOTION_COLUMNS):
        expanded = expand_signal(name, motion[:, i])
        columns.update(expanded)
        base = "translation" if name.startswith("trans") else "rotation"
        unit = "mm" if name.startswith("trans") else "rad"
        for key in expanded:
            meta[key] = {"Description": f"head-motion {base} (mcflirt, estimated before STC)" + _expansion_text(key, name), "Units": unit}

    censor = censor_vector(fd, std_dvars, censor_fd, censor_prev, censor_dvars,
                           censor_next=censor_next, min_segment=censor_min_segment)
    columns["censor"] = censor
    meta["censor"] = _censor_summary(censor, tr, censor_fd, censor_prev, censor_dvars,
                                     censor_next, censor_min_segment)
    meta["acompcor"] = {
        "n_wm_voxels": acompcor_meta.pop("n_wm_voxels"),
        "n_csf_voxels": acompcor_meta.pop("n_csf_voxels"),
        "variance_explained_wm": acompcor_meta.pop("variance_explained_wm"),
        "variance_explained_csf": acompcor_meta.pop("variance_explained_csf"),
        **acompcor_meta,
    }
    return pd.DataFrame(columns), meta


def _expansion_text(key: str, name: str) -> str:
    suffix = key[len(name):]
    return {
        "": "",
        "_derivative1": "; backward difference, first row 0",
        "_power2": "; squared",
        "_derivative1_power2": "; squared backward difference",
    }[suffix]


def _censor_summary(censor: np.ndarray, tr: float, fd_thr: float, prev: bool, dvars_thr: float,
                    next_frames: int = 0, min_segment: int = 0) -> dict[str, Any]:
    n_t = int(censor.size)
    n_censored = int(np.sum(censor == 0))
    return {
        "Description": "1 = frame kept, 0 = frame censored",
        "fd_threshold": fd_thr,
        "prev": bool(prev),
        "next": int(next_frames),
        "min_segment": int(min_segment),
        "dvars_threshold": dvars_thr,
        "n_censored": n_censored,
        "n_volumes": n_t,
        "minutes_retained": (n_t - n_censored) * tr / 60.0,
        "longest_segment": longest_true_run(censor == 1),
        "pct_censored": 100.0 * n_censored / n_t if n_t else 0.0,
        "tr": tr,
    }


def run(args: argparse.Namespace) -> None:
    if args.tr <= 0:
        raise ValueError(f"--tr must be positive: {args.tr}")
    # keep_file_open: consecutive blocks are then read in one pass through the gzip stream
    bold = nib.load(str(args.bold), keep_file_open=True)
    if len(bold.shape) != 4 or bold.shape[3] < 3:
        raise ValueError(f"expected a 4D BOLD series with at least 3 volumes: {args.bold}")
    n_t = bold.shape[3]

    brain = load_mask(args.brain_mask, bold, "brain")
    wm = load_mask(args.wm_mask, bold, "WM")
    csf = load_mask(args.csf_mask, bold, "CSF")
    if not brain.any():
        raise ValueError(f"brain mask is empty: {args.brain_mask}")
    for name, mask in (("WM", wm), ("CSF", csf)):
        if not mask.any():
            _warn(f"{name} mask is empty: its mean signal is n/a and it yields no aCompCor components")

    motion = read_motion_par(args.motion_par)
    relrms = read_1d(args.relrms)
    outliers = read_1d(args.outliers)[:, 0]
    for name, rows in (("motion parameters", motion.shape[0]), ("outlier fraction", outliers.size)):
        if rows != n_t:
            raise ValueError(f"{name}: {rows} rows, but the BOLD series has {n_t} volumes")

    union = brain | wm | csf
    series = load_masked_series(bold, union)
    finite = np.isfinite(series).all(axis=0)
    if not finite.all():
        _warn(f"{int((~finite).sum())} voxels with non-finite values are ignored")
    in_brain, in_wm, in_csf = (mask[union] & finite for mask in (brain, wm, csf))

    frame, meta = build_confounds(
        series, in_brain, in_wm, in_csf, motion, relrms, outliers,
        tr=args.tr, acompcor_n=args.acompcor_n, highpass_sec=args.highpass_sec,
        censor_fd=args.censor_fd, censor_prev=args.censor_prev, censor_dvars=args.censor_dvars,
        censor_next=args.censor_next, censor_min_segment=args.censor_min_segment,
    )
    censor = frame["censor"].to_numpy()
    _atomic_write(Path(args.out_censor), "".join(f"{int(v)}\n" for v in censor))
    write_json(args.out_json, meta)
    write_tsv(args.out_tsv, frame, float_format="%.10g")
    summary = meta["censor"]
    print(
        f"[confounds] {n_t} volumes, {frame.shape[1]} columns, mean FD "
        f"{frame['framewise_displacement'].mean():.3f} mm, censored {summary['n_censored']} "
        f"({summary['pct_censored']:.1f}%), retained {summary['minutes_retained']:.2f} min",
        file=sys.stderr,
    )


def _yes_no(value: str) -> bool:
    text = value.strip().lower()
    if text in {"yes", "true", "1", "on"}:
        return True
    if text in {"no", "false", "0", "off", ""}:
        return False
    raise argparse.ArgumentTypeError(f"expected yes or no: {value}")


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number >= 0: {value}") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"expected a whole number >= 0: {value}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bold", required=True, type=Path, help="space-T1w desc-preproc BOLD (unsmoothed, scaled)")
    parser.add_argument("--brain-mask", required=True, type=Path)
    parser.add_argument("--wm-mask", required=True, type=Path)
    parser.add_argument("--csf-mask", required=True, type=Path)
    parser.add_argument("--motion-par", required=True, type=Path, help="mcflirt .par: rx ry rz (rad) tx ty tz (mm)")
    parser.add_argument("--relrms", required=True, type=Path, help="mcflirt _rel.rms (T-1 rows)")
    parser.add_argument("--outliers", required=True, type=Path, help="3dToutcount -fraction, one value per volume")
    parser.add_argument("--tr", required=True, type=float)
    parser.add_argument("--acompcor-n", type=int, default=5)
    parser.add_argument("--highpass-sec", type=float, default=128.0)
    parser.add_argument("--censor-fd", type=float, default=0.5, help="mm; 0 = off")
    parser.add_argument("--censor-prev", type=_yes_no, default=False, metavar="yes|no")
    parser.add_argument("--censor-dvars", type=float, default=0.0, help="standardised DVARS; 0 = off")
    parser.add_argument("--censor-next", type=_non_negative_int, default=0,
                        help="also censor this many frames after a high-FD frame; 0 = off")
    parser.add_argument("--censor-min-segment", type=_non_negative_int, default=0,
                        help="censor kept stretches shorter than this many frames; 0 = off")
    parser.add_argument("--out-tsv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-censor", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError) as error:
        print(f"[confounds] ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
