"""Shared helpers for the fmriproc package (I/O, ROI means, connectivity)."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import pandas as pd

NA = "n/a"


# ----------------------------------------------------------------------------
# atomic text output
# ----------------------------------------------------------------------------

def _atomic_write(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, data: dict) -> None:
    """JSON with indent=2; NaN/inf become null; numpy types are converted."""
    _atomic_write(Path(path), json.dumps(_jsonable(data), indent=2, ensure_ascii=False) + "\n")


def read_json(path: str | Path, default: dict | None = None) -> dict:
    path = Path(path)
    if not path.is_file():
        if default is None:
            raise FileNotFoundError(path)
        return default
    with open(path, encoding="utf-8-sig") as fh:
        return json.load(fh)


def write_tsv(path: str | Path, frame: pd.DataFrame, index: bool = False, float_format: str = "%.6g") -> None:
    text = frame.to_csv(sep="\t", index=index, na_rep=NA, float_format=float_format, lineterminator="\n")
    _atomic_write(Path(path), text)


def read_tsv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", na_values=[NA], keep_default_na=True, encoding="utf-8-sig", **kwargs)


def read_1d(path: str | Path) -> np.ndarray:
    """AFNI .1D / plain numeric text -> 2D float array (rows = time points)."""
    data = np.loadtxt(path, dtype=float, comments="#", ndmin=2)
    return data


# ----------------------------------------------------------------------------
# images
# ----------------------------------------------------------------------------

def load_img(path: str | Path) -> nib.Nifti1Image:
    img = nib.load(str(path))
    if not isinstance(img, (nib.Nifti1Image, nib.Nifti2Image)):
        img = nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine)
    return img


def same_grid(a: nib.spatialimages.SpatialImage, b: nib.spatialimages.SpatialImage, atol: float = 1e-3) -> bool:
    return tuple(a.shape[:3]) == tuple(b.shape[:3]) and np.allclose(a.affine, b.affine, atol=atol)


def save_like(data: np.ndarray, like: nib.spatialimages.SpatialImage, path: str | Path, dtype: Any = np.float32) -> None:
    """Save `data` on the grid of `like` (atomic: temporary name, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = like.header.copy()
    header.set_data_dtype(dtype)
    header["scl_slope"] = np.nan
    header["scl_inter"] = np.nan
    img = nib.Nifti1Image(np.asarray(data, dtype=dtype), like.affine, header)
    suffix = ".nii.gz" if path.name.endswith(".nii.gz") else path.suffix
    tmp = path.with_name(f".tmp{os.getpid()}_{path.name[: -len(suffix)]}{suffix}")
    nib.save(img, str(tmp))
    os.replace(tmp, path)


def masked_2d(bold: nib.spatialimages.SpatialImage, mask: np.ndarray) -> np.ndarray:
    """4D image + 3D boolean mask -> (T, V) float32 array."""
    data = np.asanyarray(bold.dataobj, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError("expected a 4D image")
    if mask.shape != data.shape[:3]:
        raise ValueError("mask and image grids differ")
    return data[mask].T


# ----------------------------------------------------------------------------
# ROI means and connectivity (shared by timeseries and QC)
# ----------------------------------------------------------------------------

def extract_roi_means(
    data: np.ndarray,
    atlas: np.ndarray,
    mask: np.ndarray,
    min_coverage: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Mean time series per atlas label.

    data: 4D array (X, Y, Z, T); atlas: 3D integer labels (0 = background);
    mask: 3D boolean coverage mask on the same grid.
    A voxel is valid if it lies in the mask, is finite at every time point and is
    not constant over time. An ROI whose valid fraction is below `min_coverage`
    (or that has no valid voxel) is returned as NaN - never imputed.

    Returns (labels [R], series [T, R] float64, coverage DataFrame with columns
    roi, atlas_voxels, valid_voxels, coverage_fraction, included).
    """
    if data.ndim != 4 or atlas.shape != data.shape[:3] or mask.shape != data.shape[:3]:
        raise ValueError("data, atlas and mask must share one grid")
    atlas = np.rint(atlas).astype(np.int64)
    labels = np.unique(atlas[atlas > 0])
    n_t = data.shape[3]
    series = np.full((n_t, labels.size), np.nan, dtype=np.float64)
    rows = []
    mask = mask.astype(bool)
    for col, label in enumerate(labels):
        region = atlas == label
        n_atlas = int(region.sum())
        values = data[region & mask].astype(np.float64)
        if values.size:
            finite = np.isfinite(values).all(axis=1)
            values = values[finite]
            values = values[np.ptp(values, axis=1) > 0] if values.size else values
        n_valid = int(values.shape[0]) if values.size else 0
        fraction = n_valid / n_atlas if n_atlas else 0.0
        included = n_valid > 0 and fraction >= min_coverage
        if included:
            series[:, col] = values.mean(axis=0)
        rows.append((int(label), n_atlas, n_valid, fraction, bool(included)))
    coverage = pd.DataFrame(rows, columns=["roi", "atlas_voxels", "valid_voxels", "coverage_fraction", "included"])
    return labels, series, coverage


def fc_matrix(series: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """Pearson correlation between columns of `series` (T, R) over retained frames.

    Columns containing NaN (uncovered ROIs) or zero variance give NaN rows/columns.
    """
    series = np.asarray(series, dtype=np.float64)
    if keep is not None:
        series = series[np.asarray(keep, dtype=bool)]
    n_roi = series.shape[1]
    out = np.full((n_roi, n_roi), np.nan)
    good = np.isfinite(series).all(axis=0) & (series.std(axis=0) > 0)
    if good.sum() >= 2 and series.shape[0] >= 3:
        out[np.ix_(good, good)] = np.corrcoef(series[:, good], rowvar=False)
    return out


def fisher_z(r: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def split_half_reliability(series: np.ndarray, keep: np.ndarray | None = None) -> float:
    """Correlation between Fisher-z FC of the first and second contiguous half
    of the retained frames. NaN when fewer than 20 frames per half."""
    series = np.asarray(series, dtype=np.float64)
    if keep is not None:
        series = series[np.asarray(keep, dtype=bool)]
    half = series.shape[0] // 2
    if half < 20:
        return float("nan")
    a = upper_triangle(fisher_z(fc_matrix(series[:half])))
    b = upper_triangle(fisher_z(fc_matrix(series[half: 2 * half])))
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def longest_true_run(flags: Iterable[bool]) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best
