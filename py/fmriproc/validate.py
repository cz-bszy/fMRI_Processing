"""Validity metrics of the final ROI time series, per run (stage 10).

Reads only the ROI tables of stage 07, the censor vector, the confounds TSV, the
``_denoise.json`` files and ``<atlas-dir>/<A>/`` - never 4D data::

    python -m fmriproc.validate --func-dir D --run RUN --template TPL \
        --strategies "wmcsf24 wmcsf24gsr" --atlases "Schaefer2018_100Parcels_7Networks" \
        --atlas-dir RESOURCES/atlases --tr 2.0 --censor-mode NTRP \
        --out-tsv RUN_desc-validation.tsv --out-json RUN_desc-validation.json \
        --out-compare-tsv RUN_desc-streamcompare.tsv

A *stream* is ``volume`` (``space-<TPL>`` tables) or ``surface`` (``space-fsLR``).
``_desc-validation.tsv`` is long: ``stream strategy atlas metric value``.
``_desc-streamcompare.tsv`` (only when both streams exist) is long as well:
``strategy atlas scope roi metric volume surface value``; ``scope=run`` rows hold
every metric recomputed over the ROIs that are valid in BOTH streams
(``value`` = surface - volume; for ``fc_similarity`` and ``n_roi_common`` the
metric itself), ``scope=roi`` rows the per-ROI ``roi_tsnr`` of both streams.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

from fmriproc.utils import (
    NA,
    fc_matrix,
    fisher_z,
    read_1d,
    read_json,
    read_tsv,
    split_half_reliability,
    upper_triangle,
    write_json,
    write_tsv,
)

STREAMS = ("volume", "surface")
ATLAS_SPACE = "MNI152NLin6Asym"
DEFAULT_POLORT = 2
BAND_HZ = (0.01, 0.1)
HOMOTOPIC_MAX_MM = 20.0
MIN_FRAMES = 10

# order of the rows in the validation table
METRICS = (
    "roi_tsnr_median", "roi_tsnr_p10", "variance_removed_median", "split_half_r",
    "network_contrast", "homotopic_contrast", "dmn_contrast", "lowfreq_power_fraction",
    "fd_fc_coupling", "gs_residual_sd", "n_roi_nan", "n_retained", "dof_remaining",
)
# identical by construction once both streams are reduced to the common ROIs
PER_STREAM_ONLY = ("n_roi_nan", "n_retained", "dof_remaining")


def log(message: str) -> None:
    print(f"[validate] {message}", file=sys.stderr)


# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------

def split_list(text: str | None) -> list[str]:
    """'a b,c' -> ['a', 'b', 'c']; 'name=/path' entries (CUSTOM_ATLASES) keep the name."""
    items = [t.split("=", 1)[0] for t in re.split(r"[\s,]+", text or "") if t]
    return list(dict.fromkeys(items))


def read_series(path: Path) -> pd.DataFrame | None:
    """ROI table (header = ROI names, n/a = missing) as floats; None when absent."""
    if not path.is_file():
        return None
    frame = read_tsv(path)
    return frame.apply(pd.to_numeric, errors="coerce").astype(np.float64)


def read_keep(path: Path) -> np.ndarray | None:
    """Censor vector (1 = keep) as booleans; None when absent."""
    if not path.is_file():
        return None
    values = read_1d(path)[:, 0]
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("censor vector must contain finite binary values")
    return values.astype(bool)


def read_fd(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    frame = read_tsv(path)
    if "framewise_displacement" not in frame.columns:
        return None
    return pd.to_numeric(frame["framewise_displacement"], errors="coerce").to_numpy(dtype=np.float64)


def resolve_frames(n_rows: int, keep: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Retained rows of a series and the original frame number of every row.

    A series as long as the censor vector still contains the censored rows; a
    series as long as the number of kept frames was shortened by ``3dTproject
    -cenmode KILL``: every row is retained and row k is the k-th kept frame, so
    frame-wise covariates (FD) must be subset with the returned frame numbers.
    """
    if keep is None:
        return np.ones(n_rows, dtype=bool), np.arange(n_rows)
    keep = np.asarray(keep, dtype=bool).ravel()
    if keep.size == n_rows:
        return keep.copy(), np.arange(n_rows)
    if int(keep.sum()) == n_rows:
        return np.ones(n_rows, dtype=bool), np.flatnonzero(keep)
    raise ValueError(
        f"series has {n_rows} rows; the censor vector has {keep.size} entries of which "
        f"{int(keep.sum())} are kept: neither length matches"
    )


# ----------------------------------------------------------------------------
# ROI annotation: network, hemisphere, centroid
# ----------------------------------------------------------------------------

def network_of(name: str) -> str:
    """``7Networks_LH_Vis_1`` -> ``Vis``; anything else -> ''."""
    match = re.match(r"^\d+Networks_[LR]H_([A-Za-z]+)", str(name))
    return match.group(1) if match else ""


def hemisphere_of(name: str) -> str:
    text = f"_{name}_"
    if "_LH_" in text:
        return "L"
    if "_RH_" in text:
        return "R"
    return ""


def read_labels(path: Path) -> pd.DataFrame | None:
    """``labels.tsv`` (index, name[, network]) read as plain strings; None when absent/unusable."""
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
        frame.columns = [str(c).strip() for c in frame.columns]
        table = pd.DataFrame({
            "index": np.rint(frame["index"].astype(float)).astype(np.int64).to_numpy(),
            "name": [" ".join(str(n).split()) for n in frame["name"]],
        })
    except (KeyError, ValueError, pd.errors.ParserError) as err:
        log(f"WARNING: cannot use {path}: {err}")
        return None
    network = frame["network"].tolist() if "network" in frame.columns else [""] * len(frame)
    table["network"] = ["" if str(n).strip().lower() in ("", NA, "nan", "none") else str(n).strip() for n in network]
    table = table[table["index"] > 0].drop_duplicates("index")
    return table.sort_values("index").reset_index(drop=True)


def column_order(reference, candidate):
    """Explicit one-to-one ROI-name alignment; never infer identity from position."""
    reference, candidate = list(map(str, reference)), list(map(str, candidate))
    if len(set(reference)) != len(reference) or len(set(candidate)) != len(candidate):
        raise ValueError("duplicate ROI identities")
    if set(reference) != set(candidate):
        raise ValueError("ROI identities differ; an explicit atlas mapping is required")
    lookup = {name: i for i, name in enumerate(candidate)}
    return np.array([lookup[name] for name in reference], dtype=int)


def aligned_coverage(columns, coverage):
    if coverage is None:
        return None
    if "name" not in coverage:
        raise ValueError("coverage lacks explicit ROI names; cannot align by position")
    return coverage.iloc[column_order(columns, coverage["name"])].reset_index(drop=True)


def roi_table(columns: Sequence[str], labels: pd.DataFrame | None, coverage: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per time-series column: name, index (atlas label, -1 = unknown), network, hemi.

    The atlas label comes from the stage-07 coverage table when it has one row
    per column and a numeric ``roi`` column, otherwise from ``labels.tsv`` by
    name, otherwise by position when the label table has as many rows as there
    are columns.
    """
    names = [str(c) for c in columns]
    coverage = aligned_coverage(names, coverage)
    n_roi = len(names)
    index = np.full(n_roi, -1, dtype=np.int64)
    from_coverage = None
    if coverage is not None and len(coverage) == n_roi and "roi" in coverage.columns:
        from_coverage = pd.to_numeric(coverage["roi"], errors="coerce")
    if from_coverage is not None and bool((from_coverage > 0).all()):
        index = np.rint(from_coverage.to_numpy(dtype=np.float64)).astype(np.int64)
    elif labels is not None:
        by_name = {name: int(idx) for idx, name in zip(labels["index"], labels["name"])}
        matched = [by_name.get(" ".join(n.split()), -1) for n in names]
        if all(m > 0 for m in matched):
            index = np.array(matched, dtype=np.int64)

    label_name: dict[int, str] = {}
    label_net: dict[int, str] = {}
    if labels is not None:
        label_name = {int(i): str(n) for i, n in zip(labels["index"], labels["name"])}
        label_net = {int(i): str(n) for i, n in zip(labels["index"], labels["network"])}
    networks, hemis = [], []
    for name, idx in zip(names, index):
        ref = label_name.get(int(idx), name)
        networks.append(label_net.get(int(idx), "") or network_of(ref) or network_of(name))
        hemis.append(hemisphere_of(ref) or hemisphere_of(name))
    return pd.DataFrame({"name": names, "index": index, "network": networks, "hemi": hemis})


def compute_centroids(img: nib.spatialimages.SpatialImage) -> pd.DataFrame:
    """Centre of mass of every positive label, in world (mm) coordinates."""
    data = np.rint(np.asanyarray(img.dataobj)).astype(np.int64)
    data = data.reshape(data.shape[:3])
    voxels = np.argwhere(data > 0)
    values = data[data > 0]
    labels, inverse = np.unique(values, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    ijk = np.stack([np.bincount(inverse, weights=voxels[:, k]) / counts for k in range(3)], axis=1)
    xyz = nib.affines.apply_affine(img.affine, ijk)
    return pd.DataFrame({"index": labels, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})


def atlas_centroids(atlas_dir: Path, atlas: str) -> pd.DataFrame | None:
    """Parcel centroids of ``<atlas-dir>/<A>/<A>_space-..._dseg.nii.gz``, cached next to it."""
    base = Path(atlas_dir) / atlas
    volume = base / f"{atlas}_space-{ATLAS_SPACE}_res-02_dseg.nii.gz"
    cache = base / "centroids.tsv"
    if cache.is_file() and (not volume.is_file() or cache.stat().st_mtime >= volume.stat().st_mtime):
        try:
            table = read_tsv(cache)
            if {"index", "x", "y", "z"} <= set(table.columns) and len(table):
                return table
        except (OSError, ValueError, pd.errors.ParserError) as err:
            log(f"WARNING: unreadable centroid cache {cache}: {err}")
    if not volume.is_file():
        return None
    try:
        table = compute_centroids(nib.load(str(volume)))
    except Exception as err:  # noqa: BLE001 - a damaged atlas volume only costs homotopic_contrast
        log(f"WARNING: cannot compute centroids from {volume}: {err}")
        return None
    try:
        write_tsv(cache, table, float_format="%.3f")
    except OSError as err:   # read-only resource directory: the table lives in memory only
        log(f"centroid cache not written ({err}); computed in memory")
    return table


def roi_centroids(table: pd.DataFrame, centroids: pd.DataFrame | None) -> np.ndarray:
    """(R, 3) world coordinates of the rows of `table` (NaN = unknown)."""
    out = np.full((len(table), 3), np.nan)
    if centroids is None:
        return out
    lookup = {int(i): (x, y, z) for i, x, y, z in zip(centroids["index"], centroids["x"], centroids["y"], centroids["z"])}
    for row, idx in enumerate(table["index"]):
        if int(idx) in lookup:
            out[row] = lookup[int(idx)]
    return out


def find_homotopic_pairs(
    hemis: Sequence[str], networks: Sequence[str], centroids: np.ndarray, max_mm: float = HOMOTOPIC_MAX_MM
) -> list[tuple[int, int]]:
    """For each LH parcel the RH parcel of the same network whose centroid is nearest
    to the mirrored LH centroid (x -> -x), accepted within `max_mm`."""
    hemis = np.asarray(list(hemis), dtype=object)
    networks = np.asarray(list(networks), dtype=object)
    known = np.isfinite(centroids).all(axis=1)
    right = np.flatnonzero((hemis == "R") & known)
    pairs: list[tuple[int, int]] = []
    for left in np.flatnonzero((hemis == "L") & known):
        candidates = right[networks[right] == networks[left]]
        if candidates.size == 0:
            continue
        mirrored = centroids[left] * np.array([-1.0, 1.0, 1.0])
        distance = np.linalg.norm(centroids[candidates] - mirrored, axis=1)
        best = int(np.argmin(distance))
        if distance[best] <= max_mm:
            pairs.append((int(left), int(candidates[best])))
    return pairs


# ----------------------------------------------------------------------------
# metrics (pure functions; series are (T, R) arrays, keep vectors are boolean)
# ----------------------------------------------------------------------------

def _finite(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def nan_quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else float("nan")


def valid_rois(series: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Columns that are finite in every row and not constant over retained frames."""
    series = np.asarray(series, dtype=np.float64)
    finite = np.isfinite(series).all(axis=0)
    kept = series[np.asarray(keep, dtype=bool)]
    spread = np.zeros(series.shape[1])
    if kept.shape[0] >= 2 and finite.any():
        spread[finite] = kept[:, finite].std(axis=0)
    return finite & (spread > 0)


def detrend(series: np.ndarray, keep: np.ndarray, order: int = DEFAULT_POLORT) -> np.ndarray:
    """Remove a Legendre trend of `order`, fitted on retained frames, from every frame."""
    series = np.asarray(series, dtype=np.float64)
    keep = np.asarray(keep, dtype=bool)
    n_frames = series.shape[0]
    if keep.sum() < order + 2:
        keep, order = np.ones(n_frames, dtype=bool), 0
    x = np.linspace(-1.0, 1.0, n_frames) if n_frames > 1 else np.zeros(1)
    design = np.polynomial.legendre.legvander(x, order)
    beta = np.linalg.pinv(design[keep]) @ series[keep]
    return series - design @ beta


def roi_tsnr(pre: np.ndarray, post: np.ndarray, keep_pre: np.ndarray, keep_post: np.ndarray) -> np.ndarray:
    """mean(pre-denoise ROI series) / SD(denoised ROI series), both over retained frames.

    The final series is zero-mean, so its own mean/SD is meaningless; the signal
    level comes from the scaled pre-denoise series.
    """
    pre = np.asarray(pre, dtype=np.float64)[np.asarray(keep_pre, dtype=bool)]
    post = np.asarray(post, dtype=np.float64)[np.asarray(keep_post, dtype=bool)]
    out = np.full(post.shape[1], np.nan)
    if pre.shape[0] < 1 or post.shape[0] < 3:
        return out
    level = pre.mean(axis=0)
    noise = post.std(axis=0, ddof=1)
    ok = np.isfinite(level) & np.isfinite(noise) & (noise > 0) & (level > 0)
    out[ok] = level[ok] / noise[ok]
    return out


def variance_removed(
    pre: np.ndarray, post: np.ndarray, keep_pre: np.ndarray, keep_post: np.ndarray, polort: int = DEFAULT_POLORT
) -> np.ndarray:
    """1 - var(denoised) / var(pre-denoise, polort-detrended), over retained frames."""
    keep_pre = np.asarray(keep_pre, dtype=bool)
    post = np.asarray(post, dtype=np.float64)[np.asarray(keep_post, dtype=bool)]
    out = np.full(post.shape[1], np.nan)
    if keep_pre.sum() < 3 or post.shape[0] < 3:
        return out
    before = detrend(pre, keep_pre, polort)[keep_pre].var(axis=0, ddof=1)
    after = post.var(axis=0, ddof=1)
    ok = np.isfinite(before) & np.isfinite(after) & (before > 0)
    out[ok] = 1.0 - after[ok] / before[ok]
    return out


def network_contrast(fc_z: np.ndarray, networks: Sequence[str]) -> float:
    """(mean z within network - mean z between networks) / SD(between)."""
    labels = np.asarray([n if isinstance(n, str) else "" for n in networks], dtype=object)
    if len({n for n in labels if n}) < 2:
        return float("nan")
    rows, cols = np.triu_indices(fc_z.shape[0], k=1)
    values = fc_z[rows, cols]
    known = (labels[rows] != "") & (labels[cols] != "") & np.isfinite(values)
    same = labels[rows] == labels[cols]
    within, between = values[known & same], values[known & ~same]
    if within.size < 2 or between.size < 2:
        return float("nan")
    spread = between.std(ddof=1)
    if not spread > 0:
        return float("nan")
    return _finite((within.mean() - between.mean()) / spread)


def homotopic_contrast(fc_z: np.ndarray, hemis: Sequence[str], pairs: Sequence[tuple[int, int]], min_pairs: int = 3) -> float:
    """Mean z of homotopic pairs minus mean z of the other inter-hemispheric pairs."""
    hemis = np.asarray(list(hemis), dtype=object)
    inter = np.outer(hemis == "L", hemis == "R")
    homo = np.zeros_like(inter)
    for left, right in pairs:
        homo[left, right] = True
    finite = np.isfinite(fc_z)
    homotopic = fc_z[homo & inter & finite]
    other = fc_z[inter & ~homo & finite]
    if homotopic.size < min_pairs or other.size < min_pairs:
        return float("nan")
    return _finite(homotopic.mean() - other.mean())


def dmn_contrast(fc_z: np.ndarray, names: Sequence[str], networks: Sequence[str]) -> float:
    """Mean z between the posterior (PCC / pCunPCC) and the prefrontal (PFC) parcels of
    the Default network, minus the mean z of those parcels with SomMot parcels."""
    names = [str(n) for n in names]
    default = np.array([str(n).startswith("Default") for n in networks])
    pcc = default & np.array(["PCC" in n for n in names])
    pfc = default & np.array(["PFC" in n for n in names]) & ~pcc
    sommot = np.array([str(n).startswith("SomMot") for n in networks])
    if not (pcc.any() and pfc.any() and sommot.any()):
        return float("nan")
    core = fc_z[np.ix_(pcc, pfc)]
    control = fc_z[np.ix_(pcc | pfc, sommot)]
    core, control = core[np.isfinite(core)], control[np.isfinite(control)]
    if core.size == 0 or control.size == 0:
        return float("nan")
    return _finite(core.mean() - control.mean())


def spectral_power(series: np.ndarray, times: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Least-squares (Lomb-Scargle) power of the columns of `series` sampled at `times`.

    Returns (F, R): the sum of squares explained by a sinusoid of each frequency.
    For complete regular sampling on the Fourier grid this is the periodogram;
    with censored frames it needs no interpolation, which would add low-frequency power.
    """
    phase = 2.0 * np.pi * np.outer(freqs, times)
    cos, sin = np.cos(phase), np.sin(phase)
    cc, ss, cs = (cos * cos).sum(axis=1), (sin * sin).sum(axis=1), (cos * sin).sum(axis=1)
    yc, ys = cos @ series, sin @ series
    det = cc * ss - cs * cs
    det[det <= 1e-12 * np.maximum(cc * ss, 1e-300)] = np.nan
    return (yc ** 2 * ss[:, None] - 2.0 * yc * ys * cs[:, None] + ys ** 2 * cc[:, None]) / det[:, None]


def lowfreq_power_fraction(
    pre: np.ndarray, tr: float, keep: np.ndarray | None = None,
    polort: int = DEFAULT_POLORT, band: tuple[float, float] = BAND_HZ,
) -> float:
    """Median over ROIs of power(band) / power(band[0] .. Nyquist) of the detrended
    pre-denoise series. White noise gives (band[1]-band[0]) / (Nyquist-band[0])."""
    pre = np.asarray(pre, dtype=np.float64)
    n_frames = pre.shape[0]
    keep = np.ones(n_frames, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    if not tr > 0 or keep.sum() < 2 * MIN_FRAMES or pre.shape[1] == 0:
        return float("nan")
    data = detrend(pre, keep, polort)[keep]
    times = np.flatnonzero(keep) * float(tr)
    freqs = np.arange(1, (n_frames - 1) // 2 + 1) / (n_frames * float(tr))   # Fourier grid, Nyquist excluded
    freqs = freqs[freqs >= band[0]]
    in_band = freqs <= band[1]
    if not in_band.any() or in_band.all():
        return float("nan")
    power = spectral_power(data, times, freqs)
    total = np.nansum(power, axis=0)
    fraction = np.full(pre.shape[1], np.nan)
    ok = total > 0
    fraction[ok] = np.nansum(power[in_band], axis=0)[ok] / total[ok]
    return nan_quantile(fraction, 0.5)


def zscore_columns(series: np.ndarray) -> np.ndarray:
    """Columns centred and scaled to unit SD; a constant (or non-finite) column becomes NaN."""
    series = np.asarray(series, dtype=np.float64)
    sd = series.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (series - series.mean(axis=0)) / np.where(sd > 0, sd, np.nan)


def finite_columns(z: np.ndarray) -> np.ndarray:
    return z[:, np.isfinite(z).all(axis=0)]


def cofluctuation_rss(z: np.ndarray) -> np.ndarray:
    """Frame-wise root sum of squares of the edge time series z_i(t) * z_j(t), i < j.

    sum_{i<j} (z_i z_j)^2 = ((sum_i z_i^2)^2 - sum_i z_i^4) / 2, so the (T, R, R)
    edge array is never built.
    """
    squares = z ** 2
    return np.sqrt(np.maximum(squares.sum(axis=1) ** 2 - (squares ** 2).sum(axis=1), 0.0) / 2.0)


def fd_fc_coupling(post: np.ndarray, keep: np.ndarray, fd: np.ndarray | None) -> float:
    """|Spearman rho| between FD and the co-fluctuation amplitude over retained frames.

    `fd` has one value per ROW of `post` (already subset for KILL-mode series).
    """
    if fd is None:
        return float("nan")
    post = np.asarray(post, dtype=np.float64)
    fd = np.asarray(fd, dtype=np.float64).ravel()
    if fd.size != post.shape[0]:
        raise ValueError(f"FD has {fd.size} values for a series of {post.shape[0]} rows")
    keep = np.asarray(keep, dtype=bool)
    if post.shape[1] < 2 or keep.sum() < MIN_FRAMES:
        return float("nan")
    z = finite_columns(zscore_columns(post[keep]))
    if z.shape[1] < 2:
        return float("nan")
    amplitude = cofluctuation_rss(z)
    motion = fd[keep]
    ok = np.isfinite(motion) & np.isfinite(amplitude)
    if ok.sum() < MIN_FRAMES or np.ptp(motion[ok]) == 0 or np.ptp(amplitude[ok]) == 0:
        return float("nan")
    rho = stats.spearmanr(motion[ok], amplitude[ok])[0]
    return _finite(abs(rho))


def gs_residual_sd(post: np.ndarray, keep: np.ndarray) -> float:
    """SD over retained frames of the mean over z-scored ROIs (1/sqrt(R) if ROIs were
    independent, 1 if they were identical): what is left of a global signal."""
    post = np.asarray(post, dtype=np.float64)[np.asarray(keep, dtype=bool)]
    if post.shape[0] < 3 or post.shape[1] < 2:
        return float("nan")
    z = finite_columns(zscore_columns(post))
    if z.shape[1] < 2:
        return float("nan")
    return _finite(z.mean(axis=1).std(ddof=1))


def fc_similarity(fc_z_a: np.ndarray, fc_z_b: np.ndarray) -> float:
    """Pearson r between the Fisher-z upper triangles of two FC matrices."""
    if fc_z_a.shape != fc_z_b.shape or fc_z_a.shape[0] < 3:
        return float("nan")
    a, b = upper_triangle(fc_z_a), upper_triangle(fc_z_b)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return _finite(np.corrcoef(a[ok], b[ok])[0, 1])


# ----------------------------------------------------------------------------
# one stream x strategy x atlas
# ----------------------------------------------------------------------------

class StreamData:
    """Everything needed to score one stream x strategy x atlas."""

    def __init__(
        self, post: np.ndarray, pre: np.ndarray | None, keep_post: np.ndarray, keep_pre: np.ndarray | None,
        fd_rows: np.ndarray | None, table: pd.DataFrame, centroids: np.ndarray,
        pre_same_stream: bool = True,
    ) -> None:
        self.post = post
        self.pre = pre
        self.keep_post = keep_post
        self.keep_pre = keep_pre
        self.fd_rows = fd_rows
        self.table = table
        self.centroids = centroids
        self.pre_same_stream = pre_same_stream
        self.valid = valid_rois(post, keep_post)

    def tsnr(self) -> np.ndarray:
        """Per-ROI tSNR for every column (NaN where undefined)."""
        out = np.full(self.post.shape[1], np.nan)
        if self.pre is None or self.keep_pre is None:
            return out
        ok = self.valid & np.isfinite(self.pre).all(axis=0)
        out[ok] = roi_tsnr(self.pre[:, ok], self.post[:, ok], self.keep_pre, self.keep_post)
        return out


def stream_metrics(
    data: StreamData, tr: float, polort: int = DEFAULT_POLORT, roi_mask: np.ndarray | None = None,
    max_mm: float = HOMOTOPIC_MAX_MM,
) -> dict[str, float]:
    """All section-12 metrics of one stream; `roi_mask` restricts them to a subset
    of the valid ROIs (the ROIs shared by both streams). NaN ROIs never enter FC."""
    use = data.valid if roi_mask is None else (data.valid & np.asarray(roi_mask, dtype=bool))
    table = data.table.loc[use].reset_index(drop=True)
    post = data.post[:, use]
    keep = data.keep_post
    out = {name: float("nan") for name in METRICS}
    out["n_roi_nan"] = float((~data.valid).sum())
    out["n_retained"] = float(keep.sum())
    if post.shape[1] == 0:
        return out

    tsnr = data.tsnr()[use]
    out["roi_tsnr_median"] = nan_quantile(tsnr, 0.5)
    out["roi_tsnr_p10"] = nan_quantile(tsnr, 0.10)
    if data.pre is not None and data.keep_pre is not None and data.pre_same_stream:
        pre = data.pre[:, use]
        pre_ok = np.isfinite(pre).all(axis=0)
        if pre_ok.any():
            removed = variance_removed(pre[:, pre_ok], post[:, pre_ok], data.keep_pre, keep, polort)
            out["variance_removed_median"] = nan_quantile(removed, 0.5)
            out["lowfreq_power_fraction"] = lowfreq_power_fraction(pre[:, pre_ok], tr, data.keep_pre, polort)

    if post.shape[1] >= 2:
        fc_z = fisher_z(fc_matrix(post, keep))
        names, networks, hemis = table["name"].tolist(), table["network"].tolist(), table["hemi"].tolist()
        out["split_half_r"] = _finite(split_half_reliability(post, keep))
        out["network_contrast"] = network_contrast(fc_z, networks)
        pairs = find_homotopic_pairs(hemis, networks, data.centroids[use], max_mm)
        out["homotopic_contrast"] = homotopic_contrast(fc_z, hemis, pairs)
        out["dmn_contrast"] = dmn_contrast(fc_z, names, networks)
        out["fd_fc_coupling"] = fd_fc_coupling(post, keep, data.fd_rows)
        out["gs_residual_sd"] = gs_residual_sd(post, keep)
    return out


def compare_streams(
    volume: StreamData, surface: StreamData, tr: float, polort: int = DEFAULT_POLORT,
    per_stream: dict[str, dict[str, float]] | None = None,
) -> dict | None:
    """Paired comparison over valid ROIs, explicitly aligned by unique ROI names.

    Raises ValueError when the identities cannot be matched without guessing.
    """
    order = column_order(volume.table["name"], surface.table["name"])
    surface = StreamData(surface.post[:, order], None if surface.pre is None else surface.pre[:, order],
                         surface.keep_post, surface.keep_pre, surface.fd_rows,
                         surface.table.iloc[order].reset_index(drop=True), surface.centroids[order],
                         surface.pre_same_stream)
    common = volume.valid & surface.valid
    scores = {
        "volume": stream_metrics(volume, tr, polort, roi_mask=common),
        "surface": stream_metrics(surface, tr, polort, roi_mask=common),
    }
    for stream in STREAMS:       # counts describe the stream, not the common ROI set
        for name in PER_STREAM_ONLY:
            if per_stream is not None and stream in per_stream:
                scores[stream][name] = per_stream[stream].get(name, float("nan"))
    similarity = float("nan")
    if common.sum() >= 3:
        similarity = fc_similarity(
            fisher_z(fc_matrix(volume.post[:, common], volume.keep_post)),
            fisher_z(fc_matrix(surface.post[:, common], surface.keep_post)),
        )
    roi = pd.DataFrame({
        "roi": volume.table["name"].to_numpy(),
        "volume": volume.tsnr(),
        "surface": surface.tsnr(),
    })
    return {"fc_similarity": similarity, "n_roi_common": int(common.sum()), "metrics": scores, "roi_tsnr": roi}


# ----------------------------------------------------------------------------
# run level
# ----------------------------------------------------------------------------

def table_paths(func_dir: Path, run: str, stream: str, template: str, atlas: str, strategy: str) -> dict[str, Path]:
    space = template if stream == "volume" else "fsLR"
    stem = f"{run}_space-{space}_atlas-{atlas}"
    return {
        "post": func_dir / f"{stem}_desc-{strategy}_timeseries.tsv",
        "pre": func_dir / f"{stem}_desc-preproc_timeseries.tsv",
        "coverage": func_dir / f"{stem}_desc-{strategy}_coverage.tsv",
    }


def load_stream(
    func_dir: Path, run: str, stream: str, template: str, atlas: str, strategy: str,
    keep: np.ndarray | None, fd: np.ndarray | None, labels: pd.DataFrame | None, centroids: pd.DataFrame | None,
    notes: list[str],
) -> StreamData | None:
    """Tables of one stream x strategy x atlas; None when the denoised table is absent."""
    paths = table_paths(func_dir, run, stream, template, atlas, strategy)
    post_frame = read_series(paths["post"])
    if post_frame is None:
        return None
    tag = f"{stream}/{strategy}/{atlas}"
    post = post_frame.to_numpy(dtype=np.float64)
    keep_post, frames = resolve_frames(post.shape[0], keep)

    fd_rows = None
    if fd is not None:
        # one FD value per acquired frame: the length of the censor vector, or of
        # the series itself when nothing was censored (a longer table would be
        # silently misaligned, so only an exact match is accepted)
        expected = int(keep.size) if keep is not None else post.shape[0]
        if fd.size == expected:
            fd_rows = fd[frames]
        else:
            notes.append(f"{tag}: FD has {fd.size} values, expected {expected} (one per acquired frame): fd_fc_coupling is n/a")

    pre, keep_pre, same_stream = None, None, True
    pre_frame = read_series(paths["pre"])
    if pre_frame is None and stream == "surface":
        # the scaled volume series has the same signal level (one global scale
        # factor); good enough as tSNR numerator, not for variance ratios
        pre_frame = read_series(table_paths(func_dir, run, "volume", template, atlas, strategy)["pre"])
        if pre_frame is not None:
            same_stream = False
            notes.append(f"{tag}: no surface pre-denoise table; roi_tsnr uses the volume pre-denoise ROI means, "
                         "variance_removed and lowfreq_power_fraction are n/a")
    if pre_frame is not None:
        if pre_frame.shape[1] != post.shape[1]:
            notes.append(f"{tag}: pre-denoise table has {pre_frame.shape[1]} columns, denoised table {post.shape[1]}: ignored")
        else:
            pre = pre_frame.iloc[:, column_order(post_frame.columns, pre_frame.columns)].to_numpy(dtype=np.float64)
            try:
                keep_pre, _ = resolve_frames(pre.shape[0], keep)
            except ValueError as err:
                notes.append(f"{tag}: pre-denoise table unusable: {err}")
                pre = None
    else:
        notes.append(f"{tag}: no pre-denoise table: roi_tsnr, variance_removed and lowfreq_power_fraction are n/a")

    coverage = read_tsv(paths["coverage"]) if paths["coverage"].is_file() else None
    table = roi_table(list(post_frame.columns), labels, coverage)
    return StreamData(post, pre, keep_post, keep_pre, fd_rows, table, roi_centroids(table, centroids), same_stream)


def validate_run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    func_dir = Path(args.func_dir)
    run = args.run
    strategies, atlases = split_list(args.strategies), split_list(args.atlases)
    notes: list[str] = []
    keep = read_keep(func_dir / f"{run}_desc-censor.1D")
    if keep is None:
        raise ValueError("required censor vector is missing; retained frames are unknown: run stage 04 first")
    fd = read_fd(func_dir / f"{run}_desc-confounds_timeseries.tsv")
    if fd is None:
        notes.append("no framewise_displacement column: fd_fc_coupling is n/a")
    if keep is not None and args.censor_mode.upper() == "KILL":
        log(f"censor mode KILL: series are expected to have {int(keep.sum())} rows")

    rows: list[tuple] = []
    compare_rows: list[tuple] = []
    nested: dict = {stream: {} for stream in STREAMS}
    nested_compare: dict = {}
    for atlas in atlases:
        labels = read_labels(Path(args.atlas_dir) / atlas / "labels.tsv") if args.atlas_dir else None
        centroids = atlas_centroids(Path(args.atlas_dir), atlas) if args.atlas_dir else None
        if labels is None:
            notes.append(f"{atlas}: no labels.tsv; network/hemisphere taken from the column names when they follow the Schaefer pattern")
        for strategy in strategies:
            denoise = read_json(func_dir / f"{run}_desc-{strategy}_denoise.json", default={})
            polort = int(denoise["polort"]) if isinstance(denoise.get("polort"), (int, float)) else DEFAULT_POLORT
            polort = max(polort, 0)
            dof = denoise.get("dof_remaining")
            loaded: dict[str, StreamData] = {}
            scored: dict[str, dict[str, float]] = {}
            for stream in STREAMS:
                try:
                    data = load_stream(func_dir, run, stream, args.template, atlas, strategy, keep, fd, labels, centroids, notes)
                    if data is None:
                        continue
                    metrics = stream_metrics(data, args.tr, polort)
                except (ValueError, OSError, pd.errors.ParserError) as err:
                    notes.append(f"{stream}/{strategy}/{atlas}: skipped: {err}")
                    continue
                metrics["dof_remaining"] = float(dof) if isinstance(dof, (int, float)) else float("nan")
                loaded[stream], scored[stream] = data, metrics
                nested[stream].setdefault(strategy, {})[atlas] = metrics
                rows.extend((stream, strategy, atlas, name, metrics[name]) for name in METRICS)

            if len(loaded) < 2:
                continue
            try:
                result = compare_streams(loaded["volume"], loaded["surface"], args.tr, polort, per_stream=scored)
            except ValueError as err:
                notes.append(f"{strategy}/{atlas}: streams not comparable: {err}")
                continue
            if result is None:
                notes.append(f"{strategy}/{atlas}: volume has {loaded['volume'].post.shape[1]} columns, surface "
                             f"{loaded['surface'].post.shape[1]}: streams not compared")
                continue
            for name in ("fc_similarity", "n_roi_common"):
                compare_rows.append((strategy, atlas, "run", NA, name, np.nan, np.nan, float(result[name])))
            paired = {}
            for name in METRICS:
                vol, surf = result["metrics"]["volume"][name], result["metrics"]["surface"][name]
                paired[name] = {"volume": vol, "surface": surf, "difference": surf - vol}
                compare_rows.append((strategy, atlas, "run", NA, name, vol, surf, surf - vol))
            for roi, vol, surf in result["roi_tsnr"].itertuples(index=False):
                compare_rows.append((strategy, atlas, "roi", roi, "roi_tsnr", vol, surf, surf - vol))
            nested_compare.setdefault(strategy, {})[atlas] = {
                "fc_similarity": result["fc_similarity"], "n_roi_common": result["n_roi_common"], "metrics": paired,
            }

    for note in notes:
        log(f"NOTE: {note}")
    table = pd.DataFrame(rows, columns=["stream", "strategy", "atlas", "metric", "value"])
    compare = pd.DataFrame(compare_rows, columns=["strategy", "atlas", "scope", "roi", "metric", "volume", "surface", "value"])
    info = {
        "run": run,
        "template": args.template,
        "tr": args.tr,
        "censor_mode": args.censor_mode,
        "n_volumes": None if keep is None else int(keep.size),
        "n_censored": None if keep is None else int((~keep).sum()),
        "band_hz": list(BAND_HZ),
        "lowfreq_white_noise_baseline": (
            (BAND_HZ[1] - BAND_HZ[0]) / (0.5 / args.tr - BAND_HZ[0]) if args.tr > 0 and 0.5 / args.tr > BAND_HZ[1] else None
        ),
        "homotopic_max_mm": HOMOTOPIC_MAX_MM,
        "streams": {stream: block for stream, block in nested.items() if block},
        "stream_comparison": nested_compare,
        "notes": notes,
    }
    return table, info, compare


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fmriproc.validate", description=__doc__.split("\n")[0])
    parser.add_argument("--func-dir", required=True, help="derivatives/sub-X/func")
    parser.add_argument("--run", required=True, help="BIDS prefix of the run (<RUN>)")
    parser.add_argument("--template", required=True, help="template name of the volume stream (space-<TPL>)")
    parser.add_argument("--strategies", required=True, help="space separated denoising strategies")
    parser.add_argument("--atlases", required=True, help="space separated atlas names")
    parser.add_argument("--atlas-dir", default="", help="$RESOURCE_DIR/atlases (labels.tsv, dseg volume, centroid cache)")
    parser.add_argument("--tr", type=float, required=True, help="repetition time in seconds")
    parser.add_argument("--censor-mode", default="NTRP", type=str.upper, choices=("NTRP", "KILL", "ZERO"))
    parser.add_argument("--out-tsv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-compare-tsv", default="", help="written only when both streams exist")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.func_dir).is_dir():
        log(f"ERROR: not a directory: {args.func_dir}")
        return 1
    try:
        table, info, compare = validate_run(args)
    except (OSError, ValueError) as err:
        log(f"ERROR: {err}")
        return 1
    if table.empty:
        log(f"ERROR: no ROI time-series table found for {args.run} in {args.func_dir} "
            f"(strategies: {args.strategies}; atlases: {args.atlases}): run stage 07 first")
        return 1
    write_tsv(args.out_tsv, table)
    write_json(args.out_json, info)
    if args.out_compare_tsv:
        if not compare.empty:
            write_tsv(args.out_compare_tsv, compare)
        elif os.path.isfile(args.out_compare_tsv):
            # a comparison of an earlier configuration would be picked up by the group step
            os.unlink(args.out_compare_tsv)
            log(f"removed stale {args.out_compare_tsv}: only one stream is present now")
    streams = sorted(set(table["stream"]))
    log(f"{args.run}: {len(table)} values, streams: {', '.join(streams)}"
        + (f", {int((compare['scope'] == 'run').sum())} paired values" if not compare.empty else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
