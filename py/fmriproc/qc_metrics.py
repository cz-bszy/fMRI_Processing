"""Per-run QC metrics (docs/DESIGN.md section 9).

Writes ``<RUN>_desc-qc_metrics.json`` (flat keys) and one
``<RUN>_atlas-<A>_desc-<S>_roiqc.tsv`` per strategy/atlas. Every optional input
that is missing degrades to a null value; nothing here may stop the report.

Signal-to-noise definitions (the final series are zero-mean, so mean/SD of the
final series would be meaningless):

* ``tsnr_*_median``        mean / SD of the polort-2 detrended pre-denoise series
* ``S.tsnr_gm_median_post`` mean of the pre-denoise series / SD of the denoised series
* ``S.A.roi_tsnr_*``       the same ratio on ROI-mean series
* ``variance_removed``     1 - var(denoised) / var(detrended pre-denoise)

All of them use retained (uncensored) frames only.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import nibabel as nib
import numpy as np
import pandas as pd

from fmriproc import utils

LOG = logging.getLogger("fmriproc.qc_metrics")

QC_VERSION = "1"
POLORT = 2
CARPET_MAX_ROWS = 6000
CHUNK_BYTES = 256e6
TISSUE_CODES = {"GM": 1, "WM": 2, "CSF": 3}

# metric -> (direction in which the metric gets worse, default warn, default fail)
THRESHOLDS: dict[str, tuple[str, float, float]] = {
    "fd_mean": ("high", 0.2, 0.5),
    "pct_censored": ("high", 20.0, 50.0),
    "tsnr_gm_median": ("low", 40.0, 20.0),
    "coreg_dice": ("low", 0.90, 0.80),
    "norm_dice": ("low", 0.93, 0.88),
    "holes_total": ("high", 100.0, 200.0),
}
# metric -> stem of the command line option / QC_* config variable
THRESHOLD_OPTIONS = {
    "fd_mean": "fd-mean",
    "pct_censored": "pct-censored",
    "tsnr_gm_median": "tsnr-gm",
    "coreg_dice": "coreg-dice",
    "norm_dice": "norm-dice",
    "holes_total": "euler-holes",
}
FLAG_RANK = {"n/a": -1, "pass": 0, "warn": 1, "incomplete": 2, "fail": 3}

STRATEGY_KEYS = (
    "n_regressors", "dof_remaining", "low_dof", "tsnr_gm_median_post", "tsnr_pre_same_voxels",
    "tsnr_gain", "variance_removed_gm", "fd_dvars_corr_post", "tsnr_post_mask",
)
ATLAS_KEYS = (
    "n_roi", "roi_tsnr_median", "roi_tsnr_min", "roi_tsnr_pre_median", "roi_variance_removed_median",
    "n_roi_nan", "coverage_min", "split_half_r", "network_contrast", "fc_mean", "fc_sd",
    "surf_split_half_r", "vol_surf_fc_r",
)


# ----------------------------------------------------------------------------
# file names (docs/DESIGN.md section 6)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class RunPaths:
    """Names of every product of one run, as fixed by the design contract."""

    func_dir: Path
    anat_dir: Path
    run: str
    template: str
    res: str

    @property
    def subject(self) -> str:
        return self.run.split("_")[0]

    def func(self, suffix: str) -> Path:
        return self.func_dir / f"{self.run}_{suffix}"

    def tpl(self, suffix: str) -> Path:
        """Template-space product; the zero-padded BIDS spelling (res-02) is accepted too."""
        path = self.func(f"space-{self.template}_res-{self.res}_{suffix}")
        padded = self.func(f"space-{self.template}_res-0{self.res}_{suffix}")
        return padded if not path.exists() and padded.exists() else path

    def anat(self, suffix: str) -> Path:
        return self.anat_dir / f"{self.subject}_{suffix}"

    def timeseries(self, atlas: str, desc: str, kind: str = "timeseries") -> Path:
        return self.func(f"space-{self.template}_atlas-{atlas}_desc-{desc}_{kind}.tsv")

    def surf_timeseries(self, atlas: str, desc: str) -> Path:
        return self.func(f"space-fsLR_atlas-{atlas}_desc-{desc}_timeseries.tsv")

    def roiqc(self, atlas: str, strategy: str) -> Path:
        return self.func(f"atlas-{atlas}_desc-{strategy}_roiqc.tsv")

    @property
    def prep_info(self) -> Path:
        return self.func("desc-prep_info.json")

    @property
    def confounds(self) -> Path:
        return self.func("desc-confounds_timeseries.tsv")

    @property
    def confounds_json(self) -> Path:
        return self.func("desc-confounds_timeseries.json")

    @property
    def censor(self) -> Path:
        return self.func("desc-censor.1D")

    @property
    def metrics(self) -> Path:
        return self.func("desc-qc_metrics.json")

    @property
    def tsnr_dscalar(self) -> Path:
        return self.func("space-fsLR_den-91k_desc-preproc_tsnr.dscalar.nii")


@dataclass(frozen=True)
class CachePaths:
    """Private hand-over from qc_metrics to plots (work/sub-X/func/<RUN>/qc)."""

    root: Path

    @property
    def carpet(self) -> Path:
        return self.root / "carpet.npz"

    def tsnr(self, desc: str, space: str) -> Path:
        return self.root / f"tsnr_desc-{desc}_space-{space}.nii.gz"


# ----------------------------------------------------------------------------
# small numeric helpers
# ----------------------------------------------------------------------------

def _num(value: Any) -> float | None:
    """Finite float or None (JSON null)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _nanstat(func: Callable[[np.ndarray], float], values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return _num(func(values)) if values.size else None


def pearson(a: np.ndarray, b: np.ndarray, min_n: int = 5) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_n or a[ok].std() == 0 or b[ok].std() == 0:
        return None
    return _num(np.corrcoef(a[ok], b[ok])[0, 1])


def legendre_design(n_frames: int, order: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, n_frames) if n_frames > 1 else np.zeros(1)
    return np.polynomial.legendre.legvander(x, order)


def detrend_inplace(data: np.ndarray, retained: np.ndarray, order: int = POLORT) -> np.ndarray:
    """Remove a Legendre trend from the rows of ``data`` (V, T) in place.

    The trend is fitted on retained frames only, so motion spikes do not bend
    it, and removed from every frame. Returns the temporal mean (retained
    frames) taken before detrending.
    """
    n_frames = data.shape[1]
    retained = np.asarray(retained, dtype=bool)
    if retained.sum() < order + 2:
        retained = np.ones(n_frames, dtype=bool)
        order = 0
    design = legendre_design(n_frames, order)
    projector = np.linalg.pinv(design[retained])          # (order + 1, Tk)
    mean = np.empty(data.shape[0], dtype=np.float64)
    step = 50000
    for start in range(0, data.shape[0], step):
        block = data[start:start + step].astype(np.float64)
        kept = block[:, retained]
        mean[start:start + step] = kept.mean(axis=1)
        block -= (kept @ projector.T) @ design.T
        data[start:start + step] = block
    return mean


def sd_over(data: np.ndarray, retained: np.ndarray) -> np.ndarray:
    """Per-row SD (ddof=1, float64) over retained columns; 0 -> NaN."""
    retained = np.asarray(retained, dtype=bool)
    out = np.full(data.shape[0], np.nan, dtype=np.float64)
    if retained.sum() < 3:
        return out
    step = 50000
    for start in range(0, data.shape[0], step):
        block = data[start:start + step][:, retained].astype(np.float64)
        out[start:start + step] = block.std(axis=1, ddof=1)
    out[~(out > 0)] = np.nan
    return out


def gcor(detrended: np.ndarray, retained: np.ndarray) -> float | None:
    """Global correlation (Saad 2013): squared norm of the average unit-norm series."""
    retained = np.asarray(retained, dtype=bool)
    if retained.sum() < 5 or detrended.shape[0] < 2:
        return None
    total = np.zeros(int(retained.sum()), dtype=np.float64)
    count = 0
    step = 50000
    for start in range(0, detrended.shape[0], step):
        block = detrended[start:start + step][:, retained].astype(np.float64)
        block -= block.mean(axis=1, keepdims=True)
        norm = np.sqrt((block ** 2).sum(axis=1))
        good = norm > 0
        total += (block[good] / norm[good, None]).sum(axis=0)
        count += int(good.sum())
    if count < 2:
        return None
    return _num(float((total / count) @ (total / count)))


def dvars(data: np.ndarray) -> np.ndarray:
    """RMS over rows of the backward temporal difference; element 0 is NaN."""
    n_frames = data.shape[1]
    out = np.full(n_frames, np.nan, dtype=np.float64)
    for t in range(1, n_frames):
        diff = data[:, t].astype(np.float64) - data[:, t - 1]
        out[t] = np.sqrt(np.mean(diff ** 2)) if diff.size else np.nan
    return out


@dataclass(frozen=True)
class FrameMap:
    """Relation between the frames of a (possibly KILL-censored) series and the run."""

    index: np.ndarray      # original frame index of every frame of the series
    retained: np.ndarray   # True where the frame is uncensored

    @classmethod
    def build(cls, keep: np.ndarray, n_frames: int) -> "FrameMap":
        keep = np.asarray(keep, dtype=bool)
        if n_frames == keep.size:
            return cls(np.arange(n_frames), keep.copy())
        if n_frames == int(keep.sum()):
            return cls(np.flatnonzero(keep), np.ones(n_frames, dtype=bool))
        raise ValueError(f"series has {n_frames} frames, censor vector {keep.size} (kept {int(keep.sum())}): incompatible time axes")

    def adjacent_pairs(self) -> np.ndarray:
        """True at t when frames t-1 and t are both retained and consecutive in the run."""
        ok = np.zeros(self.index.size, dtype=bool)
        if self.index.size > 1:
            ok[1:] = (np.diff(self.index) == 1) & self.retained[1:] & self.retained[:-1]
        return ok


# ----------------------------------------------------------------------------
# image helpers
# ----------------------------------------------------------------------------

def load_mask(path: Path, like_shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        data = np.asanyarray(nib.load(str(path)).dataobj)
    except Exception as err:  # unreadable file = missing file
        LOG.warning("cannot read %s: %s", path, err)
        return None
    data = np.squeeze(data) > 0
    if like_shape is not None and data.shape != tuple(like_shape):
        LOG.warning("%s is on another grid %s (expected %s): ignored", path.name, data.shape, like_shape)
        return None
    return data


def load_masked(path: Path, mask: np.ndarray) -> np.ndarray:
    """4D image -> (V, T) float32, read in time chunks so that the full 4D array
    (about 1 GB in 2 mm template space) never sits in memory."""
    try:
        img = nib.load(str(path), keep_file_open=True)   # one pass through the gzip stream
    except TypeError:
        img = nib.load(str(path))
    shape = img.shape
    if len(shape) != 4 or tuple(shape[:3]) != mask.shape:
        raise ValueError(f"{path.name}: grid {shape} does not match the mask {mask.shape}")
    n_frames = shape[3]
    chunk = max(1, int(CHUNK_BYTES // (4 * int(np.prod(shape[:3])))))
    out = np.empty((int(mask.sum()), n_frames), dtype=np.float32)
    for start in range(0, n_frames, chunk):
        block = np.asanyarray(img.dataobj[..., start:start + chunk])
        out[:, start:start + chunk] = block[mask]
        del block
    if not np.isfinite(out).all():
        raise ValueError(f"{path.name}: nonfinite signal inside analysis mask")
    return out


def save_map(values: np.ndarray, mask: np.ndarray, like: Path, out: Path) -> None:
    volume = np.zeros(mask.shape, dtype=np.float32)
    volume[mask] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    utils.save_like(volume, nib.load(str(like)), out)


def sample_rows(data: np.ndarray, groups: np.ndarray, max_rows: int = CARPET_MAX_ROWS) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spaced subsample of the rows, proportional to the group sizes."""
    picked: list[np.ndarray] = []
    codes = [c for c in np.unique(groups) if c >= 0]
    total = int(np.isin(groups, codes).sum())
    floor = min(150, max_rows // max(len(codes), 1))       # small tissues (CSF) stay visible
    budget = max(max_rows - floor * len(codes), 0)
    for code in codes:
        rows = np.flatnonzero(groups == code)
        share = min(rows.size, floor + int(budget * rows.size / max(total, 1)))
        picked.append(rows[np.linspace(0, rows.size - 1, share).astype(int)])
    if not picked:
        return np.empty((0, data.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int8)
    order = np.concatenate(picked)
    return data[order].astype(np.float32), groups[order].astype(np.int8)


def tissue_groups(masks: dict[str, np.ndarray | None], support: np.ndarray) -> np.ndarray:
    """Carpet group code per voxel of ``support``: 1 GM, 2 WM, 3 CSF, 0 other brain."""
    volume = np.zeros(support.shape, dtype=np.int8)
    for name in ("CSF", "WM", "GM"):                     # GM wins where masks overlap
        mask = masks.get(name)
        if mask is not None:
            volume[mask] = TISSUE_CODES[name]
    groups = volume[support]
    if (groups > 0).any():
        groups[groups == 0] = -1                         # unlabeled voxels stay out of the carpet
    return groups


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp{os.getpid()}_{path.name}")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def epi_support_mask(boldref: np.ndarray) -> np.ndarray:
    """Where the EPI has signal, from intensities alone (independent of the T1 mask)."""
    finite = np.nan_to_num(boldref.astype(np.float32))
    mask = None
    if min(finite.shape) >= 16:
        try:
            from nilearn.masking import compute_epi_mask

            img = nib.Nifti1Image(finite, np.eye(4))
            mask = np.asanyarray(compute_epi_mask(img, opening=1).dataobj) > 0
        except Exception as err:
            LOG.debug("compute_epi_mask failed (%s); using a plain threshold", err)
    if mask is None or not mask.any():
        positive = finite[finite > 0]
        if positive.size == 0:
            return np.zeros(finite.shape, dtype=bool)
        mask = finite > 0.25 * np.percentile(positive, 98)
    return mask


def resample_mask(mask_path: Path, like: nib.spatialimages.SpatialImage) -> np.ndarray | None:
    """Binary mask on the grid of ``like`` (both in the same world space)."""
    if not mask_path.is_file():
        return None
    from nibabel.processing import resample_from_to

    src = nib.load(str(mask_path))
    data = (np.squeeze(np.asanyarray(src.dataobj)) > 0).astype(np.float32)
    src = nib.Nifti1Image(data, src.affine)
    if utils.same_grid(src, like):
        return data > 0.5
    out = resample_from_to(src, (like.shape[:3], like.affine), order=1, mode="constant", cval=0.0)
    return np.asanyarray(out.dataobj) > 0.5


# ----------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------

def read_json_safe(path: Path) -> dict:
    """Provenance JSON; a missing or damaged file is an empty dict."""
    try:
        data = utils.read_json(path, default={})
    except (OSError, ValueError) as err:
        LOG.warning("cannot read %s: %s", path, err)
        return {}
    return data if isinstance(data, dict) else {}


def read_series(path: Path) -> pd.DataFrame | None:
    """ROI time series TSV (header = ROI labels); None when absent or unreadable."""
    if not path.is_file():
        return None
    try:
        frame = utils.read_tsv(path)
    except Exception as err:
        LOG.warning("cannot read %s: %s", path, err)
        return None
    drop = [c for c in frame.columns if str(c).lower() in {"frame", "time", "volume", "t", "index"}]
    frame = frame.drop(columns=drop)
    return frame.apply(pd.to_numeric, errors="coerce")


def _first_column(frame: pd.DataFrame, names: tuple[str, ...], numeric: bool) -> str | None:
    lower = {str(c).lower(): c for c in frame.columns}
    for name in names:
        col = lower.get(name)
        if col is None:
            continue
        is_num = pd.api.types.is_numeric_dtype(frame[col])
        if is_num == numeric:
            return col
    return None


def network_from_name(name: str) -> str | None:
    """Schaefer-style names: 7Networks_LH_Vis_1 -> Vis."""
    match = re.match(r"^\d+Networks_[LR]H_([A-Za-z]+)", str(name))
    return match.group(1) if match else None


def find_atlas_labels(labels_dir: Path | None, atlas: str) -> pd.DataFrame | None:
    """Label table of an atlas as columns roi (int), name, network; None if not found."""
    if labels_dir is None or not Path(labels_dir).is_dir():
        return None
    root = Path(labels_dir)
    candidates = [
        root / f"{atlas}_labels.tsv", root / atlas / "labels.tsv", root / f"atlas-{atlas}_labels.tsv",
        root / f"atlas-{atlas}_dseg.tsv", root / f"{atlas}.tsv",
    ]
    candidates += sorted(root.glob(f"*{atlas}*.tsv")) + sorted(root.glob(f"{atlas}/*.tsv"))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            table = utils.read_tsv(path)
        except Exception as err:
            LOG.warning("cannot read atlas labels %s: %s", path, err)
            continue
        roi_col = _first_column(table, ("roi", "index", "label", "id", "value"), numeric=True)
        name_col = _first_column(table, ("name", "label_name", "region", "roi_name", "label"), numeric=False)
        net_col = _first_column(table, ("network", "net", "yeo_network"), numeric=False)
        if roi_col is None:
            continue
        out = pd.DataFrame({"roi": table[roi_col].astype(int)})
        out["name"] = table[name_col].astype(str) if name_col else out["roi"].astype(str)
        if net_col:
            out["network"] = table[net_col].astype(object).where(table[net_col].notna(), None)
        else:
            out["network"] = [network_from_name(n) for n in out["name"]]
        return out
    return None


def _is_label(value: Any) -> bool:
    return isinstance(value, str) and value.strip() not in {"", utils.NA, "nan", "None"}


def roi_table(columns: list[str], coverage: pd.DataFrame | None, labels: pd.DataFrame | None) -> pd.DataFrame:
    """roi / name / network / coverage_fraction, one row per time-series column."""
    from fmriproc.validate import aligned_coverage
    coverage = aligned_coverage(columns, coverage)
    n_roi = len(columns)
    header_ids = pd.to_numeric(pd.Series(columns, dtype=object), errors="coerce")
    if coverage is not None and len(coverage) == n_roi and "roi" in coverage.columns:
        roi = coverage["roi"].astype(int).to_numpy()
    elif header_ids.notna().all():
        roi = header_ids.astype(int).to_numpy()
    else:
        roi = np.arange(1, n_roi + 1)
    table = pd.DataFrame({"roi": roi})
    names = [str(c) for c in columns]
    networks: list[Any] = [network_from_name(n) for n in names]
    # stage 07 copies the network of labels.tsv into the coverage table; the
    # label table given on the command line wins over it, the ROI name is the last resort
    sources = []
    if coverage is not None and len(coverage) == n_roi and {"roi", "network"} <= set(coverage.columns):
        sources.append(coverage[["roi", "network"]])
    if labels is not None:
        sources.append(labels)
    for source in sources:
        lookup = source.drop_duplicates("roi").set_index("roi")
        if "name" in lookup.columns:
            names = [str(lookup["name"].get(r, n)) for r, n in zip(roi, names)]
        found = [lookup["network"].get(r, None) for r in roi]
        networks = [new if _is_label(new) else old for new, old in zip(found, networks)]
    table["name"] = names
    table["network"] = [n if _is_label(n) else None for n in networks]
    if coverage is not None and len(coverage) == n_roi and "coverage_fraction" in coverage.columns:
        table["coverage_fraction"] = pd.to_numeric(coverage["coverage_fraction"], errors="coerce").to_numpy()
    else:
        table["coverage_fraction"] = np.nan
    return table


def network_contrast(fc_z: np.ndarray, networks: list[str | None]) -> float | None:
    """(mean within-network z - mean between-network z) / SD(between)."""
    labels = np.array([n if n else "" for n in networks], dtype=object)
    if len({n for n in labels if n}) < 2:
        return None
    rows, cols = np.triu_indices(fc_z.shape[0], k=1)
    values = fc_z[rows, cols]
    known = (labels[rows] != "") & (labels[cols] != "") & np.isfinite(values)
    same = labels[rows] == labels[cols]
    within, between = values[known & same], values[known & ~same]
    if within.size < 2 or between.size < 2 or between.std() == 0:
        return None
    return _num((within.mean() - between.mean()) / between.std(ddof=1))


# ----------------------------------------------------------------------------
# metric blocks
# ----------------------------------------------------------------------------

def acquisition_metrics(prep: dict) -> dict[str, Any]:
    tr, n_vols = _num(prep.get("tr")), _num(prep.get("n_volumes"))
    voxel = prep.get("voxel_size")
    voxel_txt = "x".join(f"{float(v):.3g}" for v in voxel) if isinstance(voxel, (list, tuple)) and voxel else None
    return {
        "tr": tr,
        "n_volumes_raw": _num(prep.get("n_volumes_raw")),
        "n_dropped": _num(prep.get("n_dropped")),
        "n_volumes": n_vols,
        "minutes": _num(tr * n_vols / 60.0) if tr and n_vols else None,
        "voxel_size": voxel_txt,
        "obliquity_deg": _num(prep.get("obliquity_deg")),
        "stc_applied": prep.get("stc_applied"),
        "stc_reason": prep.get("stc_reason"),
        "stc_evidence": prep.get("slice_timing_evidence"),
        "stc_source": prep.get("slice_timing_source"),
        "nss_detected": _num(prep.get("nss_detected")),
        "despike": prep.get("despike"),
        "despike_fraction": _num(prep.get("despike_fraction")),
    }


def _column(frame: pd.DataFrame | None, *names: str) -> np.ndarray | None:
    if frame is None:
        return None
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
    return None


def _read_vector(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        return utils.read_1d(path)[:, 0]
    except Exception as err:
        LOG.warning("cannot read %s: %s", path, err)
        return None


def motion_metrics(paths: RunPaths, confounds: pd.DataFrame | None, keep: np.ndarray | None, tr: float | None) -> dict[str, Any]:
    fd = _column(confounds, "framewise_displacement")
    out: dict[str, Any] = {
        "fd_mean": None, "fd_median": None, "fd_max": None, "fd_pct_gt_02": None, "fd_pct_gt_05": None,
        "relrms_mean": None, "absrms_max": None, "n_censored": None, "pct_censored": None,
        "minutes_retained": None, "longest_segment": None,
    }
    if fd is not None and np.isfinite(fd).any():
        valid = fd[np.isfinite(fd)]
        out.update(
            fd_mean=_num(valid.mean()), fd_median=_num(np.median(valid)), fd_max=_num(valid.max()),
            fd_pct_gt_02=_num(100.0 * np.mean(valid > 0.2)), fd_pct_gt_05=_num(100.0 * np.mean(valid > 0.5)),
        )
    relrms = _read_vector(paths.func("desc-hmc_relrms.txt"))
    absrms = _read_vector(paths.func("desc-hmc_absrms.txt"))
    if relrms is None:
        relrms = _column(confounds, "fd_jenkinson", "rmsd")     # stage 04 pads the mcflirt relative RMS into this column
    out["relrms_mean"] = _nanstat(np.mean, relrms) if relrms is not None else None
    out["absrms_max"] = _nanstat(np.max, absrms) if absrms is not None else None
    if keep is not None and keep.size:
        n_cens = int((~keep).sum())
        out["n_censored"] = n_cens
        out["pct_censored"] = _num(100.0 * n_cens / keep.size)
        out["longest_segment"] = utils.longest_true_run(keep.tolist())
        if tr:
            out["minutes_retained"] = _num(keep.sum() * tr / 60.0)
    return out


def quality_index(paths: RunPaths, confounds: pd.DataFrame | None, prep: dict) -> float | None:
    """Mean 3dTqual index. Its file name is not fixed by the contract, so look around."""
    if _num(prep.get("quality_index_mean")) is not None:
        return _num(prep.get("quality_index_mean"))
    for suffix in ("desc-quality_timeseries.1D", "desc-tqual_timeseries.1D", "desc-qualityindex_timeseries.1D"):
        vector = _read_vector(paths.func(suffix))
        if vector is not None:
            return _nanstat(np.mean, vector)
    column = _column(confounds, "quality_index", "aqi", "tqual")
    return _nanstat(np.mean, column) if column is not None else None


def timeseries_metrics(paths: RunPaths, confounds: pd.DataFrame | None, prep: dict) -> dict[str, Any]:
    fd = _column(confounds, "framewise_displacement")
    dv = _column(confounds, "std_dvars")
    raw_dv = _column(confounds, "dvars")
    gs = _column(confounds, "global_signal")
    outliers = _read_vector(paths.func("desc-outliers_timeseries.1D"))
    if outliers is None:
        outliers = _column(confounds, "outlier_fraction", "outlier_frac", "outliers", "aor")
    out: dict[str, Any] = {
        # the first DVARS value is a placeholder (no previous frame), not a measurement
        "dvars_std_mean": _nanstat(np.mean, dv[1:]) if dv is not None and dv.size > 1 else None,
        "outlier_frac_mean": _nanstat(np.mean, outliers) if outliers is not None else None,
        "quality_index_mean": quality_index(paths, confounds, prep),
        "fd_dvars_corr": None,
        "gs_fd_corr": None,
    }
    use_dv = dv if dv is not None else raw_dv
    if fd is not None and use_dv is not None:
        out["fd_dvars_corr"] = pearson(fd[1:], use_dv[1:])
    if fd is not None and gs is not None and gs.size == fd.size:
        # the level of the global signal is arbitrary; its frame-to-frame jump is what motion drives
        out["gs_fd_corr"] = pearson(fd[1:], np.abs(np.diff(gs)))
    return out


def registration_metrics(prep: dict, anatqc: dict) -> dict[str, Any]:
    return {
        "coreg_method": prep.get("coreg_method"),
        "bbr_cost": _num(prep.get("bbr_cost")),
        "bbr_vs_init_mm": _num(prep.get("bbr_vs_init_mm")),
        "bbr_rejected": prep.get("bbr_rejected"),
        "epi_mask_method": prep.get("epi_mask_method"),
        "anat_mode": anatqc.get("anat_mode"),
        "norm_quality": anatqc.get("norm_quality"),
        "norm_dice": _num(anatqc.get("norm_dice")),
        "template_corr": _num(anatqc.get("template_corr")),
        "jacobian_p01": _num(anatqc.get("jacobian_p01")),
        "jacobian_p99": _num(anatqc.get("jacobian_p99")),
        "jacobian_nonpos_frac": _num(anatqc.get("jacobian_nonpos_frac")),
        "euler_lh": _num(anatqc.get("euler_lh")),
        "euler_rh": _num(anatqc.get("euler_rh")),
        "holes_total": _num(anatqc.get("holes_total")),
    }


def overlap_metrics(paths: RunPaths, prep: dict) -> dict[str, Any]:
    """coreg_dice and dropout_fraction from the T1w-space boldref and the T1 brain mask.

    Both compare an intensity-derived EPI support mask with the anatomical brain
    mask, inside the acquired field of view (voxels where the boldref is non-zero).
    """
    out: dict[str, Any] = {"coreg_dice": _num(prep.get("coreg_dice")), "coreg_dice_source": None, "dropout_fraction": None}
    if out["coreg_dice"] is not None:
        out["coreg_dice_source"] = "prep_info"
    boldref_path = paths.func("space-T1w_boldref.nii.gz")
    if not boldref_path.is_file():
        return out
    boldref_img = nib.load(str(boldref_path))
    boldref = np.squeeze(np.asanyarray(boldref_img.dataobj)).astype(np.float32)
    t1_mask = resample_mask(paths.anat("desc-brain_mask.nii.gz"), boldref_img)
    if t1_mask is None:
        return out
    fov = np.nan_to_num(boldref) != 0
    t1_in_fov = t1_mask & fov
    support = epi_support_mask(boldref)
    n_t1, n_epi = int(t1_in_fov.sum()), int(support.sum())
    if n_t1 == 0 or n_epi == 0:
        return out
    inter = int((support & t1_in_fov).sum())
    if out["coreg_dice"] is None:
        out["coreg_dice"] = _num(2.0 * inter / (n_t1 + n_epi))
        out["coreg_dice_source"] = "qc_boldref_support"
    out["dropout_fraction"] = _num(1.0 - inter / n_t1)
    return out


def t1w_space_metrics(paths: RunPaths, keep: np.ndarray, cache: CachePaths | None) -> dict[str, Any]:
    """Pre-denoise tSNR per tissue, GCOR and the 'before' carpet, from the T1w-space series."""
    out: dict[str, Any] = {
        "tsnr_gm_median": None, "tsnr_wm_median": None, "tsnr_brain_median": None, "gcor": None,
        "brain_mask_voxels": None,
    }
    bold_path = paths.func("space-T1w_desc-preproc_bold.nii.gz")
    brain_path = paths.func("space-T1w_desc-brain_mask.nii.gz")
    brain = load_mask(brain_path)
    if brain is not None:
        out["brain_mask_voxels"] = int(brain.sum())
    if brain is None or not bold_path.is_file():
        return out
    tissues = {name: load_mask(paths.func(f"space-T1w_label-{name}_mask.nii.gz"), brain.shape) for name in TISSUE_CODES}
    support = brain.copy()
    for mask in tissues.values():
        if mask is not None:
            support |= mask
    data = load_masked(bold_path, support)
    frames = FrameMap.build(keep, data.shape[1])
    mean = detrend_inplace(data, frames.retained)
    tsnr = mean / sd_over(data, frames.retained)
    in_brain = brain[support]
    out["tsnr_brain_median"] = _nanstat(np.median, tsnr[in_brain])
    for name, key in (("GM", "tsnr_gm_median"), ("WM", "tsnr_wm_median")):
        if tissues[name] is not None and tissues[name].any():
            out[key] = _nanstat(np.median, tsnr[tissues[name][support]])
    out["gcor"] = gcor(data[in_brain], frames.retained)
    if cache is not None:
        save_map(tsnr, support, brain_path, cache.tsnr("preproc", "T1w"))
        rows, groups = sample_rows(data, tissue_groups(tissues, support))
        _update_carpet(cache, {"pre": rows, "pre_groups": groups, "pre_frames": frames.index})
    return out


def _update_carpet(cache: CachePaths, arrays: dict[str, np.ndarray]) -> None:
    merged: dict[str, np.ndarray] = {}
    if cache.carpet.is_file():
        try:
            with np.load(cache.carpet) as old:
                merged = {k: old[k] for k in old.files}
        except Exception:
            merged = {}
    merged.update(arrays)
    save_npz(cache.carpet, merged)


def template_space_metrics(
    paths: RunPaths,
    strategies: list[str],
    keep: np.ndarray,
    fd: np.ndarray | None,
    tpl_masks: dict[str, Path | None],
    cache: CachePaths | None,
) -> dict[str, Any]:
    """Post-denoise tSNR, variance removed and residual FD-DVARS coupling per strategy."""
    out: dict[str, Any] = {f"{s}.{k}": None for s in strategies for k in STRATEGY_KEYS}
    for strategy in strategies:
        info = read_json_safe(paths.func(f"desc-{strategy}_denoise.json"))
        out[f"{strategy}.n_regressors"] = _num(info.get("n_regressors"))
        out[f"{strategy}.dof_remaining"] = _num(info.get("dof_remaining"))
        out[f"{strategy}.low_dof"] = info.get("low_dof")
    brain_path = paths.tpl("desc-brain_mask.nii.gz")
    pre_path = paths.tpl("desc-preproc_bold.nii.gz")
    brain = load_mask(brain_path)
    if brain is None or not pre_path.is_file():
        return out
    tissues = {name: (load_mask(p, brain.shape) if p else None) for name, p in tpl_masks.items()}
    gm = tissues.get("GM")
    use_gm = gm is not None and int((gm & brain).sum()) >= 50
    select = (gm & brain)[brain] if use_gm else np.ones(int(brain.sum()), dtype=bool)
    mask_name = "brain_x_warped_gm" if use_gm else "brain"
    groups = tissue_groups(tissues, brain)

    data = load_masked(pre_path, brain)
    frames = FrameMap.build(keep, data.shape[1])
    mean = detrend_inplace(data, frames.retained)
    sd_pre = sd_over(data, frames.retained)
    del data
    tsnr_pre = mean / sd_pre
    pre_median = _nanstat(np.median, tsnr_pre[select])
    if cache is not None:
        save_map(tsnr_pre, brain, brain_path, cache.tsnr("preproc", paths.template))

    for strategy in strategies:
        den_path = paths.tpl(f"desc-{strategy}_bold.nii.gz")
        if not den_path.is_file():
            LOG.warning("%s: denoised series missing (%s)", strategy, den_path.name)
            continue
        den = load_masked(den_path, brain)
        fmap = FrameMap.build(keep, den.shape[1])
        sd_post = sd_over(den, fmap.retained)
        tsnr_post = mean / sd_post
        post_median = _nanstat(np.median, tsnr_post[select])
        removed = 1.0 - (sd_post ** 2) / (sd_pre ** 2)
        out[f"{strategy}.tsnr_gm_median_post"] = post_median
        out[f"{strategy}.tsnr_pre_same_voxels"] = pre_median
        out[f"{strategy}.tsnr_gain"] = _num(post_median / pre_median) if post_median and pre_median else None
        out[f"{strategy}.variance_removed_gm"] = _nanstat(np.median, removed[select])
        out[f"{strategy}.tsnr_post_mask"] = mask_name
        if fd is not None:
            pairs = fmap.adjacent_pairs()
            index = fmap.index[pairs]
            index = index[index < fd.size]
            if index.size:
                out[f"{strategy}.fd_dvars_corr_post"] = pearson(fd[index], dvars(den)[pairs][: index.size])
        if cache is not None:
            save_map(tsnr_post, brain, brain_path, cache.tsnr(strategy, paths.template))
            rows, row_groups = sample_rows(den, groups)
            _update_carpet(cache, {f"post_{strategy}": rows, f"post_{strategy}_groups": row_groups,
                                   f"post_{strategy}_frames": fmap.index})
        del den
    return out


def roi_metrics(
    paths: RunPaths, strategy: str, atlas: str, keep: np.ndarray, labels: pd.DataFrame | None,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    prefix = f"{strategy}.{atlas}."
    out: dict[str, Any] = {prefix + k: None for k in ATLAS_KEYS}
    post = read_series(paths.timeseries(atlas, strategy))
    if post is None or post.shape[1] == 0:
        return out, None
    pre = read_series(paths.timeseries(atlas, "preproc"))
    coverage = None
    for desc in (strategy, "preproc"):
        path = paths.timeseries(atlas, desc, "coverage")
        if path.is_file():
            coverage = utils.read_tsv(path)
            break
    table = roi_table([str(c) for c in post.columns], coverage, labels)
    post_values = post.to_numpy(dtype=np.float64)
    post_map = FrameMap.build(keep, post_values.shape[0])
    sd_post = np.full(post_values.shape[1], np.nan)
    if post_map.retained.sum() >= 3:
        sd_post = np.nanstd(post_values[post_map.retained], axis=0, ddof=1)
        sd_post[~(sd_post > 0)] = np.nan

    table["tsnr_pre"] = np.nan
    table["roi_tsnr"] = np.nan
    table["variance_removed"] = np.nan
    if pre is not None and pre.shape[1] == post.shape[1]:
        from fmriproc.validate import column_order
        pre_values = pre.iloc[:, column_order(post.columns, pre.columns)].to_numpy(dtype=np.float64).T.copy()            # (R, T)
        pre_map = FrameMap.build(keep, pre_values.shape[1])
        finite = np.isfinite(pre_values).all(axis=1)
        mean_pre = np.full(pre_values.shape[0], np.nan)
        sd_pre = np.full(pre_values.shape[0], np.nan)
        if finite.any():
            block = pre_values[finite]
            mean_pre[finite] = detrend_inplace(block, pre_map.retained)
            sd_pre[finite] = sd_over(block, pre_map.retained)
        table["tsnr_pre"] = mean_pre / sd_pre
        table["roi_tsnr"] = mean_pre / sd_post
        table["variance_removed"] = 1.0 - sd_post ** 2 / sd_pre ** 2
    elif pre is not None:
        LOG.warning("%s/%s: pre- and post-denoise ROI tables differ in size", strategy, atlas)

    fc = utils.fc_matrix(post_values, post_map.retained)
    edges = utils.upper_triangle(fc)
    out.update({
        prefix + "n_roi": int(post_values.shape[1]),
        prefix + "roi_tsnr_median": _nanstat(np.median, table["roi_tsnr"].to_numpy()),
        prefix + "roi_tsnr_min": _nanstat(np.min, table["roi_tsnr"].to_numpy()),
        prefix + "roi_tsnr_pre_median": _nanstat(np.median, table["tsnr_pre"].to_numpy()),
        prefix + "roi_variance_removed_median": _nanstat(np.median, table["variance_removed"].to_numpy()),
        prefix + "n_roi_nan": int((~np.isfinite(post_values).all(axis=0)).sum()),
        prefix + "coverage_min": _nanstat(np.min, table["coverage_fraction"].to_numpy()),
        prefix + "split_half_r": _num(utils.split_half_reliability(post_values, post_map.retained)),
        prefix + "network_contrast": network_contrast(utils.fisher_z(fc), table["network"].tolist()),
        prefix + "fc_mean": _nanstat(np.mean, edges),
        prefix + "fc_sd": _nanstat(np.std, edges),
    })

    surf = read_series(paths.surf_timeseries(atlas, strategy))
    if surf is not None and surf.shape[1] > 1:
        surf_values = surf.to_numpy(dtype=np.float64)
        surf_map = FrameMap.build(keep, surf_values.shape[0])
        out[prefix + "surf_split_half_r"] = _num(utils.split_half_reliability(surf_values, surf_map.retained))
        if surf_values.shape[1] == post_values.shape[1]:
            out[prefix + "vol_surf_fc_r"] = pearson(edges, utils.upper_triangle(utils.fc_matrix(surf_values, surf_map.retained)))
    return out, table


def surface_metrics(paths: RunPaths) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = ("tsnr_cortex_median", "pct_badvertices", "pct_goodvoxels_excluded")
    info = read_json_safe(paths.func("desc-surfqc.json"))
    for key in keys:
        if _num(info.get(key)) is not None:
            out[key] = _num(info.get(key))
    if "tsnr_cortex_median" not in out and paths.tsnr_dscalar.is_file():
        try:
            img = nib.load(str(paths.tsnr_dscalar))
            values = np.asanyarray(img.dataobj, dtype=np.float32)[0]
            axis = img.header.get_axis(1)
            cortex = np.zeros(values.size, dtype=bool)
            for name, index, _ in axis.iter_structures():
                if "CORTEX" in name:
                    cortex[index] = True
            out["tsnr_cortex_median"] = _nanstat(np.median, values[cortex & (values > 0)])
        except Exception as err:
            LOG.warning("cannot read %s: %s", paths.tsnr_dscalar.name, err)
    if out:
        for key in keys:
            out.setdefault(key, None)
    return out


def parse_fwhm(path: Path | None) -> float | None:
    """Effective ACF FWHM: 4th number of the last numeric line of `3dFWHMx -acf` output."""
    if path is None or not Path(path).is_file():
        return None
    value = None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 4 and not line.lstrip().startswith("#"):
            try:
                value = float(parts[3])
            except ValueError:
                continue
    return _num(value) if value and value > 0 else None


# ----------------------------------------------------------------------------
# flags
# ----------------------------------------------------------------------------

def flag_value(value: Any, direction: str, warn: float, fail: float) -> str:
    value = _num(value)
    if value is None:
        return "n/a"
    if direction == "high":
        return "fail" if value > fail else "warn" if value > warn else "pass"
    return "fail" if value < fail else "warn" if value < warn else "pass"


def compute_flags(
    metrics: dict[str, Any], thresholds: dict[str, tuple[str, float, float]], strategies: list[str],
    min_dof: float, min_retained_min: float,
) -> dict[str, str]:
    flags = {name: flag_value(metrics.get(name), *spec) for name, spec in thresholds.items()}
    minutes = _num(metrics.get("minutes_retained"))
    flags["minutes_retained"] = "n/a" if minutes is None else ("warn" if minutes < min_retained_min else "pass")
    if metrics.get("bbr_rejected") is not None:
        flags["bbr_accepted"] = "warn" if metrics.get("bbr_rejected") else "pass"
    for strategy in strategies:
        dof = _num(metrics.get(f"{strategy}.dof_remaining"))
        if dof is None:
            flags[f"{strategy}.dof_remaining"] = "n/a"
        else:
            flags[f"{strategy}.dof_remaining"] = "fail" if dof <= 0 else "warn" if dof < min_dof else "pass"
    return flags


def worst_flag(flags: dict[str, str]) -> str:
    worst = max((FLAG_RANK.get(v, -1) for v in flags.values()), default=-1)
    return {rank: name for name, rank in FLAG_RANK.items()}[worst]


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def _guarded(name: str, func: Callable[[], dict[str, Any]], problems: list[str]) -> dict[str, Any]:
    """One broken product must not take the whole QC down."""
    try:
        return func()
    except MemoryError:
        problems.append(f"{name}: out of memory")
    except Exception as err:
        problems.append(f"{name}: {type(err).__name__}: {err}")
    LOG.warning("metric block failed - %s", problems[-1])
    return {}


def n_frames_of(paths: RunPaths, confounds: pd.DataFrame | None, prep: dict) -> int | None:
    if confounds is not None and len(confounds):
        return int(len(confounds))
    if _num(prep.get("n_volumes")):
        return int(prep["n_volumes"])
    for path in (paths.func("space-T1w_desc-preproc_bold.nii.gz"), paths.tpl("desc-preproc_bold.nii.gz")):
        if path.is_file():
            shape = nib.load(str(path)).shape
            if len(shape) == 4:
                return int(shape[3])
    return None


def load_keep(paths: RunPaths, n_frames: int | None) -> np.ndarray | None:
    vector = _read_vector(paths.censor)
    if vector is not None:
        if not np.isfinite(vector).all() or not np.isin(vector, [0, 1]).all():
            raise ValueError("censor vector must contain finite binary values")
        keep = vector.astype(bool)
        if n_frames is not None and keep.size != n_frames:
            raise ValueError(f"censor vector has {keep.size} entries, run has {n_frames} frames")
        return keep
    raise ValueError("censor vector missing or unreadable; retained frames are unknown")


def compute_metrics(args: argparse.Namespace) -> dict[str, Any]:
    paths = RunPaths(Path(args.func_dir), Path(args.anat_dir), args.run, args.template, str(args.mni_res))
    strategies, atlases = args.strategies.split(), args.atlases.split()
    cache = CachePaths(Path(args.cache_dir)) if args.cache_dir else None
    labels_dir = Path(args.atlas_labels_dir) if args.atlas_labels_dir else None
    problems: list[str] = []
    if cache is not None and cache.carpet.is_file():
        cache.carpet.unlink()          # samples of an earlier run must not reach the figures

    prep = read_json_safe(paths.prep_info)
    anatqc = read_json_safe(paths.anat("desc-anatqc.json"))
    conf_json = read_json_safe(paths.confounds_json)
    confounds = None
    if paths.confounds.is_file():
        try:
            confounds = utils.read_tsv(paths.confounds)
        except Exception as err:
            problems.append(f"confounds: {err}")
    n_frames = n_frames_of(paths, confounds, prep)
    try:
        keep = load_keep(paths, n_frames)
    except (ValueError, OSError) as err:
        problems.append(f"censor: {err}")
        keep = None
    tr = _num(prep.get("tr"))
    fd = _column(confounds, "framewise_displacement")

    metrics: dict[str, Any] = {
        "qc_version": QC_VERSION, "subject": paths.subject, "run": paths.run, "template": paths.template,
        "mni_res": str(args.mni_res), "strategies": " ".join(strategies), "atlases": " ".join(atlases),
        "censor_mode": args.censor_mode,
        "censor_fd_threshold": _num((conf_json.get("censor") or {}).get("fd_threshold")),
    }
    metrics.update(acquisition_metrics(prep))
    if metrics["n_volumes"] is None and n_frames:
        metrics["n_volumes"] = n_frames
    metrics.update(_guarded("motion", lambda: motion_metrics(paths, confounds, keep, tr), problems))
    metrics.update(_guarded("timeseries", lambda: timeseries_metrics(paths, confounds, prep), problems))
    metrics["fwhm_acf"] = parse_fwhm(Path(args.fwhm_file) if args.fwhm_file else None)
    metrics.update(registration_metrics(prep, anatqc))
    metrics.update(_guarded("overlap", lambda: overlap_metrics(paths, prep), problems))
    for key in ("wm_mask_voxels", "csf_mask_voxels", "gm_mask_voxels"):
        metrics[key] = _num(prep.get(key))
    metrics["tissue_erosion_relaxed"] = prep.get("tissue_erosion_relaxed")

    if n_frames is None:
        problems.append("number of frames unknown: no confounds, censor vector or BOLD series")
    signal_keys = {"tsnr_gm_median": None, "tsnr_wm_median": None, "tsnr_brain_median": None, "gcor": None,
                   "brain_mask_voxels": None}
    metrics.update(signal_keys)
    if keep is not None:
        metrics.update(_guarded("t1w_space", lambda: t1w_space_metrics(paths, keep, cache), problems))
    tpl_masks = {"GM": args.tpl_gm_mask, "WM": args.tpl_wm_mask, "CSF": args.tpl_csf_mask}
    tpl_masks = {k: (Path(v) if v else None) for k, v in tpl_masks.items()}
    metrics.update({f"{s}.{k}": None for s in strategies for k in STRATEGY_KEYS})
    if keep is not None:
        metrics.update(_guarded("template_space",
                                lambda: template_space_metrics(paths, strategies, keep, fd, tpl_masks, cache), problems))

    for atlas in atlases:
        labels = find_atlas_labels(labels_dir, atlas)
        for strategy in strategies:
            metrics.update({f"{strategy}.{atlas}.{k}": None for k in ATLAS_KEYS})

            def block(strategy: str = strategy, atlas: str = atlas, labels: pd.DataFrame | None = labels) -> dict[str, Any]:
                values, table = roi_metrics(paths, strategy, atlas, keep, labels)
                if table is not None:
                    utils.write_tsv(paths.roiqc(atlas, strategy), table)
                return values

            if keep is not None:
                metrics.update(_guarded(f"roi {strategy}/{atlas}", block, problems))
    metrics.update(_guarded("surface", lambda: surface_metrics(paths), problems))

    thresholds = {
        name: (spec[0], getattr(args, f"qc_{THRESHOLD_OPTIONS[name].replace('-', '_')}_warn"),
               getattr(args, f"qc_{THRESHOLD_OPTIONS[name].replace('-', '_')}_fail"))
        for name, spec in THRESHOLDS.items()
    }
    flags = compute_flags(metrics, thresholds, strategies, args.min_dof, args.min_retained_min)
    metrics["flags"] = flags
    required = ["fd_mean", "minutes_retained", "tsnr_gm_median", "coreg_dice", "norm_dice"]
    required += [f"{strategy}.{metric}" for strategy in strategies
                 for metric in ("dof_remaining", "tsnr_gm_median_post")]
    missing = [name for name in required if _num(metrics.get(name)) is None]
    if missing:
        problems.append("required QC unavailable: " + ", ".join(missing))
    if problems:
        flags["qc_complete"] = "incomplete"
    metrics["overall_flag"] = worst_flag(flags)
    metrics["thresholds"] = {name: {"worse_when": spec[0], "warn": spec[1], "fail": spec[2]}
                             for name, spec in thresholds.items()}
    metrics["thresholds"]["min_dof"] = args.min_dof
    metrics["thresholds"]["min_retained_min"] = args.min_retained_min
    metrics["qc_problems"] = "; ".join(problems) if problems else None
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--func-dir", required=True, help="derivatives/sub-X/func")
    parser.add_argument("--anat-dir", required=True, help="derivatives/sub-X/anat")
    parser.add_argument("--run", required=True, help="BIDS run prefix, e.g. sub-01_task-rest")
    parser.add_argument("--template", required=True, help="template name used in the file names")
    parser.add_argument("--mni-res", required=True, help="template-space resolution entity (res-<R>)")
    parser.add_argument("--strategies", default="", help="space separated denoising strategies")
    parser.add_argument("--atlases", default="", help="space separated atlas names")
    parser.add_argument("--censor-mode", default="NTRP", help="3dTproject -cenmode used by the denoising stage")
    parser.add_argument("--out-json", default=None, help="default: <func-dir>/<RUN>_desc-qc_metrics.json")
    parser.add_argument("--atlas-labels-dir", default=None, help="directory with atlas label tables (network column)")
    parser.add_argument("--cache-dir", default=None, help="tSNR maps and carpet samples for fmriproc.plots")
    parser.add_argument("--fwhm-file", default=None, help="captured stdout of 3dFWHMx -acf")
    for tissue in ("gm", "wm", "csf"):
        parser.add_argument(f"--tpl-{tissue}-mask", default=None,
                            help=f"subject {tissue.upper()} mask warped to the template-space BOLD grid")
    for name, (_, warn, fail) in THRESHOLDS.items():
        stem = THRESHOLD_OPTIONS[name]
        parser.add_argument(f"--qc-{stem}-warn", type=float, default=warn)
        parser.add_argument(f"--qc-{stem}-fail", type=float, default=fail)
    parser.add_argument("--min-dof", type=float, default=15.0)
    parser.add_argument("--min-retained-min", type=float, default=4.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    func_dir = Path(args.func_dir)
    if not func_dir.is_dir():
        LOG.error("func directory not found: %s", func_dir)
        return 2
    metrics = compute_metrics(args)
    out_json = Path(args.out_json) if args.out_json else func_dir / f"{args.run}_desc-qc_metrics.json"
    utils.write_json(out_json, metrics)
    LOG.info("%s: overall %s -> %s", args.run, metrics["overall_flag"], out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
