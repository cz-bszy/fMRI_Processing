"""Atlas time series and functional connectivity (stage 07).

Sub-commands::

    python -m fmriproc.timeseries volume    --bold B --atlas A --mask M --out-prefix P ...
    python -m fmriproc.timeseries cifti     --ptseries X.ptseries.nii --out-prefix P ...
    python -m fmriproc.timeseries gridcheck --image A --reference B     (prints same|different)

Outputs of ``volume``: ``<P>_timeseries.tsv`` (header = ROI names, uncovered ROIs
= n/a), ``<P>_coverage.tsv``, ``<P>_timeseries.json`` and ``<P>_connectivity.tsv``
(square Pearson r matrix over retained frames, header row only, same column
order as the time series). ``cifti`` writes the same files; its columns are the
dlabel parcels in ascending label-key order (the order of wb_command
-cifti-parcellate) under the dlabel names, and the coverage table needs
``--dlabel`` and ``--dtseries``.
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from fmriproc.utils import (
    NA,
    extract_roi_means,
    fc_matrix,
    load_img,
    read_1d,
    same_grid,
    write_json,
    write_tsv,
)

CENSOR_MODES = ("NTRP", "KILL", "ZERO")
SERIES_FORMAT = "%.8g"

VOLUME_DEFINITION = (
    "Each column is the mean, per volume, over the voxels that carry the atlas label on the BOLD "
    "grid, lie inside the brain mask, are finite at every time point and are not constant over "
    "time. The input is the unsmoothed series. An ROI whose fraction of such voxels is below "
    "min_coverage (or that is absent from the resampled atlas) is n/a in every row; it is never "
    "imputed. Rows are all volumes of the input series: with censor_handling=applied the censored "
    "rows are still present (interpolated or zeroed by the denoising step) and must be dropped "
    "with the censor vector; with censor_handling=already_removed the denoising step deleted them. "
    "Connectivity is the Pearson correlation over retained frames only."
)
CIFTI_DEFINITION = (
    "Each column is the mean over the grayordinates of one dlabel parcel (wb_command "
    "-cifti-parcellate, method MEAN) of the unsmoothed dense series; columns follow the ascending "
    "label keys of the dlabel and carry the label-table name of the same key when both files hold "
    "the same keys with matching hemisphere and network (roi_names_source; the dlabel names "
    "otherwise, renamed_from_dlabel lists the differences). Parcels without data or with a constant "
    "series are n/a. With coverage_checked=true the rule of the volume stream is applied as well: "
    "grayordinates that are non-finite or constant over time (outside the field of view) do not "
    "count, a parcel whose fraction of valid grayordinates is below min_coverage is n/a, and a "
    "partly covered parcel (recomputed_rois) is the mean over its valid grayordinates. Censored rows "
    "are handled as described by censor_handling (applied = rows still present, drop them with the "
    "censor vector; already_removed = deleted by the denoising step). Connectivity is the Pearson "
    "correlation over retained frames only."
)
UNLABELED = "???"   # name of the unassigned label in Workbench label tables


def log(message: str) -> None:
    print(f"[timeseries] {message}", file=sys.stderr)


# ----------------------------------------------------------------------------
# label tables and column layout
# ----------------------------------------------------------------------------

def is_none(value: str | None) -> bool:
    return value is None or str(value).strip().lower() in ("", "none")


def read_labels(path: str | Path | None) -> pd.DataFrame | None:
    """Label table (columns index, name[, network]) -> DataFrame[index:int, name, network].

    Read as plain strings: a region called "NA" must not turn into a missing value.
    Rows with index <= 0 (background) are dropped.
    """
    if path is None or is_none(str(path)):
        return None
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame.columns]
    missing = [c for c in ("index", "name") if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: label table lacks column(s) {missing}; expected index, name, network")
    try:
        index = frame["index"].str.strip().astype(float)
    except ValueError as err:
        raise ValueError(f"{path}: non-numeric value in column 'index'") from err
    if not np.allclose(index, np.rint(index)):
        raise ValueError(f"{path}: column 'index' must hold integers")
    network = frame["network"] if "network" in frame.columns else pd.Series([NA] * len(frame))
    table = pd.DataFrame(
        {
            "index": np.rint(index).astype(np.int64).to_numpy(),
            "name": frame["name"].astype(str).to_numpy(),
            "network": network.astype(str).to_numpy(),
        }
    )
    table = table[table["index"] > 0]
    if table["index"].duplicated().any():
        dup = sorted(set(table.loc[table["index"].duplicated(), "index"].tolist()))
        raise ValueError(f"{path}: duplicated label index {dup}")
    return table.sort_values("index").reset_index(drop=True)


def _clean_name(name: str, index: int) -> str:
    text = " ".join(str(name).split())
    if text == "" or text.lower() in (NA, "nan"):
        return f"roi_{index}"
    return text


HEMISPHERES = ("LH", "RH")


def _hemi_network(name: str) -> tuple[str, str] | None:
    """('LH' | 'RH', network) of a Schaefer-style name such as 7Networks_LH_Default_PCC_1."""
    parts = str(name).split("_")
    for i, part in enumerate(parts[:-1]):
        if part in HEMISPHERES:
            return part, parts[i + 1]
    return None


def canonical_names(parcels: pd.DataFrame, table: pd.DataFrame | None) -> tuple[dict[str, str], str]:
    """dlabel parcel name -> label-table name of the same atlas label key.

    The volumetric label table and the dlabel of one atlas release may spell the
    same parcel differently: the CBIG fsLR dlabel renames 19 of the Schaefer-100
    parcels (e.g. Default_PCC_1 -> Default_pCunPCC_1) with unchanged label keys and
    boundaries. The label key is the parcel identity; the label-table names are
    taken only when both files hold exactly the same keys, every renamed key keeps
    its hemisphere and network, and the names stay unique. Otherwise ({}, reason).
    """
    if table is None:
        return {}, "no label table"
    by_key = {int(k): _clean_name(n, int(k)) for k, n in zip(table["index"], table["name"])}
    keys = [int(k) for k in parcels["roi"]]
    if set(keys) != set(by_key):
        return {}, "the label keys of the dlabel and of the label table differ"
    mapping: dict[str, str] = {}
    for key, name in zip(keys, parcels["name"]):
        canon = by_key[key]
        if canon != name:
            here, there = _hemi_network(name), _hemi_network(canon)
            if here is None or here != there:
                return {}, f"label key {key}: '{name}' (dlabel) and '{canon}' (label table) differ in hemisphere or network"
        mapping[name] = canon
    if len(set(mapping.values())) != len(mapping):
        return {}, "the label-table names are not unique"
    return mapping, "label table, matched by atlas label key"


def _unique_names(names: list[str], indices: list[int]) -> list[str]:
    counts = pd.Series(names).value_counts()
    return [f"{n}_{i}" if counts[n] > 1 else n for n, i in zip(names, indices)]


def build_columns(atlas_labels: np.ndarray, table: pd.DataFrame | None) -> pd.DataFrame:
    """One row per output column: roi (label index), name, network.

    Without a table the columns are the labels present in the atlas, named
    ``roi_<index>``. With a table the columns are the union of table and atlas
    labels in ascending index order, so that every subject has the same columns
    even when nearest-neighbour resampling loses a small parcel (it becomes n/a).
    """
    present = [int(v) for v in np.asarray(atlas_labels).ravel()]
    if table is None:
        indices = sorted(set(present))
        names = [f"roi_{i}" for i in indices]
        networks = [NA] * len(indices)
    else:
        by_index = table.set_index("index")
        indices = sorted(set(present) | set(int(v) for v in table["index"]))
        names, networks = [], []
        for i in indices:
            if i in by_index.index:
                names.append(_clean_name(by_index.at[i, "name"], i))
                net = " ".join(str(by_index.at[i, "network"]).split())
                networks.append(net if net and net.lower() != "nan" else NA)
            else:
                names.append(f"roi_{i}")
                networks.append(NA)
        unknown = [i for i in present if i not in by_index.index]
        if unknown:
            log(f"WARNING: {len(set(unknown))} atlas label(s) not in the label table: {sorted(set(unknown))[:10]}")
    return pd.DataFrame({"roi": indices, "name": _unique_names(names, indices), "network": networks})


# ----------------------------------------------------------------------------
# censoring
# ----------------------------------------------------------------------------

def read_censor(path: str | Path | None) -> np.ndarray | None:
    if path is None or is_none(str(path)):
        return None
    data = read_1d(path)
    if data.shape[1] != 1:
        raise ValueError(f"{path}: expected one column (1 = keep, 0 = censored), found {data.shape[1]}")
    return data[:, 0]


def retained_frames(n_volumes: int, censor: np.ndarray | None, mode: str = "NTRP") -> tuple[np.ndarray, str]:
    """Boolean keep vector for a series of `n_volumes` rows, and how censoring was handled.

    The decision is made on lengths, not on `mode`: a series as long as the
    censor vector still contains the censored rows ("applied"); a series as long
    as the number of kept frames was shortened by 3dTproject -cenmode KILL
    ("already_removed", every row counts as retained).
    """
    if censor is None:
        return np.ones(n_volumes, dtype=bool), "none"
    keep = np.asarray(censor, dtype=float).ravel() > 0.5
    n_keep = int(keep.sum())
    if keep.size == n_volumes:
        if mode.upper() == "KILL" and n_keep < n_volumes:
            # expected for the pre-denoise series, which 3dTproject never shortened
            log("censor mode KILL, but this series still has every volume: censored rows stay in the table")
        return keep, "applied"
    if n_keep == n_volumes:
        if mode.upper() != "KILL":
            log(f"WARNING: censor mode {mode} but the series is already shortened to the kept frames")
        return np.ones(n_volumes, dtype=bool), "already_removed"
    raise ValueError(
        f"series has {n_volumes} volumes; censor vector has {keep.size} entries of which {n_keep} are kept: "
        "neither length matches"
    )


# ----------------------------------------------------------------------------
# volume extraction
# ----------------------------------------------------------------------------

def load_roi_voxels(bold: nib.spatialimages.SpatialImage, select: np.ndarray, max_chunk_bytes: int = 256 * 2**20) -> np.ndarray:
    """(V, T) float32 array of the voxels in `select`, read in blocks of volumes so
    that the full 4D array (1 GB for a 2 mm template-space run) is never in memory."""
    if len(bold.shape) != 4:
        raise ValueError(f"expected a 4D BOLD image, got shape {bold.shape}")
    if select.shape != tuple(bold.shape[:3]):
        raise ValueError("voxel selection and BOLD grids differ")
    n_t = int(bold.shape[3])
    out = np.empty((int(select.sum()), n_t), dtype=np.float32)
    # 8 bytes per value: scaled (int16 + slope) data are delivered as float64
    step = max(1, int(max_chunk_bytes // (int(np.prod(bold.shape[:3])) * 8)))
    for start in range(0, n_t, step):
        block = np.asanyarray(bold.dataobj[..., start:start + step])
        out[:, start:start + block.shape[3]] = block[select]
        del block
    return out


def roi_series(
    voxels: np.ndarray,
    voxel_labels: np.ndarray,
    voxel_in_mask: np.ndarray,
    columns: pd.DataFrame,
    min_coverage: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """ROI means for the label voxels (V, T) -> (series [T, n_columns], coverage table).

    Uses utils.extract_roi_means on a (V, 1, 1, T) view, so the coverage rule is
    the shared one; columns missing from the atlas grid are n/a with 0 voxels.
    """
    n_vox, n_t = voxels.shape
    labels, series, coverage = extract_roi_means(
        voxels.reshape(n_vox, 1, 1, n_t),
        voxel_labels.reshape(n_vox, 1, 1),
        voxel_in_mask.reshape(n_vox, 1, 1),
        min_coverage=min_coverage,
    )
    position = {int(roi): k for k, roi in enumerate(columns["roi"])}
    target = np.array([position[int(label)] for label in labels], dtype=np.int64)
    full = np.full((n_t, len(columns)), np.nan, dtype=np.float64)
    full[:, target] = series
    table = columns.copy()
    for key, dtype in (("atlas_voxels", np.int64), ("valid_voxels", np.int64), ("coverage_fraction", np.float64), ("included", bool)):
        values = np.zeros(len(columns), dtype=dtype)
        values[target] = coverage[key].to_numpy(dtype=dtype)
        table[key] = values
    return full, table


def entity_from_prefix(prefix: str, key: str) -> str:
    """Value of `_<key>-<value>` in a file prefix; atlas names may contain '_'."""
    name = Path(prefix).name
    marker = f"_{key}-"
    if marker not in name:
        return NA
    value = name.split(marker, 1)[1]
    if key == "atlas":
        return value.split("_desc-", 1)[0]
    return value.split("_", 1)[0]


def write_outputs(
    prefix: str,
    series: np.ndarray,
    names: list[str],
    keep: np.ndarray,
    info: dict,
    coverage: pd.DataFrame | None = None,
    connectivity: bool = True,
) -> None:
    write_tsv(f"{prefix}_timeseries.tsv", pd.DataFrame(series, columns=names), float_format=SERIES_FORMAT)
    if coverage is not None:
        write_tsv(f"{prefix}_coverage.tsv", coverage)
    if connectivity:
        write_tsv(f"{prefix}_connectivity.tsv", pd.DataFrame(fc_matrix(series, keep), columns=names))
    write_json(f"{prefix}_timeseries.json", info)


def run_volume(args: argparse.Namespace) -> int:
    if not 0.0 <= args.min_coverage <= 1.0:
        raise ValueError("--min-coverage must lie in [0, 1]")
    atlas_img = load_img(args.atlas)
    mask_img = load_img(args.mask)
    # keep_file_open: consecutive blocks continue in the gzip stream instead of
    # decompressing from the start for every block
    bold_img = nib.load(str(args.bold), keep_file_open=True)
    if len(atlas_img.shape) != 3 and not (len(atlas_img.shape) == 4 and atlas_img.shape[3] == 1):
        raise ValueError(f"{args.atlas}: expected a 3D label image, got shape {atlas_img.shape}")
    if not same_grid(atlas_img, bold_img):
        raise ValueError(f"atlas and BOLD grids differ ({args.atlas} vs {args.bold}); resample the atlas first")
    if not same_grid(mask_img, bold_img):
        raise ValueError(f"mask and BOLD grids differ ({args.mask} vs {args.bold})")

    atlas = np.rint(np.asanyarray(atlas_img.dataobj).reshape(atlas_img.shape[:3])).astype(np.int64)
    mask = np.asanyarray(mask_img.dataobj).reshape(mask_img.shape[:3]) > 0
    select = atlas > 0
    if not select.any():
        raise ValueError(f"{args.atlas}: no positive labels")
    table = read_labels(args.labels)
    columns = build_columns(np.unique(atlas[select]), table)

    voxels = load_roi_voxels(bold_img, select, max_chunk_bytes=int(args.chunk_mb * 2**20))
    del bold_img
    series, coverage = roi_series(voxels, atlas[select], mask[select], columns, args.min_coverage)
    del voxels

    keep, handling = retained_frames(series.shape[0], read_censor(args.censor), args.censor_mode)
    names = columns["name"].tolist()
    missing = coverage.loc[~coverage["included"].astype(bool), "name"].tolist()
    if missing:
        log(f"{len(missing)} of {len(names)} ROIs are n/a (coverage < {args.min_coverage:g}): {missing[:8]}")
    info = {
        "atlas": args.atlas_name or entity_from_prefix(args.out_prefix, "atlas"),
        "strategy": args.strategy or entity_from_prefix(args.out_prefix, "desc"),
        "space": "volume",
        "n_volumes": int(series.shape[0]),
        "n_retained": int(keep.sum()),
        "n_rois": len(names),
        "roi_names": names,
        "missing_rois": missing,
        "min_coverage": float(args.min_coverage),
        "censor_mode": args.censor_mode,
        "censor_handling": handling,
        "connectivity_written": not args.no_connectivity,
        "definition": VOLUME_DEFINITION,
        "sources": {
            "bold": str(args.bold),
            "atlas": str(args.atlas),
            "labels": None if is_none(args.labels) else str(args.labels),
            "mask": str(args.mask),
            "censor": None if is_none(args.censor) else str(args.censor),
        },
    }
    write_outputs(args.out_prefix, series, names, keep, info, coverage=coverage, connectivity=not args.no_connectivity)
    return 0


# ----------------------------------------------------------------------------
# CIFTI
# ----------------------------------------------------------------------------

def _cifti_axes(path: str | Path, kind: str) -> tuple[nib.Cifti2Image, list]:
    img = nib.load(str(path))
    if not isinstance(img, nib.Cifti2Image) or img.ndim != 2:
        raise ValueError(f"{path}: not a 2D CIFTI-2 file ({kind} expected)")
    return img, [img.header.get_axis(i) for i in range(2)]


def _single_axis(path: str | Path, axes: list, axis_type: type, kind: str) -> int:
    dims = [i for i, axis in enumerate(axes) if isinstance(axis, axis_type)]
    if len(dims) != 1:
        raise ValueError(f"{path}: expected exactly one {axis_type.__name__} ({kind})")
    return dims[0]


def load_ptseries(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Parcellated series -> ((T, P) float64 array, parcel names from the parcels axis)."""
    img, axes = _cifti_axes(path, "ptseries")
    dim = _single_axis(path, axes, nib.cifti2.ParcelsAxis, "ptseries")
    data = np.asarray(img.get_fdata(dtype=np.float64))
    if dim == 0:
        data = data.T
    return data, [str(name) for name in axes[dim].name]


def load_dlabel(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, nib.cifti2.BrainModelAxis]:
    """dlabel -> (parcels [roi, name], label key of every grayordinate, brain-model axis).

    The parcels are the ones wb_command -cifti-parcellate creates: every label of
    the (first) map except the unlabeled one, sorted by ascending key.
    """
    img, axes = _cifti_axes(path, "dlabel")
    label_dim = _single_axis(path, axes, nib.cifti2.LabelAxis, "dlabel")
    model_dim = _single_axis(path, axes, nib.cifti2.BrainModelAxis, "dlabel")
    data = np.asarray(img.dataobj)
    if label_dim == 1:
        data = data.T
    keys = np.rint(data[0]).astype(np.int64)
    table = axes[label_dim].label[0]
    items = sorted((int(key), str(value[0])) for key, value in table.items() if str(value[0]) != UNLABELED)
    if not items:
        raise ValueError(f"{path}: the label table holds no parcel")
    return pd.DataFrame(items, columns=["roi", "name"]), keys, axes[model_dim]


def load_dtseries(path: str | Path) -> tuple[np.ndarray, nib.cifti2.BrainModelAxis]:
    """Dense series -> ((T, G) float32 array, brain-model axis); 91k x 300 frames = 110 MB."""
    img, axes = _cifti_axes(path, "dtseries")
    model_dim = _single_axis(path, axes, nib.cifti2.BrainModelAxis, "dtseries")
    data = np.asarray(img.dataobj, dtype=np.float32)
    if model_dim == 0:
        data = data.T
    return data, axes[model_dim]


def grayordinate_codes(axis: nib.cifti2.BrainModelAxis) -> np.ndarray:
    """One int64 per grayordinate, comparable between files: a vertex is identified
    by structure and vertex number, a voxel by its indices (as CIFTI parcels do)."""
    codes = np.empty(len(axis), dtype=np.int64)
    surface = np.asarray(axis.surface_mask, dtype=bool)
    structures, inverse = np.unique(np.asarray(axis.name, dtype=str), return_inverse=True)
    ids = np.array([zlib.crc32(name.encode()) & 0x3FFFFFF for name in structures], dtype=np.int64)[inverse]
    codes[surface] = (ids[surface] << 32) + np.asarray(axis.vertex, dtype=np.int64)[surface]
    voxel = np.asarray(axis.voxel, dtype=np.int64)[~surface]
    codes[~surface] = -(1 + voxel[:, 0] + (voxel[:, 1] << 20) + (voxel[:, 2] << 40))
    return codes


def parcel_coverage(
    data: np.ndarray,
    data_axis: nib.cifti2.BrainModelAxis,
    label_keys: np.ndarray,
    label_axis: nib.cifti2.BrainModelAxis,
    parcel_keys: np.ndarray,
    sampled: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Coverage of every parcel by a dense series (T, G).

    Returns (table [roi, atlas_voxels, data_voxels, valid_voxels], mean over all
    parcel grayordinates present in the series [T, P] (= what -cifti-parcellate
    computes), mean over the valid ones [T, P]). Valid = finite at every time
    point and not constant, the rule of the volume stream, and - with `sampled`
    (bool per grayordinate of the series) - sampled from the grayordinate's own
    data rather than filled in by the surface dilation.
    """
    codes = grayordinate_codes(data_axis)
    order = np.argsort(codes, kind="stable")
    wanted = grayordinate_codes(label_axis)
    position = np.clip(np.searchsorted(codes[order], wanted), 0, codes.size - 1)
    found = codes[order][position] == wanted
    column = order[position]
    valid = np.isfinite(data).all(axis=0) & (np.ptp(data, axis=0) > 0)
    if sampled is not None:
        valid &= sampled

    mean_all = np.full((data.shape[0], len(parcel_keys)), np.nan, dtype=np.float64)
    mean_valid = mean_all.copy()
    rows = []
    for col, key in enumerate(parcel_keys):
        members = label_keys == key
        present = column[members & found]
        good = present[valid[present]]
        if present.size:
            mean_all[:, col] = data[:, present].mean(axis=1, dtype=np.float64)
        if good.size:
            mean_valid[:, col] = data[:, good].mean(axis=1, dtype=np.float64)
        rows.append((int(key), int(members.sum()), int(present.size), int(good.size)))
    table = pd.DataFrame(rows, columns=["roi", "atlas_voxels", "data_voxels", "valid_voxels"])
    return table, mean_all, mean_valid


def expand_to_parcels(series: np.ndarray, names: list[str], expected: list[str]) -> tuple[np.ndarray, list[str], bool]:
    """Columns in dlabel parcel order, with n/a columns for the parcels that
    -legacy-mode discarded. Returns (series, names, matched); when the names of
    the file are not a subset of the dlabel parcels nothing is changed."""
    if names == expected:
        return series, names, True
    if len(set(expected)) != len(expected) or len(set(names)) != len(names) or not set(names) <= set(expected):
        log("WARNING: parcel names of the ptseries do not match the dlabel label table; columns kept as they are")
        return series, names, False
    out = np.full((series.shape[0], len(expected)), np.nan, dtype=np.float64)
    source = {name: k for k, name in enumerate(names)}
    for col, name in enumerate(expected):
        if name in source:
            out[:, col] = series[:, source[name]]
    return out, list(expected), True


def load_sampled_mask(path: str | Path, data_axis: nib.cifti2.BrainModelAxis) -> np.ndarray:
    """Sampled-mask dscalar (stage 06) -> bool per grayordinate of `data_axis`,
    matched by grayordinate identity (structure + vertex/voxel), never by position."""
    img, axes = _cifti_axes(path, "sampled mask")
    model_dim = _single_axis(path, axes, nib.cifti2.BrainModelAxis, "dscalar")
    values = np.asarray(img.dataobj, dtype=np.float64)
    values = values[0] if model_dim == 1 else values[:, 0]
    codes = grayordinate_codes(axes[model_dim])
    lookup = dict(zip(codes.tolist(), values.tolist()))
    wanted = grayordinate_codes(data_axis).tolist()
    missing = sum(code not in lookup for code in wanted)
    if missing:
        raise ValueError(f"{path}: {missing} grayordinates of the dense series are not in the sampled mask")
    return np.array([lookup[code] > 0.5 for code in wanted], dtype=bool)


def apply_cifti_coverage(
    series: np.ndarray,
    parcels: pd.DataFrame,
    dtseries: str | Path,
    label_keys: np.ndarray,
    label_axis: nib.cifti2.BrainModelAxis,
    min_coverage: float,
    sampled_mask: str | Path | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Coverage rule of the volume stream for parcellated CIFTI data.

    wb_command averages every grayordinate of a parcel, including those without
    signal (outside the field of view of the EPI: constant series) and, on the
    surface, vertices that only received a neighbour's copy through the dilation
    of stage 06 (`sampled_mask`). Partly covered parcels are re-averaged over their
    valid grayordinates, parcels below `min_coverage` become n/a. All other
    parcels double as a check of -cifti-parcellate against the mean computed here.
    """
    data, data_axis = load_dtseries(dtseries)
    if data.shape[0] != series.shape[0]:
        raise ValueError(f"{dtseries}: {data.shape[0]} frames, but the parcellated series has {series.shape[0]}")
    sampled = None if is_none(sampled_mask) else load_sampled_mask(sampled_mask, data_axis)
    table, mean_all, mean_valid = parcel_coverage(data, data_axis, label_keys, label_axis, parcels["roi"].to_numpy(),
                                                  sampled)
    del data
    n_atlas = table["atlas_voxels"].to_numpy(dtype=np.float64)
    n_valid = table["valid_voxels"].to_numpy()
    fraction = np.divide(n_valid, n_atlas, out=np.zeros_like(n_atlas), where=n_atlas > 0)
    included = (n_valid > 0) & (fraction >= min_coverage)

    both = np.isfinite(series).all(axis=0) & np.isfinite(mean_all).all(axis=0)
    max_diff = float(np.abs(series[:, both] - mean_all[:, both]).max()) if both.any() else float("nan")
    scale = float(np.abs(mean_all[:, both]).max()) if both.any() else 0.0
    if both.any() and max_diff > 1e-4 * max(scale, 1.0):
        log(f"WARNING: -cifti-parcellate differs from the mean over the parcel grayordinates (max |diff| = {max_diff:.6g})")

    partial = included & (n_valid < table["data_voxels"].to_numpy())
    out = series.copy()
    out[:, partial] = mean_valid[:, partial]
    out[:, ~included] = np.nan
    coverage = parcels.copy()
    coverage["atlas_voxels"] = table["atlas_voxels"].to_numpy()
    coverage["data_voxels"] = table["data_voxels"].to_numpy()
    coverage["valid_voxels"] = n_valid
    coverage["coverage_fraction"] = fraction
    coverage["included"] = included
    report = {
        "recomputed_rois": parcels.loc[partial, "name"].tolist(),
        "parcellate_max_abs_diff": max_diff,
        "sampled_mask": None if sampled is None else str(sampled_mask),
        "unsampled_grayordinates": None if sampled is None else int((~sampled).sum()),
    }
    return out, coverage, report


def run_cifti(args: argparse.Namespace) -> int:
    if not 0.0 <= args.min_coverage <= 1.0:
        raise ValueError("--min-coverage must lie in [0, 1]")
    if not is_none(args.dtseries) and is_none(args.dlabel):
        raise ValueError("--dtseries needs --dlabel (coverage is counted per dlabel parcel)")
    series, names = load_ptseries(args.ptseries)
    names = [_clean_name(name, k + 1) for k, name in enumerate(names)]
    # empty parcels are filled with a constant by wb_command: that is not data
    bad = ~np.isfinite(series).all(axis=0) | (np.ptp(series, axis=0) == 0)
    series[:, bad] = np.nan

    matched, coverage, report = False, None, {}
    table = read_labels(args.labels)
    if not is_none(args.dlabel):
        parcels, label_keys, label_axis = load_dlabel(args.dlabel)
        parcels["name"] = [_clean_name(name, roi) for name, roi in zip(parcels["name"], parcels["roi"])]
        n_file = len(names)
        series, names, matched = expand_to_parcels(series, names, parcels["name"].tolist())
        if len(names) > n_file:
            log(f"{len(names) - n_file} dlabel parcel(s) are absent from the ptseries (no data): n/a")
        if matched and not is_none(args.dtseries):
            network = {} if table is None else dict(zip(table["index"].tolist(), table["network"].tolist()))
            parcels["network"] = [network.get(int(roi), NA) for roi in parcels["roi"]]
            series, coverage, report = apply_cifti_coverage(
                series, parcels, args.dtseries, label_keys, label_axis, args.min_coverage, args.sampled_mask)
        elif not is_none(args.dtseries):
            log("WARNING: coverage not checked (the parcels could not be matched to the dlabel)")

    # the volume stream names ROIs after the label table: same names here, so that
    # stage 10 can pair the two streams by ROI identity
    renamed: dict[str, str] = {}
    names_source = "dlabel" if matched else "parcels axis of the ptseries"
    if matched:
        mapping, reason = canonical_names(parcels, table)
        if mapping:
            renamed = {mapping[n]: n for n in names if mapping.get(n, n) != n}
            names = [mapping.get(n, n) for n in names]
            if coverage is not None:
                coverage["name"] = [mapping.get(n, n) for n in coverage["name"]]
            names_source = reason
            if renamed:
                log(f"{len(renamed)} parcel(s) take the label-table name of their label key: {list(renamed)[:4]}")
        elif table is not None:
            log(f"NOTE: dlabel parcel names kept ({reason}): the volume and surface streams cannot be paired")

    keep, handling = retained_frames(series.shape[0], read_censor(args.censor), args.censor_mode)
    missing = [name for name, ok in zip(names, np.isfinite(series).all(axis=0)) if not ok]
    if missing:
        log(f"{len(missing)} of {len(names)} parcels are n/a: {missing[:8]}")
    info = {
        "atlas": args.atlas_name or entity_from_prefix(args.out_prefix, "atlas"),
        "strategy": args.strategy or entity_from_prefix(args.out_prefix, "desc"),
        "space": "fsLR",
        "n_volumes": int(series.shape[0]),
        "n_retained": int(keep.sum()),
        "n_rois": len(names),
        "roi_names": names,
        "missing_rois": missing,
        "min_coverage": float(args.min_coverage),
        "coverage_checked": coverage is not None,
        "column_order": "dlabel label keys, ascending" if matched else "parcels axis of the ptseries",
        "roi_names_source": names_source,
        "renamed_from_dlabel": renamed,
        # the volumetric label table may spell the same parcels differently
        "label_table_names": None if table is None else table["name"].tolist(),
        "censor_mode": args.censor_mode,
        "censor_handling": handling,
        "connectivity_written": not args.no_connectivity,
        "definition": CIFTI_DEFINITION,
        "sources": {
            "ptseries": str(args.ptseries),
            "dlabel": None if is_none(args.dlabel) else str(args.dlabel),
            "dtseries": None if is_none(args.dtseries) else str(args.dtseries),
            "labels": None if is_none(args.labels) else str(args.labels),
            "censor": None if is_none(args.censor) else str(args.censor),
        },
    }
    info.update(report)
    write_outputs(args.out_prefix, series, names, keep, info, coverage=coverage, connectivity=not args.no_connectivity)
    return 0


# ----------------------------------------------------------------------------
# grid comparison (used by the stage script to decide on resampling)
# ----------------------------------------------------------------------------

def run_gridcheck(args: argparse.Namespace) -> int:
    print("same" if same_grid(load_img(args.image), load_img(args.reference)) else "different")
    return 0


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fmriproc.timeseries", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    vol = sub.add_parser("volume", help="ROI means of a template-space BOLD series")
    vol.add_argument("--bold", required=True, help="4D BOLD on the template grid (unsmoothed)")
    vol.add_argument("--atlas", required=True, help="3D integer label image on the same grid")
    vol.add_argument("--labels", default="none", help="TSV with columns index, name, network; or 'none'")
    vol.add_argument("--mask", required=True, help="brain/coverage mask on the same grid")
    vol.add_argument("--censor", default="none", help="1 = keep, 0 = censored, one value per line; or 'none'")
    vol.add_argument("--censor-mode", default="NTRP", type=str.upper, choices=CENSOR_MODES)
    vol.add_argument("--min-coverage", type=float, default=0.5)
    vol.add_argument("--atlas-name", default="", help="recorded in the JSON (default: parsed from --out-prefix)")
    vol.add_argument("--strategy", default="", help="recorded in the JSON (default: parsed from --out-prefix)")
    vol.add_argument("--no-connectivity", action="store_true", help="skip the FC matrix (pre-denoise series)")
    vol.add_argument("--chunk-mb", type=float, default=256.0, help="memory budget for one block of volumes")
    vol.add_argument("--out-prefix", required=True)
    vol.set_defaults(func=run_volume)

    cif = sub.add_parser("cifti", help="TSV + FC from a parcellated CIFTI series (.ptseries.nii)")
    cif.add_argument("--ptseries", required=True, help="output of wb_command -cifti-parcellate <dtseries> <dlabel> COLUMN")
    cif.add_argument("--dlabel", default="none",
                     help="the dlabel used for the parcellation: restores parcels dropped by -legacy-mode as n/a columns")
    cif.add_argument("--dtseries", default="none",
                     help="the dense series that was parcellated: enables the coverage table and the min-coverage rule")
    cif.add_argument("--labels", default="none",
                     help="volumetric label table (index, name, network): network column of the coverage table, names recorded in the JSON")
    cif.add_argument("--sampled-mask", default="none",
                     help="dscalar from stage 06: 1 = grayordinate sampled from its own data, 0 = only filled in "
                          "by the surface dilation (does not count as covered); needs --dtseries")
    cif.add_argument("--censor", default="none")
    cif.add_argument("--censor-mode", default="NTRP", type=str.upper, choices=CENSOR_MODES)
    cif.add_argument("--min-coverage", type=float, default=0.5)
    cif.add_argument("--atlas-name", default="")
    cif.add_argument("--strategy", default="")
    cif.add_argument("--no-connectivity", action="store_true", help="skip the FC matrix (pre-denoise series)")
    cif.add_argument("--out-prefix", required=True)
    cif.set_defaults(func=run_cifti)

    grid = sub.add_parser("gridcheck", help="print 'same' or 'different' for two image grids")
    grid.add_argument("--image", required=True)
    grid.add_argument("--reference", required=True)
    grid.set_defaults(func=run_gridcheck)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError) as err:
        log(f"ERROR: {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
