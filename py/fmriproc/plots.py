"""QC figures of one subject (docs/DESIGN.md section 9).

Two sub-commands write PNG files into ``derivatives/sub-X/figures``:

    python -m fmriproc.plots anat --anat-dir ... --fig-dir ... --subject sub-X ...
    python -m fmriproc.plots run  --func-dir ... --anat-dir ... --fig-dir ... --run <RUN> ...

File names are ``sub-X_<figure>.png`` and ``<RUN>_<figure>.png`` (see
``anat_figures`` / ``run_figures``). Every figure function returns the path it
wrote, or None when its inputs are missing or unusable: a figure never stops
the report. The carpet plots and tSNR maps come from the cache written by
``fmriproc.qc_metrics --cache-dir`` so that no 4D series is read twice.

Figure text is English only: the container has no CJK font.
"""
from __future__ import annotations

import argparse
import copy
import functools
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from fmriproc import utils  # noqa: E402
from fmriproc.qc_metrics import CachePaths, RunPaths, find_atlas_labels, read_series, roi_table  # noqa: E402

LOG = logging.getLogger("fmriproc.plots")

DPI = 110
# categorical series colours, fixed order (never cycled); status colours are reserved for flags
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS = {"pass": "#0ca30c", "warn": "#fab219", "fail": "#d03b3b", "n/a": "#898781"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
# contour colours are chosen for a dark grey-scale image, not for the light chart surface
CONTOUR = {"brain": "#ffd21f", "GM": "#ff6fb5", "WM": "#4db5ff", "CSF": "#35e0a5", "edge": "#ff5a4f", "pial": "#ff5a4f",
           "white": "#4db5ff"}
TISSUE_GROUPS = {1: ("GM", SERIES[0]), 2: ("WM", SERIES[1]), 3: ("CSF", SERIES[2]), 0: ("brain", MUTED)}
WM_CODES = (2, 41, 7, 46)          # FreeSurfer/SynthSeg cerebral + cerebellar white matter
PLANE_NAMES = {0: "sagittal", 1: "coronal", 2: "axial"}


def anat_figures() -> tuple[str, ...]:
    return ("anat-mosaic", "anat-template", "anat-surfaces")


def run_figures(atlases: Sequence[str]) -> tuple[str, ...]:
    """Figure names of one run, in report order."""
    names = ["boldref-mask", "epi-to-t1", "epi-to-template", "motion", "carpet", "tsnr", "confound-corr"]
    for atlas in atlases:
        names += [f"atlas-{atlas}_fc", f"atlas-{atlas}_roiqc"]
    names.append("surface-tsnr")
    return tuple(names)


def figure_path(fig_dir: Path | str, prefix: str, name: str) -> Path:
    return Path(fig_dir) / f"{prefix}_{name}.png"


# ----------------------------------------------------------------------------
# infrastructure
# ----------------------------------------------------------------------------

def tolerant(func: Callable[..., Path | None]) -> Callable[..., Path | None]:
    """A figure that cannot be drawn is reported and skipped, never fatal."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Path | None:
        try:
            return func(*args, **kwargs)
        except MemoryError:
            LOG.warning("%s: out of memory, figure skipped", func.__name__)
        except Exception as err:  # noqa: BLE001 - any broken input must only cost its own figure
            LOG.warning("%s: %s: %s (figure skipped)", func.__name__, type(err).__name__, err)
        finally:
            plt.close("all")
        return None

    return wrapper


def save_figure(fig: plt.Figure, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".tmp{os.getpid()}_{out.name}")
    fig.savefig(tmp, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor(), format="png")
    plt.close(fig)
    os.replace(tmp, out)
    return out


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def load3d(path: Path | str | None) -> nib.Nifti1Image | None:
    """3D image (first volume of a 4D one) in closest-canonical (RAS+) voxel order."""
    if path is None or not Path(path).is_file():
        return None
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    while data.ndim > 3:
        data = data[..., 0]
    data = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return nib.as_closest_canonical(nib.Nifti1Image(data, img.affine))


def on_grid(img: nib.Nifti1Image | None, ref: nib.Nifti1Image, order: int) -> np.ndarray | None:
    """Data of ``img`` on the grid of ``ref`` (same world space); order 0 = labels/masks."""
    if img is None:
        return None
    if utils.same_grid(img, ref):
        return np.asanyarray(img.dataobj)
    from nibabel.processing import resample_from_to

    out = resample_from_to(img, (ref.shape[:3], ref.affine), order=order, mode="constant", cval=0.0)
    return np.asanyarray(out.dataobj)


def crop_to(img: nib.Nifti1Image, mask: np.ndarray | None, pad: int = 4) -> nib.Nifti1Image:
    """Image cropped to the bounding box of ``mask`` (keeps the world coordinates)."""
    if mask is None or not mask.any():
        return img
    bounds = []
    for axis in range(3):
        hit = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
        bounds.append(slice(max(int(hit[0]) - pad, 0), min(int(hit[-1]) + pad + 1, mask.shape[axis])))
    return img.slicer[bounds[0], bounds[1], bounds[2]]


def tissue_boundary(values: np.ndarray) -> float:
    """Intensity between the two brightest of three classes (CSF / GM / WM in a T1w brain).

    Plain 1-D k-means on a subsample; falls back to a percentile.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 50:
        return float(np.percentile(values, 65)) if values.size else 0.5
    sample = values[:: max(1, values.size // 20000)]
    centres = np.percentile(sample, [15, 50, 85])
    for _ in range(25):
        nearest = np.abs(sample[:, None] - centres[None, :]).argmin(axis=1)
        updated = np.array([sample[nearest == k].mean() if (nearest == k).any() else centres[k] for k in range(3)])
        if np.allclose(updated, centres):
            break
        centres = updated
    centres = np.sort(centres)
    return float((centres[1] + centres[2]) / 2.0)


# ----------------------------------------------------------------------------
# mosaic engine
# ----------------------------------------------------------------------------

@dataclass
class Contour:
    data: np.ndarray
    color: str
    label: str
    level: float = 0.5
    width: float = 0.9


@dataclass
class Mesh:
    vertices: np.ndarray            # voxel coordinates of the block grid, (N, 3)
    faces: np.ndarray               # (M, 3)
    color: str
    label: str


@dataclass
class Block:
    """One background volume with overlays; drawn as one row of cuts per plane."""

    bg: np.ndarray
    zooms: tuple[float, float, float]
    label: str
    contours: list[Contour] = field(default_factory=list)
    meshes: list[Mesh] = field(default_factory=list)
    focus: np.ndarray | None = None          # chooses the cuts and the crop
    cmap: str = "gray"
    vmin: float | None = None
    vmax: float | None = None


def _plane(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(volume, index, axis=axis).T


def _bounds(block: Block, pad: int = 3) -> list[tuple[int, int]]:
    focus = block.focus if block.focus is not None and block.focus.any() else block.bg > 0
    if not focus.any():
        return [(0, n) for n in block.bg.shape]
    out = []
    for axis in range(3):
        hit = np.flatnonzero(focus.any(axis=tuple(a for a in range(3) if a != axis)))
        out.append((max(int(hit[0]) - pad, 0), min(int(hit[-1]) + pad + 1, focus.shape[axis])))
    return out


def _cuts(lo: int, hi: int, n_cuts: int) -> list[int]:
    span = hi - lo
    positions = np.linspace(lo + 0.14 * span, hi - 1 - 0.14 * span, n_cuts)
    return [int(np.clip(round(p), lo, hi - 1)) for p in positions]


def mesh_segments(vertices: np.ndarray, faces: np.ndarray, axis: int, index: float) -> np.ndarray:
    """Intersection of a triangle mesh with the plane ``coordinate[axis] == index``.

    Returns (S, 2, 2) line segments in the two remaining coordinates (ascending axis order).
    """
    dist = vertices[:, axis] - float(index)
    dist[dist == 0] = 1e-6                          # a vertex on the plane would give a degenerate case
    per_face = dist[faces]
    cross = (per_face.min(axis=1) < 0) & (per_face.max(axis=1) > 0)
    if not cross.any():
        return np.empty((0, 2, 2))
    tri, d = faces[cross], per_face[cross]
    points = np.empty((tri.shape[0], 3, 3))
    hit = np.empty((tri.shape[0], 3), dtype=bool)
    for k, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
        da, db = d[:, a], d[:, b]
        hit[:, k] = da * db < 0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(hit[:, k], da / (da - db), 0.0)
        points[:, k] = vertices[tri[:, a]] + t[:, None] * (vertices[tri[:, b]] - vertices[tri[:, a]])
    first_two = np.argsort(~hit, axis=1, kind="stable")[:, :2]       # a crossing triangle has exactly two hit edges
    segs = points[np.arange(tri.shape[0])[:, None], first_two]        # (S, 2, 3)
    keep = [a for a in range(3) if a != axis]
    return segs[:, :, keep]


def render_mosaic(blocks: list[Block], title: str, out: Path, planes: Sequence[int] = (2, 1, 0), n_cuts: int = 7,
                  colorbar_label: str | None = None, note: str | None = None) -> Path | None:
    blocks = [b for b in blocks if b is not None and b.bg is not None and np.any(b.bg)]
    if not blocks:
        return None
    width = 13.0
    panel = width / n_cuts
    rows: list[tuple[Block, int, list[tuple[int, int]]]] = []
    heights: list[float] = []
    for block in blocks:
        bounds = _bounds(block)
        for axis in planes:
            horiz, vert = [a for a in range(3) if a != axis]
            w_mm = (bounds[horiz][1] - bounds[horiz][0]) * block.zooms[horiz]
            h_mm = (bounds[vert][1] - bounds[vert][0]) * block.zooms[vert]
            heights.append(panel * float(np.clip(h_mm / max(w_mm, 1e-6), 0.5, 1.5)))
            rows.append((block, axis, bounds))
    fig = plt.figure(figsize=(width, sum(heights) + 0.9), facecolor="black")
    grid = fig.add_gridspec(len(rows), n_cuts, height_ratios=heights, hspace=0.03, wspace=0.01,
                            left=0.01, right=0.93 if colorbar_label else 0.99, top=1 - 0.55 / (sum(heights) + 0.9),
                            bottom=0.45 / (sum(heights) + 0.9))
    image = None
    previous: Block | None = None
    for r, (block, axis, bounds) in enumerate(rows):
        horiz, vert = [a for a in range(3) if a != axis]
        h0, h1 = bounds[horiz]
        v0, v1 = bounds[vert]
        aspect = block.zooms[vert] / block.zooms[horiz]
        inside = block.bg[block.bg > 0]
        vmin = block.vmin if block.vmin is not None else 0.0
        vmax = block.vmax if block.vmax is not None else (float(np.percentile(inside, 99.5)) if inside.size else 1.0)
        for c, index in enumerate(_cuts(*bounds[axis], n_cuts)):
            ax = fig.add_subplot(grid[r, c])
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            image = ax.imshow(_plane(block.bg, axis, index)[v0:v1, h0:h1], cmap=block.cmap, vmin=vmin, vmax=vmax,
                              origin="lower", aspect=aspect, interpolation="nearest")
            for contour in block.contours:
                plane = _plane(contour.data, axis, index)[v0:v1, h0:h1].astype(np.float32)
                if plane.min() < contour.level < plane.max():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ax.contour(plane, levels=[contour.level], colors=[contour.color], linewidths=contour.width)
            for mesh in block.meshes:
                segs = mesh_segments(mesh.vertices, mesh.faces, axis, index)
                if segs.size:
                    segs = segs - np.array([h0, v0], dtype=np.float64)
                    ax.add_collection(LineCollection(segs, colors=mesh.color, linewidths=0.6))
            ax.set_xlim(-0.5, h1 - h0 - 0.5)
            ax.set_ylim(-0.5, v1 - v0 - 0.5)
            if c == 0:
                if block is not previous:
                    ax.set_title(block.label, color="white", fontsize=9, loc="left", pad=3)
                if axis != 0:
                    ax.text(0.02, 0.5, "L", color="white", fontsize=8, transform=ax.transAxes, va="center")
        previous = block
    handles, seen = [], set()
    for block in blocks:
        for item in [*block.contours, *block.meshes]:
            if item.label not in seen:
                seen.add(item.label)
                handles.append(Line2D([0], [0], color=item.color, linewidth=1.6, label=item.label))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), frameon=False, fontsize=8,
                   labelcolor="white")
    if colorbar_label and image is not None:
        cax = fig.add_axes([0.945, 0.2, 0.012, 0.6])
        bar = fig.colorbar(image, cax=cax)
        bar.set_label(colorbar_label, color="white", fontsize=8)
        bar.ax.tick_params(colors="white", labelsize=7)
    fig.suptitle(title + (f"\n{note}" if note else ""), color="white", fontsize=10, x=0.01, ha="left", y=0.995)
    return save_figure(fig, out)


def _block(bg_img: nib.Nifti1Image, label: str, **kwargs: Any) -> Block:
    zooms = tuple(float(z) for z in bg_img.header.get_zooms()[:3])
    return Block(bg=np.asanyarray(bg_img.dataobj), zooms=zooms, label=label, **kwargs)


def _mask_contour(path: Path | None, ref: nib.Nifti1Image, color: str, label: str, **kwargs: Any) -> Contour | None:
    data = on_grid(load3d(path), ref, order=0)
    if data is None or not np.any(data > 0):
        return None
    return Contour((data > 0).astype(np.uint8), color, label, **kwargs)


def _edge_contour(img: nib.Nifti1Image | None, ref: nib.Nifti1Image, color: str, label: str) -> Contour | None:
    """Grey/white boundary of a T1w brain image as an iso-intensity contour."""
    data = on_grid(img, ref, order=1)
    if data is None or not np.any(data > 0):
        return None
    return Contour(data, color, label, level=tissue_boundary(data[data > 0]), width=0.7)


# ----------------------------------------------------------------------------
# anatomical figures
# ----------------------------------------------------------------------------

@tolerant
def plot_anat_mosaic(anat_dir: Path, subject: str, out: Path) -> Path | None:
    """T1w with the brain mask and the tissue masks used for the nuisance signals."""
    anat_dir = Path(anat_dir)
    t1 = load3d(anat_dir / f"{subject}_desc-preproc_T1w.nii.gz")
    if t1 is None:
        return None
    brain = on_grid(load3d(anat_dir / f"{subject}_desc-brain_mask.nii.gz"), t1, order=0)
    t1 = crop_to(t1, brain > 0 if brain is not None else None)
    contours = [
        _mask_contour(anat_dir / f"{subject}_desc-brain_mask.nii.gz", t1, CONTOUR["brain"], "brain mask"),
        _mask_contour(anat_dir / f"{subject}_label-GM_mask.nii.gz", t1, CONTOUR["GM"], "GM mask", width=0.6),
        _mask_contour(anat_dir / f"{subject}_label-WM_mask.nii.gz", t1, CONTOUR["WM"], "WM mask (eroded)", width=0.6),
        _mask_contour(anat_dir / f"{subject}_label-CSF_mask.nii.gz", t1, CONTOUR["CSF"], "CSF mask (eroded)", width=0.6),
    ]
    block = _block(t1, "bias-corrected T1w", contours=[c for c in contours if c is not None])
    return render_mosaic([block], f"{subject}: T1w, brain mask and tissue masks", out)


@tolerant
def plot_anat_template(anat_dir: Path, subject: str, template: str, template_brain: Path | None,
                       template_mask: Path | None, out: Path) -> Path | None:
    """T1w -> template: template edges on the warped T1w and warped-T1w edges on the template."""
    anat_dir = Path(anat_dir)
    warped = load3d(anat_dir / f"{subject}_space-{template}_desc-preproc_T1w.nii.gz")
    if warped is None:
        return None
    tpl = load3d(template_brain)
    mask_path = template_mask if template_mask and Path(template_mask).is_file() else None
    focus = on_grid(load3d(mask_path), warped, order=0) if mask_path else None
    warped = crop_to(warped, focus > 0 if focus is not None else np.asanyarray(warped.dataobj) > 0)
    own_mask = _mask_contour(anat_dir / f"{subject}_space-{template}_desc-brain_mask.nii.gz", warped,
                             CONTOUR["brain"], "warped subject brain mask")
    tpl_mask = _mask_contour(mask_path, warped, CONTOUR["CSF"], "template brain mask")
    first = [c for c in (_edge_contour(tpl, warped, CONTOUR["edge"], "template GM/WM edge"), tpl_mask, own_mask)
             if c is not None]
    blocks = [_block(warped, f"warped T1w ({template}) + template contours", contours=first)]
    if tpl is not None:
        tpl_on = nib.Nifti1Image(on_grid(tpl, warped, order=1), warped.affine, warped.header)
        second = [c for c in (_edge_contour(warped, warped, CONTOUR["WM"], "subject GM/WM edge"),) if c is not None]
        blocks.append(_block(tpl_on, "template + warped-T1w contours", contours=second))
    return render_mosaic(blocks, f"{subject}: T1w -> {template} normalisation", out)


def _load_surface(path: Path, ref: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    gii = nib.load(str(path))
    coords = np.asarray(gii.agg_data("pointset"), dtype=np.float64)
    faces = np.asarray(gii.agg_data("triangle"), dtype=np.int64)
    vox = nib.affines.apply_affine(np.linalg.inv(ref.affine), coords)
    return vox, faces


@tolerant
def plot_anat_surfaces(anat_dir: Path, subject: str, out: Path) -> Path | None:
    """White and pial surfaces (scanner coordinates) cut by the T1w slices."""
    anat_dir = Path(anat_dir)
    t1 = load3d(anat_dir / f"{subject}_desc-preproc_T1w.nii.gz")
    if t1 is None:
        return None
    brain = on_grid(load3d(anat_dir / f"{subject}_desc-brain_mask.nii.gz"), t1, order=0)
    t1 = crop_to(t1, brain > 0 if brain is not None else None)
    meshes = []
    for kind in ("white", "pial"):
        for hemi in ("L", "R"):
            loaded = _load_surface(anat_dir / f"{subject}_hemi-{hemi}_{kind}.surf.gii", t1)
            if loaded is not None:
                meshes.append(Mesh(loaded[0], loaded[1], CONTOUR[kind], f"{kind} surface"))
    if not meshes:
        return None
    block = _block(t1, "T1w + FreeSurfer surfaces", meshes=meshes)
    return render_mosaic([block], f"{subject}: white and pial surfaces on the T1w", out, planes=(2, 1))


# ----------------------------------------------------------------------------
# functional registration figures
# ----------------------------------------------------------------------------

@tolerant
def plot_boldref_mask(paths: RunPaths, out: Path) -> Path | None:
    boldref = load3d(paths.func("space-T1w_boldref.nii.gz"))
    if boldref is None:
        return None
    contours = [_mask_contour(paths.func("space-T1w_desc-brain_mask.nii.gz"), boldref, CONTOUR["brain"], "BOLD brain mask")]
    for name in ("GM", "WM", "CSF"):
        contours.append(_mask_contour(paths.func(f"space-T1w_label-{name}_mask.nii.gz"), boldref, CONTOUR[name],
                                      f"{name} mask on the BOLD grid", width=0.6))
    contours = [c for c in contours if c is not None]
    focus = contours[0].data > 0 if contours else None
    block = _block(boldref, "BOLD reference (T1w space)", contours=contours, focus=focus)
    return render_mosaic([block], f"{paths.run}: BOLD reference, brain mask and nuisance masks", out)


def _anat_wm(paths: RunPaths, ref: nib.Nifti1Image) -> Contour | None:
    """Un-eroded white matter: aseg labels, else the BBR mask, else the eroded nuisance mask."""
    dseg = on_grid(load3d(paths.anat("desc-aseg_dseg.nii.gz")), ref, order=0)
    if dseg is not None and np.isin(np.rint(dseg), WM_CODES).any():
        return Contour(np.isin(np.rint(dseg), WM_CODES).astype(np.uint8), CONTOUR["WM"], "T1w white matter", width=0.7)
    for suffix, label in (("label-WMbbr_mask.nii.gz", "T1w white matter"), ("label-WM_mask.nii.gz", "T1w WM mask (eroded)")):
        contour = _mask_contour(paths.anat(suffix), ref, CONTOUR["WM"], label, width=0.7)
        if contour is not None:
            return contour
    return None


@tolerant
def plot_epi_to_t1(paths: RunPaths, out: Path) -> Path | None:
    """EPI -> T1w: anatomical white-matter and brain outlines on the BOLD reference, and
    the BOLD mask outline on the T1w (field of view, signal loss)."""
    boldref = load3d(paths.func("space-T1w_boldref.nii.gz"))
    t1 = load3d(paths.anat("desc-preproc_T1w.nii.gz"))
    if boldref is None:
        return None
    if t1 is None:
        return plot_boldref_mask.__wrapped__(paths, out)
    brain = on_grid(load3d(paths.anat("desc-brain_mask.nii.gz")), t1, order=0)
    t1 = crop_to(t1, brain > 0 if brain is not None else None)
    bold_on_t1 = nib.Nifti1Image(on_grid(boldref, t1, order=1), t1.affine, t1.header)
    first = [c for c in (_anat_wm(paths, t1),
                         _mask_contour(paths.anat("desc-brain_mask.nii.gz"), t1, CONTOUR["brain"], "T1w brain mask"))
             if c is not None]
    second = [c for c in (_mask_contour(paths.func("space-T1w_desc-brain_mask.nii.gz"), t1, CONTOUR["edge"],
                                        "BOLD brain mask"),) if c is not None]
    focus = first[-1].data > 0 if first else None
    blocks = [
        _block(bold_on_t1, "BOLD reference resampled to the T1w grid + T1w contours", contours=first, focus=focus),
        _block(t1, "T1w + BOLD mask outline (coverage / signal loss)", contours=second, focus=focus),
    ]
    return render_mosaic(blocks, f"{paths.run}: EPI -> T1w coregistration", out)


@tolerant
def plot_epi_to_template(paths: RunPaths, template_brain: Path | None, template_mask: Path | None, out: Path) -> Path | None:
    boldref = load3d(paths.tpl("boldref.nii.gz"))
    if boldref is None:
        return None
    tpl = load3d(template_brain)
    contours = [
        _mask_contour(template_mask, boldref, CONTOUR["brain"], "template brain mask"),
        _edge_contour(tpl, boldref, CONTOUR["WM"], "template GM/WM edge"),
    ]
    contours = [c for c in contours if c is not None]
    focus = contours[0].data > 0 if contours and contours[0].label == "template brain mask" else None
    blocks = [_block(boldref, f"BOLD reference in {paths.template} space + template contours", contours=contours, focus=focus)]
    if tpl is not None:
        tpl_on = nib.Nifti1Image(on_grid(tpl, boldref, order=1), boldref.affine, boldref.header)
        own = [c for c in (_mask_contour(paths.tpl("desc-brain_mask.nii.gz"), boldref, CONTOUR["edge"], "BOLD brain mask"),)
               if c is not None]
        blocks.append(_block(tpl_on, "template + BOLD mask outline", contours=own, focus=focus))
    return render_mosaic(blocks, f"{paths.run}: EPI -> {paths.template}", out)


# ----------------------------------------------------------------------------
# time-series figures
# ----------------------------------------------------------------------------

def _read_table(path: Path) -> pd.DataFrame | None:
    if not Path(path).is_file():
        return None
    try:
        return utils.read_tsv(path)
    except Exception as err:  # noqa: BLE001
        LOG.warning("cannot read %s: %s", path, err)
        return None


def _keep_vector(paths: RunPaths, n_frames: int) -> np.ndarray:
    if paths.censor.is_file():
        try:
            keep = utils.read_1d(paths.censor)[:, 0] > 0.5
            if keep.size == n_frames:
                return keep
        except Exception as err:  # noqa: BLE001
            LOG.warning("cannot read %s: %s", paths.censor, err)
    return np.ones(n_frames, dtype=bool)


def _shade(ax: plt.Axes, keep: np.ndarray, alpha: float = 0.22) -> None:
    """Shade censored frames (contiguous stretches are merged)."""
    censored = np.flatnonzero(~keep)
    if censored.size == 0:
        return
    breaks = np.flatnonzero(np.diff(censored) > 1)
    starts = np.r_[censored[0], censored[breaks + 1]]
    stops = np.r_[censored[breaks], censored[-1]]
    for a, b in zip(starts, stops):
        ax.axvspan(a - 0.5, b + 0.5, color=STATUS["fail"], alpha=alpha, linewidth=0)


def _fd_threshold(paths: RunPaths) -> float | None:
    try:
        value = float((utils.read_json(paths.confounds_json, default={}).get("censor") or {}).get("fd_threshold"))
    except (TypeError, ValueError, OSError):
        return None
    return value if value > 0 else None


@tolerant
def plot_motion(paths: RunPaths, out: Path) -> Path | None:
    table = _read_table(paths.confounds)
    if table is None or not len(table):
        return None
    n_frames = len(table)
    keep = _keep_vector(paths, n_frames)
    frames = np.arange(n_frames)
    panels: list[tuple[str, list[tuple[str, np.ndarray]]]] = []
    trans = [(c[-1], table[c].to_numpy(float)) for c in ("trans_x", "trans_y", "trans_z") if c in table]
    rots = [(c[-1], np.degrees(table[c].to_numpy(float))) for c in ("rot_x", "rot_y", "rot_z") if c in table]
    if trans:
        panels.append(("translation (mm)", trans))
    if rots:
        panels.append(("rotation (deg)", rots))
    for column, label in (("framewise_displacement", "FD, Power (mm)"), ("std_dvars", "standardised DVARS"),
                          ("outlier_fraction", "outlier fraction (3dToutcount)")):
        if column in table:
            panels.append((label, [(column, pd.to_numeric(table[column], errors="coerce").to_numpy(float))]))
    if not panels:
        return None
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 1.55 * len(panels) + 0.8), sharex=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    threshold = _fd_threshold(paths)
    for ax, (label, series) in zip(axes, panels):
        style_axes(ax)
        _shade(ax, keep)
        for k, (name, values) in enumerate(series):
            ax.plot(frames, values, color=SERIES[k] if len(series) > 1 else INK, linewidth=1.0,
                    label=name if len(series) > 1 else None)
        if len(series) > 1:
            ax.legend(loc="upper right", ncol=3, fontsize=7, frameon=False)
        if label.startswith("FD") and threshold:
            ax.axhline(threshold, color=STATUS["fail"], linewidth=0.9, linestyle="--")
            ax.text(n_frames - 1, threshold, f" censor > {threshold:g} mm", color=MUTED, fontsize=7, va="bottom", ha="right")
        ax.set_ylabel(label, fontsize=8, color=INK)
    axes[-1].set_xlabel("frame (after dropping the initial volumes)", fontsize=8, color=INK)
    axes[-1].set_xlim(-0.5, n_frames - 0.5)
    n_cens = int((~keep).sum())
    fig.suptitle(f"{paths.run}: head motion and frame-wise quality; shaded = censored frames ({n_cens} of {n_frames})",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(fig, out)


def _zscore_rows(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float32)
    data = data - data.mean(axis=1, keepdims=True)
    sd = data.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return data / sd


def _group_order(groups: np.ndarray) -> np.ndarray:
    rank = {1: 0, 2: 1, 3: 2, 0: 3}
    return np.argsort(np.array([rank.get(int(g), 4) for g in groups]), kind="stable")


@tolerant
def plot_carpet(paths: RunPaths, cache: CachePaths | None, strategies: Sequence[str], out: Path) -> Path | None:
    """Voxel x time carpets before and after denoising, rows ordered GM, WM, CSF, FD on top."""
    if cache is None or not cache.carpet.is_file():
        return None
    with np.load(cache.carpet) as store:
        arrays = {k: store[k] for k in store.files}
    panels = []
    if "pre" in arrays and arrays["pre"].size:
        panels.append(("before denoising (T1w space, detrended)", arrays["pre"], arrays["pre_groups"], arrays.get("pre_frames")))
    for strategy in strategies:
        key = f"post_{strategy}"
        if key in arrays and arrays[key].size:
            panels.append((f"after denoising: {strategy} ({paths.template} space)", arrays[key], arrays[f"{key}_groups"],
                           arrays.get(f"{key}_frames")))
    if not panels:
        return None
    table = _read_table(paths.confounds)
    fd = None
    if table is not None and "framewise_displacement" in table:
        fd = pd.to_numeric(table["framewise_displacement"], errors="coerce").to_numpy(float)
    n_total = fd.size if fd is not None else max(int(p[1].shape[1]) for p in panels)
    keep = _keep_vector(paths, n_total)

    heights = [1.0] + [2.3] * len(panels)
    fig = plt.figure(figsize=(11, sum(heights) + 0.9), facecolor=SURFACE)
    grid = fig.add_gridspec(len(heights), 2, height_ratios=heights, width_ratios=[0.012, 1], hspace=0.28, wspace=0.01)
    top = fig.add_subplot(grid[0, 1])
    style_axes(top)
    _shade(top, keep)
    if fd is not None:
        top.plot(np.arange(fd.size), fd, color=INK, linewidth=0.9)
        threshold = _fd_threshold(paths)
        if threshold:
            top.axhline(threshold, color=STATUS["fail"], linewidth=0.8, linestyle="--")
    top.set_ylabel("FD (mm)", fontsize=8)
    top.set_xlim(-0.5, n_total - 0.5)
    colours = ListedColormap([TISSUE_GROUPS[k][1] for k in (0, 1, 2, 3)])
    for row, (label, data, groups, frames) in enumerate(panels, start=1):
        order = _group_order(groups)
        data, groups = _zscore_rows(data[order]), groups[order]
        ax = fig.add_subplot(grid[row, 1])
        side = fig.add_subplot(grid[row, 0])
        full_length = frames is None or data.shape[1] == n_total
        extent = (-0.5, data.shape[1] - 0.5, data.shape[0], 0)
        ax.imshow(data, cmap="gray", vmin=-2.0, vmax=2.0, aspect="auto", interpolation="nearest", extent=extent)
        if full_length:
            _shade(ax, keep, alpha=0.18)
            ax.set_xlim(-0.5, n_total - 0.5)
        else:
            label += " - censored frames already removed"
        ax.set_title(label, fontsize=8.5, loc="left", color=INK, pad=3)
        ax.set_yticks([])
        ax.tick_params(colors=MUTED, labelsize=8)
        side.imshow(np.clip(groups, 0, 3)[:, None], cmap=colours, vmin=0, vmax=3, aspect="auto", interpolation="nearest")
        side.set_xticks([])
        side.set_yticks([])
        if row == len(panels):
            ax.set_xlabel("frame", fontsize=8)
    present = sorted({int(g) for p in panels for g in np.unique(p[2]) if int(g) in TISSUE_GROUPS})
    handles = [Line2D([0], [0], color=TISSUE_GROUPS[g][1], linewidth=5, label=TISSUE_GROUPS[g][0]) for g in present]
    handles.append(Line2D([0], [0], color=STATUS["fail"], alpha=0.4, linewidth=5, label="censored frame"))
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle(f"{paths.run}: carpet plots (rows = voxels, z-scored, grey scale -2..2 SD)", fontsize=10, x=0.01,
                 ha="left", color=INK)
    return save_figure(fig, out)


@tolerant
def plot_tsnr(paths: RunPaths, cache: CachePaths | None, strategies: Sequence[str], out: Path) -> Path | None:
    """tSNR maps before and after denoising on one colour scale."""
    if cache is None:
        return None
    entries = [("pre-denoise tSNR, T1w space", cache.tsnr("preproc", "T1w")),
               (f"pre-denoise tSNR, {paths.template} space", cache.tsnr("preproc", paths.template))]
    entries += [(f"post-denoise tSNR: {s}  (pre-denoise mean / residual SD)", cache.tsnr(s, paths.template)) for s in strategies]
    images = [(label, load3d(path)) for label, path in entries]
    images = [(label, img) for label, img in images if img is not None and np.any(np.asanyarray(img.dataobj) > 0)]
    if not images:
        return None
    values = np.concatenate([np.asanyarray(i.dataobj)[np.asanyarray(i.dataobj) > 0][::7] for _, i in images])
    vmax = float(np.percentile(values, 98)) if values.size else 1.0
    blocks = [_block(img, label, cmap="viridis", vmin=0.0, vmax=vmax, focus=np.asanyarray(img.dataobj) > 0)
              for label, img in images]
    return render_mosaic(blocks, f"{paths.run}: temporal SNR before and after denoising (common colour scale)", out,
                         planes=(2,), n_cuts=9, colorbar_label="tSNR")


CONFOUND_COLUMNS = (
    "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "global_signal", "white_matter", "csf",
    "w_comp_cor_00", "w_comp_cor_01", "c_comp_cor_00", "c_comp_cor_01", "framewise_displacement", "std_dvars",
    "outlier_fraction",
)


@tolerant
def plot_confound_corr(paths: RunPaths, out: Path) -> Path | None:
    table = _read_table(paths.confounds)
    if table is None:
        return None
    columns = [c for c in CONFOUND_COLUMNS if c in table.columns]
    if len(columns) < 2:
        return None
    data = table[columns].apply(pd.to_numeric, errors="coerce")
    data = data.loc[:, data.std(axis=0, skipna=True) > 0]
    if data.shape[1] < 2:
        return None
    corr = data.corr().to_numpy()
    n = corr.shape[0]
    fig, ax = plt.subplots(figsize=(0.42 * n + 2.6, 0.42 * n + 2.0), facecolor=SURFACE)
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    # two-call form: the container's matplotlib may predate set_xticks(ticks, labels)
    ax.set_xticks(range(n))
    ax.set_xticklabels(data.columns, rotation=90, fontsize=7.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(data.columns, fontsize=7.5)
    if n <= 18:
        for i in range(n):
            for j in range(n):
                if i != j and np.isfinite(corr[i, j]):
                    ax.text(j, i, f"{corr[i, j]:.1f}".replace("0.", "."), ha="center", va="center", fontsize=6,
                            color="white" if abs(corr[i, j]) > 0.6 else INK)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("Pearson r", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    ax.set_title(f"{paths.run}: correlation between confounds", fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    return save_figure(fig, out)


def _roi_info(paths: RunPaths, atlas: str, strategy: str, columns: list[str], labels_dir: Path | None) -> pd.DataFrame:
    coverage = _read_table(paths.timeseries(atlas, strategy, "coverage"))
    if coverage is None:
        coverage = _read_table(paths.timeseries(atlas, "preproc", "coverage"))
    return roi_table(columns, coverage, find_atlas_labels(labels_dir, atlas))


def _network_order(networks: list[Any]) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Stable ROI order grouped by network, and (name, start, stop) of every group."""
    names = [n if isinstance(n, str) and n else "" for n in networks]
    if len({n for n in names if n}) < 2:
        return np.arange(len(names)), []
    first_seen: dict[str, int] = {}
    for n in names:
        first_seen.setdefault(n, len(first_seen))
    order = np.argsort(np.array([first_seen[n] for n in names]), kind="stable")
    groups, start = [], 0
    ordered = [names[i] for i in order]
    for k in range(1, len(ordered) + 1):
        if k == len(ordered) or ordered[k] != ordered[start]:
            groups.append((ordered[start] or "other", start, k))
            start = k
    return order, groups


@tolerant
def plot_fc(paths: RunPaths, atlas: str, strategies: Sequence[str], labels_dir: Path | None, out: Path) -> Path | None:
    """FC matrix per strategy (ROIs grouped by network) and the distribution of the edges."""
    matrices: list[tuple[str, np.ndarray, list[str]]] = []
    for strategy in strategies:
        table = _read_table(paths.timeseries(atlas, strategy, "connectivity"))
        if table is None or table.shape[0] < 2:
            continue
        table = table.drop(columns=[c for c in table.columns if str(c).lower() in {"roi", "name", "index"}], errors="ignore")
        values = table.apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if values.shape[0] == values.shape[1]:
            matrices.append((strategy, values, [str(c) for c in table.columns]))
    if not matrices:
        return None
    info = _roi_info(paths, atlas, matrices[0][0], matrices[0][2], labels_dir)
    order, groups = _network_order(info["network"].tolist())
    n_panels = len(matrices) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(4.1 * n_panels, 4.4), facecolor=SURFACE)
    image = None
    for ax, (strategy, values, _) in zip(axes[:-1], matrices):
        shown = values[np.ix_(order, order)] if values.shape[0] == order.size else values
        shown = shown.copy()
        np.fill_diagonal(shown, np.nan)
        cmap = copy.copy(plt.get_cmap("RdBu_r"))
        cmap.set_bad("#d9d8d2")
        image = ax.imshow(shown, cmap=cmap, vmin=-0.8, vmax=0.8, interpolation="nearest")
        for name, start, stop in groups:
            ax.axhline(stop - 0.5, color=INK, linewidth=0.4)
            ax.axvline(stop - 0.5, color=INK, linewidth=0.4)
        if groups and len(groups) <= 20:
            ax.set_yticks([(a + b - 1) / 2 for _, a, b in groups])
            ax.set_yticklabels([g[0] for g in groups], fontsize=7)
        else:
            ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(strategy, fontsize=9, color=INK)
    if image is not None:
        bar = fig.colorbar(image, ax=list(axes[:-1]), fraction=0.03, pad=0.02, orientation="horizontal")
        bar.set_label("Pearson r (retained frames); grey = ROI without coverage", fontsize=8)
        bar.ax.tick_params(labelsize=7)
    hist = axes[-1]
    style_axes(hist)
    bins = np.linspace(-1, 1, 61)
    for k, (strategy, values, _) in enumerate(matrices):
        edges = utils.upper_triangle(values)
        edges = edges[np.isfinite(edges)]
        if edges.size:
            hist.hist(edges, bins=bins, histtype="step", density=True, color=SERIES[k % len(SERIES)], linewidth=1.6,
                      label=f"{strategy}: mean {edges.mean():.2f}, SD {edges.std():.2f}")
    hist.axvline(0, color=MUTED, linewidth=0.8)
    hist.set_xlabel("edge r", fontsize=8)
    hist.set_ylabel("density", fontsize=8)
    hist.legend(fontsize=7, frameon=False, loc="upper left")
    hist.set_title("distribution of edges", fontsize=9, color=INK)
    fig.suptitle(f"{paths.run}: functional connectivity, atlas {atlas}", fontsize=10, x=0.01, ha="left", color=INK)
    return save_figure(fig, out)


@tolerant
def plot_roiqc(paths: RunPaths, atlas: str, strategies: Sequence[str], out: Path) -> Path | None:
    """Per-ROI tSNR of the final series, variance removed and coverage."""
    tables = [(s, _read_table(paths.roiqc(atlas, s))) for s in strategies]
    tables = [(s, t) for s, t in tables if t is not None and len(t)]
    if not tables:
        return None
    base = tables[0][1]
    networks = base["network"].tolist() if "network" in base else [None] * len(base)
    networks = [n if isinstance(n, str) else None for n in networks]
    order, groups = _network_order(networks)
    x = np.arange(len(base))
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.6), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [2.2, 1.4, 1.0]})
    for ax in axes:
        style_axes(ax)
        for _, _, stop in groups[:-1]:
            ax.axvline(stop - 0.5, color="#c3c2b7", linewidth=0.7)
    if "tsnr_pre" in base:
        axes[0].step(x, pd.to_numeric(base["tsnr_pre"], errors="coerce").to_numpy(float)[order], where="mid", color=MUTED,
                     linewidth=1.0, label="pre-denoise (mean / SD of the detrended ROI series)")
    for k, (strategy, table) in enumerate(tables):
        if len(table) != len(base):
            continue
        colour = SERIES[k % len(SERIES)]
        axes[0].step(x, pd.to_numeric(table["roi_tsnr"], errors="coerce").to_numpy(float)[order], where="mid",
                     color=colour, linewidth=1.3, label=f"final series: {strategy}")
        axes[1].step(x, pd.to_numeric(table["variance_removed"], errors="coerce").to_numpy(float)[order], where="mid",
                     color=colour, linewidth=1.3, label=strategy)
    axes[0].set_ylabel("ROI tSNR", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    axes[1].set_ylabel("variance removed", fontsize=8)
    axes[1].set_ylim(min(0.0, axes[1].get_ylim()[0]), 1.0)
    coverage = pd.to_numeric(base["coverage_fraction"], errors="coerce").to_numpy(float)[order]
    axes[2].bar(x, np.nan_to_num(coverage), width=0.9, color=SERIES[0])
    missing = np.flatnonzero(~np.isfinite(pd.to_numeric(tables[0][1]["roi_tsnr"], errors="coerce").to_numpy(float)[order]))
    if missing.size:
        axes[2].plot(missing, np.full(missing.size, 1.04), linestyle="none", marker="v", markersize=4,
                     color=STATUS["fail"], label="ROI without usable series (n/a)")
        axes[2].legend(fontsize=7, frameon=False, loc="lower right")
    axes[2].set_ylim(0, 1.1)
    axes[2].set_ylabel("coverage", fontsize=8)
    if groups and len(groups) <= 20:
        axes[2].set_xticks([(a + b - 1) / 2 for _, a, b in groups])
        axes[2].set_xticklabels([g[0] for g in groups], fontsize=7.5)
        axes[2].set_xlabel("ROIs grouped by network", fontsize=8)
    else:
        axes[2].set_xlabel("ROI (atlas order)", fontsize=8)
    axes[2].set_xlim(-0.5, len(base) - 0.5)
    fig.suptitle(f"{paths.run}: ROI quality, atlas {atlas}", fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, out)


def _fslr_mesh(resource_dir: Path | None, hemi: str) -> Path | None:
    if resource_dir is None or not Path(resource_dir).is_dir():
        return None
    for kind in ("inflated", "midthickness"):
        hits = sorted(Path(resource_dir).rglob(f"tpl-fsLR*hemi-{hemi}*{kind}.surf.gii"))
        hits = [h for h in hits if "32k" in h.name]
        if hits:
            return hits[0]
    return None


@tolerant
def plot_surface_tsnr(paths: RunPaths, resource_dir: Path | None, out: Path) -> Path | None:
    """Cortical tSNR on the fsLR-32k surface (needs the meshes fetched by fetch_resources)."""
    if not paths.tsnr_dscalar.is_file():
        return None
    meshes = {hemi: _fslr_mesh(resource_dir, hemi) for hemi in ("L", "R")}
    if any(m is None for m in meshes.values()):
        LOG.info("fsLR meshes not found under %s: surface tSNR figure skipped", resource_dir)
        return None
    from nilearn import plotting

    img = nib.load(str(paths.tsnr_dscalar))
    values = np.asanyarray(img.dataobj, dtype=np.float32)[0]
    axis = img.header.get_axis(1)
    per_hemi: dict[str, np.ndarray] = {}
    for name, index, model in axis.iter_structures():
        for hemi, key in (("L", "CORTEX_LEFT"), ("R", "CORTEX_RIGHT")):
            if name.endswith(key):
                full = np.full(int(model.nvertices[name]), np.nan, dtype=np.float32)
                full[model.vertex] = values[index]
                per_hemi[hemi] = full
    if not per_hemi:
        return None
    finite = np.concatenate([v[np.isfinite(v)] for v in per_hemi.values()])
    vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
    fig = plt.figure(figsize=(11, 3.2), facecolor="white")
    k = 0
    for hemi, side in (("L", "left"), ("R", "right")):
        if hemi not in per_hemi:
            continue
        gii = nib.load(str(meshes[hemi]))
        mesh = (np.asarray(gii.agg_data("pointset")), np.asarray(gii.agg_data("triangle")))
        for view in ("lateral", "medial"):
            k += 1
            ax = fig.add_subplot(1, 4, k, projection="3d")
            plotting.plot_surf(mesh, surf_map=np.nan_to_num(per_hemi[hemi]), hemi=side, view=view, cmap="viridis",
                               vmin=0.0, vmax=vmax, colorbar=(k == 4), axes=ax, figure=fig)
            ax.set_title(f"{side} {view}", fontsize=8)
    fig.suptitle(f"{paths.run}: pre-denoise tSNR on the fsLR-32k cortex", fontsize=10, x=0.01, ha="left")
    return save_figure(fig, out)


# ----------------------------------------------------------------------------
# drivers
# ----------------------------------------------------------------------------

def _fresh(path: Path) -> Path:
    """A figure that is not redrawn must not survive from an earlier run."""
    if path.exists():
        path.unlink()
    return path


def make_anat_figures(anat_dir: Path, fig_dir: Path, subject: str, template: str, template_brain: Path | None,
                      template_mask: Path | None) -> dict[str, Path | None]:
    target = {name: _fresh(figure_path(fig_dir, subject, name)) for name in anat_figures()}
    return {
        "anat-mosaic": plot_anat_mosaic(anat_dir, subject, target["anat-mosaic"]),
        "anat-template": plot_anat_template(anat_dir, subject, template, template_brain, template_mask,
                                            target["anat-template"]),
        "anat-surfaces": plot_anat_surfaces(anat_dir, subject, target["anat-surfaces"]),
    }


def make_run_figures(paths: RunPaths, fig_dir: Path, strategies: Sequence[str], atlases: Sequence[str],
                     cache: CachePaths | None, template_brain: Path | None, template_mask: Path | None,
                     labels_dir: Path | None, resource_dir: Path | None) -> dict[str, Path | None]:
    target = {name: _fresh(figure_path(fig_dir, paths.run, name)) for name in run_figures(atlases)}
    done: dict[str, Path | None] = {
        "boldref-mask": plot_boldref_mask(paths, target["boldref-mask"]),
        "epi-to-t1": plot_epi_to_t1(paths, target["epi-to-t1"]),
        "epi-to-template": plot_epi_to_template(paths, template_brain, template_mask, target["epi-to-template"]),
        "motion": plot_motion(paths, target["motion"]),
        "carpet": plot_carpet(paths, cache, strategies, target["carpet"]),
        "tsnr": plot_tsnr(paths, cache, strategies, target["tsnr"]),
        "confound-corr": plot_confound_corr(paths, target["confound-corr"]),
    }
    for atlas in atlases:
        done[f"atlas-{atlas}_fc"] = plot_fc(paths, atlas, strategies, labels_dir, target[f"atlas-{atlas}_fc"])
        done[f"atlas-{atlas}_roiqc"] = plot_roiqc(paths, atlas, strategies, target[f"atlas-{atlas}_roiqc"])
    done["surface-tsnr"] = plot_surface_tsnr(paths, resource_dir, target["surface-tsnr"])
    return done


def _optional(value: str | None) -> Path | None:
    return Path(value) if value and value.lower() != "none" else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("anat", "run"):
        cmd = sub.add_parser(name, help=f"{name} figures")
        cmd.add_argument("--anat-dir", required=True)
        cmd.add_argument("--fig-dir", required=True)
        cmd.add_argument("--template", required=True, help="template name used in the file names")
        cmd.add_argument("--template-brain", default=None, help="template T1w brain (contours)")
        cmd.add_argument("--template-mask", default=None, help="template brain mask")
        cmd.add_argument("-v", "--verbose", action="store_true")
        if name == "anat":
            cmd.add_argument("--subject", required=True)
            continue
        cmd.add_argument("--func-dir", required=True)
        cmd.add_argument("--run", required=True)
        cmd.add_argument("--mni-res", required=True)
        cmd.add_argument("--strategies", default="")
        cmd.add_argument("--atlases", default="")
        cmd.add_argument("--cache-dir", default=None, help="the --cache-dir given to fmriproc.qc_metrics")
        cmd.add_argument("--atlas-labels-dir", default=None)
        cmd.add_argument("--resource-dir", default=None, help="searched for the fsLR-32k meshes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    if args.command == "anat":
        done = make_anat_figures(Path(args.anat_dir), Path(args.fig_dir), args.subject, args.template,
                                 _optional(args.template_brain), _optional(args.template_mask))
    else:
        paths = RunPaths(Path(args.func_dir), Path(args.anat_dir), args.run, args.template, str(args.mni_res))
        cache = CachePaths(Path(args.cache_dir)) if args.cache_dir else None
        done = make_run_figures(paths, Path(args.fig_dir), args.strategies.split(), args.atlases.split(), cache,
                                _optional(args.template_brain), _optional(args.template_mask),
                                _optional(args.atlas_labels_dir), _optional(args.resource_dir))
    for name, path in done.items():
        LOG.info("%-40s %s", name, path.name if path else "skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
