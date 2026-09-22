"""Helpers of the surface branch (stage 06): grid checks, medial-wall ROI, surface QC.

Sub-commands (``python -m fmriproc.surface_utils <cmd> ...``):

check-grid     print same | reordered | different for two volume grids (exit 0 only for same)
voxel-subdiv   print the -voxel-subdiv value for wb_command -volume-to-surface-mapping
label-to-roi   GIFTI label file (non-zero key = cortex) -> metric ROI (1 = cortex)
surfqc         write <RUN>_desc-surfqc.json
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np

from fmriproc.utils import read_json, write_json

CORTEX_STRUCTURES = ("CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT")


# ----------------------------------------------------------------------------
# volume grids
# ----------------------------------------------------------------------------

def grid_of(path: str | Path) -> tuple[tuple[int, int, int], np.ndarray]:
    """Spatial shape and voxel-to-world matrix from the header (no data is read)."""
    img = nib.load(str(path))
    shape = tuple(int(n) for n in img.shape[:3])
    return shape, np.asarray(img.affine, dtype=np.float64)  # type: ignore[return-value]


def _describe(path: str | Path, shape: tuple[int, ...], affine: np.ndarray) -> str:
    with np.printoptions(precision=4, suppress=True):
        return f"{path}: dims {shape}\n{affine}"


def grid_relation(image: str | Path, reference: str | Path, atol: float = 1e-3) -> tuple[str, str]:
    """(relation, human-readable description of both grids).

    ``same``       dimensions and voxel-to-world matrix agree: usable as they are.
    ``reordered``  the voxel centres coincide but the storage order differs (axes
                   flipped or permuted, e.g. RAS written by one tool and LAS by
                   another): an ENCLOSING_VOXEL resampling is a lossless fix.
    ``different``  another lattice (resolution, origin or field of view).
    """
    img_a, img_b = nib.load(str(image)), nib.load(str(reference))
    shape_a, shape_b = tuple(int(n) for n in img_a.shape[:3]), tuple(int(n) for n in img_b.shape[:3])
    affine_a = np.asarray(img_a.affine, dtype=np.float64)
    affine_b = np.asarray(img_b.affine, dtype=np.float64)
    text = _describe(image, shape_a, affine_a) + "\n" + _describe(reference, shape_b, affine_b)
    if shape_a == shape_b and np.allclose(affine_a, affine_b, atol=atol):
        return "same", text
    canon_a, shape_ca = _canonical_grid(affine_a, shape_a)
    canon_b, shape_cb = _canonical_grid(affine_b, shape_b)
    if shape_ca == shape_cb and np.allclose(canon_a, canon_b, atol=atol):
        return "reordered", text
    return "different", text


def _canonical_grid(affine: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, tuple[int, ...]]:
    """Affine and shape after reordering the axes to RAS+ (header arithmetic only)."""
    ornt = nib.orientations.io_orientation(affine)
    transform = nib.orientations.inv_ornt_aff(ornt, shape)
    new_shape = tuple(int(shape[int(axis)]) for axis in np.argsort(ornt[:, 0]))
    return affine @ transform, new_shape


def compare_grids(image: str | Path, reference: str | Path, atol: float = 1e-3) -> tuple[bool, str]:
    """(same grid?, human-readable description of both grids)."""
    relation, text = grid_relation(image, reference, atol)
    return relation == "same", text


def choose_voxel_subdiv(voxel_size: Sequence[float], threshold_mm: float = 3.5, fine: int = 5, coarse: int = 7) -> int:
    """-voxel-subdiv for the ribbon mapping.

    Measured on FreeSurfer's bert: the wb_command default (3) leaves 0.5 % of the
    cortical vertices without any voxel at 3 mm and 5.7 % at 4 mm; 5 resp. 7
    bring that below 0.1 %.
    """
    sizes = [abs(float(v)) for v in list(voxel_size)[:3]]
    if not sizes:
        raise ValueError("voxel_size is empty")
    return coarse if max(sizes) >= threshold_mm else fine


# ----------------------------------------------------------------------------
# GIFTI / CIFTI
# ----------------------------------------------------------------------------

def load_metric(path: str | Path) -> np.ndarray:
    """First data array of a GIFTI metric/label file as a 1D array."""
    img = nib.load(str(path))
    if not isinstance(img, nib.gifti.GiftiImage) or not img.darrays:
        raise ValueError(f"not a GIFTI data file: {path}")
    return np.asarray(img.darrays[0].data).reshape(-1)


def save_metric(data: np.ndarray, path: str | Path, structure: str | None = None) -> None:
    """Write a single-column float32 metric (atomic). ``structure`` = CortexLeft | CortexRight."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    darray = nib.gifti.GiftiDataArray(
        np.asarray(data, dtype=np.float32), intent="NIFTI_INTENT_SHAPE", datatype="NIFTI_TYPE_FLOAT32"
    )
    img = nib.gifti.GiftiImage(darrays=[darray])
    if structure:
        img.meta["AnatomicalStructurePrimary"] = structure
    tmp = path.with_name(f".tmp{os.getpid()}_{path.name}")
    try:
        nib.save(img, str(tmp))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def label_gii_to_roi(label_path: str | Path, out_path: str | Path, structure: str | None = None) -> int:
    """TemplateFlow ``desc-nomedialwall_dparc.label.gii`` -> ROI metric; returns the cortex vertex count.

    The key/name of the cortex label is not documented, so every non-zero key
    counts as cortex (the rule niworkflows uses). fsLR-32k must give 29696 (L)
    or 29716 (R) vertices.
    """
    roi = (load_metric(label_path) > 0).astype(np.float32)
    save_metric(roi, out_path, structure)
    return int(roi.sum())


def cifti_structure_values(path: str | Path, structures: Sequence[str] | None = None,
                           exclude: Sequence[str] = ()) -> np.ndarray:
    """Values of the first map of a dscalar for the given brain structures."""
    img = nib.load(str(path))
    if not isinstance(img, nib.Cifti2Image):
        raise ValueError(f"not a CIFTI file: {path}")
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    axis = img.header.get_axis(1)
    if not isinstance(axis, nib.cifti2.BrainModelAxis):
        raise ValueError(f"{path}: second dimension is not a brain-model axis")
    chunks = [
        data[0, index]
        for name, index, _ in axis.iter_structures()
        if (structures is None or name in structures) and name not in exclude
    ]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


# ----------------------------------------------------------------------------
# surface QC
# ----------------------------------------------------------------------------

def pct_bad_vertices(bad: Sequence[np.ndarray], roi: Sequence[np.ndarray]) -> float:
    """% of native cortex-ROI vertices whose ribbon polyhedron met no usable voxel (hemispheres pooled)."""
    n_bad = n_roi = 0
    for bad_h, roi_h in zip(bad, roi):
        if bad_h.shape != roi_h.shape:
            raise ValueError("bad-vertex metric and cortex ROI have different vertex counts")
        inside = roi_h > 0
        n_roi += int(inside.sum())
        n_bad += int(((bad_h > 0) & inside).sum())
    return 100.0 * n_bad / n_roi if n_roi else float("nan")


def goodvoxel_stats(ribbon: np.ndarray, goodvoxels: np.ndarray, datamask: np.ndarray) -> dict[str, float | int]:
    """Exclusions of the HCP CoV rule inside the cortical ribbon.

    ``pct_goodvoxels_excluded`` counts only ribbon voxels that have data, so a
    cut field of view shows up in ``pct_ribbon_outside_mask`` instead.
    """
    if not (ribbon.shape == goodvoxels.shape == datamask.shape):
        raise ValueError("ribbon, goodvoxels and mask must share one grid")
    rib = ribbon > 0
    with_data = rib & (datamask > 0)
    n_rib, n_data = int(rib.sum()), int(with_data.sum())
    n_excluded = int((with_data & ~(goodvoxels > 0)).sum())
    return {
        "n_ribbon_voxels": n_rib,
        "pct_ribbon_outside_mask": 100.0 * (n_rib - n_data) / n_rib if n_rib else float("nan"),
        "pct_goodvoxels_excluded": 100.0 * n_excluded / n_data if n_data else float("nan"),
    }


def _load_volume(path: str | Path) -> np.ndarray:
    data = np.asanyarray(nib.load(str(path)).dataobj)
    return data[..., 0] if data.ndim == 4 else data


def compute_surfqc(
    badvert: Sequence[str | Path],
    roi: Sequence[str | Path],
    ribbon: str | Path,
    goodvoxels: str | Path,
    datamask: str | Path,
    tsnr: str | Path,
    valid: Sequence[str | Path] = (),
    atlasroi: Sequence[str | Path] = (),
) -> dict[str, float | int]:
    qc: dict[str, float | int] = {}
    qc["pct_badvertices"] = pct_bad_vertices([load_metric(p) for p in badvert], [load_metric(p) for p in roi])
    qc.update(goodvoxel_stats(_load_volume(ribbon), _load_volume(goodvoxels), _load_volume(datamask)))

    cortex = cifti_structure_values(tsnr, CORTEX_STRUCTURES)
    usable = np.isfinite(cortex) & (cortex > 0)
    qc["tsnr_cortex_median"] = float(np.median(cortex[usable])) if usable.any() else float("nan")
    qc["n_vertices_valid"] = int(usable.sum())
    qc["n_vertices_total"] = int(cortex.size)
    subcortex = cifti_structure_values(tsnr, None, exclude=CORTEX_STRUCTURES)
    usable_sub = np.isfinite(subcortex) & (subcortex > 0)
    qc["tsnr_subcortex_median"] = float(np.median(subcortex[usable_sub])) if usable_sub.any() else float("nan")
    qc["n_subcortex_voxels_valid"] = int(usable_sub.sum())
    qc["n_subcortex_voxels_total"] = int(subcortex.size)

    # fsLR vertices inside the atlas cortex ROI that received no data from the
    # native cortex ROI (medial-wall mismatch between the subject and the fsLR
    # atlasroi). Nothing dilates after -metric-resample (HCP does not either):
    # these vertices are zero in every dtseries and count as "no data" downstream.
    if valid and atlasroi:
        n_atlas = n_nodata = 0
        for valid_path, atlas_path in zip(valid, atlasroi):
            inside = load_metric(atlas_path) > 0
            n_atlas += int(inside.sum())
            n_nodata += int((inside & ~(load_metric(valid_path) > 0)).sum())
        qc["pct_fslr_vertices_nodata"] = 100.0 * n_nodata / n_atlas if n_atlas else float("nan")
    return qc


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fmriproc.surface_utils", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    grid = sub.add_parser("check-grid", help="compare the grids of two volumes")
    grid.add_argument("image", type=Path)
    grid.add_argument("reference", type=Path)
    grid.add_argument("--atol", type=float, default=1e-3)

    subdiv = sub.add_parser("voxel-subdiv", help="print the -voxel-subdiv value")
    subdiv.add_argument("--prep-info", required=True, type=Path, help="<RUN>_desc-prep_info.json (key voxel_size)")
    subdiv.add_argument("--threshold", type=float, default=3.5, help="mm; largest voxel edge at/above this -> 7")

    label = sub.add_parser("label-to-roi", help="GIFTI label -> metric ROI")
    label.add_argument("label", type=Path)
    label.add_argument("out", type=Path)
    label.add_argument("--structure", choices=["CortexLeft", "CortexRight"], default=None)

    qc = sub.add_parser("surfqc", help="write the surface QC JSON")
    qc.add_argument("--badvert", nargs=2, required=True, type=Path, metavar=("L", "R"))
    qc.add_argument("--roi", nargs=2, required=True, type=Path, metavar=("L", "R"), help="native cortex ROI metrics")
    qc.add_argument("--ribbon", required=True, type=Path)
    qc.add_argument("--goodvoxels", required=True, type=Path)
    qc.add_argument("--mask", required=True, type=Path, help="voxels with data (brain mask and mean > 0)")
    qc.add_argument("--tsnr", required=True, type=Path, help="desc-preproc_tsnr.dscalar.nii")
    qc.add_argument("--valid", nargs=2, type=Path, default=[], metavar=("L", "R"),
                    help="-valid-roi-out metrics of the preproc -metric-resample (fsLR vertices that got data)")
    qc.add_argument("--atlasroi", nargs=2, type=Path, default=[], metavar=("L", "R"))
    qc.add_argument("--voxel-subdiv", type=int, default=None)
    qc.add_argument("--strategies", nargs="*", default=[])
    qc.add_argument("--smooth-fwhm", type=float, default=0.0)
    qc.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "check-grid":
        relation, text = grid_relation(args.image, args.reference, args.atol)
        print(text, file=sys.stderr)
        print(relation)
        return 0 if relation == "same" else 1

    if args.command == "voxel-subdiv":
        voxel_size = read_json(args.prep_info).get("voxel_size")
        if not isinstance(voxel_size, (list, tuple)) or not voxel_size:
            print(f"voxel_size missing in {args.prep_info}", file=sys.stderr)
            return 1
        print(choose_voxel_subdiv(voxel_size, args.threshold))
        return 0

    if args.command == "label-to-roi":
        n_cortex = label_gii_to_roi(args.label, args.out, args.structure)
        print(f"{args.out}: {n_cortex} cortical vertices", file=sys.stderr)
        return 0 if n_cortex > 0 else 1

    qc = compute_surfqc(args.badvert, args.roi, args.ribbon, args.goodvoxels, args.mask, args.tsnr,
                        args.valid, args.atlasroi)
    if args.voxel_subdiv is not None:
        qc["voxel_subdiv"] = args.voxel_subdiv
    qc["strategies"] = " ".join(args.strategies)
    qc["surf_smooth_fwhm"] = args.smooth_fwhm
    write_json(args.out, qc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
