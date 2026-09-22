"""Anatomical QC metrics for stage 02 (writes sub-X_desc-anatqc.json).

Keys follow docs/DESIGN.md section 10. Everything template related is computed
on the 1 mm template grid: Dice of the warped subject brain mask with the
template brain mask, Pearson correlation of the warped brain with the template
brain inside the template mask, and statistics of the Jacobian determinant of
the T1w->template warp inside the template mask.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from fmriproc.utils import load_img, same_grid, write_json

NA_STRINGS = {"", "n/a", "na", "nan", "none", "null", "-"}
# ITK and FSL write the same 1 mm grid with slightly different float rounding of
# the sform/qform; a real grid mismatch is at least a voxel (1 mm).
GRID_ATOL_MM = 1e-2
JACOBIAN_MISSING = {
    "jacobian_p01": float("nan"),
    "jacobian_p50": float("nan"),
    "jacobian_p99": float("nan"),
    "jacobian_nonpos_frac": float("nan"),
}


# ----------------------------------------------------------------------------
# metric functions (pure numpy, unit tested)
# ----------------------------------------------------------------------------

def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice overlap of two masks (non-zero = inside). NaN when both are empty."""
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} vs {b.shape}")
    total = int(a.sum()) + int(b.sum())
    if total == 0:
        return float("nan")
    return 2.0 * int(np.logical_and(a, b).sum()) / total


def masked_correlation(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Pearson r between two images over the finite voxels of `mask`."""
    mask = np.asarray(mask) > 0
    if x.shape != mask.shape or y.shape != mask.shape:
        raise ValueError("images and mask must share one grid")
    xv = np.asarray(x[mask], dtype=np.float64)
    yv = np.asarray(y[mask], dtype=np.float64)
    ok = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[ok], yv[ok]
    if xv.size < 3 or xv.std() == 0 or yv.std() == 0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def jacobian_stats(jacobian: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Percentiles (1, 50, 99) of the Jacobian determinant inside `mask` and the
    fraction of in-mask voxels that are folded or collapsed (<= 0, or not finite)."""
    mask = np.asarray(mask) > 0
    if jacobian.shape != mask.shape:
        raise ValueError("Jacobian and mask must share one grid")
    values = np.asarray(jacobian[mask], dtype=np.float64)
    nan = float("nan")
    if values.size == 0:
        return dict(JACOBIAN_MISSING)
    finite = np.isfinite(values)
    bad = int((~finite).sum()) + int((values[finite] <= 0).sum())
    if finite.any():
        p01, p50, p99 = (float(v) for v in np.percentile(values[finite], [1, 50, 99]))
    else:
        p01 = p50 = p99 = nan
    return {
        "jacobian_p01": p01,
        "jacobian_p50": p50,
        "jacobian_p99": p99,
        "jacobian_nonpos_frac": bad / values.size,
    }


def voxel_volume_mm3(img: nib.spatialimages.SpatialImage) -> float:
    return float(abs(np.linalg.det(img.affine[:3, :3])))


def mask_voxels(img: nib.spatialimages.SpatialImage) -> int:
    return int(np.count_nonzero(np.asanyarray(img.dataobj) > 0))


def parse_optional_int(text: str | None) -> int | None:
    """'-12' -> -12; 'n/a', '' or None -> None (FreeSurfer-free runs have no Euler number)."""
    if text is None or text.strip().lower() in NA_STRINGS:
        return None
    return int(float(text))


def holes_from_euler(euler: int | None) -> int | None:
    """Number of topological defects of one closed surface: euler = 2 - 2 * holes."""
    if euler is None:
        return None
    return max(0, (2 - euler) // 2)


# ----------------------------------------------------------------------------
# assembly
# ----------------------------------------------------------------------------

def _data(img: nib.spatialimages.SpatialImage) -> np.ndarray:
    data = img.get_fdata(dtype=np.float32)
    return data[..., 0] if data.ndim == 4 else data


def _on_grid(path: Path, reference: nib.spatialimages.SpatialImage, what: str) -> np.ndarray:
    img = load_img(path)
    if not same_grid(img, reference, atol=GRID_ATOL_MM):
        raise ValueError(f"{what} ({path}) is not on the template grid")
    return _data(img)


def template_metrics(
    warped_brain: Path, warped_mask: Path, template_brain: Path, template_mask: Path, jacobian: Path | None
) -> dict[str, float]:
    tpl_mask_img = load_img(template_mask)
    tpl_mask = _data(tpl_mask_img) > 0
    if not tpl_mask.any():
        raise ValueError(f"template mask is empty: {template_mask}")
    out: dict[str, float] = {
        "norm_dice": dice(_on_grid(warped_mask, tpl_mask_img, "warped brain mask"), tpl_mask),
        "template_corr": masked_correlation(
            _on_grid(warped_brain, tpl_mask_img, "warped brain"),
            _on_grid(template_brain, tpl_mask_img, "template brain"),
            tpl_mask,
        ),
    }
    if jacobian is not None:
        out.update(jacobian_stats(_on_grid(jacobian, tpl_mask_img, "Jacobian"), tpl_mask))
    else:
        out.update(JACOBIAN_MISSING)
    return out


def compute(args: argparse.Namespace) -> dict:
    euler_lh = parse_optional_int(args.euler_lh)
    euler_rh = parse_optional_int(args.euler_rh)
    holes_lh = parse_optional_int(args.holes_lh)
    holes_rh = parse_optional_int(args.holes_rh)
    holes_lh = holes_from_euler(euler_lh) if holes_lh is None else holes_lh
    holes_rh = holes_from_euler(euler_rh) if holes_rh is None else holes_rh
    holes_total = None if holes_lh is None or holes_rh is None else holes_lh + holes_rh

    brain_mask = load_img(args.brain_mask)
    result: dict = {
        "anat_mode": args.anat_mode,
        "euler_lh": euler_lh,
        "euler_rh": euler_rh,
        "holes_total": holes_total,
        "brain_volume_mm3": round(mask_voxels(brain_mask) * voxel_volume_mm3(brain_mask), 3),
        "norm_quality": args.norm_quality,
    }
    result.update(
        template_metrics(args.warped_brain, args.warped_mask, args.template_brain, args.template_mask, args.jacobian)
    )
    result["wm_voxels"] = mask_voxels(load_img(args.wm_mask))
    result["csf_voxels"] = mask_voxels(load_img(args.csf_mask))
    result["gm_voxels"] = mask_voxels(load_img(args.gm_mask))

    # provenance (scalar extras; not part of the minimal contract)
    result["template"] = args.template_name
    result["t1w_source"] = args.t1w_source
    result["wm_erode"] = args.wm_erode
    result["csf_erode"] = args.csf_erode
    result["csf_erode_requested"] = args.csf_erode_requested
    result["csf_erosion_relaxed"] = (
        args.csf_erode is not None and args.csf_erode_requested is not None and args.csf_erode < args.csf_erode_requested
    )
    result["pipeline_version"] = args.pipeline_version
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fmriproc.anat_qc", description=__doc__.splitlines()[0])
    p.add_argument("--anat-mode", required=True, choices=["freesurfer", "synth"])
    p.add_argument("--norm-quality", required=True, choices=["precise", "quick"])
    p.add_argument("--brain-mask", required=True, type=Path, help="T1w-space brain mask")
    p.add_argument("--wm-mask", required=True, type=Path)
    p.add_argument("--csf-mask", required=True, type=Path)
    p.add_argument("--gm-mask", required=True, type=Path)
    p.add_argument("--warped-brain", required=True, type=Path, help="subject brain on the 1 mm template grid")
    p.add_argument("--warped-mask", required=True, type=Path, help="subject brain mask on the 1 mm template grid")
    p.add_argument("--template-brain", required=True, type=Path)
    p.add_argument("--template-mask", required=True, type=Path)
    p.add_argument("--jacobian", type=Path, default=None, help="Jacobian determinant of the warp (template grid)")
    p.add_argument("--euler-lh", default="n/a", help="Euler number of lh.orig.nofix, or n/a")
    p.add_argument("--euler-rh", default="n/a")
    p.add_argument("--holes-lh", default="n/a", help="defects reported by mris_euler_number; derived from Euler if n/a")
    p.add_argument("--holes-rh", default="n/a")
    p.add_argument("--template-name", default=None)
    p.add_argument("--t1w-source", default=None)
    p.add_argument("--wm-erode", type=int, default=None, help="erosion iterations actually used")
    p.add_argument("--csf-erode", type=int, default=None, help="erosion iterations actually used")
    p.add_argument("--csf-erode-requested", type=int, default=None)
    p.add_argument("--pipeline-version", default=None)
    p.add_argument("--out", required=True, type=Path, help="sub-X_desc-anatqc.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = compute(args)
    except (OSError, ValueError, nib.filebasedimages.ImageFileError) as err:
        print(f"anat_qc: {err}", file=sys.stderr)
        return 1
    write_json(args.out, result)
    print(
        "anat_qc: norm_dice={norm_dice:.4f} template_corr={template_corr:.4f} "
        "jacobian_nonpos_frac={jacobian_nonpos_frac:.3g} holes_total={holes_total}".format(**result)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
