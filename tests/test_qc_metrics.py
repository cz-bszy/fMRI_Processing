"""Tests of fmriproc.qc_metrics (stage 08) on a tiny synthetic derivatives tree.

The tree follows the file names of docs/DESIGN.md section 6 (few-voxel NIfTIs,
fake TSV/JSON). ``build_run`` / ``build_subject`` are shared with tests/test_report.py.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from fmriproc import qc_metrics
from fmriproc.utils import fc_matrix, read_tsv, write_json, write_tsv

SUB = "sub-01"
RUN = f"{SUB}_task-rest"
TPL = "MNI152NLin6Asym"
RES = "2"
ATLAS = "Schaefer2018_100Parcels_7Networks"
STRATEGIES = ("wmcsf24", "wmcsf24gsr")
N_T = 60
TR = 2.0
ROI_NAMES = ["7Networks_LH_Vis_1", "7Networks_LH_Vis_2", "7Networks_LH_SomMot_1", "7Networks_RH_Vis_1",
             "7Networks_RH_SomMot_1", "7Networks_RH_Default_1"]
NETWORKS = [n.split("_")[2] for n in ROI_NAMES]
UNCOVERED = 5                     # ROI column that stage 07 would set to n/a (coverage below the minimum)

ANAT_SHAPE, T1W_SHAPE, TPL_SHAPE = (24, 24, 24), (10, 10, 10), (12, 12, 12)
ANAT_AFFINE = np.eye(4)
T1W_AFFINE = np.diag([3.0, 3.0, 3.0, 1.0])
T1W_AFFINE[:3, 3] = -3.0                          # the 3 mm BOLD grid sits inside the 1 mm anat grid
TPL_AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
TPL_AFFINE[:3, 3] = -12.0


# ----------------------------------------------------------------------------
# synthetic data
# ----------------------------------------------------------------------------

def _cube(shape: tuple[int, ...], lo: int, hi: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[lo:hi + 1, lo:hi + 1, lo:hi + 1] = True
    return mask


def _save(data: np.ndarray, affine: np.ndarray, path: Path, dtype: type = np.float32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=dtype), affine), str(path))


def t1w_masks() -> dict[str, np.ndarray]:
    brain = _cube(T1W_SHAPE, 2, 7)
    dropout = brain.copy()
    dropout[:, :, :7] = False                     # top slab: inside the T1 brain, no EPI signal
    brain &= ~dropout
    wm = _cube(T1W_SHAPE, 4, 5)
    csf = np.zeros(T1W_SHAPE, dtype=bool)
    csf[2, 2, 2] = csf[2, 3, 2] = True
    gm = brain & ~wm & ~csf
    return {"brain": brain, "GM": gm, "WM": wm, "CSF": csf, "dropout": dropout}


def tpl_masks() -> dict[str, np.ndarray]:
    brain = _cube(TPL_SHAPE, 2, 9)
    wm = _cube(TPL_SHAPE, 5, 6)
    csf = np.zeros(TPL_SHAPE, dtype=bool)
    csf[2, 2, 2] = True
    gm = brain & ~_cube(TPL_SHAPE, 4, 7)
    return {"brain": brain, "GM": gm, "WM": wm, "CSF": csf}


def legendre_trend(n_t: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, n_t)
    return 200.0 * x + 100.0 * (3.0 * x ** 2 - 1.0) / 2.0


def make_pre(rng: np.random.Generator, mask: np.ndarray, n_t: int, level: float = 10000.0,
             noise: float = 50.0) -> np.ndarray:
    """4D pre-denoise series: level + polynomial trend + a shared signal + noise inside ``mask``."""
    shared = 30.0 * rng.normal(size=n_t)
    data = np.zeros(mask.shape + (n_t,), dtype=np.float32)
    n_vox = int(mask.sum())
    data[mask] = level + legendre_trend(n_t) + shared + noise * rng.normal(size=(n_vox, n_t))
    return data


def make_post(rng: np.random.Generator, mask: np.ndarray, n_t: int, keep: np.ndarray, sd: float,
              kill: bool) -> np.ndarray:
    """Zero-mean residual series (what 3dTproject leaves); KILL mode drops the censored frames."""
    frames = int(keep.sum()) if kill else n_t
    data = np.zeros(mask.shape + (frames,), dtype=np.float32)
    data[mask] = sd * rng.normal(size=(int(mask.sum()), frames))
    return data


def roi_series(rng: np.random.Generator, n_t: int, keep: np.ndarray, post_sd: float) -> tuple[np.ndarray, np.ndarray]:
    """(pre [T, R], post [T, R]) ROI-mean tables with a network structure; one ROI uncovered."""
    net_signal = {net: rng.normal(size=n_t) for net in set(NETWORKS)}
    post = np.stack([1.2 * net_signal[net] + rng.normal(size=n_t) for net in NETWORKS], axis=1) * post_sd
    post = post - post[keep].mean(axis=0)
    nuisance = 40.0 * rng.normal(size=(n_t, 1))
    pre = 10000.0 + legendre_trend(n_t)[:, None] + 1.5 * post + nuisance + 5.0 * rng.normal(size=post.shape)
    pre[:, UNCOVERED] = np.nan
    post[:, UNCOVERED] = np.nan
    return pre, post


def write_atlas_dir(root: Path, atlas: str = ATLAS) -> Path:
    """$RESOURCE_DIR/atlases/<A>/: labels.tsv (index, name, network) + dseg on the template grid."""
    base = root / atlas
    base.mkdir(parents=True, exist_ok=True)
    labels = pd.DataFrame({"index": range(1, len(ROI_NAMES) + 1), "name": ROI_NAMES, "network": NETWORKS})
    write_tsv(base / "labels.tsv", labels)
    dseg = np.zeros(TPL_SHAPE, dtype=np.int16)
    centres = [(3, 3, 3), (3, 8, 3), (3, 3, 8), (8, 3, 3), (8, 8, 3), (8, 3, 8)]
    for label, (i, j, k) in enumerate(centres, start=1):
        dseg[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2] = label
    _save(dseg, TPL_AFFINE, base / f"{atlas}_space-{TPL}_res-02_dseg.nii.gz", np.int16)
    return root


def build_anat(anat: Path, sub: str = SUB) -> None:
    """Stage-02 products on a 1 mm grid that shares world coordinates with the BOLD grid."""
    rng = np.random.default_rng(7)
    brain = _cube(ANAT_SHAPE, 2, 20)
    core = _cube(ANAT_SHAPE, 8, 14)
    t1 = np.zeros(ANAT_SHAPE, dtype=np.float32)
    t1[brain] = 100.0 + 5.0 * rng.normal(size=int(brain.sum()))
    t1[core] += 50.0
    _save(t1, ANAT_AFFINE, anat / f"{sub}_desc-preproc_T1w.nii.gz")
    _save(t1 * brain, ANAT_AFFINE, anat / f"{sub}_desc-brain_T1w.nii.gz")
    _save(brain, ANAT_AFFINE, anat / f"{sub}_desc-brain_mask.nii.gz", np.uint8)
    dseg = np.zeros(ANAT_SHAPE, dtype=np.int16)
    dseg[brain] = 3
    dseg[core] = 2
    _save(dseg, ANAT_AFFINE, anat / f"{sub}_desc-aseg_dseg.nii.gz", np.int16)
    _save(_cube(ANAT_SHAPE, 9, 13), ANAT_AFFINE, anat / f"{sub}_label-WM_mask.nii.gz", np.uint8)
    csf = np.zeros(ANAT_SHAPE, dtype=bool)
    csf[4:6, 4:6, 4:6] = True
    _save(csf, ANAT_AFFINE, anat / f"{sub}_label-CSF_mask.nii.gz", np.uint8)
    _save(brain & ~core, ANAT_AFFINE, anat / f"{sub}_label-GM_mask.nii.gz", np.uint8)
    tpl = tpl_masks()
    warped = np.zeros(TPL_SHAPE, dtype=np.float32)
    warped[tpl["brain"]] = 100.0
    warped[_cube(TPL_SHAPE, 4, 7)] = 150.0
    _save(warped, TPL_AFFINE, anat / f"{sub}_space-{TPL}_desc-preproc_T1w.nii.gz")
    _save(tpl["brain"], TPL_AFFINE, anat / f"{sub}_space-{TPL}_desc-brain_mask.nii.gz", np.uint8)
    write_json(anat / f"{sub}_desc-anatqc.json", {
        "anat_mode": "freesurfer", "euler_lh": -40, "euler_rh": -60, "holes_total": 50, "brain_volume_mm3": 1.2e6,
        "norm_quality": "precise", "norm_dice": 0.96, "template_corr": 0.91, "jacobian_p01": 0.6, "jacobian_p50": 1.0,
        "jacobian_p99": 1.8, "jacobian_nonpos_frac": 0.0, "wm_voxels": 125, "csf_voxels": 8, "gm_voxels": 6000,
    })


def build_run(func: Path, run: str, seed: int = 0, fd_level: float = 0.10, kill: bool = False,
              strategies: tuple[str, ...] = STRATEGIES, images: bool = True, validation: bool = True,
              surfqc: bool = False, censored: tuple[int, ...] = (20, 21, 40)) -> dict[str, np.ndarray]:
    """Stage 03-07 (+10) products of one run. Returns the keep vector and the ROI tables."""
    func.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    keep = np.ones(N_T, dtype=bool)
    keep[list(censored)] = False
    n_keep = int(keep.sum())
    pre = func / run
    tpl = f"{run}_space-{TPL}_res-{RES}"

    # ---- stage 03: provenance, motion files, images
    write_json(pre.with_name(f"{run}_desc-prep_info.json"), {
        "source": f"/data/{run}_bold.nii.gz", "run_label": run, "tr": TR, "n_volumes_raw": N_T + 4, "n_dropped": 4,
        "n_volumes": N_T, "nss_detected": 2, "despike": True, "despike_fraction": 0.012, "stc_applied": True,
        "stc_reason": "verified slice timing (acquisition table)", "stc_interp": "quintic",
        "slice_timing_source": "acquisition_table", "slice_timing_evidence": "B: site protocol", "tzero": 1.0,
        "hmc_reference": "median of mcflirt pass 1", "coreg_method": "bbregister", "bbr_cost": 0.45,
        "bbr_vs_init_mm": 1.2, "bbr_rejected": False, "epi_mask_method": "synthstrip", "scale_factor": 1.7,
        "func_t1w_res": 3, "mni_res": 2, "template": TPL, "voxel_size": [3.0, 3.0, 3.5], "obliquity_deg": 0.5,
        "wm_mask_voxels": 8, "csf_mask_voxels": 2, "gm_mask_voxels": 170, "tissue_erosion_relaxed": False,
        "coreg_dice": 0.95, "brain_mask_voxels": 180, "quality_index_mean": 0.02, "outlier_frac_mean": 0.01,
        "tool_versions": {"fsl": "6.0.7", "afni": "25.3"},
    })
    motion = 0.02 * rng.normal(size=(N_T, 6))
    motion[~keep] += 0.4
    np.savetxt(func / f"{run}_desc-hmc_motion.par", np.c_[motion[:, 3:], motion[:, :3]], fmt="%.6f")
    np.savetxt(func / f"{run}_desc-hmc_relrms.txt", np.abs(rng.normal(0.05, 0.02, N_T - 1)), fmt="%.6f")
    np.savetxt(func / f"{run}_desc-hmc_absrms.txt", np.abs(rng.normal(0.3, 0.1, N_T)), fmt="%.6f")
    np.savetxt(func / f"{run}_desc-outliers_timeseries.1D", np.abs(rng.normal(0.01, 0.005, N_T)), fmt="%.6f")
    np.savetxt(func / f"{run}_desc-quality_timeseries.1D", np.abs(rng.normal(0.02, 0.005, N_T)), fmt="%.6f")
    t1w, tpl_m = t1w_masks(), tpl_masks()
    if images:
        bold_t1w = make_pre(rng, t1w["brain"], N_T)
        bold_t1w[t1w["dropout"]] = 500.0 + rng.normal(size=(int(t1w["dropout"].sum()), N_T))
        _save(bold_t1w, T1W_AFFINE, func / f"{run}_space-T1w_desc-preproc_bold.nii.gz")
        _save(bold_t1w.mean(axis=3), T1W_AFFINE, func / f"{run}_space-T1w_boldref.nii.gz")
        _save(t1w["brain"], T1W_AFFINE, func / f"{run}_space-T1w_desc-brain_mask.nii.gz", np.uint8)
        for name in ("GM", "WM", "CSF"):
            _save(t1w[name], T1W_AFFINE, func / f"{run}_space-T1w_label-{name}_mask.nii.gz", np.uint8)
        bold_tpl = make_pre(rng, tpl_m["brain"], N_T)
        _save(bold_tpl, TPL_AFFINE, func / f"{tpl}_desc-preproc_bold.nii.gz")
        _save(bold_tpl.mean(axis=3), TPL_AFFINE, func / f"{tpl}_boldref.nii.gz")
        _save(tpl_m["brain"], TPL_AFFINE, func / f"{tpl}_desc-brain_mask.nii.gz", np.uint8)

    # ---- stage 04: confounds + censor vector
    fd = np.abs(rng.normal(fd_level, 0.04, N_T))
    fd[0] = 0.0
    fd[~keep] = 0.9
    dvars = np.abs(rng.normal(20.0, 3.0, N_T))
    dvars[0] = 0.0
    dvars[~keep] += 15.0
    std_dvars = dvars / 18.0
    table = {f"trans_{a}": motion[:, k] for k, a in enumerate("xyz")}
    table.update({f"rot_{a}": motion[:, 3 + k] for k, a in enumerate("xyz")})
    for column in list(table):
        table[f"{column}_derivative1"] = np.r_[0.0, np.diff(table[column])]
    gs = 10000.0 + 10.0 * rng.normal(size=N_T)
    table.update({
        "global_signal": gs, "csf": gs + rng.normal(size=N_T), "white_matter": gs - rng.normal(size=N_T),
        "w_comp_cor_00": rng.normal(size=N_T), "w_comp_cor_01": rng.normal(size=N_T),
        "c_comp_cor_00": rng.normal(size=N_T), "c_comp_cor_01": rng.normal(size=N_T),
        "cosine_00": np.cos(np.linspace(0, np.pi, N_T)), "dvars": dvars, "std_dvars": std_dvars,
        "framewise_displacement": fd, "fd_jenkinson": np.r_[0.0, np.abs(rng.normal(0.05, 0.02, N_T - 1))],
        "outlier_fraction": np.abs(rng.normal(0.01, 0.005, N_T)), "censor": keep.astype(int),
    })
    write_tsv(func / f"{run}_desc-confounds_timeseries.tsv", pd.DataFrame(table), float_format="%.10g")
    write_json(func / f"{run}_desc-confounds_timeseries.json", {
        "framewise_displacement": {"Description": "Power FD", "Units": "mm"},
        "censor": {"fd_threshold": 0.5, "prev": False, "dvars_threshold": 0.0, "n_censored": N_T - n_keep,
                   "n_volumes": N_T, "minutes_retained": n_keep * TR / 60.0, "longest_segment": 20,
                   "pct_censored": 100.0 * (N_T - n_keep) / N_T},
        "acompcor": {"n_wm_voxels": 8, "n_csf_voxels": 2, "variance_explained_wm": [0.4, 0.2],
                     "variance_explained_csf": [0.5, 0.2]},
    })
    (func / f"{run}_desc-censor.1D").write_text("".join(f"{int(v)}\n" for v in keep), encoding="utf-8")

    # ---- stage 05 + 07: denoised series, ROI tables, FC
    tables: dict[str, np.ndarray] = {"keep": keep}
    for s, strategy in enumerate(strategies):
        write_json(func / f"{run}_desc-{strategy}_denoise.json", {
            "strategy": strategy, "columns": ["trans_x", "white_matter", "csf"], "n_regressors": 26 + s, "polort": 2,
            "filter_mode": "bandpass", "band": [0.01, 0.1], "n_volumes": N_T, "n_censored": N_T - n_keep,
            "n_retained": n_keep, "censor_mode": "KILL" if kill else "NTRP", "n_volumes_out": n_keep if kill else N_T,
            "dof_bandpass_cost": 6, "dof_remaining": 22 - 4 * s, "low_dof": False, "smooth_fwhm": 6.0,
            "inputs": {"confounds": "x.tsv"},
        })
        sd = 25.0 if s == 0 else 15.0
        if images:
            _save(make_post(rng, tpl_m["brain"], N_T, keep, sd, kill), TPL_AFFINE, func / f"{tpl}_desc-{strategy}_bold.nii.gz")
        pre_roi, post_roi = roi_series(rng, N_T, keep, post_sd=sd / 2.0)
        if s > 0:
            pre_roi = tables["pre"]              # stage 07 writes one pre-denoise ROI table per atlas, not per strategy
        stem = func / f"{run}_space-{TPL}_atlas-{ATLAS}"
        rows = post_roi[keep] if kill else post_roi
        write_tsv(f"{stem}_desc-{strategy}_timeseries.tsv", pd.DataFrame(rows, columns=ROI_NAMES), float_format="%.8g")
        write_tsv(f"{stem}_desc-{strategy}_connectivity.tsv", pd.DataFrame(fc_matrix(post_roi, keep), columns=ROI_NAMES))
        coverage = pd.DataFrame({
            "roi": range(1, len(ROI_NAMES) + 1), "name": ROI_NAMES, "network": NETWORKS, "atlas_voxels": 27,
            "valid_voxels": [0 if k == UNCOVERED else 27 for k in range(len(ROI_NAMES))],
            "coverage_fraction": [0.0 if k == UNCOVERED else 1.0 for k in range(len(ROI_NAMES))],
            "included": [k != UNCOVERED for k in range(len(ROI_NAMES))],
        })
        write_tsv(f"{stem}_desc-{strategy}_coverage.tsv", coverage)
        if s == 0:
            write_tsv(f"{stem}_desc-preproc_timeseries.tsv", pd.DataFrame(pre_roi, columns=ROI_NAMES), float_format="%.8g")
            write_tsv(f"{stem}_desc-preproc_coverage.tsv", coverage)
            tables["pre"] = pre_roi
        tables[f"post_{strategy}"] = post_roi

    # ---- stage 10 (optional) + stage 06 surface QC (optional)
    if validation:
        rows, compare = [], []
        for stream in ("volume", "surface"):
            for strategy in strategies:
                for metric, value in (("roi_tsnr_median", 120.0), ("split_half_r", 0.7), ("network_contrast", 1.1)):
                    rows.append((stream, strategy, ATLAS, metric, value + (5.0 if stream == "surface" else 0.0)))
        for strategy in strategies:
            compare.append((strategy, ATLAS, "run", "n/a", "fc_similarity", np.nan, np.nan, 0.8))
            compare.append((strategy, ATLAS, "run", "n/a", "roi_tsnr_median", 120.0, 125.0, 5.0))
            for roi, name in enumerate(ROI_NAMES, start=1):
                compare.append((strategy, ATLAS, "roi", name, "roi_tsnr", 100.0 + roi, 104.0 + roi, 4.0))
        write_tsv(func / f"{run}_desc-validation.tsv", pd.DataFrame(rows, columns=["stream", "strategy", "atlas", "metric", "value"]))
        write_tsv(func / f"{run}_desc-streamcompare.tsv",
                  pd.DataFrame(compare, columns=["strategy", "atlas", "scope", "roi", "metric", "volume", "surface", "value"]))
    if surfqc:
        write_json(func / f"{run}_desc-surfqc.json", {"tsnr_cortex_median": 88.0, "pct_badvertices": 1.5,
                                                     "pct_goodvoxels_excluded": 4.0})
    return tables


def build_subject(root: Path, sub: str = SUB, runs: tuple[str, ...] = (RUN,), seed: int = 0,
                  **kwargs: object) -> dict[str, dict[str, np.ndarray]]:
    """derivatives/<sub>/{anat,func}; returns the ROI tables per run."""
    build_anat(root / "derivatives" / sub / "anat", sub)
    return {run: build_run(root / "derivatives" / sub / "func", run, seed=seed + 11 * k, **kwargs)
            for k, run in enumerate(runs)}


def write_manifest(root: Path, entries: list[tuple[str, str, str]]) -> Path:
    """rawdata/manifest.tsv from (subject, run_label, group) triples."""
    rows = [{"subject": sub, "session": "-", "task": "rest", "run": "-", "group": group, "bold": f"/data/{run}_bold.nii.gz",
             "t1w": f"/data/{sub}_T1w.nii.gz", "run_label": run} for sub, run, group in entries]
    path = root / "rawdata" / "manifest.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, lineterminator="\n")
    return path


def qc_args(root: Path, sub: str, run: str, cache: Path | None = None, strategies: str = " ".join(STRATEGIES),
            censor_mode: str = "NTRP", tpl_masks_dir: Path | None = None, extra: list[str] | None = None) -> list[str]:
    func, anat = root / "derivatives" / sub / "func", root / "derivatives" / sub / "anat"
    # the synthetic run is 2 minutes long: MIN_RETAINED_MIN is lowered so that it passes
    args = ["--func-dir", str(func), "--anat-dir", str(anat), "--run", run, "--template", TPL, "--mni-res", RES,
            "--strategies", strategies, "--atlases", ATLAS, "--censor-mode", censor_mode,
            "--out-json", str(func / f"{run}_desc-qc_metrics.json"), "--atlas-labels-dir", str(root / "resources" / "atlases"),
            "--min-retained-min", "1"]
    if cache is not None:
        args += ["--cache-dir", str(cache)]
    if tpl_masks_dir is not None:
        for name in ("GM", "WM", "CSF"):
            args += [f"--tpl-{name.lower()}-mask", str(tpl_masks_dir / f"space-{TPL}_label-{name}_mask.nii.gz")]
    return args + (extra or [])


def write_tpl_tissue_masks(directory: Path) -> Path:
    """What stages/08_qc.sh produces with antsApplyTransforms (subject tissue masks on the template BOLD grid)."""
    directory.mkdir(parents=True, exist_ok=True)
    masks = tpl_masks()
    for name in ("GM", "WM", "CSF"):
        _save(masks[name], TPL_AFFINE, directory / f"space-{TPL}_label-{name}_mask.nii.gz", np.uint8)
    return directory


# ----------------------------------------------------------------------------
# independent reference implementations
# ----------------------------------------------------------------------------

def ref_detrend(series: np.ndarray, keep: np.ndarray, order: int = 2) -> np.ndarray:
    """Legendre (polort) trend fitted on the retained frames with lstsq, removed from every frame. series: (V, T)."""
    x = np.linspace(-1.0, 1.0, series.shape[1])
    design = np.polynomial.legendre.legvander(x, order)
    coef = np.linalg.lstsq(design[keep], series[:, keep].T, rcond=None)[0]
    return series - (design @ coef).T


def ref_tsnr(series: np.ndarray, keep: np.ndarray) -> np.ndarray:
    resid = ref_detrend(series, keep)
    return series[:, keep].mean(axis=1) / resid[:, keep].std(axis=1, ddof=1)


def load4d(path: Path, mask: np.ndarray) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj).astype(np.float64)[mask]


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="fmriproc_qc_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------

class EndToEndTest(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        write_atlas_dir(self.tmp / "resources" / "atlases")
        self.tables = build_subject(self.tmp, surfqc=True)[RUN]
        self.cache = self.tmp / "work" / SUB / "func" / RUN / "qc"
        write_tpl_tissue_masks(self.cache)
        fwhm = self.cache / "fwhm_acf.txt"
        fwhm.write_text(" 0  0  0  0\n 0.5  3.0  12.0  7.5\n", encoding="utf-8")
        rc = qc_metrics.main(qc_args(self.tmp, SUB, RUN, self.cache, tpl_masks_dir=self.cache,
                                     extra=["--fwhm-file", str(fwhm)]))
        self.assertEqual(rc, 0)
        self.func = self.tmp / "derivatives" / SUB / "func"
        self.metrics = json.loads((self.func / f"{RUN}_desc-qc_metrics.json").read_text(encoding="utf-8"))

    def test_section9_keys_present(self) -> None:
        m = self.metrics
        for key in ("tr", "n_volumes_raw", "n_dropped", "n_volumes", "minutes", "voxel_size", "stc_applied", "stc_evidence",
                    "nss_detected", "despike_fraction", "fd_mean", "fd_median", "fd_max", "fd_pct_gt_02", "fd_pct_gt_05",
                    "relrms_mean", "absrms_max", "n_censored", "pct_censored", "minutes_retained", "longest_segment",
                    "tsnr_gm_median", "tsnr_wm_median", "tsnr_brain_median", "dvars_std_mean", "outlier_frac_mean",
                    "quality_index_mean", "gcor", "fd_dvars_corr", "gs_fd_corr", "fwhm_acf", "coreg_method", "bbr_cost",
                    "bbr_vs_init_mm", "coreg_dice", "norm_dice", "jacobian_p01", "jacobian_p99", "jacobian_nonpos_frac",
                    "template_corr", "brain_mask_voxels", "wm_mask_voxels", "csf_mask_voxels", "dropout_fraction",
                    "tsnr_cortex_median", "pct_badvertices", "pct_goodvoxels_excluded"):
            self.assertIn(key, m, key)
            self.assertIsNotNone(m[key], key)
        for s in STRATEGIES:
            for key in ("n_regressors", "dof_remaining", "tsnr_gm_median_post", "tsnr_gain", "variance_removed_gm",
                        "fd_dvars_corr_post"):
                self.assertIsNotNone(m[f"{s}.{key}"], f"{s}.{key}")
            for key in ("roi_tsnr_median", "roi_tsnr_min", "n_roi_nan", "coverage_min", "split_half_r", "network_contrast",
                        "fc_mean", "fc_sd"):
                self.assertIsNotNone(m[f"{s}.{ATLAS}.{key}"], f"{s}.{ATLAS}.{key}")
        self.assertEqual(m["stc_evidence"], "B: site protocol")
        self.assertEqual(m["fwhm_acf"], 7.5)
        self.assertEqual(m["n_censored"], 3)
        self.assertAlmostEqual(m["pct_censored"], 5.0)
        self.assertAlmostEqual(m["minutes_retained"], 57 * TR / 60.0)
        self.assertEqual(m[f"{STRATEGIES[0]}.{ATLAS}.n_roi_nan"], 1)
        self.assertEqual(m[f"{STRATEGIES[0]}.{ATLAS}.coverage_min"], 0.0)
        self.assertEqual(m[f"{STRATEGIES[0]}.tsnr_post_mask"], "brain_x_warped_gm")
        self.assertGreater(m["dropout_fraction"], 0.1)          # the low-signal slab inside the T1 brain
        self.assertEqual(m["coreg_dice"], 0.95)                  # stage 03 value wins
        self.assertIsNone(m["qc_problems"])

    def test_flags_and_thresholds(self) -> None:
        flags = self.metrics["flags"]
        for key in ("fd_mean", "pct_censored", "tsnr_gm_median", "coreg_dice", "norm_dice", "holes_total",
                    "minutes_retained", "bbr_accepted", f"{STRATEGIES[0]}.dof_remaining"):
            self.assertIn(flags.get(key), ("pass", "warn", "fail"), key)
        self.assertEqual(flags["tsnr_gm_median"], "pass")
        self.assertEqual(self.metrics["overall_flag"], "pass")
        self.assertEqual(self.metrics["thresholds"]["fd_mean"]["warn"], 0.2)
        # overriding the QC_* thresholds moves the flags
        rc = qc_metrics.main(qc_args(self.tmp, SUB, RUN, extra=["--qc-tsnr-gm-warn", "1000", "--qc-tsnr-gm-fail", "500",
                                                                 "--qc-fd-mean-warn", "0.05", "--min-dof", "30"]))
        self.assertEqual(rc, 0)
        again = json.loads((self.func / f"{RUN}_desc-qc_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(again["flags"]["tsnr_gm_median"], "fail")
        self.assertEqual(again["flags"]["fd_mean"], "warn")
        self.assertEqual(again["flags"][f"{STRATEGIES[0]}.dof_remaining"], "warn")
        self.assertEqual(again["overall_flag"], "fail")

    def test_pre_denoise_tsnr_and_gcor_definitions(self) -> None:
        masks = t1w_masks()
        support = masks["brain"] | masks["GM"] | masks["WM"] | masks["CSF"]
        data = load4d(self.func / f"{RUN}_space-T1w_desc-preproc_bold.nii.gz", support)
        keep = self.tables["keep"]
        tsnr = ref_tsnr(data, keep)
        for name, key in (("GM", "tsnr_gm_median"), ("WM", "tsnr_wm_median"), ("brain", "tsnr_brain_median")):
            expected = np.median(tsnr[masks[name][support]])
            self.assertAlmostEqual(self.metrics[key], expected, delta=1e-6 * expected, msg=key)
        resid = ref_detrend(data, keep)[masks["brain"][support]][:, keep]
        expected_gcor = float(np.corrcoef(resid).mean())        # Saad 2013: mean of all pairwise correlations
        self.assertAlmostEqual(self.metrics["gcor"], expected_gcor, places=6)
        self.assertGreater(self.metrics["gcor"], 0.05)          # the planted shared signal is visible

    def test_post_denoise_tsnr_and_variance_removed(self) -> None:
        masks = tpl_masks()
        brain, gm = masks["brain"], masks["GM"] & masks["brain"]
        pre = load4d(self.func / f"{RUN}_space-{TPL}_res-{RES}_desc-preproc_bold.nii.gz", brain)
        keep = self.tables["keep"]
        mean_pre = pre[:, keep].mean(axis=1)
        sd_pre = ref_detrend(pre, keep)[:, keep].std(axis=1, ddof=1)
        select = gm[brain]
        for strategy in STRATEGIES:
            den = load4d(self.func / f"{RUN}_space-{TPL}_res-{RES}_desc-{strategy}_bold.nii.gz", brain)
            sd_post = den[:, keep].std(axis=1, ddof=1)
            post = np.median((mean_pre / sd_post)[select])
            self.assertAlmostEqual(self.metrics[f"{strategy}.tsnr_gm_median_post"], post, delta=1e-6 * post, msg=strategy)
            removed = np.median((1.0 - sd_post ** 2 / sd_pre ** 2)[select])
            self.assertAlmostEqual(self.metrics[f"{strategy}.variance_removed_gm"], removed, places=6)
            pre_same = np.median((mean_pre / sd_pre)[select])
            self.assertAlmostEqual(self.metrics[f"{strategy}.tsnr_gain"], post / pre_same, places=6)
        # the stronger denoising (smaller residual) gives the higher post-denoise tSNR
        self.assertGreater(self.metrics["wmcsf24gsr.tsnr_gm_median_post"], self.metrics["wmcsf24.tsnr_gm_median_post"])

    def test_roi_table_definitions(self) -> None:
        keep = self.tables["keep"]
        for strategy in STRATEGIES:
            table = read_tsv(self.func / f"{RUN}_atlas-{ATLAS}_desc-{strategy}_roiqc.tsv")
            self.assertEqual(list(table.columns),
                             ["roi", "name", "network", "coverage_fraction", "tsnr_pre", "roi_tsnr", "variance_removed"])
            self.assertEqual(table["name"].tolist(), ROI_NAMES)
            self.assertEqual(table["network"].tolist(), NETWORKS)
            pre, post = self.tables["pre"].T, self.tables[f"post_{strategy}"].T          # (R, T)
            good = [k for k in range(len(ROI_NAMES)) if k != UNCOVERED]
            mean_pre = pre[good][:, keep].mean(axis=1)
            sd_pre = ref_detrend(pre[good], keep)[:, keep].std(axis=1, ddof=1)
            sd_post = post[good][:, keep].std(axis=1, ddof=1)
            np.testing.assert_allclose(table["roi_tsnr"].to_numpy()[good], mean_pre / sd_post, rtol=1e-5)
            np.testing.assert_allclose(table["tsnr_pre"].to_numpy()[good], mean_pre / sd_pre, rtol=1e-5)
            np.testing.assert_allclose(table["variance_removed"].to_numpy()[good], 1 - sd_post ** 2 / sd_pre ** 2, rtol=1e-5)
            self.assertTrue(np.isnan(table["roi_tsnr"].to_numpy()[UNCOVERED]))
            np.testing.assert_allclose(self.metrics[f"{strategy}.{ATLAS}.roi_tsnr_median"], np.median(mean_pre / sd_post), rtol=1e-7)  # input TSV uses 8 significant digits
            np.testing.assert_allclose(self.metrics[f"{strategy}.{ATLAS}.roi_tsnr_min"], np.min(mean_pre / sd_post), rtol=1e-7)
        self.assertGreater(self.metrics[f"{STRATEGIES[0]}.{ATLAS}.network_contrast"], 0.0)

    def test_missing_or_invalid_censor_keeps_independent_qc(self):
        censor = self.tmp / "derivatives" / SUB / "func" / f"{RUN}_desc-censor.1D"
        censor.unlink()
        for content in (None, "1\nNaN\n0\n"):
            if content is not None:
                censor.write_text(content, encoding="utf-8")
            qc_metrics.main(qc_args(self.tmp, SUB, RUN, self.cache, tpl_masks_dir=self.cache))
            path = censor.parent / f"{RUN}_desc-qc_metrics.json"
            values = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(values["overall_flag"], "incomplete")
            self.assertIsNone(values["n_censored"])
            self.assertIsNone(values["minutes_retained"])
            self.assertIsNone(values["tsnr_gm_median"])
            self.assertEqual(values["coreg_dice"], 0.95)
            self.assertIn("censor:", values["qc_problems"])

    def test_cache_for_figures(self) -> None:
        with np.load(self.cache / "carpet.npz") as store:
            keys = set(store.files)
            self.assertLessEqual({"pre", "pre_groups", "pre_frames", "post_wmcsf24", "post_wmcsf24_groups"}, keys)
            self.assertEqual(store["pre"].shape[1], N_T)
            self.assertEqual(sorted(set(store["pre_groups"].tolist())), [1, 2, 3])       # GM, WM, CSF rows
            self.assertEqual(sorted(set(store["post_wmcsf24_groups"].tolist())), [1, 2])  # template CSF overlaps GM; GM wins
        for name in ("tsnr_desc-preproc_space-T1w", f"tsnr_desc-preproc_space-{TPL}", f"tsnr_desc-wmcsf24_space-{TPL}"):
            self.assertTrue((self.cache / f"{name}.nii.gz").is_file(), name)


class KillModeTest(TempDirCase):
    def test_shortened_series(self) -> None:
        write_atlas_dir(self.tmp / "resources" / "atlases")
        build_subject(self.tmp, kill=True)
        cache = self.tmp / "work" / SUB / "func" / RUN / "qc"
        rc = qc_metrics.main(qc_args(self.tmp, SUB, RUN, cache, censor_mode="KILL"))
        self.assertEqual(rc, 0)
        func = self.tmp / "derivatives" / SUB / "func"
        m = json.loads((func / f"{RUN}_desc-qc_metrics.json").read_text(encoding="utf-8"))
        self.assertIsNone(m["qc_problems"])
        for strategy in STRATEGIES:
            self.assertIsNotNone(m[f"{strategy}.tsnr_gm_median_post"])
            self.assertIsNotNone(m[f"{strategy}.fd_dvars_corr_post"])
            self.assertIsNotNone(m[f"{strategy}.{ATLAS}.roi_tsnr_median"])
            self.assertIsNotNone(m[f"{strategy}.{ATLAS}.split_half_r"])
            self.assertEqual(m[f"{strategy}.tsnr_post_mask"], "brain")     # no warped GM mask given
        with np.load(cache / "carpet.npz") as store:
            self.assertEqual(store["post_wmcsf24"].shape[1], N_T - 3)
            self.assertEqual(store["post_wmcsf24_frames"].size, N_T - 3)
            self.assertEqual(store["pre"].shape[1], N_T)


class MissingInputsTest(TempDirCase):
    def test_only_provenance_and_confounds(self) -> None:
        func = self.tmp / "derivatives" / SUB / "func"
        build_run(func, RUN, images=False, validation=False)
        for path in list(func.glob(f"{RUN}_space-*")) + [func / f"{RUN}_desc-censor.1D"]:
            path.unlink()
        for strategy in STRATEGIES:
            (func / f"{RUN}_desc-{strategy}_denoise.json").unlink()
        rc = qc_metrics.main(qc_args(self.tmp, SUB, RUN))
        self.assertEqual(rc, 0)
        m = json.loads((func / f"{RUN}_desc-qc_metrics.json").read_text(encoding="utf-8"))
        self.assertIsNotNone(m["fd_mean"])
        self.assertIsNone(m["n_censored"])  # missing censor is unknown, never all-kept
        for key in ("tsnr_gm_median", "gcor", "norm_dice", "dropout_fraction",
                    "wmcsf24.tsnr_gm_median_post", "wmcsf24.dof_remaining", f"wmcsf24.{ATLAS}.roi_tsnr_median"):
            self.assertIsNone(m[key], key)
        self.assertEqual(m["flags"]["tsnr_gm_median"], "n/a")
        self.assertEqual(m["flags"]["norm_dice"], "n/a")
        self.assertEqual(m["coreg_dice"], 0.95)  # retained stage-03 provenance
        self.assertEqual(m["flags"]["fd_mean"], "pass")
        self.assertEqual(m["overall_flag"], "incomplete")

    def test_empty_func_dir(self) -> None:
        func = self.tmp / "derivatives" / SUB / "func"
        func.mkdir(parents=True)
        rc = qc_metrics.main(qc_args(self.tmp, SUB, RUN))
        self.assertEqual(rc, 0)
        m = json.loads((func / f"{RUN}_desc-qc_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(m["run"], RUN)
        self.assertEqual(m["overall_flag"], "incomplete")
        self.assertIn("number of frames unknown", m["qc_problems"])

    def test_missing_func_dir_is_an_error(self) -> None:
        self.assertEqual(qc_metrics.main(qc_args(self.tmp, SUB, RUN)), 2)


class PureFunctionTest(unittest.TestCase):
    def test_gcor(self) -> None:
        rng = np.random.default_rng(3)
        base = rng.normal(size=50)
        identical = np.tile(base, (8, 1)) * rng.uniform(1, 5, size=(8, 1)) + rng.uniform(-5, 5, size=(8, 1))
        keep = np.ones(50, dtype=bool)
        self.assertAlmostEqual(qc_metrics.gcor(identical.astype(np.float32), keep), 1.0, places=5)
        independent = rng.normal(size=(200, 50))
        value = qc_metrics.gcor(independent, keep)
        self.assertAlmostEqual(value, float(np.corrcoef(independent).mean()), places=6)
        self.assertLess(value, 0.05)
        self.assertIsNone(qc_metrics.gcor(independent, keep[:4]))       # too few frames

    def test_dvars_and_detrend(self) -> None:
        rng = np.random.default_rng(1)
        data = rng.normal(size=(30, 12)).astype(np.float32)
        dv = qc_metrics.dvars(data)
        self.assertTrue(np.isnan(dv[0]))
        self.assertAlmostEqual(dv[5], float(np.sqrt(np.mean((data[:, 5].astype(float) - data[:, 4]) ** 2))), places=5)
        keep = np.ones(12, dtype=bool)
        keep[3] = False
        series = (1000.0 + legendre_trend(12)[None, :] + data).astype(np.float32)
        expected_mean = series[:, keep].astype(float).mean(axis=1)
        work = series.copy()
        mean = qc_metrics.detrend_inplace(work, keep)
        np.testing.assert_allclose(mean, expected_mean, rtol=1e-6)
        np.testing.assert_allclose(work, ref_detrend(series.astype(float), keep), atol=1e-2)

    def test_frame_map(self) -> None:
        keep = np.array([1, 1, 0, 1, 1, 0, 1], dtype=bool)
        full = qc_metrics.FrameMap.build(keep, 7)
        self.assertEqual(full.retained.sum(), 5)
        self.assertEqual(full.adjacent_pairs().tolist(), [False, True, False, False, True, False, False])
        short = qc_metrics.FrameMap.build(keep, 5)
        self.assertEqual(short.index.tolist(), [0, 1, 3, 4, 6])
        self.assertEqual(short.adjacent_pairs().tolist(), [False, True, False, True, False])
        with self.assertRaises(ValueError):
            qc_metrics.FrameMap.build(keep, 6)

    def test_flag_value(self) -> None:
        self.assertEqual(qc_metrics.flag_value(0.1, "high", 0.2, 0.5), "pass")
        self.assertEqual(qc_metrics.flag_value(0.3, "high", 0.2, 0.5), "warn")
        self.assertEqual(qc_metrics.flag_value(0.6, "high", 0.2, 0.5), "fail")
        self.assertEqual(qc_metrics.flag_value(50, "low", 40, 20), "pass")
        self.assertEqual(qc_metrics.flag_value(30, "low", 40, 20), "warn")
        self.assertEqual(qc_metrics.flag_value(10, "low", 40, 20), "fail")
        self.assertEqual(qc_metrics.flag_value(None, "low", 40, 20), "n/a")
        self.assertEqual(qc_metrics.flag_value(float("nan"), "low", 40, 20), "n/a")
        self.assertEqual(qc_metrics.worst_flag({"a": "pass", "b": "warn", "c": "n/a"}), "warn")

    def test_parse_fwhm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fwhm.txt"
            path.write_text("++ some banner\n 0 0 0 0\n 0.6 4.2 11.0 6.25\n", encoding="utf-8")
            self.assertEqual(qc_metrics.parse_fwhm(path), 6.25)
            path.write_text("garbage\n", encoding="utf-8")
            self.assertIsNone(qc_metrics.parse_fwhm(path))
            self.assertIsNone(qc_metrics.parse_fwhm(Path(tmp) / "absent.txt"))

    def test_network_contrast(self) -> None:
        rng = np.random.default_rng(5)
        z = rng.normal(0.0, 0.1, size=(6, 6))
        z = (z + z.T) / 2
        nets = ["A", "A", "A", "B", "B", "B"]
        for i in range(6):
            for j in range(6):
                if nets[i] == nets[j] and i != j:
                    z[i, j] += 0.8
        self.assertGreater(qc_metrics.network_contrast(z, nets), 3.0)
        self.assertIsNone(qc_metrics.network_contrast(z, ["A"] * 6))
        self.assertIsNone(qc_metrics.network_contrast(z, [None] * 6))

    def test_find_atlas_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_atlas_dir(Path(tmp))
            table = qc_metrics.find_atlas_labels(root, ATLAS)
            self.assertIsNotNone(table)
            self.assertEqual(table["network"].tolist(), NETWORKS)
            self.assertEqual(table["roi"].tolist(), list(range(1, 7)))
            self.assertIsNone(qc_metrics.find_atlas_labels(root, "NoSuchAtlas"))
            self.assertIsNone(qc_metrics.find_atlas_labels(None, ATLAS))


class UpgradeQCRegressionTest(unittest.TestCase):
    def test_nonfinite_masked_image_is_not_zero_filled(self):
        from unittest.mock import patch
        image = nib.Nifti1Image(np.array([[[[1., np.nan, 3.]]]]), np.eye(4))
        with patch.object(qc_metrics.nib, "load", return_value=image):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                qc_metrics.load_masked(Path("synthetic.nii"), np.ones((1, 1, 1), bool))

    def test_coverage_alignment_uses_identity(self):
        coverage = pd.DataFrame({"roi": [2, 1], "name": ["B", "A"], "coverage_fraction": [.2, .9]})
        table = qc_metrics.roi_table(["A", "B"], coverage, None)
        self.assertEqual(table["roi"].tolist(), [1, 2])
        self.assertEqual(table["coverage_fraction"].tolist(), [.9, .2])


if __name__ == "__main__":
    unittest.main()
