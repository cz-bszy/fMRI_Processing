# fMRI_Processing v2.1 — design contract

This file is the **interface contract** between all scripts. If code and this
file disagree, fix one of them in the same commit.

Target data: legacy single-band resting-state fMRI (TR 2–3 s, 3–4 mm voxels,
no fieldmaps, often no JSON sidecars: ABIDE, ADNI, DPABI-layout cohorts).
Language: bash stage scripts + a small Python package (`py/fmriproc`) for
everything that is not a neuroimaging CLI call (ingest, confounds, metrics,
figures, reports).

## 1. Principles (hard-won; do not regress)

1. Never guess slice order from the vendor. STC only with verified timing
   (sidecar `SliceTiming`, or the dataset acquisition table); otherwise skip and
   record it.
2. Motion is **estimated** on pre-STC data, and **applied once**, together with
   EPI→T1 and T1→MNI, in a single interpolation (LanczosWindowedSinc).
3. No header-only deobliquing, no forced reorientation of BOLD voxel storage.
4. Data are never multiplied by a mask. Masks are separate files.
5. Nuisance signals come from **unsmoothed** data. Smoothing is the last,
   optional step. ROI time series always come from unsmoothed data.
6. Temporal filter + polynomial trend + nuisance regressors + censoring are one
   simultaneous projection (`3dTproject`).
7. BBR is never trusted blindly: compare with its initialisation, record cost,
   fall back, and flag.
8. Every ROI reports its coverage; low-coverage ROIs are `NaN`, never invented.
9. Every stage records what it did (JSON provenance) and is restartable; a
   stage is skipped only if its parameter hash is unchanged.
10. Denoising strategies are named by what they do: `...gsr` regresses the
    global signal. (Legacy `NoGRS` = GSR applied, `Retain_GRS` = no GSR.)

## 2. Repository layout

```
run_pipeline.sh              single entry point (subject loop, stage loop, parallel)
config/default.conf          every parameter, documented (bash KEY=VALUE)
config/datasets/*.conf       per-dataset overrides
config/datasets/*_acquisition.tsv   per-site acquisition table (timing, drop, PE)
lib/common.sh                logging, run(), config, stage hashing, naming helpers
stages/00_ingest.sh          raw layout -> rawdata/ (BIDS-like) + sidecars + manifest
stages/01_anat_recon.sh      FreeSurfer recon-all (or reuse)
stages/02_anat_prep.sh       T1 reference, brain/tissue masks, T1->MNI (ANTs)
stages/03_func_prep.sh       drop, despike, HMC estimate, STC, coreg, one-shot resampling
stages/04_confounds.sh       confounds TSV, FD, DVARS, aCompCor, censor vector
stages/05_denoise.sh         3dTproject per strategy (+ optional smoothing)
stages/06_surface.sh         optional fsLR-32k CIFTI branch
stages/07_timeseries.sh      atlas time series + FC (volume and CIFTI)
stages/08_qc.sh              per-run metrics JSON + per-subject HTML report
stages/09_group_qc.sh        group table, outliers, QC-FC, group HTML
stages/10_validate.sh        time-series validity metrics + volume-vs-surface comparison
stages/fetch_resources.sh    one-time download of templates/atlases
py/fmriproc/*.py             Python package, run as: $PYTHON_BIN -m fmriproc.<module>
docker/run_docker.sh|.ps1    launch inside the container from Windows/Linux
tests/                       unittest-based tests + run_tests.sh
parcellations/               legacy Yeo-100 atlas (kept as optional custom atlas)
legacy/                      v1 scripts, untouched, for reference only
```

## 3. Runtime environment

* Container image `zhaochang07/myubuntu:neuro-v2`; scripts must run under
  `bash -lc` (login shell sets FSL/FreeSurfer/AFNI/ANTs PATH).
* `PYTHON_BIN=/opt/micromamba/envs/neuro/bin/python` (nilearn 0.12, nibabel 5.3,
  niworkflows, templateflow, jinja2, pandas, scipy, sklearn, matplotlib).
  Bare `python3` is FSL's python and lacks nilearn — never use it.
  `PYTHONPATH` gets `$REPO_DIR/py` prepended by `lib/common.sh`.
* No pytest: tests use `unittest`.
* FreeSurfer license: `FS_LICENSE` env var or `/opt/freesurfer/license.txt`.
* All scripts: `#!/bin/bash`, `set -euo pipefail`, LF line endings.

## 4. Directory contract (all under `$OUT_DIR`)

```
rawdata/manifest.tsv                       one row per BOLD run (see 5)
rawdata/participants.tsv
rawdata/**/<image>.nii.gz.ingest.json       source/target copy identity (path,size,mtime_ns)
rawdata/ingest_report.tsv                  header/geometry validation table
rawdata/sub-X[/ses-Y]/anat/sub-X[_ses-Y]_T1w.nii.gz
rawdata/sub-X[/ses-Y]/func/<RUN>_bold.nii.gz + <RUN>_bold.json
freesurfer/sub-X/                          SUBJECTS_DIR=$FS_DIR (default $OUT_DIR/freesurfer)
work/sub-X/.done/<stage>[__<RUN>].hash     stage completion markers
work/sub-X/anat/                           anat intermediates
work/sub-X/func/<RUN>/                     func intermediates (safe to delete)
derivatives/sub-X/anat/                    final anat files (see 6)
derivatives/sub-X/func/                    final func files (see 6)
derivatives/sub-X/figures/                 PNG/SVG used by the report
derivatives/sub-X.html                     per-subject QC report
derivatives/group/                         group_qc.tsv, group_report.html, qcfc_*.tsv
logs/sub-X/<stage>.log                     per-subject, per-stage log (no ANSI codes)
logs/pipeline_<timestamp>.log
```

`<RUN>` = `sub-X[_ses-Y]_task-<task>[_run-<N>]` (BIDS prefix, from the manifest).
Anatomy is per subject (first T1w in the manifest for that subject); anat
derivatives carry no `ses` entity.

## 5. Manifest (`rawdata/manifest.tsv`, tab-separated, header row)

`subject  session  task  run  group  bold  t1w  run_label`

* `subject` = `sub-<id>`; `session`/`run` = `-` when absent; `group` = site /
  acquisition group used to look up the acquisition table (`-` if none).
* `participants.tsv` writes `acq_group`, not a diagnostic `group`; conflicting
  sites for one participant are rejected. DPABI same-stem BOLD JSON is read.
* `bold`, `t1w` = absolute paths (inside the container).
* `lib/common.sh: manifest_runs SUB` prints the rows of one subject.

### Acquisition table (`ACQ_TABLE`, TSV, header row)

`group  tr  n_slices  slice_order  stc  drop_volumes  pe_dir  evidence  note`

* `group`: site name matching the manifest `group`, or `*` = default row.
* `slice_order`: `SA SD IA IA2 ID ID2` (DPABI codes, 1-based slice numbers,
  `IA`=1,3,5…2,4…; `IA2`=2,4,…1,3,…; `ID`=N,N-2,…,N-1,N-3,…; `ID2`=N-1,N-3,…,N,N-2,…),
  `file:<path>` (one slice time in seconds per line), or `unknown`.
* `stc`: `apply` | `skip`. `unknown` order forces `skip`.
* Explicitly unconfirmed vendor-inferred timing rows default to `skip` even
  when a candidate order is recorded. A numeric validity check is not external
  protocol verification; evidence and notes retain that distinction.
* Slice times: slice acquired at rank r (0-based) gets `r * TR / n_slices`.
* Mismatch between table `n_slices`/`tr` and the NIfTI header ⇒ STC skipped for
  that run with a warning in `ingest_report.tsv` (never "fixed").
* Sidecar written by ingest: `RepetitionTime`, `SliceTiming` (only when applied),
  `SliceEncodingDirection` (`k`), `PhaseEncodingDirection` (if known),
  `SliceTimingSource`, `SliceTimingEvidence`, `DropVolumes`, `AcquisitionGroup`.
* For `INPUT_LAYOUT=bids`, existing sidecars win (same-stem JSON, then
  dataset-root `task-<task>_bold.json`); the table only fills gaps.

## 6. Output file names

Anat (`derivatives/sub-X/anat/`), T1w space = the reference T1 grid
(FreeSurfer conformed grid in `ANAT_MODE=freesurfer`; world coordinates equal
scanner coordinates of the input T1):

```
sub-X_desc-preproc_T1w.nii.gz          bias-corrected T1 (FS nu.mgz | N4 of raw T1)
sub-X_desc-brain_mask.nii.gz
sub-X_desc-brain_T1w.nii.gz
sub-X_desc-aseg_dseg.nii.gz            FreeSurfer aseg | SynthSeg labels (same codes)
sub-X_label-WM_mask.nii.gz             eroded, for nuisance signals
sub-X_label-CSF_mask.nii.gz            eroded lateral ventricles
sub-X_label-GM_mask.nii.gz
sub-X_label-WMbbr_mask.nii.gz          un-eroded WM (FLIRT-BBR fallback only)
xfm/T1w_to_MNI_0GenericAffine.mat      ANTs
xfm/T1w_to_MNI_1Warp.nii.gz
xfm/T1w_to_MNI_1InverseWarp.nii.gz
sub-X_space-<TPL>_desc-preproc_T1w.nii.gz     warped brain, 1 mm
sub-X_space-<TPL>_desc-brain_mask.nii.gz
sub-X_desc-anatqc.json                 Euler number, Dice, Jacobian stats, volumes
sub-X_hemi-{L,R}_{white,pial,midthickness}.surf.gii        (surface branch)
sub-X_hemi-{L,R}_space-fsLR_den-32k_midthickness.surf.gii  (surface branch)
```

`<TPL>` = `$TEMPLATE_NAME` (default `MNI152NLin6Asym` = FSL MNI152).

Func (`derivatives/sub-X/func/`):

```
<RUN>_desc-prep_info.json                         provenance of stage 03
<RUN>_desc-hmc_motion.par                         mcflirt: rx ry rz (rad) tx ty tz (mm)
<RUN>_desc-hmc_relrms.txt / _absrms.txt
<RUN>_desc-outliers_timeseries.1D                 3dToutcount fraction (pre-despike)
<RUN>_from-bold_to-T1w_itk.txt                    boldref -> T1w (ITK)
<RUN>_space-T1w_boldref.nii.gz
<RUN>_space-T1w_desc-brain_mask.nii.gz
<RUN>_space-T1w_desc-preproc_bold.nii.gz
<RUN>_space-T1w_label-{WM,CSF,GM}_mask.nii.gz     tissue masks on the BOLD grid
<RUN>_space-<TPL>_res-<R>_boldref.nii.gz
<RUN>_space-<TPL>_res-<R>_desc-brain_mask.nii.gz
<RUN>_space-<TPL>_res-<R>_desc-preproc_bold.nii.gz
<RUN>_desc-confounds_timeseries.tsv + .json
<RUN>_desc-censor.1D                              1 = keep, 0 = censored
<RUN>_desc-<S>_regressors.1D + _denoise.json      per strategy S
<RUN>_space-<TPL>_res-<R>_desc-<S>_bold.nii.gz    denoised, unsmoothed
<RUN>_space-<TPL>_res-<R>_desc-<S>sm<F>_bold.nii.gz   optional smoothed copy
<RUN>_space-fsLR_den-91k_desc-preproc_bold.dtseries.nii       (surface branch; pre-denoise, scaled)
<RUN>_space-fsLR_den-91k_desc-preproc_tsnr.dscalar.nii        (surface branch)
<RUN>_desc-surfqc.json                                        (surface branch)
<RUN>_space-fsLR_den-91k_desc-<S>_bold.dtseries.nii           (surface branch)
<RUN>_space-<TPL>_atlas-<A>_desc-<S>_timeseries.tsv + _coverage.tsv + .json
<RUN>_space-<TPL>_atlas-<A>_desc-<S>_connectivity.tsv         Pearson r, retained frames
<RUN>_space-<TPL>_atlas-<A>_desc-preproc_timeseries.tsv       pre-denoise ROI means (for ROI tSNR)
<RUN>_space-fsLR_atlas-<A>_desc-<S>_timeseries.tsv + _connectivity.tsv   (surface branch)
<RUN>_space-fsLR_atlas-<A>_desc-preproc_timeseries.tsv        (surface branch)
<RUN>_desc-validation.json + <RUN>_desc-validation.tsv        stage 10, one row per stream x strategy x atlas
<RUN>_desc-qc_metrics.json
<RUN>_atlas-<A>_desc-<S>_roiqc.tsv                per-ROI tSNR, variance removed, coverage
```

## 7. Processing order

**00 ingest** (Python) — discover runs (`INPUT_LAYOUT=dpabi|bids`), validate
headers (4D, TR, units, qform/sform codes, obliquity, dtype, FOV), write a
normalised copy for DPABI layout (codes set to 1, sform:=qform if sform_code=0,
TR stored in seconds; voxel data untouched), write sidecars from the
acquisition table, write manifest/participants/ingest_report.

**01 anat_recon** — `ANAT_MODE=freesurfer`: `recon-all -all` with
`-parallel -openmp $NTHREADS`; complete = `scripts/recon-all.done` and
`surf/{lh,rh}.pial` and `mri/aseg.mgz`. Refuse to touch a subject with
`scripts/IsRunning*`. `ANAT_MODE=synth`: nothing to do (no surfaces possible).

**02 anat_prep** — freesurfer mode: `nu.mgz`→T1w, `brainmask.mgz`→mask
(binarise, fill holes), `aseg.mgz`→dseg. synth mode: N4 → `mri_synthstrip` →
`mri_synthseg --robust` resampled to the T1 grid. Tissue masks with
`mri_binarize --match … --erode N`: WM = 2 41 (erode `WM_ERODE`), CSF = 4 43
(erode `CSF_ERODE`), GM = 3 42 8 47 10 11 12 13 17 18 26 28 49 50 51 52 53 54 58 60.
T1→MNI: `antsRegistrationSyN.sh` (`NORM_QUALITY=precise`) or
`antsRegistrationSyNQuick.sh` (`quick`), brain-to-brain, 1 mm template, `-t s`.
QC JSON: Euler number/holes, Dice(warped mask, template mask), Jacobian
percentiles and fraction ≤ 0, correlation with template in mask.

**03 func_prep** (per run)
1. float32 copy, drop `DropVolumes` (sidecar) or `DROP_VOLUMES`; non-steady-state
   detector reports (warns if detected > dropped).
2. Raw QC before any cleaning: `3dToutcount -automask -fraction -polort`, `3dTqual`.
3. `DESPIKE=yes`: `3dDespike -NEW -nomask`.
4. HMC estimate (pre-STC): ref0 = min-outlier volume → `mcflirt` pass 1 →
   boldref = temporal median of pass-1 output → `mcflirt -reffile boldref -mats
   -plots -rmsrel -rmsabs` pass 2. Only matrices/parameters are kept.
5. STC (if sidecar has `SliceTiming` and `STC!=off`): `3dTshift -TR -tzero
   (mid acquisition) -tpattern @file -$STC_INTERP`. `STC=require` fails when
   timing is missing; `auto` skips with a warning.
6. boldref → `N4BiasFieldCorrection`; EPI mask: `mri_synthstrip` (fallback
   `3dAutomask`).
7. Coreg: freesurfer mode = `mri_coreg` → `bbregister --init-reg … --bold`;
   reject BBR (keep mri_coreg) if `lta_diff … --dist 4` > `BBR_MAX_DISP_MM` or
   BBR min cost > `BBR_MAX_COST`; synth mode = FLIRT 6-dof + FLIRT-BBR with the
   same rule. Result stored as ITK text transform (`lta_convert --outitk` /
   `wb_command -convert-affine -from-flirt … -to-itk`).
8. mcflirt `MAT_*` → ITK (`wb_command -convert-affine -from-flirt MAT boldref
   boldref -to-itk`).
9. One-shot resampling per volume with `antsApplyTransforms -n
   LanczosWindowedSinc --float` to (a) T1w grid at `FUNC_T1W_RES` mm and (b)
   template grid at `MNI_RES` mm; merge with the true TR; clip negatives.
10. Masks per space = T1 brain mask on that grid ∩ dilated EPI support mask ∩
    extents (temporal min > 0). One global scale factor (median in-brain of the
    T1w-space mean → 10000) applied to both outputs and recorded.

**04 confounds** (Python, T1w-space BOLD) — 24 motion parameters (fMRIPrep
column names), WM/CSF/global means (+ derivative, power2, derivative1_power2),
aCompCor (`w_comp_cor_00..`, `c_comp_cor_00..`; DCT 128 s high-pass before PCA;
fixed `ACOMPCOR_N` each), cosines, FD (Power, 50 mm) and Jenkinson relative RMS,
DVARS + standardised DVARS, outlier fraction, censor vector (`CENSOR_FD`,
optional `CENSOR_PREV`, `CENSOR_DVARS`).

**05 denoise** — mean-centre data once per run and space, reuse that uncompressed
work input across strategies. For each `S`: select columns →
`regressors.1D` (centred, unit-scaled) → `3dTproject -ort
-polort -dt [-passband] [-censor -cenmode]` on the template-space BOLD (and on
the T1w-space BOLD when `SURFACE=yes`). Degrees-of-freedom accounting in
`_denoise.json`; `dof_remaining < MIN_DOF` is flagged (not fatal). NTRP uses
all fit rows after interpolation; ZERO/KILL use retained rows. aCompCor includes
all DCT bases used before PCA, since Fourier stop bands do not exactly span them.
`dof_remaining = fit_rows - numerical joint-design rank` is an algebraic
dimension, not effective sample size or regularized-smoother DOF. AFNI separately
requires >=9 retained observations and fewer nominal columns than retained rows;
that failure stops the strategy regardless of numerical rank. Optional
`3dBlurInMask -FWHM $SMOOTH_FWHM` copy.

Strategies: `wmcsf24` (24P + WM + CSF), `wmcsf24gsr`, `36p` (24P + WM/CSF/GS
with derivatives and squares), `acompcor` (12P + 5 WM + 5 CSF aCompCor),
`acompcorgsr`, `legacy8` (6P + WM + CSF), `legacy9gsr` (6P + WM + CSF + GS).

**06 surface** (optional, freesurfer mode, needs `fetch_resources`) — see
`docs/SURFACE.md`: FS surfaces → GIFTI (`mris_convert --to-scanner`),
midthickness, cortex ROI, HCP goodvoxels on the T1w-space preproc BOLD,
ribbon-constrained mapping (`-voxel-subdiv 5`, 7 when slice ≥ 3.5 mm), dilate,
mask, `-metric-resample ADAP_BARY_AREA` to fsLR-32k, subcortex from the 2 mm
template-space series and HCP `Atlas_ROIs.2`, dense time series per strategy,
optional `-cifti-smoothing`. Requires `MNI_RES=2`.

**07 timeseries** — volume: atlas on the BOLD grid (NN resample if needed),
mean over covered, finite, non-constant voxels; ROI with coverage <
`MIN_ROI_COVERAGE` ⇒ NaN; TSV with label header; FC on retained frames.
CIFTI: `wb_command -cifti-parcellate` → TSV.

**08 qc / 09 group_qc** — see section 9.

## 8. `lib/common.sh` API (source it; then call `fp_init "$@"`)

```
fp_init "$@"            parse "-c CONF" (or $FMRIPROC_CONFIG), remaining args -> FP_ARGS;
                        loads default.conf then CONF, env overrides win; derives dirs;
                        exports thread variables; sets SUBJECTS_DIR=$FS_DIR
log LEVEL msg...        LEVEL in INFO WARN ERROR OK DEBUG; stderr + $FP_LOGFILE
die msg...              log ERROR and exit 1
run cmd args...         log the command, execute (DRY_RUN=yes: only log)
require_cmds c...       die if a command is missing
require_files f...      die if a file is missing/empty
fp_set_log SUB STAGE    route log()/run() output to logs/SUB/STAGE.log
stage_should_run STAGE SUB [RUN] -- VAR...   0 = run, 1 = skip (hash of pipeline
                        version + stage script + shared Bash + stage Python dependencies + listed config values unchanged,
                        SKIP_EXISTING=yes, FORCE!=yes)
stage_mark_done STAGE SUB [RUN]              write marker (uses the last hash)
manifest_runs SUB       rows of the manifest for SUB (tab-separated, no header)
subject_t1w SUB         path of the subject's T1w in rawdata
anat_dir SUB | func_dir SUB | fig_dir SUB | work_anat SUB | work_func SUB RUN
pyrun module args...    $PYTHON_BIN -m fmriproc.<module> args
json_get FILE KEY [DEFAULT]   scalar from a JSON file
nvols IMG | img_tr IMG  via nibabel-free tools (fslval)
mask_count IMG          number of non-zero voxels
template_path KIND      KIND in brain|head|mask at 1 mm, or brain_res|mask_res at $MNI_RES
is_yes VALUE            true for yes/true/1/on
fp_cpu_limit            minimum of affinity, cgroup, Slurm and optional CPU_BUDGET
fp_mem_limit_kb          minimum memory capacity/quota (KiB), including MEMORY_BUDGET_GB
fp_check_resources      check worker count x per-worker budget before heavy stages
```

Stage script skeleton:

```bash
#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
fp_init "$@"
SUB="${FP_ARGS[0]:?usage: $0 [-c conf] sub-XXXX}"
fp_set_log "$SUB" 03_func_prep
...
```

Stages that loop over runs read `manifest_runs "$SUB"`; columns as in section 5.
A failing run fails the stage (non-zero exit) after the remaining runs were
attempted.

## 9. QC

Per run `<RUN>_desc-qc_metrics.json` (flat keys, numbers or strings):

* acquisition: tr, n_volumes_raw, n_dropped, n_volumes, minutes, voxel size,
  stc_applied, stc_evidence, nss_detected, despike_fraction
* motion: fd_mean, fd_median, fd_max, fd_pct_gt_02, fd_pct_gt_05, relrms_mean,
  absrms_max, n_censored, pct_censored, minutes_retained, longest_segment
* signal: tsnr_gm_median (pre-denoise), tsnr_wm_median, tsnr_brain_median,
  dvars_std_mean, outlier_frac_mean (AOR), quality_index_mean (AQI), gcor,
  fd_dvars_corr, gs_fd_corr, fwhm_acf (3dFWHMx, optional)
* registration: coreg_method, bbr_cost, bbr_vs_init_mm, coreg_dice (EPI support
  vs T1 brain mask on the BOLD grid), norm_dice, jacobian_p01/p99,
  jacobian_nonpos_frac, template_corr
* masks: brain_mask_voxels, wm_mask_voxels, csf_mask_voxels, dropout_fraction
* per strategy `S` (prefix `S.`): n_regressors, dof_remaining, tsnr_gm_median_post
  (pre-denoise mean / residual SD), tsnr_gain, variance_removed_gm,
  fd_dvars_corr_post, and per atlas `A` (prefix `S.A.`): roi_tsnr_median,
  roi_tsnr_min, n_roi_nan, coverage_min, split_half_r (contiguous halves,
  retained frames), network_contrast (when the atlas has network labels),
  fc_mean, fc_sd
* surface (when present): tsnr_cortex_median, pct_badvertices, pct_goodvoxels_excluded

Time-series SNR definitions (final data are zero-mean, so mean/SD of the final
series is meaningless):

* `roi_tsnr` = mean over time of the ROI-mean **pre-denoise scaled** BOLD divided
  by the SD over retained frames of the **final denoised** ROI series.
* `variance_removed` = 1 − var(denoised) / var(pre-denoise, polort-detrended).
* `split_half_r` = Pearson r between the upper triangles of Fisher-z FC from the
  first and second half of retained frames.

Report (`derivatives/sub-X.html`, self-contained, images embedded): summary
table with traffic-light flags, ingest/acquisition table, T1 + brain mask +
tissue contours, T1→template contours, surfaces on T1 (if present), boldref +
mask, EPI→T1 with white-matter contours, EPI→template, motion/FD/DVARS/outlier
traces with censored frames shaded, carpet plot before and after denoising
(GM/WM/CSF ordered), tSNR map before/after, confound correlation heat-map, FC
matrix + histogram per strategy, ROI coverage/tSNR bar, surface tSNR (if present).

Flag thresholds are config values (`QC_*`), defaults in `default.conf`.

Group (`derivatives/group/`): `group_qc.tsv` (one row per run), robust
z-score outlier flags within `group` (site), distributions per site, QC-FC
(edge-wise correlation of FC with mean FD, % significant, median |r|,
distance dependence) per strategy and atlas when ≥ `QCFC_MIN_SUBJECTS` runs.

## 10. Cross-stage file and JSON contracts

Work files that cross stage boundaries (everything else in `work/` is private):

```
work/sub-X/func/<RUN>/denoise/<S>_space-T1w_bold.nii.gz   05 -> 06 (only when SURFACE=yes)
work/sub-X/func/<RUN>/surface/                            06 private
```

`<RUN>_desc-prep_info.json` (stage 03) keys:
`source, run_label, tr, n_volumes_raw, n_dropped, n_volumes, nss_detected,
despike (bool), despike_fraction, stc_applied (bool), stc_reason, stc_interp,
slice_timing_source, slice_timing_evidence, tzero, hmc_reference,
coreg_method ("bbregister" | "mri_coreg" | "flirt_bbr" | "flirt"), bbr_cost,
bbr_vs_init_mm, bbr_rejected (bool), epi_mask_method, scale_factor,
func_t1w_res, mni_res, template, voxel_size (list), obliquity_deg,
wm_mask_voxels, csf_mask_voxels, gm_mask_voxels, tissue_erosion_relaxed (bool),
tool_versions (object)`.

`sub-X_desc-anatqc.json` (stage 02) keys:
`anat_mode, euler_lh, euler_rh, holes_total, brain_volume_mm3, norm_quality,
norm_dice, template_corr, jacobian_p01, jacobian_p50, jacobian_p99,
jacobian_nonpos_frac, wm_voxels, csf_voxels, gm_voxels`.

`<RUN>_desc-confounds_timeseries.json` (stage 04): per-column description plus
top-level `"censor": {fd_threshold, prev, dvars_threshold, n_censored, n_volumes,
minutes_retained, longest_segment}` and `"acompcor": {n_wm_voxels, n_csf_voxels,
variance_explained_wm (list), variance_explained_csf (list)}`.

`<RUN>_desc-<S>_denoise.json` (stage 05): `strategy, columns (list), n_regressors,
polort, filter_mode, band (list), n_volumes, n_censored, censor_mode,
dof_bandpass_cost, dof_remaining, low_dof (bool), smooth_fwhm, inputs (object)`.
v2.1 adds `retained_observations, fit_rows, design_columns, design_rank,
design_rank_tolerance, algebraic_dof, dof_definition, afni_nominal_dof,
afni_model_feasible`; see stage 05 above. Model feasibility is distinct from rank.

Python modules are CLI programs (`argparse`, `main()` returning an exit code,
`if __name__ == "__main__": sys.exit(main())`), take explicit file paths and
parameters as arguments (they do not read the bash config), write TSV with a
header row and `n/a` for missing values, and JSON with `indent=2`.

## 11. Robustness rules

* Heavy stages call `fp_check_mem` (config `MIN_MEM_GB`).
* Erosion of tissue masks is done on a 1 mm grid (FreeSurfer conformed space or
  the SynthSeg 1 mm output), so iterations ≈ millimetres; masks reach the BOLD
  grid by linear interpolation and a 0.9 threshold. If a mask has fewer than
  `MIN_TISSUE_VOX` voxels on the BOLD grid the threshold is relaxed (0.7, 0.5)
  and `tissue_erosion_relaxed` is recorded.
* Header-derived alignment is never trusted (ABIDE headers carry generic
  origins): coregistration always starts from `mri_coreg` / FLIRT search.
* `antsRegistration` runs with `--random-seed $ANTS_SEED`
  (`antsRegistrationSyN*.sh -e`).
* Outputs are written to a temporary name and moved into place, so an
  interrupted stage never leaves a plausible-looking final file.
* Per-volume temporaries are removed after merging unless `KEEP_WORK=yes`.

## 12. Validation of the final time series and volume-vs-surface comparison (stage 10)

`stages/10_validate.sh <sub>` (per subject, after 07; `--group` = dataset level,
after all subjects). Python: `fmriproc.validate` (per run) and
`fmriproc.compare_streams` (group). It only reads the ROI tables of stage 07
(`*_timeseries.tsv`, `*_desc-preproc_timeseries.tsv`, `*_coverage.tsv`),
`desc-censor.1D`, the confounds TSV and `$RESOURCE_DIR/atlases/<A>/` - never 4D data.

A *stream* is `volume` (`space-<TPL>` tables) or `surface` (`space-fsLR` tables).
Schaefer volumetric atlases and their fsLR dlabels contain the same cortical
parcels in the same order, so the two streams are compared parcel by parcel.

Per run x stream x strategy x atlas (`<RUN>_desc-validation.tsv`, long format:
`stream strategy atlas metric value`; the JSON holds the same numbers nested):

* `roi_tsnr_median`, `roi_tsnr_p10` - mean(pre-denoise ROI series) / SD(denoised ROI
  series, retained frames).
* `variance_removed_median`.
* `split_half_r` - reliability of FC between contiguous halves of retained frames.
* `network_contrast` - (mean z within network - mean z between) / SD(between);
  needs `network` labels.
* `homotopic_contrast` - mean Fisher-z FC of homotopic parcel pairs minus mean
  inter-hemispheric non-homotopic FC. Pairs: for each LH parcel the RH parcel of
  the same network whose centroid (from the volumetric atlas) is nearest to the
  mirrored LH centroid (|x| flipped), accepted within 20 mm.
* `dmn_contrast` - mean z between Default-network parcels containing `PCC`/`pCunPCC`
  and those containing `PFC` minus their mean z with `SomMot` parcels.
* `lowfreq_power_fraction` - on the pre-denoise, polort-detrended ROI series:
  power in 0.01-0.1 Hz / power in 0.01 Hz-Nyquist (median over ROIs).
* `fd_fc_coupling` - |Spearman| between FD and the frame-wise co-fluctuation
  amplitude (RSS of z-scored ROI products / edge time series) over retained frames:
  residual motion coupling of the final series (lower is better).
* `gs_residual_sd` - SD of the mean over ROIs of the final series (z-scored ROIs).
* `n_roi_nan`, `n_retained`, `dof_remaining` (copied from `_denoise.json`).

Stream comparison per run (`<RUN>_desc-streamcompare.tsv`, when both streams
exist and unique ROI names can be aligned): `fc_similarity` (Pearson r between the Fisher-z upper triangles of the two
streams), and the paired difference surface - volume of every metric above; per-ROI
table with `roi_tsnr` of both streams.

Group level (`derivatives/group/`): `validation_long.tsv` (all runs),
`stream_comparison.tsv` (per strategy x atlas x metric: n, median volume, median
surface, median paired difference, Wilcoxon signed-rank p and BH q within each
strategy x atlas metric family). Repeated paired runs are averaged within subject
before inference; n is independent subjects. QC-FC similarly uses subject means
and BH q across edges. `fc_typicality.tsv` excludes all runs of the target subject,
and weights the remaining subjects equally, per stream. Figure
`stream_comparison.png` (paired dot plots) and `validation_report.html`.
Interpretation rules: these comparisons remain exploratory. No single metric
decides biological quality; within-run split-half consistency is not test-retest
reliability. Strategy summaries do not automatically rank different available
subject sets. Missing essential QC or failed metric blocks produce `incomplete`,
not `pass`; missing/invalid censor leaves retained denominators unknown.
