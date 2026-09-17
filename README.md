# fMRI_Processing

## ADNI pilot branch (2026-09-17)

The local `codex/adni-preprocessing-pilot` branch starts from historical commit
`2eaa2ba2eda5ae6792bbf4331201159f5d112c27`. Session-path and per-run metadata
handling were brought forward from the existing CNS-001 CN scripts. A bounded
real-data pilot has completed across GE, Philips, and Siemens long-TR and
multiband acquisitions. This is not a validated cohort release.

The corrected path reuses completed FreeSurfer `brain.mgz` and `T1.mgz` without
running `recon-all`. Structural voxel storage is reoriented to the standard
orientation, preserving world anatomy, before affine searches. It reruns FAST with the inverse standard-to-subject prior
matrix, uses the complete structural brain mask, estimates motion before STC,
applies verified JSON slice timing, and reuses one EPI-to-T1 BBR transform.
T1-to-MNI normalization uses FNIRT; nonpositive brain Jacobians reject a failed
warp before downstream processing. Functional-to-MNI warps are composed before
the final resampling. Temporal bandpass, polynomial trends and nuisance
regressors are projected together using AFNI `3dTproject`.
Before projection, data are centered and nuisance columns are centered/scaled
in double precision. This preserves the model's column space while reducing
float32 cancellation from the large BOLD baseline. A redundant, unused first
FAST fit has been removed; only the full-prior fit is run.

The global/CSF/WM masks are static 3D masks. CSF and WM are thresholded at 0.9,
eroded once in T1 space, and mapped once to EPI. The EPI signal-support mask uses
fixed `clfrac=0.2`, `peels=0`, and one dilation, intersected with the mapped whole
brain mask and closed/filled. These pilot choices must be assessed using both
brain coverage and anatomical alignment; a larger mask alone is not success.
ROI means exclude uncovered/constant voxels, export coverage counts and retain
NaN for a missing ROI rather than inventing signal. `QC_nor` respects NIfTI axes.

Use a **new derived output root**; legacy functional results already contain
irreversible masking and cannot be the input to this corrected path. The main
entry point rejects mixing old outputs with this revision. `STC=required` fails
on missing/incompatible timing; `STC=off` is an explicit documented alternative,
not an automatic fallback. `DROP_VOLUMES` is explicit and defaults to 0. The
spatial diagnostic pilot retains all volumes for comparison; this does not
settle non-steady-state or motion-censoring decisions. No fieldmap correction
is implemented in this branch, so native susceptibility artifacts remain a
limitation. No new dependencies are installed by these scripts.

Legacy labels remain for compatibility: `NoGRS` **regresses global signal**;
`Retain_GRS` retains it. FC_step5 now uses a joint AFNI projection rather than
the old FEAT design templates.

The pilot uses a separate work root, independent of the source dataset. The
entry point, inside a container with AFNI/FSL/FreeSurfer installed, is:

```bash
bash /project/pipeline/run_adni_pilot.sh /project sub-example ses-example
```

Inputs are staged under `/project/input`; saved FreeSurfer inputs are under
`/project/recon`; an existing FreeSurfer license is read from `/project/license.txt`.
The private validation launcher bounds CPU/RAM per scan and concurrency.
FC_step1–6 produce registration QC and `quality.json`. The completed pilot
passed output-dimension and ROI extraction checks for both denoising branches;
the shared spatial transforms passed positive brain-Jacobian checks.
Saved native residuals also passed the recorded
numerical projection check. Canonically oriented QC and old/new spatial outputs
were visually reviewed. These checks establish execution on the sampled
protocols, not anatomical ground truth or suitability of every scan for analysis.
Motion and incomplete regional coverage still require explicit inclusion rules.
Run logs, participant identifiers, licenses and images stay outside this repository.

For a verified checkpoint in the same run, `START_STEP=5` reruns only nuisance
regression, ROI extraction and checks. Reuse requires unchanged upstream inputs,
parameters and implementation. A two-column subject/session TSV can be dispatched
with GNU Parallel; use `--jobs` to bound concurrency and retain per-scan logs.

`python test_helpers.py` checks multiband times, negative slice-axis ordering,
rejection of missing timing, and ROI means with constant/missing voxels. Bash
syntax checks pass. Runtime and visual QC are recorded with the pilot outputs.
The input-scaled projection check uses a float32 accumulation reference; it is
neither a formal error bound for the full fitting algorithm nor a measure of
biological denoising. Larger masks and higher fitted-template similarity alone
do not establish recovery of true signal or independent registration accuracy.

## Status
This repository has been consolidated into **MRI_Processing**.

## Primary Repository
- Main repo: https://github.com/cz-bszy/MRI_Processing
- Merged location in main repo: merged/fMRI_Processing/

## What This Means
- This repository remains as a historical source.
- New maintenance and integration should be done in MRI_Processing.

## Legacy Notes
Original scripts are preserved and can still be referenced for reproducibility.
