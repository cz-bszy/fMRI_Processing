# HCP-Inspired fMRI Processing Pipeline

This repository provides a Bash-driven orchestration layer for neuroimaging workflows. The default configuration reproduces a Human Connectome Project-style functional MRI (fMRI) stream: structural T1w/T2w pre-processing, FreeSurfer reconstruction, anatomical parcel generation, functional clean-up for task and resting-state runs, and FEAT-based model execution. The modular step registry makes it easy to insert, remove, or reorder stages without rewriting the control logic.

## Prerequisites
- **Operating system**: Linux (tested on Ubuntu-based environments)
- **Core toolkits**  
  - [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) ≥ 7.3 (requires `FS_LICENSE`)  
  - [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) ≥ 6.0 with `bet`, `fast`, `flirt`, `fnirt` on `$PATH`  
  - [AFNI](https://afni.nimh.nih.gov/) for nuisance regression utilities  
  - [GNU Parallel](https://www.gnu.org/software/parallel/) for subject-level concurrency
- Add `FREESURFER_HOME`, `FSLDIR`, and `AFNI` binaries to your shell environment before running `main.sh`.

## Quick Start
1. **Inspect data**  
   Place BIDS-formatted inputs under `data/` (default). Example:  
   ```
   data/
   └── sub-A00086238/
       └── ses-BAS1/
           ├── anat/
           │   ├── sub-A00086238_ses-BAS1_T1w.nii.gz
           │   └── sub-A00086238_ses-BAS1_T2w.nii.gz
           └── func/
               └── sub-A00086238_ses-BAS1_task-rest_bold.nii.gz
   ```
2. **Prime the environment**  
   ```bash
   source $FREESURFER_HOME/SetUpFreeSurfer.sh
   source $FSLDIR/etc/fslconf/fsl.sh
   export AFNI_TOOLBOX_PATH=/path/to/afni/bin   # ensure afni binaries are on PATH
   ```
3. **Run a smoke test**  
   By default only `sub-A00086238` is processed for a quick validation:
   ```bash
   cd fMRI_Processing-master
   bash main.sh
   ```
4. **Expand the scope**  
   ```bash
   TARGET_SUBJECTS=all bash main.sh                          # process every subject under INPUT_DIR
   TARGET_SUBJECTS="sub-A00086238 sub-A00086259" bash main.sh # explicit subset
   ```

Key outputs are written to `data/processing/`, including per-step logs, FreeSurfer surfaces (`recon_all/`), anatomical derivatives (`anat/`), functional artefact reductions (`func/`), and FEAT/timeseries results (`results/`).

## Pipeline Anatomy
Every subject/session pair passes through a fixed registry of shell helpers. Each helper runs under `set -euo pipefail` and communicates via explicit input/output folders, making failures easy to diagnose.

| Order | Step / Script | Core Functions | Inputs → Outputs |
|-------|---------------|----------------|------------------|
| 1 | **Structural Preprocess** (`FC_step1b`) | `bet`, `fast`, `flirt`, `3dcalc` | - Reads `anat/<sub>/<ses>/*T1w.nii*` and optional `*T2w.nii*`<br>- Produces bias-corrected T1w/T2w (`anat/preproc/T1w_preproc.nii.gz`, `T2w_preproc.nii.gz`), brain mask, T1/T2 mean |
| 2 | **FreeSurfer Recon-All** (`FC_step0`) | `recon-all` | - Consumes `T1w_preproc.nii.gz` (+ `T2w_preproc` when present)<br>- Writes FreeSurfer subject tree under `recon_all/<sub[_ses]>` (`brain.mgz`, surfaces) |
| 3 | **Anatomical Export** (`FC_step1`) | `mri_convert`, `flirt`, multi-channel `fast` | - Uses `recon_all/<sub>/mri/brain.mgz` and optional T2w<br>- Outputs structural brain (`anat/Stru_Brain.nii.gz`), affine to MNI (`std2sub.mat`), and tissue priors (`segment_prob_{0,1,2}.nii.gz`) |
| 4 | **Functional Preprocess** (`FC_step2`) | AFNI (`3dcalc`, `3dvolreg`, `3dTproject`, …), FSL (`flirt`) | - Ingests raw EPI (`func/*bold*.nii*`), structural brain & masks<br>- Emits motion-corrected, filtered data (`func/rest_pp.nii.gz`), example functional, nuisance masks, registration matrices |
| 5 | **Functional→Standard Registration** (`FC_step3`) | `flirt`, `convert_xfm` | - Requires `Stru_Brain`, `example_func`, registration matrices<br>- Generates standard-space transform + warped functional (`reg_dir/example_func2standard.nii.gz`, `.mat`) |
| 6 | **Tissue Segmentation for Nuisance** (`FC_step4`) | `flirt`, `fslmaths`, `parallel` | - Projects FAST priors into functional space<br>- Creates cleaned WM/CSF/global masks in `func/seg/` for regression |
| 7 | **FSF Processing** (`FC_step5`) | `3dmaskave`, `feat_model`, `film_gls`, `flirt` | - Reads motion parameters, nuisance masks, preprocessed time series<br>- Builds FEAT design, runs GLM, exports residuals to standard space (`results/<FSF_TYPE>/*.nii.gz`) |
| 8 | **Atlas Time Series** (`FC_step6`) | `3dresample`, `3dmaskdump`, `3dmaskave`, `1dcat` | - Consumes residual volumes per FSF type, atlas template<br>- Produces ROI time series (`results/<FSF_TYPE>/timeseries/*_timeseries.1D`) |

Completion checks (e.g., `fmri_has_functional_output`) guard each stage. When `SKIP_EXISTING=true`, reruns skip steps whose outputs already exist, dramatically shortening reprocessing cycles.

### Main Entrypoint (`main.sh`)
`main.sh` performs orchestration and validation:

1. **Environment & parameter checks** – Verifies required binaries (`recon-all`, `flirt`, AFNI tools) and directories (`INPUT_DIR`, `STANDARD_DIR`, etc.).
2. **Configuration normalisation** – Hydrates arrays (pipelines, FSF types), normalises session naming (`ses-XXX`), prepares per-subject directories.
3. **Pipeline registration** – Registers each step with a command builder and completion check. Additional modalities can be added by appending to `initialize_pipelines()`.
4. **Subject discovery** – Scans `INPUT_DIR/sub-*[/ses-*]`, filtered by `TARGET_SUBJECTS`.
5. **GNU Parallel execution** – Launches `process_subject_parallel` for each subject/session combination, passing configuration via exported functions and environment variables.
6. **Reporting and error capture** – Logs successes/failures per step, writes consolidated reports (`processing_report_YYYYMMDD.txt`), and captures failing step details under `error_logs/`.

Because `main.sh` orchestrates via builders/handlers, adding a new step usually means:
- authoring `FC_stepX`,
- defining `build_<pipeline>_<step>_command`,
- declaring a completion check,
- registering everything inside `initialize_pipelines()`.

## Configuration Reference
Modify `main.sh` or export variables at runtime:

- **Data locations**  
  `INPUT_DIR` (default `$(pwd)/data`), `OUTPUT_DIR` (default `$INPUT_DIR/processing`), `STANDARD_DIR`, `TISSUES_DIR`, `TEMPLATE_DIR`, `RECONALL_DIR`.
- **Subject selection**  
  `TARGET_SUBJECTS="sub-XXX sub-YYY"` or `TARGET_SUBJECTS=all` to bypass the default single-subject smoke test.
- **Structural pre-pass**  
  `STRUCT_PREPROC_ENABLED=true|false`, `T1_FILE_PATTERN`, `T2_FILE_PATTERN`, `STRUCT_PREPROC_BET_FRACTION` (BET threshold). Disabling falls back to the original T1-only flow.
- **Functional filtering**  
  `FUNC_FILE_PATTERN` (comma-separated globs), `HIGHP`, `LOWP`, `FWHM`, `SIGMA`.
- **Execution controls**  
  `NUM_THREADS` (per subject), `PIPELINES_TO_RUN`, `FSF_TYPES`, `SKIP_EXISTING`, `DRY_RUN`, `VERBOSE`, `GENERAL_FLAGS`.

Example:
```bash
STRUCT_PREPROC_BET_FRACTION=0.23 \
TARGET_SUBJECTS="sub-A00086238 sub-A00086259" \
FSF_TYPES="NoGRS Retain_GRS" \
bash main.sh
```

## Data Products
```
processing/
├── logs/pipeline_YYYYMMDD.log
├── error_logs/
│   ├── failed_subjects_YYYYMMDD.txt
│   └── errors_YYYYMMDD.log
├── recon_all/sub-XXX[_ses]/mri/brain.mgz
├── sub-XXX/
│   ├── anat/
│   │   ├── Stru_Brain.nii.gz
│   │   ├── T1w_preproc.nii.gz
│   │   └── segment_prob_{0,1,2}.nii.gz
│   └── func/
│       ├── example_func.nii.gz
│       ├── reg_dir/
│       └── seg/{wm,csf,global}_mask.nii.gz
└── results/
    └── <FSF_TYPE>/
        ├── feat/
        └── timeseries/
```
The run summary `processing_report_YYYYMMDD.txt` captures parameters, FSF types processed, and any step failures.

## Troubleshooting
- **`recon-all` cannot find T2w inputs**: ensure `FC_step1b` produced `T2w_preproc.nii.gz` in `anat/preproc/`; otherwise adjust `T2_FILE_PATTERN` or acquire T2w data.
- **Missing `brain.mgz` or recon directory**: clear `processing/recon_all/<subject>` and rerun; the script auto-cleans incomplete outputs when `SKIP_EXISTING=false`.
- **Functional step fails to locate EPI**: confirm `FUNC_FILE_PATTERN` matches file names (`*bold.nii.gz`, `*rest*.nii*`, etc.).
- **Workers exit immediately**: `TARGET_SUBJECTS` may filter everything. Set `TARGET_SUBJECTS=all` during bulk runs.
- **`parallel` not found**: install GNU Parallel or set `PARALLEL=serial` (future enhancement) to run sequentially.

## Extending the Pipeline
1. Add new step scripts (`FC_stepX`) or custom runners.
2. Register the step in `initialize_pipelines()` with an associated output check.
3. Update `prepare_pipeline_subject()` if you need additional directory scaffolding.
4. Document the change in `AGENTS.md` and, when appropriate, expose toggles in the configuration section.

The registry enforces execution order, handles dry runs, and surfaces per-step errors, so new modalities inherit the existing control-plane features automatically.

## License
This project is released under the MIT License. See `LICENSE` for details.
