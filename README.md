# Modular Neuroimaging Processing Pipeline

This repository contains a Bash-based orchestration layer for neuroimaging workflows. The entrypoint (`main.sh`) currently drives a functional MRI (fMRI) pipeline and is structured so additional modalities (e.g., structural MRI, PET) can be plugged in with minimal effort.

## Features
- **Pipeline registry** that enumerates ordered steps, associated commands, and completion checks per modality.
- **Extensible architecture**: register new modalities/steps without touching the subject-processing core.
- **Robust execution controls**: skip-completed logic, dry-run support, parallelisation, and consistent logging.
- **Centralised error tracking** with per-subject summaries and daily reports.

## Prerequisites
Install and configure the following toolkits before running the pipeline:

- [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)
- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)
- [AFNI](https://afni.nimh.nih.gov/)
- [GNU Parallel](https://www.gnu.org/software/parallel/)

Ensure `FREESURFER_HOME` and `FSLDIR` are set in the environment and their setup scripts have been sourced.

## Directory Layout
```
fMRI_Processing/
├── main.sh
├── FC_step0 ... FC_step5        # Modality-specific helper scripts
├── standard/                    # Standard-space templates
├── template/                    # FEAT/FSF templates
├── tissuepriors/                # Tissue priors for segmentation
└── ...
```

### Input structure
```
/input_dir/
└── sub-001/
    ├── anat/
    │   └── sub-001_T1w.nii.gz
    └── func/
        └── sub-001_task-rest_bold.nii.gz
```
Sessions (`ses-001`, `ses-002`, …) are automatically detected if present under each subject directory.

### Output structure
```
/output_dir/
├── logs/pipeline_YYYYMMDD.log
├── error_logs/
│   ├── failed_subjects_YYYYMMDD.txt
│   └── errors_YYYYMMDD.log
├── recon_all/
└── sub-001/
    ├── anat/
    └── func/
```
A per-run summary is written to `processing_report_YYYYMMDD.txt`.

## Configuration
Key settings reside near the top of `main.sh`:
- `INPUT_DIR`, `OUTPUT_DIR`, `STANDARD_DIR`, `TISSUES_DIR`, `TEMPLATE_DIR`, `RECONALL_DIR`
- Processing parameters (`NUM_THREADS`, `FWHM`, `SIGMA`, `HIGHP`, `LOWP`, `TR`, `TE`, `N_VOLS`)
- Pipeline controls (`PIPELINES_TO_RUN`, `FSF_TYPES`, `GENERAL_FLAGS`, `SKIP_EXISTING`, `DRY_RUN`, `VERBOSE`)

Override any variable at invocation time by exporting or prefixing the call:
```bash
DRY_RUN=true PIPELINES_TO_RUN="fmri" bash main.sh
```

## Usage
1. Adjust configuration values in `main.sh` or export overrides.
2. Run the pipeline:
   ```bash
   bash main.sh
   ```
3. Review logs under `OUTPUT_DIR/logs` and the generated processing report.

### Common scenarios
- Inspect commands without executing:
  ```bash
  DRY_RUN=true VERBOSE=true bash main.sh
  ```
- Reprocess even if outputs exist:
  ```bash
  SKIP_EXISTING=false bash main.sh
  ```
- Limit processing to a subset of modalities (e.g., future `smri`):
  ```bash
  PIPELINES_TO_RUN="fmri smri" bash main.sh
  ```

## Extending to New Modalities
1. Create modality-specific helper scripts or command builders.
2. Register the steps inside `initialize_pipelines()` using `register_shell_step` or `register_custom_step`.
3. Add any modality-specific preparation to `prepare_pipeline_subject()`.
4. Update configuration defaults (e.g., include the new modality in `PIPELINES_TO_RUN`).

The registry automatically orders steps, handles skip logic, and ensures dry-run support.

## Error Handling
- Detailed step output is captured via temporary logs; failures append to `error_logs/errors_YYYYMMDD.log`.
- Unique subject/pipeline/step failures are tracked in `failed_subjects_YYYYMMDD.txt` for quick triage.
- Aggregate metrics, including per-step error counts, appear in the processing report.

## Troubleshooting
- **Missing `brain.mgz`**: recon-all output not located at `RECONALL_DIR`; verify FreeSurfer processing.
- **Functional outputs absent**: confirm `FC_step2` finished and input data follow expected naming.
- **Permission errors**: the pipeline attempts to `chmod` key folders; ensure the user has write access.
- **`parallel: command not found`**: install GNU Parallel or adjust the script to use serial execution.

## Contributing
Contributions are welcome. Submit pull requests with clear descriptions and, when introducing new modalities, include example configuration and documentation updates.

## License
Released under the MIT License. See `LICENSE` for details.
