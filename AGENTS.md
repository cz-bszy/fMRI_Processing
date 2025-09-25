# Repository Guidelines

## Project Structure & Module Organization
The pipeline is orchestrated from `main.sh`, which registers modality-specific steps and handles logging. Helper scripts `FC_step0`–`FC_step5` encapsulate fMRI stages (recon-all, preprocessing, nuisance regression, etc.), and live in the repository root for quick inspection. Reference data sit under `standard/`, while FEAT templates and nuisance regressors are in `template/`; tissue priors live under `tissuepriors/` grouped by resolution. Use these directories when wiring new modalities so paths stay consistent.

## Build, Test, and Development Commands
- `bash main.sh` — run the default fMRI pipeline with configured directories.
- `DRY_RUN=true VERBOSE=true bash main.sh` — print every scheduled command without executing, useful for quick validation.
- `PIPELINES_TO_RUN="fmri smri" bash main.sh` — multi-modal example illustrating how to opt into additional registries once implemented.
- `bash FC_step2 -h` — each helper script exposes `-h`; consult it before tweaking step-level behavior.

## Coding Style & Naming Conventions
All orchestration is Bash; always start new scripts with `#!/bin/bash` and `set -euo pipefail`. Indent with four spaces, keep functions in lower_snake_case (`prepare_pipeline_subject`), and reserve UPPER_SNAKE_CASE for environment-tuned constants. Leverage existing logging helpers (`log`, `record_error`) instead of ad-hoc `echo`. When adding options, mirror current `getopts` patterns and document defaults in the script header. Run `shellcheck <file>` locally to catch common pitfalls before review.

## Testing Guidelines
There is no dedicated test harness yet; rely on staged pipeline execution. For logic changes, perform a `DRY_RUN` first, then process a single subject by setting `SUBJECT_INCLUDE` in `main.sh` or filtering the input directory. Review `OUTPUT_DIR/logs/pipeline_*.log` and `error_logs/*` to confirm success. When adding new steps, supply sample input/output summaries in the PR.

## Commit & Pull Request Guidelines
Commit messages follow short, imperative summaries (e.g., `Refactor pipeline and update README`). Scope each commit to one concern and include configuration migrations or data updates in the message body. Pull requests should describe the motivation, list affected steps or scripts, note any required environment overrides, and link related issues. Attach snippets of representative command lines or log excerpts so reviewers can reproduce your validation.
