# Changelog

## 2.2.0 (2026-09-24)

First release tested end to end on real data: two ABIDE subjects (ABIDEII-GU_1 and
ABIDEII-NYU_2) through both the synth and the FreeSurfer + surface configurations,
on Windows 11 with Docker Desktop and the image `zhaochang07/myubuntu:neuro-v2`.
The test record is in README section 15.2.

Stage outputs change, so the version is part of every stage hash: a rerun
recomputes every stage. A finished recon-all is adopted, not repeated.

### Fixed

- Ingest strips the AFNI NIfTI extension. AFNI read the TR from it (1 s for
  ABIDEII-GU_1) instead of the header (2 s), and slice timing was silently skipped.
  The STC check now reads the rawdata header.
- `dof_remaining` counts the retained frames only. With `CENSOR_MODE=NTRP` it used
  to count the interpolated censored frames as well, so the `MIN_DOF` flag missed
  heavily censored runs. `algebraic_dof` keeps the fit-row count.
- Surface coverage: vertices that stage 06 only filled in by the 10 mm dilation
  (`desc-sampled_mask` = 0) no longer count as covered and are left out of parcel
  means. A parcel outside the field of view is n/a in both streams.
- CIFTI parcel names are matched to the atlas label table by label key; the CBIG
  dlabel names differed for 19 Schaefer-100 parcels and stage 10 refused to compare.
- `mris_convert -c` output name in stage 06.
- SynthSeg `--robust` ran out of memory in a 12 GB VM: the T1 is cropped to the
  brain first; flags are configurable (`SYNTHSEG_FLAGS`).
- Group QC no longer crashes when no run has metrics.
- The reading of a negative post-denoising FD-DVARS correlation in the QC report
  and docs: it reflects the high leverage of motion regressors at high-motion
  frames (a numpy rebuild of the 3dTproject design with a random-regressor
  control), not over-correction by derivative terms.
- Tests: locale-independent sorting, a CR check that works on Windows, and the
  runner prefers the pipeline's Python interpreter.

### Added

- Run inclusion for the group steps (`py/fmriproc/inclusion.py`, stages 09 and
  10 `--group`): pre-specified `EXCLUDE_*` criteria (mean FD, max FD, % frames with
  FD > 0.2 mm, retained minutes, censor-aware DOF per strategy, run-level QC fail).
  Nothing is deleted: `derivatives/group/inclusion.tsv` records the decision and
  the reasons; QC-FC and the stream and strategy comparisons use the included
  runs; `validation_long.tsv` keeps every run with an `included` column.
- Optional phenotype table (`PHENOTYPE_TSV`, id, group and label settings): the
  group report compares exclusions and head motion between groups.
- Censoring options `CENSOR_NEXT` (frames after a high-FD frame) and
  `CENSOR_MIN_SEGMENT` (short kept stretches); both off by default.
- `FREEZE_CODE`: every run executes a copy of the code in `logs/code_<run_id>/`,
  with an md5 manifest and the git revision; stage hashes do not depend on where
  the code lives.
- `docs/STEPS_zh.md`: what every stage does, what to watch, how to check it.

## 2.1.0 (2026-09-22)

Rewrite of the v1 scripts as a bash stage pipeline with a Python helper package:
DPABI/BIDS ingest with an evidence-tiered acquisition table, single-interpolation
resampling, `3dTproject` joint projection with DOF accounting, fsLR-32k surface
branch, per-run QC metrics and HTML reports, group QC with QC-FC, time-series
validation metrics and a volume-versus-surface comparison, Docker and Singularity
launchers. v1 scripts are kept in `legacy/`.
