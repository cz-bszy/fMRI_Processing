#!/bin/bash
# =============================================================================
# run_pipeline.sh - single entry point of fMRI_Processing v2
#
# Dataset-level stages run once, per-subject stages run for every selected
# subject (N_JOBS subjects at a time). A failing stage stops that subject only.
# Every run ends with logs/status_<timestamp>.tsv, a summary table and exit
# status 1 when anything failed. Contract: docs/DESIGN.md.
# =============================================================================
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
# shellcheck source=lib/common.sh
source "$(dirname "$SELF")/lib/common.sh"

# Execution order is fixed by the data flow, whatever the order in STAGES.
# 'validate' precedes 'qc' so that the subject report can include its numbers.
DATASET_PRE_STAGES=(ingest fetch)
SUBJECT_STAGES=(anat_recon anat_prep func_prep confounds denoise surface timeseries validate qc)
DATASET_POST_STAGES=(group_qc)
ALL_STAGES=("${DATASET_PRE_STAGES[@]}" "${SUBJECT_STAGES[@]}" "${DATASET_POST_STAGES[@]}")
# 'validate_group' is the dataset-level half of 'validate' (10_validate.sh --group)
SUMMARY_COLUMNS=("${DATASET_PRE_STAGES[@]}" "${SUBJECT_STAGES[@]}" validate_group group_qc)
HEAVY_STAGES=(anat_recon anat_prep func_prep denoise surface)
LICENSED_STAGES=(anat_recon anat_prep func_prep surface)
RESOURCE_STAGES=(surface timeseries validate)

# Test seam: tests point this to a directory of fake stage scripts.
STAGE_DIR="${FMRIPROC_STAGE_DIR:-$REPO_DIR/stages}"

CLI_CONF=""
CLI_SUBJECTS=""
CLI_SUBJECTS_FILE=""
LIST_ONLY=no
WORKER_SUB=""
SELECTED_STAGES=()
SUBJECTS=()
MISSING_SUBJECTS=()
SELECT_ERROR=""
PREFLIGHT_FAILED=no
FINALIZED=no
STATUS_FILE=""

# ----------------------------- command line ---------------------------------

usage() {
    cat <<'EOF'
usage: run_pipeline.sh -c <dataset.conf> [options]

  -c, --config FILE        dataset configuration (or export FMRIPROC_CONFIG)
  -s, --subjects "A B"     subjects to process, with or without the sub- prefix
                           (space or comma separated, option may be repeated)
      --subjects-file F    file with one subject id per line ('#' starts a comment)
      --stages "a b ..."   stages to run (default: STAGES of the configuration)
  -j, --jobs N             subjects processed in parallel (overrides N_JOBS)
      --force              rerun the selected stages even when they are up to date
      --dry-run            log the commands instead of executing them
      --list-subjects      print the subjects that would be processed and exit
  -h, --help

stages, always executed in this order:
  dataset level   ingest fetch
  per subject     anat_recon anat_prep func_prep confounds denoise surface
                  timeseries validate qc
  dataset level   validate --group (whenever 'validate' is selected)
                  group_qc         (whenever 'group_qc' or 'qc' is selected)

'fetch' runs automatically when templates/atlases needed by the surface,
time-series or validation stage are missing. Any configuration value can be
overridden from the environment:   NTHREADS=8 SURFACE=no run_pipeline.sh -c my.conf

exit status: 0 = every selected stage succeeded, 1 = at least one failure
(details in logs/status_<timestamp>.tsv), 2 = usage error.
EOF
}

usage_error() {
    echo "run_pipeline.sh: $*" >&2
    echo "try 'run_pipeline.sh --help'" >&2
    exit 2
}

parse_cli() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -c|--config|-s|--subjects|--subjects-file|--stages|-j|--jobs|--worker)
                [[ $# -ge 2 ]] || usage_error "option $1 needs a value" ;;&
            -c|--config)      CLI_CONF="$2"; shift 2 ;;
            -s|--subjects)    CLI_SUBJECTS+=" $2"; shift 2 ;;
            --subjects-file)  CLI_SUBJECTS_FILE="$2"; shift 2 ;;
            --stages)         export STAGES="$2"; shift 2 ;;
            -j|--jobs)        export N_JOBS="$2"; shift 2 ;;
            --force)          export FORCE=yes; shift ;;
            --dry-run)        export DRY_RUN=yes; shift ;;
            --list-subjects)  LIST_ONLY=yes; shift ;;
            --worker)         WORKER_SUB="$2"; shift 2 ;;
            -h|--help)        usage; exit 0 ;;
            *)                usage_error "unknown argument: $1" ;;
        esac
    done
    if [[ -n "$CLI_SUBJECTS_FILE" && ! -f "$CLI_SUBJECTS_FILE" ]]; then
        usage_error "subjects file not found: $CLI_SUBJECTS_FILE"
    fi
}

# ----------------------------- stage bookkeeping ----------------------------

stage_script() {
    case "$1" in
        ingest)     echo "00_ingest.sh" ;;
        fetch)      echo "fetch_resources.sh" ;;
        anat_recon) echo "01_anat_recon.sh" ;;
        anat_prep)  echo "02_anat_prep.sh" ;;
        func_prep)  echo "03_func_prep.sh" ;;
        confounds)  echo "04_confounds.sh" ;;
        denoise)    echo "05_denoise.sh" ;;
        surface)    echo "06_surface.sh" ;;
        timeseries) echo "07_timeseries.sh" ;;
        qc)         echo "08_qc.sh" ;;
        group_qc)   echo "09_group_qc.sh" ;;
        validate)   echo "10_validate.sh" ;;
        *)          die "stage_script: unknown stage $1" ;;
    esac
}

parse_stages() {
    local s k known
    read -r -a SELECTED_STAGES <<< "${STAGES//,/ }"
    [[ ${#SELECTED_STAGES[@]} -gt 0 ]] || die "STAGES is empty (valid: ${ALL_STAGES[*]})"
    for s in "${SELECTED_STAGES[@]}"; do
        known=no
        for k in "${ALL_STAGES[@]}"; do
            if [[ "$s" == "$k" ]]; then known=yes; fi
        done
        [[ "$known" == yes ]] || die "unknown stage '$s' (valid: ${ALL_STAGES[*]})"
    done
}

stage_selected() {
    local s
    for s in "${SELECTED_STAGES[@]}"; do
        if [[ "$s" == "$1" ]]; then return 0; fi
    done
    return 1
}

any_stage_selected() {
    local s
    for s in "$@"; do
        if stage_selected "$s"; then return 0; fi
    done
    return 1
}

# any_stage_active STAGE... : like any_stage_selected, but 'surface' only
# counts when SURFACE=yes (otherwise the worker skips it).
any_stage_active() {
    local s
    for s in "$@"; do
        if [[ "$s" == surface ]] && ! is_yes "$SURFACE"; then continue; fi
        if stage_selected "$s"; then return 0; fi
    done
    return 1
}

# record_status FILE SUBJECT STAGE STATUS SECONDS LOG [NOTE]
record_status() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$2" "$3" "$4" "$5" "$6" "${7:--}" >> "$1"
}

# Prefix the output of a stage so that parallel subjects stay readable; the
# console transcript is also kept in the pipeline log.
tag_lines() {
    local tag="$1" line
    while IFS= read -r line || [[ -n "$line" ]]; do
        printf '[%s] %s\n' "$tag" "$line"
    done | tee -a "$FP_PIPELINE_LOG" >&2
}

# run_stage_script TAG SCRIPT [ARGS...] : returns the exit status of the script.
# FP_LOGFILE is cleared for the child, otherwise lines logged before its own
# fp_set_log would reach the pipeline log twice (directly and through tag_lines).
run_stage_script() {
    local tag="$1" script="$2" rc=0
    shift 2
    if [[ ! -f "$script" ]]; then
        log ERROR "stage script not found: $script"
        return 127
    fi
    set +e
    env FP_LOGFILE= bash "$script" -c "$FMRIPROC_CONFIG" "$@" < /dev/null 2>&1 | tag_lines "$tag"
    rc=${PIPESTATUS[0]}
    set -e
    return "$rc"
}

init_run() {
    local stamp n=1
    stamp="$(date +%Y%m%d_%H%M%S)"
    FP_RUN_ID="$stamp"
    # two runs started within the same second must not share their records
    while [[ -e "$LOG_DIR/pipeline_${FP_RUN_ID}.log" || -e "$LOG_DIR/status_${FP_RUN_ID}.tsv" ]]; do
        FP_RUN_ID="${stamp}_$n"
        n=$(( n + 1 ))
    done
    FP_PIPELINE_LOG="$LOG_DIR/pipeline_${FP_RUN_ID}.log"
    FP_STATUS_DIR="$LOG_DIR/status_${FP_RUN_ID}.d"
    FP_LOGFILE="$FP_PIPELINE_LOG"
    mkdir -p "$FP_STATUS_DIR"
    export FP_RUN_ID FP_PIPELINE_LOG FP_STATUS_DIR FP_LOGFILE
}

# ----------------------------- worker (one subject) -------------------------

worker_main() {
    local sub="$1" fragment stage script logpath t0 dt rc blocked=""
    fragment="$FP_STATUS_DIR/$sub.tsv"
    : > "$fragment"
    for stage in "${SUBJECT_STAGES[@]}"; do
        stage_selected "$stage" || continue
        script="$(stage_script "$stage")"
        logpath="$LOG_DIR/$sub/${script%.sh}.log"
        if [[ -n "$blocked" ]]; then
            record_status "$fragment" "$sub" "$stage" skipped 0 - "upstream stage $blocked failed"
            continue
        fi
        if [[ "$stage" == surface ]] && ! is_yes "$SURFACE"; then
            log INFO "[$sub] surface: skipped (SURFACE=$SURFACE)"
            record_status "$fragment" "$sub" "$stage" skipped 0 - "SURFACE is not yes"
            continue
        fi
        log INFO "[$sub] $stage: start"
        t0=$SECONDS
        rc=0
        run_stage_script "$sub" "$STAGE_DIR/$script" "$sub" || rc=$?
        dt=$(( SECONDS - t0 ))
        if [[ $rc -eq 0 ]]; then
            log OK "[$sub] $stage: ok (${dt} s)"
            record_status "$fragment" "$sub" "$stage" ok "$dt" "$logpath"
        else
            log ERROR "[$sub] $stage: FAILED (exit $rc, ${dt} s) - see $logpath; later stages of $sub are skipped"
            record_status "$fragment" "$sub" "$stage" failed "$dt" "$logpath" "exit status $rc"
            blocked="$stage"
        fi
    done
}

# ----------------------------- subject selection ----------------------------

# ids from -s / --subjects-file (or SUBJECT_LIST when neither is given),
# normalised to sub-<id>, one per line
requested_subjects() {
    {
        if [[ -n "${CLI_SUBJECTS// /}" ]]; then
            printf '%s\n' "$CLI_SUBJECTS"
        fi
        if [[ -n "$CLI_SUBJECTS_FILE" ]]; then
            cat "$CLI_SUBJECTS_FILE"
        fi
        if [[ -z "${CLI_SUBJECTS// /}" && -z "$CLI_SUBJECTS_FILE" && -n "${SUBJECT_LIST:-}" ]]; then
            cat "$SUBJECT_LIST"
        fi
    } | sed -e 's/#.*$//' | tr ',\r\t' '   ' | tr -s ' ' '\n' |
        awk 'NF { id = $1; if (id !~ /^sub-/) id = "sub-" id; print id }' | sort -u
}

# select_subjects : SUBJECTS = manifest subjects (optionally restricted),
# MISSING_SUBJECTS = requested ids that the manifest does not contain.
# Returns 1 with SELECT_ERROR set when no list can be built. It never calls
# die, because a silent failure inside $(...) would select every subject.
select_subjects() {
    local available requested
    SUBJECTS=()
    MISSING_SUBJECTS=()
    SELECT_ERROR=""
    if ! requested="$(requested_subjects)"; then
        SELECT_ERROR="the subject list cannot be read"
        return 1
    fi
    if [[ ! -s "$MANIFEST" ]]; then
        if is_yes "$DRY_RUN" && [[ -n "$requested" ]]; then
            log WARN "no manifest yet (dry run): using the requested subjects as they are"
            mapfile -t SUBJECTS <<< "$requested"
            return 0
        fi
        SELECT_ERROR="manifest not found: $MANIFEST (run the ingest stage first: --stages ingest)"
        return 1
    fi
    if ! available="$(manifest_subjects)" || [[ -z "$available" ]]; then
        SELECT_ERROR="the manifest lists no runs: $MANIFEST"
        return 1
    fi
    if [[ -z "$requested" ]]; then
        mapfile -t SUBJECTS <<< "$available"
        return 0
    fi
    mapfile -t SUBJECTS < <(awk 'NR == FNR { want[$1] = 1; next } ($1 in want)' \
        <(printf '%s\n' "$requested") <(printf '%s\n' "$available"))
    mapfile -t MISSING_SUBJECTS < <(awk 'NR == FNR { have[$1] = 1; next } !($1 in have)' \
        <(printf '%s\n' "$available") <(printf '%s\n' "$requested"))
}

# ----------------------------- pre-flight -----------------------------------

# preflight_check CMD... : run a check that die()s on failure in a subshell, so
# that every problem is reported before the run is refused.
preflight_check() {
    if ( "$@" ); then
        return 0
    fi
    PREFLIGHT_FAILED=yes
}

required_tools() {
    local tools=()
    if stage_selected anat_recon && [[ "$ANAT_MODE" == freesurfer ]]; then
        tools+=(recon-all)
    fi
    if stage_selected anat_prep; then
        tools+=(fslmaths fslstats mri_binarize antsRegistration antsApplyTransforms N4BiasFieldCorrection
                CreateJacobianDeterminantImage)
        if [[ "$ANAT_MODE" == synth ]]; then
            tools+=(mri_synthstrip mri_synthseg)
        else
            tools+=(mri_convert mris_euler_number)
        fi
    fi
    if stage_selected func_prep; then
        # wb_command converts the FLIRT/mcflirt matrices to ITK even without the surface branch
        tools+=(mcflirt flirt fslmaths fslmerge fslsplit 3dTshift 3dDespike 3dToutcount 3dTstat 3dcalc
                antsApplyTransforms N4BiasFieldCorrection wb_command)
        if [[ "$EPI_MASK_METHOD" == synthstrip ]]; then tools+=(mri_synthstrip); fi
        if [[ "$ANAT_MODE" == freesurfer ]]; then
            tools+=(bbregister mri_coreg lta_diff lta_convert)
        else
            tools+=(rmsdiff)
        fi
    fi
    if stage_selected denoise; then
        tools+=(3dTproject 3dTstat 3dcalc)
    fi
    if is_yes "$SURFACE" && any_stage_selected surface timeseries; then
        tools+=(wb_command)
    fi
    if [[ ${#tools[@]} -gt 0 ]]; then
        printf '%s\n' "${tools[@]}" | sort -u
    fi
}

check_python() {
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "PYTHON_BIN not found: $PYTHON_BIN"
    "$PYTHON_BIN" - <<'PY' || die "python environment incomplete: $PYTHON_BIN (see the message above)"
import importlib
import sys

missing = []
for name in ("numpy", "scipy", "pandas", "nibabel", "nilearn", "sklearn", "matplotlib", "jinja2", "fmriproc"):
    try:
        importlib.import_module(name)
    except Exception as exc:  # report every broken import, not only the first
        missing.append(f"{name} ({exc.__class__.__name__}: {exc})")
if missing:
    print("python modules that cannot be imported: " + "; ".join(missing), file=sys.stderr)
    sys.exit(1)
PY
}

check_license() {
    local lic="${FS_LICENSE:-}"
    [[ -n "$lic" && -s "$lic" ]] || die "FreeSurfer license not found (FS_LICENSE='$lic'); mount it at /opt/freesurfer/license.txt (docker/run_docker.ps1 -License ...)"
}

# license_needed : the stages die without a license exactly in these cases
# (01/02/03 in freesurfer mode; 03 with the SynthStrip EPI mask; 06). In synth
# mode with EPI_MASK_METHOD=automask stage 02 only warns, so the pre-flight
# must not be stricter than the stages.
license_needed() {
    if [[ "$ANAT_MODE" == freesurfer ]] && any_stage_active "${LICENSED_STAGES[@]}"; then
        return 0
    fi
    if stage_selected func_prep && [[ "$EPI_MASK_METHOD" == synthstrip ]]; then
        return 0
    fi
    return 1
}

# Windows bind mounts (9p/drvfs) are slow and cannot hold the symlinks that
# recon-all creates; named Docker volumes are the supported setup.
warn_windows_mount() {
    local dir="$1" what="$2" fstype
    fstype="$(stat -f -c %T "$dir" 2>/dev/null || echo unknown)"
    case "$fstype" in
        9p|v9fs|fuse*|drvfs|cifs|smb*|ntfs*|virtiofs)
            log WARN "$what ($dir) is on a '$fstype' file system (host bind mount): slow, and FreeSurfer symlinks fail there. Use a Docker named volume (docker/run_docker.ps1 does this)." ;;
    esac
}

warn_resources() {
    local cores mem_kb mem_gb
    cores="$(fp_cpu_limit)"
    if [[ "$cores" -gt 0 && $(( N_JOBS * NTHREADS )) -gt "$cores" ]]; then
        log WARN "N_JOBS x NTHREADS = $(( N_JOBS * NTHREADS )) exceeds the $cores available cores"
    fi
    mem_kb="$(fp_mem_limit_kb)"
    mem_gb=$(( ${mem_kb:-0} / 1024 / 1024 ))
    if [[ "$N_JOBS" -gt 1 && "$MIN_MEM_GB" =~ ^[0-9]+$ && "$mem_gb" -lt $(( N_JOBS * MIN_MEM_GB )) ]]; then
        log WARN "${mem_gb} GB RAM for N_JOBS=$N_JOBS (rule of thumb: $MIN_MEM_GB GB per concurrent subject)"
    fi
}

preflight() {
    local tools=()
    mapfile -t tools < <(required_tools)
    if [[ ${#tools[@]} -gt 0 ]]; then
        preflight_check require_cmds "${tools[@]}"
    fi
    preflight_check check_python
    if license_needed; then
        preflight_check check_license
    fi
    if any_stage_active "${HEAVY_STAGES[@]}"; then
        preflight_check fp_check_mem
        preflight_check fp_check_resources
    fi
    warn_resources
    warn_windows_mount "$WORK_DIR" "work directory"
    if [[ "$ANAT_MODE" == freesurfer ]]; then
        warn_windows_mount "$FS_DIR" "FreeSurfer SUBJECTS_DIR"
    fi
    if [[ "$PREFLIGHT_FAILED" == yes ]]; then
        if is_yes "$DRY_RUN"; then
            log WARN "pre-flight problems ignored because DRY_RUN=yes"
        else
            die "pre-flight checks failed (messages above); nothing was processed"
        fi
    fi
}

# A run executes a frozen copy of the code (logs/code_<run>): bash reads scripts
# incrementally, so editing the repository during a long run would otherwise
# corrupt the stage that is executing and mix code versions between stages. The
# copy, its md5 manifest and the git revision document what produced the results.
freeze_code() {
    local dest="$LOG_DIR/code_${FP_RUN_ID}" item rel
    if is_yes "$DRY_RUN" || ! is_yes "${FREEZE_CODE:-yes}"; then
        return 0
    fi
    mkdir -p "$dest"
    cp "$SELF" "$dest/run_pipeline.sh"
    for item in lib py config parcellations; do
        if [[ -d "$REPO_DIR/$item" ]]; then
            cp -R "$REPO_DIR/$item" "$dest/$item"
        fi
    done
    cp -R "$STAGE_DIR" "$dest/stages"          # the stage directory in use (tests: fake stages)
    find "$dest" -name __pycache__ -type d -prune -exec rm -rf {} +
    (cd "$dest" && find . -type f ! -name code_manifest.md5 -print0 | sort -z | xargs -0 md5sum) > "$dest/code_manifest.md5"
    if command -v git >/dev/null 2>&1 && git -C "$REPO_DIR" rev-parse HEAD > "$dest/git_revision.txt" 2>/dev/null; then
        git -C "$REPO_DIR" status --porcelain --untracked-files=no 2>/dev/null | sed 's/^/modified: /' >> "$dest/git_revision.txt" || true
    else
        rm -f "$dest/git_revision.txt"
    fi
    # the dataset conf is frozen as well: same relative path inside the
    # repository, otherwise a copy next to the code
    case "$FMRIPROC_CONFIG" in
        "$REPO_DIR"/*) rel="${FMRIPROC_CONFIG#"$REPO_DIR"/}" ;;
        *) rel="config/external/$(basename "$FMRIPROC_CONFIG")"
           mkdir -p "$dest/config/external"
           cp "$FMRIPROC_CONFIG" "$dest/$rel" ;;
    esac
    FMRIPROC_CONFIG="$dest/$rel"
    SELF="$dest/run_pipeline.sh"
    STAGE_DIR="$dest/stages"
    export FMRIPROC_CONFIG FMRIPROC_STAGE_DIR="$STAGE_DIR"
    log INFO "code frozen for this run: $dest"
}

write_tool_versions() {
    local out="$LOG_DIR/tool_versions.json" tmp="$LOG_DIR/.tool_versions.$$.json"
    if is_yes "$DRY_RUN"; then
        return 0
    fi
    if fp_tool_versions "$tmp" 2>/dev/null; then
        mv -f "$tmp" "$out"
    else
        rm -f "$tmp"
        log WARN "could not record the tool versions"
    fi
}

# ----------------------------- dataset-level stages -------------------------

# run_dataset_stage FRAGMENT LABEL STAGE [ARGS...] : run a dataset-level stage
# script and record it under LABEL.
run_dataset_stage() {
    local fragment="$1" label="$2" stage="$3" script t0 dt rc=0
    shift 3
    script="$(stage_script "$stage")"
    log INFO "[dataset] $label: start"
    t0=$SECONDS
    run_stage_script "$label" "$STAGE_DIR/$script" "$@" || rc=$?
    dt=$(( SECONDS - t0 ))
    if [[ $rc -eq 0 ]]; then
        log OK "[dataset] $label: ok (${dt} s)"
        record_status "$fragment" - "$label" ok "$dt" "$FP_PIPELINE_LOG"
    else
        log ERROR "[dataset] $label: FAILED (exit $rc, ${dt} s)"
        record_status "$fragment" - "$label" failed "$dt" "$FP_PIPELINE_LOG" "exit status $rc"
    fi
    return "$rc"
}

resources_needed() {
    any_stage_active "${RESOURCE_STAGES[@]}" || return 1
    if is_yes "$SURFACE" || [[ -n "${ATLASES// /}" ]]; then
        return 0
    fi
    return 1
}

# Explicit 'fetch', or automatic when fetch_resources.sh --check reports a gap.
# A failed download is recorded but does not stop the anatomical/functional
# stages, which need no downloaded resource.
run_fetch() {
    local fragment="$1" rc=0
    if ! stage_selected fetch; then
        resources_needed || return 0
        run_stage_script fetch "$STAGE_DIR/fetch_resources.sh" --check || rc=$?
        if [[ $rc -eq 0 ]]; then
            log INFO "[dataset] templates and atlases are complete"
            return 0
        fi
        log WARN "[dataset] templates/atlases are incomplete: running fetch_resources.sh"
    fi
    rc=0
    run_dataset_stage "$fragment" fetch fetch || rc=$?
    if [[ $rc -ne 0 ]]; then
        log WARN "fetch failed: the surface/timeseries/validate stages will fail until $RESOURCE_DIR is complete (network access is required once)"
    fi
    return 0
}

# ----------------------------- subject dispatch -----------------------------

have_gnu_parallel() {
    command -v parallel >/dev/null 2>&1 && parallel --version 2>/dev/null | grep -q 'GNU parallel'
}

# Workers report through their status fragment, so the exit status of the
# runner is informative only: one failed subject must never stop the others
# (v1 died here: 'set -e' + the non-zero exit status of GNU parallel).
dispatch_subjects() {
    local rc=0 sub list="$FP_STATUS_DIR/_subjects.txt"
    if [[ "$N_JOBS" -le 1 || ${#SUBJECTS[@]} -le 1 ]]; then
        for sub in "${SUBJECTS[@]}"; do
            rc=0
            bash "$SELF" --worker "$sub" < /dev/null || rc=$?
            if [[ $rc -ne 0 ]]; then
                log WARN "worker of $sub ended with status $rc"
            fi
        done
        return 0
    fi
    printf '%s\n' "${SUBJECTS[@]}" > "$list"
    if have_gnu_parallel; then
        log INFO "dispatching ${#SUBJECTS[@]} subjects, $N_JOBS at a time (GNU parallel)"
        # -q protects a repository path with spaces; without a replacement string
        # GNU parallel appends each input line as the last argument.
        parallel --will-cite -q --jobs "$N_JOBS" --line-buffer bash "$SELF" --worker :::: "$list" || rc=$?
    else
        log INFO "dispatching ${#SUBJECTS[@]} subjects, $N_JOBS at a time (xargs -P)"
        xargs -P "$N_JOBS" -I '{}' bash "$SELF" --worker '{}' < "$list" || rc=$?
    fi
    if [[ $rc -ne 0 ]]; then
        log WARN "parallel runner returned $rc; results are taken from the per-subject status records"
    fi
    return 0
}

# A worker that was killed (out of memory, docker stop) leaves no record.
fill_missing_rows() {
    local sub stage fragment gap
    for sub in "${SUBJECTS[@]}"; do
        fragment="$FP_STATUS_DIR/$sub.tsv"
        [[ -f "$fragment" ]] || : > "$fragment"
        gap=no
        for stage in "${SUBJECT_STAGES[@]}"; do
            stage_selected "$stage" || continue
            if awk -F'\t' -v s="$stage" '$2 == s { found = 1 } END { exit !found }' "$fragment"; then
                continue
            fi
            if [[ "$gap" == no ]]; then
                record_status "$fragment" "$sub" "$stage" failed 0 - "worker ended without a status record (killed / out of memory?)"
                gap=yes
            else
                record_status "$fragment" "$sub" "$stage" skipped 0 - "worker ended early"
            fi
        done
    done
    return 0
}

# run_subjects FRAGMENT : select the subjects and process them; selection
# problems are recorded in FRAGMENT instead of stopping the run.
run_subjects() {
    local fragment="$1" sub
    if ! select_subjects; then
        if is_yes "$DRY_RUN"; then
            log WARN "dry run: $SELECT_ERROR - per-subject stages are not listed"
            return 0
        fi
        log ERROR "$SELECT_ERROR"
        record_status "$fragment" - subjects failed 0 "$MANIFEST" "$SELECT_ERROR"
        return 0
    fi
    for sub in "${MISSING_SUBJECTS[@]}"; do
        log ERROR "requested subject is not in the manifest: $sub"
        record_status "$fragment" "$sub" ingest failed 0 "$MANIFEST" "not in the manifest"
    done
    if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
        log ERROR "no subject to process"
        record_status "$fragment" - subjects failed 0 "$MANIFEST" "no subject selected"
        return 0
    fi
    log INFO "${#SUBJECTS[@]} subject(s): ${SUBJECTS[*]}"
    dispatch_subjects
    fill_missing_rows
    return 0
}

# ----------------------------- status & summary -----------------------------

merge_status() {
    local tmp="$STATUS_FILE.tmp.$$" sub
    {
        printf 'subject\tstage\tstatus\tseconds\tlog\tnote\n'
        if [[ -f "$FP_STATUS_DIR/_pre.tsv" ]]; then cat "$FP_STATUS_DIR/_pre.tsv"; fi
        for sub in "${SUBJECTS[@]}"; do
            if [[ -f "$FP_STATUS_DIR/$sub.tsv" ]]; then cat "$FP_STATUS_DIR/$sub.tsv"; fi
        done
        if [[ -f "$FP_STATUS_DIR/_post.tsv" ]]; then cat "$FP_STATUS_DIR/_post.tsv"; fi
    } > "$tmp"
    mv -f "$tmp" "$STATUS_FILE"
}

print_summary() {
    echo
    echo "===== fMRI_Processing $PIPELINE_VERSION - run $FP_RUN_ID ====="
    awk -v stage_list="${SUMMARY_COLUMNS[*]}" '
        BEGIN { FS = "\t"; ns = split(stage_list, names, " "); w = 9 }
        NR == 1 { next }
        {
            if (!($1 in seen)) { seen[$1] = 1; order[++n] = $1 }
            status[$1, $2] = $3
            total[$1] += $4
            used[$2] = 1
            if (length($1) > w) w = length($1)
        }
        END {
            line = sprintf("%-" w "s", "subject")
            for (i = 1; i <= ns; i++) {
                if (!(names[i] in used)) continue
                cw[i] = length(names[i]) > 7 ? length(names[i]) : 7
                line = line sprintf("  %-" cw[i] "s", names[i])
            }
            print line "  seconds"
            for (k = 1; k <= n; k++) {
                key = order[k]
                line = sprintf("%-" w "s", key == "-" ? "(dataset)" : key)
                for (i = 1; i <= ns; i++) {
                    if (!(names[i] in used)) continue
                    s = status[key, names[i]]
                    if (s == "") s = "."
                    if (s == "failed") s = "FAILED"
                    line = line sprintf("  %-" cw[i] "s", s)
                }
                print line "  " total[key]
            }
        }' "$STATUS_FILE"
    echo
    awk -F'\t' 'NR > 1 && $3 == "failed" { printf "FAILED  %s  %s  (%s)  log: %s\n", $1, $2, $6, $5 }' "$STATUS_FILE"
    echo "status table: $STATUS_FILE"
    echo "pipeline log: $FP_PIPELINE_LOG"
}

# finalize : merge the records, print the summary, return 1 when anything failed
finalize() {
    local n_failed
    merge_status
    FINALIZED=yes
    print_summary | tee -a "$FP_PIPELINE_LOG"
    rm -rf "$FP_STATUS_DIR"
    n_failed="$(awk -F'\t' 'NR > 1 && $3 == "failed"' "$STATUS_FILE" | wc -l | tr -d ' ')"
    if [[ "$n_failed" -gt 0 ]]; then
        log ERROR "$n_failed stage(s) failed"
        return 1
    fi
    log OK "all selected stages finished"
    return 0
}

on_exit() {
    if [[ "$FINALIZED" != yes && -n "$STATUS_FILE" && -d "${FP_STATUS_DIR:-}" ]]; then
        merge_status || true
        rm -rf "$FP_STATUS_DIR" || true
        echo "run ended early; partial status table: $STATUS_FILE" >&2
    fi
}

# ----------------------------- orchestrator ---------------------------------

list_subjects() {
    select_subjects || die "$SELECT_ERROR"
    if [[ ${#MISSING_SUBJECTS[@]} -gt 0 ]]; then
        log WARN "requested but not in the manifest: ${MISSING_SUBJECTS[*]}"
    fi
    if [[ ${#SUBJECTS[@]} -gt 0 ]]; then
        printf '%s\n' "${SUBJECTS[@]}"
    fi
}

orchestrate() {
    local pre post rc=0
    init_run
    STATUS_FILE="$LOG_DIR/status_${FP_RUN_ID}.tsv"
    pre="$FP_STATUS_DIR/_pre.tsv"
    post="$FP_STATUS_DIR/_post.tsv"
    trap on_exit EXIT
    trap 'log WARN "interrupted"; exit 130' INT TERM

    log INFO "===== fMRI_Processing $PIPELINE_VERSION | run $FP_RUN_ID ====="
    log INFO "config=$FMRIPROC_CONFIG"
    log INFO "INPUT_DIR=$INPUT_DIR OUT_DIR=$OUT_DIR FS_DIR=$FS_DIR RESOURCE_DIR=$RESOURCE_DIR"
    log INFO "stages=${SELECTED_STAGES[*]}"
    log INFO "ANAT_MODE=$ANAT_MODE SURFACE=$SURFACE MNI_RES=$MNI_RES strategies='$DENOISE_STRATEGIES' atlases='$ATLASES'"
    log INFO "N_JOBS=$N_JOBS NTHREADS=$NTHREADS FORCE=$FORCE DRY_RUN=$DRY_RUN SKIP_EXISTING=$SKIP_EXISTING"

    preflight
    freeze_code
    write_tool_versions

    if stage_selected ingest; then
        rc=0
        run_dataset_stage "$pre" ingest ingest || rc=$?
        if [[ $rc -ne 0 ]]; then
            log ERROR "ingest failed: no subject can be processed"
            finalize || true
            exit 1
        fi
    fi
    run_fetch "$pre"

    if any_stage_selected "${SUBJECT_STAGES[@]}"; then
        run_subjects "$pre"
    fi

    if stage_selected validate; then
        run_dataset_stage "$post" validate_group validate --group || true
    fi
    if any_stage_selected group_qc qc; then
        run_dataset_stage "$post" group_qc group_qc || true
    fi

    rc=0
    finalize || rc=$?
    exit "$rc"
}

main() {
    parse_cli "$@"
    if [[ -n "$CLI_CONF" ]]; then
        export FMRIPROC_CONFIG="$CLI_CONF"
    fi
    if [[ -z "${FMRIPROC_CONFIG:-}" ]]; then
        usage_error "no configuration given (-c dataset.conf or FMRIPROC_CONFIG)"
    fi
    # The command-line values were exported above, so they win over the
    # configuration files (environment > dataset conf > default.conf).
    fp_init
    parse_stages
    if [[ -n "${SUBJECT_LIST:-}" && ! -f "$SUBJECT_LIST" ]]; then
        die "SUBJECT_LIST file not found: $SUBJECT_LIST"
    fi

    if [[ -n "$WORKER_SUB" ]]; then
        if [[ -z "${FP_STATUS_DIR:-}" || ! -d "${FP_STATUS_DIR:-}" ]]; then
            init_run
        fi
        worker_main "$WORKER_SUB"
        exit 0
    fi
    if [[ "$LIST_ONLY" == yes ]]; then
        list_subjects
        exit 0
    fi
    orchestrate
}

main "$@"
