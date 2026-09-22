#!/bin/bash
# =============================================================================
# lib/common.sh - shared helpers for every stage script (source, do not execute)
# API documented in docs/DESIGN.md section 8.
# =============================================================================

PIPELINE_VERSION="2.1.0"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FP_LOGFILE="${FP_LOGFILE:-}"
FP_STAGE_HASH=""
FP_ARGS=()

# ----------------------------- logging --------------------------------------

log() {
    local level="$1"; shift
    local line="[$(date +'%Y-%m-%d %H:%M:%S')] [$level] $*"
    if [[ "$level" == DEBUG ]] && ! is_yes "${VERBOSE:-no}"; then
        return 0
    fi
    echo "$line" >&2
    if [[ -n "$FP_LOGFILE" ]]; then
        echo "$line" >> "$FP_LOGFILE"
    fi
}

die() {
    log ERROR "$@"
    exit 1
}

is_yes() {
    case "${1,,}" in
        yes|true|1|on) return 0 ;;
        *) return 1 ;;
    esac
}

# run CMD ARGS... : log the command line, run it with output going to the stage
# log. Callers that need the command's stdout must not use run().
run() {
    local printable
    printf -v printable '%q ' "$@"
    log INFO "+ ${printable% }"
    if is_yes "${DRY_RUN:-no}"; then
        return 0
    fi
    local rc=0
    if [[ -n "$FP_LOGFILE" ]]; then
        "$@" >> "$FP_LOGFILE" 2>&1 || rc=$?
    else
        "$@" || rc=$?
    fi
    if [[ $rc -ne 0 ]]; then
        log ERROR "command failed (exit $rc): $1"
        if [[ -n "$FP_LOGFILE" ]]; then
            tail -n 15 "$FP_LOGFILE" >&2 || true
        fi
    fi
    return $rc
}

require_cmds() {
    local cmd missing=()
    for cmd in "$@"; do
        command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        die "missing commands: ${missing[*]} (run inside the container with 'bash -lc')"
    fi
}

require_files() {
    local f
    for f in "$@"; do
        [[ -s "$f" ]] || die "required file missing or empty: $f"
    done
}

# ----------------------------- configuration --------------------------------

fp_usage() {
    echo "usage: $(basename "$0") [-c dataset.conf] [args...]   (or export FMRIPROC_CONFIG)" >&2
}

# fp_init "$@" : load configuration, derive directories, export the environment.
fp_init() {
    local conf="${FMRIPROC_CONFIG:-}"
    FP_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -c) [[ $# -ge 2 ]] || { fp_usage; exit 2; }
                conf="$2"; shift 2 ;;
            -h|--help) fp_usage; exit 0 ;;
            *) FP_ARGS+=("$1"); shift ;;
        esac
    done

    # Dataset conf first, defaults afterwards: both only fill unset variables,
    # so precedence is environment > dataset conf > default.conf.
    if [[ -n "$conf" ]]; then
        [[ -f "$conf" ]] || die "config file not found: $conf"
        # shellcheck disable=SC1090
        source "$conf"
        FMRIPROC_CONFIG="$(cd "$(dirname "$conf")" && pwd)/$(basename "$conf")"
        export FMRIPROC_CONFIG
    fi
    # shellcheck disable=SC1091
    source "$REPO_DIR/config/default.conf"

    [[ -n "$OUT_DIR" ]] || die "OUT_DIR is not set (dataset conf or environment)"
    mkdir -p "$OUT_DIR"
    OUT_DIR="$(cd "$OUT_DIR" && pwd)"
    : "${FS_DIR:=$OUT_DIR/freesurfer}"
    : "${RESOURCE_DIR:=$OUT_DIR/resources}"
    [[ -n "$FS_DIR" ]] || FS_DIR="$OUT_DIR/freesurfer"
    [[ -n "$RESOURCE_DIR" ]] || RESOURCE_DIR="$OUT_DIR/resources"
    RAW_DIR="$OUT_DIR/rawdata"
    WORK_DIR="$OUT_DIR/work"
    DERIV_DIR="$OUT_DIR/derivatives"
    LOG_DIR="$OUT_DIR/logs"
    MANIFEST="$RAW_DIR/manifest.tsv"
    mkdir -p "$FS_DIR" "$RESOURCE_DIR" "$RAW_DIR" "$WORK_DIR" "$DERIV_DIR" "$LOG_DIR"

    [[ "$NTHREADS" =~ ^[1-9][0-9]*$ ]] || die "NTHREADS must be a positive integer: $NTHREADS"
    [[ "$N_JOBS" =~ ^[1-9][0-9]*$ ]] || die "N_JOBS must be a positive integer: $N_JOBS"
    [[ -z "$CPU_BUDGET" || "$CPU_BUDGET" =~ ^[1-9][0-9]*$ ]] || die "CPU_BUDGET must be a positive integer"
    [[ -z "$MEMORY_BUDGET_GB" || "$MEMORY_BUDGET_GB" =~ ^[1-9][0-9]*$ ]] || die "MEMORY_BUDGET_GB must be a positive integer (GiB)"
    case "$ANAT_MODE" in freesurfer|synth) ;; *) die "ANAT_MODE must be freesurfer or synth: $ANAT_MODE" ;; esac
    case "$STC" in auto|require|off) ;; *) die "STC must be auto, require or off: $STC" ;; esac
    case "$FILTER_MODE" in bandpass|highpass|none) ;; *) die "FILTER_MODE must be bandpass, highpass or none" ;; esac
    case "$MNI_RES" in 1|2|3|4) ;; *) die "MNI_RES must be 1, 2, 3 or 4: $MNI_RES" ;; esac
    if is_yes "$SURFACE"; then
        [[ "$ANAT_MODE" == freesurfer ]] || die "SURFACE=yes needs ANAT_MODE=freesurfer"
        [[ "$MNI_RES" == 2 ]] || die "SURFACE=yes needs MNI_RES=2 (CIFTI subcortex grid)"
    fi

    export SUBJECTS_DIR="$FS_DIR"
    export OMP_NUM_THREADS="$NTHREADS"
    export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$NTHREADS"
    export OPENBLAS_NUM_THREADS="$NTHREADS" MKL_NUM_THREADS="$NTHREADS"
    export FSLOUTPUTTYPE=NIFTI_GZ
    export AFNI_NIFTI_TYPE_WARN=NO AFNI_NO_OBLIQUE_WARNING=YES AFNI_COMPRESSOR=NONE
    export TEMPLATEFLOW_HOME="$RESOURCE_DIR/templateflow"
    export MPLBACKEND=Agg
    export PYTHONPATH="$REPO_DIR/py${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONDONTWRITEBYTECODE=1
    if [[ -z "${FS_LICENSE:-}" && -f "${FREESURFER_HOME:-/opt/freesurfer}/license.txt" ]]; then
        export FS_LICENSE="${FREESURFER_HOME:-/opt/freesurfer}/license.txt"
    fi
    export OUT_DIR FS_DIR RESOURCE_DIR RAW_DIR WORK_DIR DERIV_DIR LOG_DIR MANIFEST REPO_DIR PIPELINE_VERSION
}

# fp_set_log SUB STAGE : route log()/run() output to logs/SUB/STAGE.log
fp_set_log() {
    local sub="$1" stage="$2"
    mkdir -p "$LOG_DIR/$sub"
    FP_LOGFILE="$LOG_DIR/$sub/$stage.log"
    export FP_LOGFILE
    log INFO "===== $stage | $sub | pipeline $PIPELINE_VERSION | $(date) ====="
}

# Effective allocation, not the host's advertised capacity. cgroup v2 is used
# by recent Docker; v1 and Slurm limits cover older container/HPC installations.
# Optional paths permit a known-answer fixture without changing host settings.
fp_cpu_limit() {
    local root="${1:-/sys/fs/cgroup}" limit value quota period
    # GNU nproc also consults OMP_NUM_THREADS; that is a per-worker setting,
    # not the allocation. Keep process affinity, but exclude that override.
    limit="$(unset OMP_NUM_THREADS OMP_THREAD_LIMIT; nproc 2>/dev/null || echo 1)"
    for value in "${CPU_BUDGET:-}" "${SLURM_CPUS_PER_TASK:-}"; do
        if [[ "$value" =~ ^[1-9][0-9]*$ && "$value" -lt "$limit" ]]; then limit="$value"; fi
    done
    if [[ -r "$root/cpu.max" ]]; then
        read -r quota period < "$root/cpu.max"
    elif [[ -r "$root/cpu/cpu.cfs_quota_us" && -r "$root/cpu/cpu.cfs_period_us" ]]; then
        read -r quota < "$root/cpu/cpu.cfs_quota_us"
        read -r period < "$root/cpu/cpu.cfs_period_us"
    else
        quota=max; period=0
    fi
    if [[ "$quota" =~ ^[1-9][0-9]*$ && "$period" =~ ^[1-9][0-9]*$ ]]; then
        value=$(( quota / period )); [[ "$value" -gt 0 ]] || value=1
        [[ "$value" -ge "$limit" ]] || limit="$value"
    fi
    echo "$limit"
}

fp_mem_limit_kb() {
    local root="${1:-/sys/fs/cgroup}" meminfo="${2:-/proc/meminfo}" limit value path
    limit="$(awk '/^MemTotal:/ {print $2}' "$meminfo" 2>/dev/null || true)"
    [[ "$limit" =~ ^[1-9][0-9]*$ ]] || { echo 0; return; }
    for path in "$root/memory.max" "$root/memory/memory.limit_in_bytes"; do
        [[ -r "$path" ]] || continue
        read -r value < "$path"
        if [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
            value=$(( value / 1024 ))
            [[ "$value" -ge "$limit" ]] || limit="$value"
        fi
    done
    if [[ "${SLURM_MEM_PER_NODE:-}" =~ ^[1-9][0-9]*$ ]]; then
        value=$(( SLURM_MEM_PER_NODE * 1024 ))
        [[ "$value" -ge "$limit" ]] || limit="$value"
    elif [[ "${SLURM_MEM_PER_CPU:-}" =~ ^[1-9][0-9]*$ && "${SLURM_CPUS_PER_TASK:-}" =~ ^[1-9][0-9]*$ ]]; then
        value=$(( SLURM_MEM_PER_CPU * SLURM_CPUS_PER_TASK * 1024 ))
        [[ "$value" -ge "$limit" ]] || limit="$value"
    fi
    if [[ "${MEMORY_BUDGET_GB:-}" =~ ^[1-9][0-9]*$ ]]; then
        value=$(( MEMORY_BUDGET_GB * 1024 * 1024 ))
        [[ "$value" -ge "$limit" ]] || limit="$value"
    fi
    echo "$limit"
}

# fp_check_mem : refuse heavy stages on allocations with too little RAM.
fp_check_mem() {
    local need="${MIN_MEM_GB:-0}" have_kb have_gb
    [[ "$need" =~ ^[0-9]+$ && "$need" -gt 0 ]] || return 0
    have_kb="$(fp_mem_limit_kb)"
    have_gb=$(( have_kb / 1024 / 1024 ))
    if [[ "$have_gb" -lt "$need" ]]; then
        die "allocation has ${have_gb} GiB RAM, MIN_MEM_GB=$need; check Docker/cgroup/Slurm limits and the workload budget."
    fi
}

fp_check_resources() {
    local cpus mem_kb
    cpus="$(fp_cpu_limit)"; mem_kb="$(fp_mem_limit_kb)"
    log INFO "resource budget: ${cpus} CPU(s), $(( mem_kb / 1024 / 1024 )) GiB; requested $N_JOBS subject(s) x $NTHREADS threads"
    [[ $(( N_JOBS * NTHREADS )) -le "$cpus" ]] || die "N_JOBS x NTHREADS exceeds the effective CPU allocation; lower concurrency/threads or request a larger allocation"
    if [[ "$MIN_MEM_GB" =~ ^[1-9][0-9]*$ && "$mem_kb" -lt $(( N_JOBS * MIN_MEM_GB * 1024 * 1024 )) ]]; then
        die "N_JOBS x MIN_MEM_GB exceeds the effective memory allocation; lower concurrency or request more memory"
    fi
}

fp_tmpdir() {
    mktemp -d "${TMPDIR:-/tmp}/fmriproc.XXXXXX"
}

# ----------------------------- stage markers --------------------------------

# Hash only small implementation files used by this stage. Previously changing
# a Python computation could incorrectly reuse a marker based on its shell alone.
_fp_stage_sources() {
    local stage="$1" modules="" module
    printf '%s\n' "$REPO_DIR/lib/common.sh"
    case "$stage" in
        00_ingest) modules="ingest timing utils" ;;
        02_anat_prep) modules="prep_utils anat_qc utils" ;;
        03_func_prep) modules="prep_utils timing utils" ;;
        04_confounds) modules="confounds utils" ;;
        05_denoise) modules="strategies utils" ;;
        06_surface) modules="surface_utils utils" ;;
        07_timeseries) modules="timeseries utils" ;;
        08_qc) modules="qc_metrics validate report plots utils" ;;
        10_validate) modules="validate utils" ;;
    esac
    for module in $modules; do printf '%s\n' "$REPO_DIR/py/fmriproc/$module.py"; done
}

_fp_marker() {   # STAGE SUB [RUN]
    local stage="$1" sub="$2" run="${3:-}"
    echo "$WORK_DIR/$sub/.done/${stage}${run:+__$run}.hash"
}

# stage_should_run STAGE SUB [RUN] [--dep MARKERNAME]... -- VAR...
#   returns 0 when the stage must run, 1 when it can be skipped.
#   MARKERNAME is "<stage>" or "<stage>__<run>" of an upstream stage; when the
#   upstream stage is redone its marker changes and this stage is redone too.
stage_should_run() {
    local stage="$1" sub="$2" run=""
    shift 2
    if [[ $# -gt 0 && "$1" != --* ]]; then
        run="$1"; shift
    fi
    local deps=() vars=() marker text v dep source_file
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dep) deps+=("$2"); shift 2 ;;
            --) shift; vars=("$@"); break ;;
            *) die "stage_should_run: unexpected argument $1" ;;
        esac
    done
    text="version=$PIPELINE_VERSION"$'\n'"script=$(md5sum "$0" | cut -d' ' -f1)"
    while IFS= read -r source_file; do
        text+=$'\n'"source:$source_file=$(md5sum "$source_file" | cut -d' ' -f1)"
    done < <(_fp_stage_sources "$stage")
    for v in "${vars[@]}"; do
        text+=$'\n'"$v=${!v-}"
    done
    for dep in "${deps[@]}"; do
        text+=$'\n'"dep:$dep=$(cat "$WORK_DIR/$sub/.done/$dep.hash" 2>/dev/null || echo missing)"
    done
    FP_STAGE_HASH="$(printf '%s' "$text" | md5sum | cut -d' ' -f1)"
    marker="$(_fp_marker "$stage" "$sub" "$run")"
    if is_yes "${FORCE:-no}" || ! is_yes "${SKIP_EXISTING:-yes}"; then
        return 0
    fi
    if [[ -f "$marker" && "$(head -n 1 "$marker")" == "$FP_STAGE_HASH" ]]; then
        log INFO "skip $stage ${run:-$sub}: up to date"
        return 1
    fi
    return 0
}

stage_mark_done() {   # STAGE SUB [RUN]
    local marker
    marker="$(_fp_marker "$@")"
    is_yes "${DRY_RUN:-no}" && return 0
    mkdir -p "$(dirname "$marker")"
    printf '%s\n%s\n' "$FP_STAGE_HASH" "$(date +%s.%N)" > "$marker"
    log OK "done ${1} ${3:-$2}"
}

# ----------------------------- manifest & naming ----------------------------

# manifest columns: subject session task run group bold t1w run_label
manifest_runs() {
    local sub="$1"
    require_files "$MANIFEST"
    awk -F'\t' -v s="$sub" 'NR > 1 && $1 == s' "$MANIFEST"
}

manifest_subjects() {
    require_files "$MANIFEST"
    awk -F'\t' 'NR > 1 {print $1}' "$MANIFEST" | sort -u
}

subject_t1w() {
    manifest_runs "$1" | awk -F'\t' '$7 != "-" && $7 != "" {print $7; exit}'
}

anat_dir()  { echo "$DERIV_DIR/$1/anat"; }
func_dir()  { echo "$DERIV_DIR/$1/func"; }
fig_dir()   { echo "$DERIV_DIR/$1/figures"; }
work_anat() { echo "$WORK_DIR/$1/anat"; }
work_func() { echo "$WORK_DIR/$1/func/$2"; }

# sidecar of a BOLD file (same stem, .json)
bold_json() {
    local bold="$1"
    echo "${bold%.nii*}.json"
}

# ----------------------------- small utilities ------------------------------

pyrun() {
    local module="$1"; shift
    run "$PYTHON_BIN" -m "fmriproc.$module" "$@"
}

# json_get FILE KEY [DEFAULT] : scalar value (true/false printed as yes/no)
json_get() {
    "$PYTHON_BIN" - "$1" "$2" "${3-}" <<'PY'
import json, sys
path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(path, encoding="utf-8-sig") as fh:
        value = json.load(fh).get(key)
except (OSError, ValueError):
    value = None
if value is None or isinstance(value, (list, dict)):
    print(default)
elif isinstance(value, bool):
    print("yes" if value else "no")
else:
    print(value)
PY
}

# json_has_list FILE KEY : exit 0 when KEY holds a non-empty list
json_has_list() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8-sig") as fh:
        value = json.load(fh).get(sys.argv[2])
except (OSError, ValueError):
    value = None
sys.exit(0 if isinstance(value, list) and len(value) > 0 else 1)
PY
}

nvols()      { fslnvols "$1"; }
img_tr()     { fslval "$1" pixdim4 | tr -d ' '; }
mask_count() { fslstats "$1" -V | awk '{print $1}'; }

# template_path KIND : brain|head|mask (1 mm)  brain_res|mask_res ($MNI_RES mm)
template_path() {
    local kind="$1" std="$FSLDIR/data/standard" res="${MNI_RES:-2}" out
    case "$kind" in
        brain) echo "$std/MNI152_T1_1mm_brain.nii.gz" ;;
        head)  echo "$std/MNI152_T1_1mm.nii.gz" ;;
        mask)  echo "$std/MNI152_T1_1mm_brain_mask.nii.gz" ;;
        brain_res|mask_res)
            if [[ "$res" == 1 || "$res" == 2 ]]; then
                if [[ "$kind" == brain_res ]]; then
                    echo "$std/MNI152_T1_${res}mm_brain.nii.gz"
                else
                    echo "$std/MNI152_T1_${res}mm_brain_mask.nii.gz"
                fi
                return 0
            fi
            mkdir -p "$RESOURCE_DIR/templates"
            out="$RESOURCE_DIR/templates/MNI152_T1_${res}mm_${kind%_res}.nii.gz"
            if [[ ! -s "$out" ]]; then
                local tmp="$RESOURCE_DIR/templates/.tmp.$$.${kind}.nii.gz" src interp=trilinear
                src="$(template_path "${kind%_res}")"
                [[ "$kind" == mask_res ]] && interp=nearestneighbour
                flirt -in "$src" -ref "$src" -applyisoxfm "$res" -interp "$interp" -out "$tmp" >/dev/null
                [[ "$kind" == mask_res ]] && fslmaths "$tmp" -bin "$tmp" -odt char
                mv -f "$tmp" "$out"
            fi
            echo "$out" ;;
        *) die "template_path: unknown kind $kind" ;;
    esac
}

# fp_tool_versions FILE : write a JSON with the versions of the external tools
fp_tool_versions() {
    local out="$1"
    {
        echo "{"
        echo "  \"pipeline\": \"$PIPELINE_VERSION\","
        echo "  \"fsl\": \"$(cat "$FSLDIR/etc/fslversion" 2>/dev/null || echo unknown)\","
        echo "  \"afni\": \"$(afni -ver 2>/dev/null | head -n 1 | tr -d '"' || echo unknown)\","
        echo "  \"freesurfer\": \"$(cat "$FREESURFER_HOME/build-stamp.txt" 2>/dev/null || echo unknown)\","
        echo "  \"ants\": \"$(antsRegistration --version 2>/dev/null | head -n 1 | tr -d '"' || echo unknown)\","
        echo "  \"workbench\": \"$(wb_command -version 2>/dev/null | awk '/^Version/ {print $2}' || echo unknown)\","
        echo "  \"python\": \"$("$PYTHON_BIN" --version 2>&1)\""
        echo "}"
    } > "$out"
}
