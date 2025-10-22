#!/bin/bash

set -euo pipefail

# Resolve repository locations so default paths follow this checkout.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# =============================================================================
# Script: main.sh
# Description:
#   Modular neuroimaging processing pipeline orchestrator.
#   Initially configured for fMRI processing and designed to be readily
#   extended to additional modalities (e.g., sMRI, PET) via the pipeline
#   registration helpers defined below.
# =============================================================================

# ---------------------------- Configuration ----------------------------------

# Directory settings
DATASET_LABEL="${DATASET_LABEL:-after}"                         # Default dataset selector
DEFAULT_INPUT_DIR="${PROJECT_ROOT}/data"
if [ ! -d "$DEFAULT_INPUT_DIR" ]; then
    DEFAULT_INPUT_DIR="${PROJECT_ROOT}/Outputs_${DATASET_LABEL}"
    if [ ! -d "$DEFAULT_INPUT_DIR" ] && [ -d "${PROJECT_ROOT}/Outputs" ]; then
        DEFAULT_INPUT_DIR="${PROJECT_ROOT}/Outputs"
    fi
fi
INPUT_DIR="${INPUT_DIR:-${DEFAULT_INPUT_DIR}}"                  # BIDS dataset root
OUTPUT_DIR="${OUTPUT_DIR:-${INPUT_DIR}/processing}"             # Pipeline outputs
STANDARD_DIR="${STANDARD_DIR:-${SCRIPT_DIR}/standard}"          # Standard brain templates
TISSUES_DIR="${TISSUES_DIR:-${SCRIPT_DIR}/tissuepriors}"        # Tissue priors
TEMPLATE_DIR="${TEMPLATE_DIR:-${SCRIPT_DIR}/template}"          # FSF templates
RECONALL_DIR="${RECONALL_DIR:-${OUTPUT_DIR}/recon_all}"         # FreeSurfer recon-all outputs

# Step script locations
FC_STEP0="${SCRIPT_DIR}/FC_step0"
FC_STEP1="${SCRIPT_DIR}/FC_step1"
FC_STEP2="${SCRIPT_DIR}/FC_step2"
FC_STEP3="${SCRIPT_DIR}/FC_step3"
FC_STEP4="${SCRIPT_DIR}/FC_step4"
FC_STEP5="${SCRIPT_DIR}/FC_step5"
FC_STEP6="${SCRIPT_DIR}/FC_step6"
FC_STEP1B="${SCRIPT_DIR}/FC_step1b"

# Ensure recon-all destination exists before validation.
mkdir -p "$RECONALL_DIR"

# Processing parameters
NUM_THREADS=${NUM_THREADS:-4}               # Number of CPU threads to use per subject
FWHM=${FWHM:-6.0}                           # Full Width at Half Maximum for smoothing
SIGMA=${SIGMA:-2.548}                       # Sigma for smoothing (FWHM = 2.355 * SIGMA)
HIGHP=${HIGHP:-0.1}                         # High-pass filter in Hz
LOWP=${LOWP:-0.01}                          # Low-pass filter in Hz
TR=${TR:-2.0}                               # Repetition Time (seconds)
TE=${TE:-30}                                # Echo Time (milliseconds)
N_VOLS=${N_VOLS:-96}                        # Number of volumes in functional run

# Pipeline and step configuration
PIPELINES_TO_RUN=("fmri")                  # Pipelines to execute; extend with smri/pet
FSF_TYPES=("NoGRS" "Retain_GRS")          # FSF flavours for fMRI step 5
GENERAL_FLAGS=()                            # Additional flags forwarded to step scripts
FUNC_FILE_PATTERN="${FUNC_FILE_PATTERN:-*task-rest*_bold.nii*}"  # Glob(s) searched for functional runs (comma-separated)
TIMESERIES_ATLAS="${TIMESERIES_ATLAS:-ThomasYeo100}"
TIMESERIES_ATLAS_PATH="${TIMESERIES_ATLAS_PATH:-}"
T1_FILE_PATTERN="${T1_FILE_PATTERN:-*T1w*.nii*}"
T2_FILE_PATTERN="${T2_FILE_PATTERN:-*T2w*.nii*}"
STRUCT_PREPROC_ENABLED="${STRUCT_PREPROC_ENABLED:-true}"
STRUCT_PREPROC_BET_FRACTION="${STRUCT_PREPROC_BET_FRACTION:-0.25}"
TARGET_SUBJECTS="${TARGET_SUBJECTS:-sub-A00086238}"

# Processing behaviour
SKIP_EXISTING=${SKIP_EXISTING:-true}       # Skip completed steps when outputs exist
DRY_RUN=${DRY_RUN:-false}                  # Show commands without executing
VERBOSE=${VERBOSE:-false}                  # Show detailed output

# Error tracking configuration
ERROR_LOG_DIR=""                           # Set in main
ERROR_SUBJECTS_FILE=""                     # Set in main
LOG_DIR=""                                 # Defaults to $OUTPUT_DIR/logs in main
CURRENT_DATE=$(date +"%Y%m%d")

# Internal registries (populated during initialization)
PIPELINE_REGISTRY=""                       # Lines describing pipeline steps
declare -A __PIPELINE_STEP_ORDER=()         # Tracks per-pipeline step ordering during registration

# ---------------------------- Logging & Errors -------------------------------

log() {
    local level="$1"
    shift
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] [$level] $*"

    if [ "$level" = "DEBUG" ] && [ "$VERBOSE" != "true" ]; then
        return 0
    fi

    case "$level" in
        "ERROR")   echo -e "\033[31m$message\033[0m" ;;
        "WARNING") echo -e "\033[33m$message\033[0m" ;;
        "SUCCESS") echo -e "\033[32m$message\033[0m" ;;
        "INFO")    echo -e "\033[36m$message\033[0m" ;;
        "DRY-RUN") echo -e "\033[35m$message\033[0m" ;;
        *)          echo "$message" ;;
    esac

    if [ -n "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR"
        echo "$message" >> "$LOG_DIR/pipeline_${CURRENT_DATE}.log"
    fi
}

record_error() {
    local pipeline="$1"
    local step="$2"
    local subject_id="$3"
    local session="${4:-}"
    local error_msg="$5"
    local error_code="${6:-1}"

    mkdir -p "$ERROR_LOG_DIR"

    local detail_file="${ERROR_LOG_DIR}/errors_${CURRENT_DATE}.log"
    local summary_file="${ERROR_LOG_DIR}/error_summary_${CURRENT_DATE}.log"

    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    local subject_session="$subject_id${session:+_${session}}"
    local error_entry="[${timestamp}] [ERROR CODE ${error_code}]
    Subject: ${subject_session}
    Pipeline: ${pipeline}
    Step: ${step}
    Error: ${error_msg}
    ----------------------------------------"

    echo "$error_entry" >> "$detail_file"

    (
        flock -x 200
        local entry="${subject_session}|${pipeline}|${step}|${error_code}"
        if ! grep -qx "$entry" "$ERROR_SUBJECTS_FILE" 2>/dev/null; then
            echo "$entry" >> "$ERROR_SUBJECTS_FILE"
        fi
    ) 200>"${ERROR_LOG_DIR}/.lock"

    local total_errors=$(wc -l < "$ERROR_SUBJECTS_FILE" 2>/dev/null || echo 0)
    local unique_subjects=$(cut -d'|' -f1 "$ERROR_SUBJECTS_FILE" 2>/dev/null | sort -u | wc -l | tr -d ' ')

    {
        echo "Last Error: ${timestamp}"
        echo "Total Errors: ${total_errors}"
        echo "Unique Subjects with Errors: ${unique_subjects}"
    } > "$summary_file"
}

# ---------------------------- Utility Helpers -------------------------------

format_subject_session() {
    local subject_id="$1"
    local session="${2:-}"
    if [ -n "$session" ]; then
        printf '%s (session %s)' "$subject_id" "$session"
    else
        printf '%s' "$subject_id"
    fi
}

hydrate_runtime_arrays() {
    local IFS
    if [ -n "${PIPELINES_TO_RUN_STRING:-}" ]; then
        IFS=' ' read -r -a PIPELINES_TO_RUN <<< "$PIPELINES_TO_RUN_STRING"
    elif [ -z "${PIPELINES_TO_RUN+x}" ]; then
        PIPELINES_TO_RUN=()
    fi

    if [ -n "${FSF_TYPES_STRING:-}" ]; then
        IFS=' ' read -r -a FSF_TYPES <<< "$FSF_TYPES_STRING"
    elif [ -z "${FSF_TYPES+x}" ]; then
        FSF_TYPES=()
    fi

    if [ -n "${GENERAL_FLAGS_STRING:-}" ]; then
        IFS=' ' read -r -a GENERAL_FLAGS <<< "$GENERAL_FLAGS_STRING"
    elif [ -z "${GENERAL_FLAGS+x}" ]; then
        GENERAL_FLAGS=()
    fi
}

subject_output_dir() {
    local subject="$1"
    printf '%s/%s' "$OUTPUT_DIR" "$subject"
}

session_dir_name() {
    local session="$1"
    if [ -z "$session" ] || [ "$session" = "\"\"" ]; then
        printf ''
        return 0
    fi
    if [[ "$session" == ses-* ]]; then
        printf '%s' "$session"
    else
        printf 'ses-%s' "$session"
    fi
}

subject_session_dir() {
    local subject="$1"
    local session="${2:-}"
    local base_dir
    base_dir="$(subject_output_dir "$subject")"
    local session_dir
    session_dir="$(session_dir_name "$session")"
    if [ -n "$session_dir" ]; then
        printf '%s/%s' "$base_dir" "$session_dir"
    else
        printf '%s' "$base_dir"
    fi
}

subject_anat_dir() {
    local subject="$1"
    local session="${2:-}"
    printf '%s/anat' "$(subject_session_dir "$subject" "$session")"
}

subject_func_dir() {
    local subject="$1"
    local session="${2:-}"
    printf '%s/func' "$(subject_session_dir "$subject" "$session")"
}

subject_in_target_list() {
    local subject="$1"
    local list="$TARGET_SUBJECTS"

    if [ -z "$list" ] || [ "$list" = "all" ]; then
        return 0
    fi

    list="${list//,/ }"
    local token
    for token in $list; do
        if [ "$token" = "$subject" ]; then
            return 0
        fi
    done
    return 1
}

subject_struct_preproc_dir() {
    local subject="$1"
    local session="${2:-}"
    printf '%s/anat/preproc' "$(subject_session_dir "$subject" "$session")"
}

prepare_fmri_subject() {
    local subject="$1"
    local base_dir="$(subject_output_dir "$subject")"
    mkdir -p "$base_dir/anat" "$base_dir/func" "$base_dir/anat/preproc"
    chmod -R 775 "$base_dir"
}

# ---------------------------- Environment Checks ---------------------------

setup_environment() {
    local required_software=(
        "recon-all"
        "fslmaths"
        "parallel"
        "3dcalc"
        "flirt"
    )

    local software
    for software in "${required_software[@]}"; do
        if ! command -v "$software" >/dev/null 2>&1; then
            log "ERROR" "Required software not found: $software"
            return 1
        fi
    done

    if [ -z "${FREESURFER_HOME:-}" ]; then
        log "ERROR" "FREESURFER_HOME is not set"
        return 1
    fi

    if [ -z "${FSLDIR:-}" ]; then
        log "ERROR" "FSLDIR is not set"
        return 1
    fi

    if [ -z "${FS_LICENSE:-}" ]; then
        local default_license="$FREESURFER_HOME/.license"
        if [ -f "$default_license" ]; then
            export FS_LICENSE="$default_license"
        else
            local found_license
            found_license=$(find "$FREESURFER_HOME" -maxdepth 1 -type f \( -iname '*license*.txt' -o -iname 'license' -o -iname '.license' \) | head -n 1)
            if [ -n "$found_license" ]; then
                log "WARNING" "FS_LICENSE not set; using detected license file: $found_license"
                export FS_LICENSE="$found_license"
            else
                log "ERROR" "FreeSurfer license not found. Set FS_LICENSE or place a valid license in $FREESURFER_HOME/.license"
                return 1
            fi
        fi
    elif [ ! -f "$FS_LICENSE" ]; then
        log "ERROR" "FS_LICENSE is set to '$FS_LICENSE' but the file does not exist"
        return 1
    fi

    return 0
}

setup_directories() {
    local base_dir="$1"
    log "INFO" "Setting up directories with proper permissions..."

    local dirs=(
        "$base_dir"
        "$base_dir/error_logs"
        "$base_dir/logs"
        "$base_dir/results"
    )

    local dir
    for dir in "${dirs[@]}"; do
        if ! mkdir -p "$dir"; then
            log "ERROR" "Failed to create directory: $dir"
            return 1
        fi

        if [ ! -w "$dir" ]; then
            if ! chmod -R 777 "$dir" 2>/dev/null; then
                log "ERROR" "Directory not writable and permissions could not be adjusted: $dir"
                return 1
            fi
        fi
    done
    return 0
}

validate_parameters() {
    local expected_dirs=("$INPUT_DIR" "$STANDARD_DIR" "$TEMPLATE_DIR" "$TISSUES_DIR" "$RECONALL_DIR")
    local dir
    for dir in "${expected_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            log "ERROR" "Directory does not exist: $dir"
            return 1
        fi
    done

    local num_params=(
        "NUM_THREADS:$NUM_THREADS"
        "FWHM:$FWHM"
        "SIGMA:$SIGMA"
        "HIGHP:$HIGHP"
        "LOWP:$LOWP"
        "TR:$TR"
        "TE:$TE"
        "N_VOLS:$N_VOLS"
    )

    local param value name
    for param in "${num_params[@]}"; do
        name="${param%%:*}"
        value="${param#*:}"
        if ! [[ "$value" =~ ^[0-9]+\.?[0-9]*$ ]]; then
            log "ERROR" "Invalid $name: $value. Must be numeric."
            return 1
        fi
    done
    return 0
}

check_mask_validity() {
    local mask_file="$1"

    if [ ! -s "$mask_file" ]; then
        return 1
    fi

    local valid_voxels
    if ! valid_voxels=$(3dBrickStat -non-zero "$mask_file" 2>/dev/null); then
        return 1
    fi

    if [ -z "$valid_voxels" ] || [ "$valid_voxels" -eq 0 ]; then
        return 1
    fi

    return 0
}

# ---------------------------- Pipeline Registry ----------------------------

append_pipeline_registry() {
    local line="$1|$2|$3|$4|$5|$6|$7"
    if [ -n "$PIPELINE_REGISTRY" ]; then
        PIPELINE_REGISTRY+=$'\n'
    fi
    PIPELINE_REGISTRY+="$line"
}

register_shell_step() {
    local pipeline="$1"
    local step_id="$2"
    local label="$3"
    local command_builder="$4"
    local output_check="${5:-}"
    local order=$((__PIPELINE_STEP_ORDER[$pipeline]+1))
    __PIPELINE_STEP_ORDER[$pipeline]=$order

    append_pipeline_registry "$pipeline" "$order" "$step_id" "shell" "$label" "$command_builder" "$output_check"
}

register_custom_step() {
    local pipeline="$1"
    local step_id="$2"
    local label="$3"
    local runner="$4"
    local output_check="${5:-}"
    local order=$((__PIPELINE_STEP_ORDER[$pipeline]+1))
    __PIPELINE_STEP_ORDER[$pipeline]=$order

    append_pipeline_registry "$pipeline" "$order" "$step_id" "custom" "$label" "$runner" "$output_check"
}

initialize_pipelines() {
    __PIPELINE_STEP_ORDER=()
    PIPELINE_REGISTRY=""

    if [ "$STRUCT_PREPROC_ENABLED" = "true" ]; then
        register_shell_step "fmri" "structural_preproc" "T1/T2 structural preprocessing" \
            build_fmri_structural_preproc_command fmri_has_structural_preproc_output
    fi

    # fMRI pipeline definition
    register_shell_step "fmri" "recon_all" "FreeSurfer recon-all" \
        build_fmri_recon_command fmri_has_recon_output

    register_shell_step "fmri" "anatomical" "Anatomical preprocessing" \
        build_fmri_anatomical_command fmri_has_anatomical_output

    register_shell_step "fmri" "functional" "Functional preprocessing" \
        build_fmri_functional_command fmri_has_functional_output

    register_shell_step "fmri" "registration" "Registration to standard" \
        build_fmri_registration_command fmri_has_registration_output

    register_shell_step "fmri" "segmentation" "Tissue segmentation" \
        build_fmri_segmentation_command fmri_has_segmentation_output

    register_custom_step "fmri" "fsf_processing" "FSF processing" \
        execute_fmri_fsf_step

    register_shell_step "fmri" "timeseries" "Atlas time-series extraction" \
        build_fmri_timeseries_command fmri_has_timeseries_output
}

# ---------------------------- Step Builders & Checks -----------------------

fmri_has_structural_preproc_output() {
    local subject="$1"
    local session="${2:-}"

    if [ "$STRUCT_PREPROC_ENABLED" != "true" ]; then
        return 0
    fi

    local preproc_dir
    preproc_dir="$(subject_struct_preproc_dir "$subject" "$session")"

    local t1_preproc="${preproc_dir}/T1w_preproc.nii.gz"
    local brain_mask="${preproc_dir}/T1w_brain_mask.nii.gz"

    if [ ! -f "$t1_preproc" ] || [ ! -f "$brain_mask" ]; then
        return 1
    fi

    return 0
}

build_fmri_structural_preproc_command() {
    local subject="$1"
    local session="${2:-}"

    local -a cmd=(
        "$FC_STEP1B"
        -i "$INPUT_DIR"
        -o "$OUTPUT_DIR"
        -p "$T1_FILE_PATTERN"
        -t "$T2_FILE_PATTERN"
        -f "$STRUCT_PREPROC_BET_FRACTION"
        -l "$LOG_DIR"
    )

    [ "$SKIP_EXISTING" = true ] && cmd+=(-x)
    [ "$DRY_RUN" = true ]       && cmd+=(-d)
    [ "$VERBOSE" = true ]       && cmd+=(-v)

    cmd+=("$subject")
    if [ -n "$session" ] && [ "$session" != "\"\"" ]; then
        cmd+=("$session")
    fi

    local command
    printf -v command '%q ' "${cmd[@]}"
    command=${command% }
    printf '%s' "$command"
}

fmri_has_recon_output() {
    local subject="$1"
    local session="${2:-}"
    if [ "$session" = "\"\"" ]; then
        session=""
    fi

    local recon_subject="$subject"
    if [ -n "$session" ]; then
        recon_subject="${subject}_${session}"
    fi

    local brain_path
    printf -v brain_path '%s/%s/mri/brain.mgz' "$RECONALL_DIR" "$recon_subject"
    [ -f "$brain_path" ]
}

fmri_has_anatomical_output() {
    local subject="$1"
    local session="${2:-}"

    local anat_dir
    anat_dir="$(subject_anat_dir "$subject" "$session")"

    local required=(
        "$anat_dir/Stru_Brain.nii.gz"
        "$anat_dir/segment_prob_0.nii.gz"
        "$anat_dir/segment_prob_1.nii.gz"
        "$anat_dir/segment_prob_2.nii.gz"
    )

    for file in "${required[@]}"; do
        [ -f "$file" ] || return 1
    done

    return 0
}

fmri_has_functional_output() {
    local subject="$1"
    local session="${2:-}"
    [ -f "$(subject_func_dir "$subject" "$session")/example_func.nii.gz" ]
}

fmri_has_registration_output() {
    local subject="$1"
    local session="${2:-}"
    [ -f "$(subject_func_dir "$subject" "$session")/reg_dir/example_func2standard.nii.gz" ]
}

fmri_has_segmentation_output() {
    local subject="$1"
    local session="${2:-}"
    [ -f "$(subject_func_dir "$subject" "$session")/seg/wm_mask.nii.gz" ]
}

fmri_has_timeseries_output() {
    local subject="$1"
    local session="${2:-}"

    if [ "$session" = "\"\"" ]; then
        session=""
    fi

    local -a fsf_types=("${FSF_TYPES[@]}")
    if [ ${#fsf_types[@]} -eq 0 ]; then
        fsf_types=(NoGRS Retain_GRS)
    fi

    local fsf base_name ts_file
    for fsf in "${fsf_types[@]}"; do
        if [ -n "$session" ]; then
            base_name="${subject}_${session}_${fsf}"
        else
            base_name="${subject}_${fsf}"
        fi
        ts_file="$OUTPUT_DIR/results/$fsf/timeseries/${base_name}_timeseries.1D"
        if [ ! -f "$ts_file" ]; then
            return 1
        fi
    done

    return 0
}

build_fmri_recon_command() {
    local subject="$1"
    local session="${2:-}"
    local -a cmd=(
        "$FC_STEP0"
        -i "$INPUT_DIR"
        -o "$RECONALL_DIR"
        -c "$NUM_THREADS"
        -p "$T1_FILE_PATTERN"
        -l "$LOG_DIR"
        -S "$OUTPUT_DIR"
    )

    if [ ${#GENERAL_FLAGS[@]} -gt 0 ]; then
        cmd+=("${GENERAL_FLAGS[@]}")
    fi

    [ "$SKIP_EXISTING" = true ] && cmd+=(-x)
    [ "$DRY_RUN" = true ]       && cmd+=(-d)
    [ "$VERBOSE" = true ]       && cmd+=(-v)

    cmd+=("$subject")
    if [ -n "$session" ] && [ "$session" != "\"\"" ]; then
        cmd+=("$session")
    fi

    local command
    printf -v command '%q ' "${cmd[@]}"
    command=${command% }
    printf '%s' "$command"
}

build_fmri_anatomical_command() {
    local subject="$1"
    local session="${2:-}"
    local -a cmd=(
        "$FC_STEP1"
        -i "$INPUT_DIR"
        -o "$OUTPUT_DIR"
        -r "$RECONALL_DIR"
        -c "$NUM_THREADS"
        -l "$LOG_DIR"
    )

    if [ ${#GENERAL_FLAGS[@]} -gt 0 ]; then
        cmd+=("${GENERAL_FLAGS[@]}")
    fi

    [ "$SKIP_EXISTING" = true ] && cmd+=(-x)
    [ "$DRY_RUN" = true ]       && cmd+=(-d)   # use -r for FC_step2
    [ "$VERBOSE" = true ]       && cmd+=(-v)

    cmd+=("$subject")
    if [ -n "$session" ] && [ "$session" != "\"\"" ]; then
        cmd+=("$session")
    fi

    local command
    printf -v command '%q ' "${cmd[@]}"
    command=${command% }
    printf '%s' "$command"
}

build_fmri_functional_command() {
    local subject="$1"
    local session="${2:-}"
    local -a cmd=(
        "$FC_STEP2"
        -i "$INPUT_DIR"
        -o "$OUTPUT_DIR"
        -n "$NUM_THREADS"
        -w "$FWHM"
        -g "$SIGMA"
        -h "$HIGHP"
        -l "$LOWP"
        -d "$LOG_DIR"
        -p "$FUNC_FILE_PATTERN"
    )

    if [ ${#GENERAL_FLAGS[@]} -gt 0 ]; then
        cmd+=("${GENERAL_FLAGS[@]}")
    fi

    [ "$SKIP_EXISTING" = true ] && cmd+=(-x)
    [ "$DRY_RUN" = true ]       && cmd+=(-r)
    [ "$VERBOSE" = true ]       && cmd+=(-v)

    cmd+=("$subject")
    if [ -n "$session" ] && [ "$session" != "\"\"" ]; then
        cmd+=("$session")
    fi

    local command
    printf -v command '%q ' "${cmd[@]}"
    command=${command% }
    printf '%s' "$command"
}

build_fmri_registration_command() {
    local subject="$1"
    local session="${2:-}"
    local -a cmd=(
        "$FC_STEP3"
        -i "$INPUT_DIR"
        -o "$OUTPUT_DIR"
        -s "$STANDARD_DIR"
        -n "$NUM_THREADS"
        -l "$LOG_DIR"
    )

    if [ ${#GENERAL_FLAGS[@]} -gt 0 ]; then
        cmd+=("${GENERAL_FLAGS[@]}")
    fi

    [ "$SKIP_EXISTING" = true ] && cmd+=(-x)
    [ "$DRY_RUN" = true ]       && cmd+=(-d)
    [ "$VERBOSE" = true ]       && cmd+=(-v)

    cmd+=("$subject")
    if [ -n "$session" ] && [ "$session" != "\"\"" ]; then
        cmd+=("$session")
    fi

    local command
    printf -v command '%q ' "${cmd[@]}"
    command=${command% }
    printf '%s' "$command"
}

build_fmri_segmentation_command() {
    local subject="$1"
    local session="${2:-}"
    local -a cmd=(
        "$FC_STEP4"
        -i "$INPUT_DIR"
        -o "$OUTPUT_DIR"
        -s "$TISSUES_DIR"
        -n "$NUM_THREADS"
        -g "$SIGMA"
        -l "$LOG_DIR"
    )

    if [ ${#GENERAL_FLAGS[@]} -gt 0 ]; then
        cmd+=("${GENERAL_FLAGS[@]}")
    fi

    [ "$SKIP_EXISTING" = true ] && cmd+=(-x)
    [ "$DRY_RUN" = true ]       && cmd+=(-d)
    [ "$VERBOSE" = true ]       && cmd+=(-v)

    cmd+=("$subject")
    if [ -n "$session" ] && [ "$session" != "\"\"" ]; then
        cmd+=("$session")
    fi

    local command
    printf -v command '%q ' "${cmd[@]}"
    command=${command% }
    printf '%s' "$command"
}

build_fmri_timeseries_command() {
    local subject="$1"
    local session="${2:-}"
    local atlas_name="${TIMESERIES_ATLAS:-ThomasYeo100}"

    local cmd=(
        "$FC_STEP6"
        -i "$INPUT_DIR"
        -o "$OUTPUT_DIR"
        -n "$NUM_THREADS"
        -l "$LOG_DIR"
        -T "$atlas_name"
    )

    if [ -n "$TIMESERIES_ATLAS_PATH" ]; then
        cmd+=(-A "$TIMESERIES_ATLAS_PATH")
    fi

    if [ -n "${FSF_TYPES_STRING:-}" ]; then
        cmd+=(-F "$FSF_TYPES_STRING")
    fi

    [ "$SKIP_EXISTING" = true ] && cmd+=(-x)
    [ "$DRY_RUN" = true ]       && cmd+=(-d)
    [ "$VERBOSE" = true ]       && cmd+=(-v)

    if [ ${#GENERAL_FLAGS[@]} -gt 0 ]; then
        cmd+=("${GENERAL_FLAGS[@]}")
    fi

    cmd+=("$subject")
    if [ -n "$session" ] && [ "$session" != "\"\"" ]; then
        cmd+=("$session")
    fi

    local command
    printf -v command '%q ' "${cmd[@]}"
    command=${command% }
    printf '%s' "$command"
}

build_fmri_fsf_command() {
    local subject="$1"
    local session="${2:-}"
    local fsf_type="$3"
    local -a cmd=(
        "$FC_STEP5"
        -i "$INPUT_DIR"
        -o "$OUTPUT_DIR"
        -t "$TEMPLATE_DIR"
        -r "$TR"
        -e "$TE"
        -s "$N_VOLS"
        -f "$fsf_type"
        -l "$LOG_DIR"
    )

    if [ ${#GENERAL_FLAGS[@]} -gt 0 ]; then
        cmd+=("${GENERAL_FLAGS[@]}")
    fi

    [ "$SKIP_EXISTING" = true ] && cmd+=(-x)
    [ "$DRY_RUN" = true ]       && cmd+=(-d)
    [ "$VERBOSE" = true ]       && cmd+=(-v)

    cmd+=("$subject")
    if [ -n "$session" ] && [ "$session" != "\"\"" ]; then
        cmd+=("$session")
    fi

    local command
    printf -v command '%q ' "${cmd[@]}"
    command=${command% }
    printf '%s' "$command"
}

execute_fmri_fsf_step() {
    local pipeline="$1"
    local step_id="$2"
    local label="$3"
    local subject="$4"
    local session="${5:-}"

    local func_dir="$(subject_func_dir "$subject" "$session")"
    local mask
    for mask in global csf wm; do
        local mask_file="$func_dir/seg/${mask}_mask.nii.gz"
        if ! check_mask_validity "$mask_file"; then
            local msg="Invalid or empty ${mask} mask for subject $(format_subject_session "$subject" "$session")"
            log "ERROR" "[$pipeline] $msg"
            record_error "$pipeline" "$step_id" "$subject" "$session" "$msg" 1
            return 1
        fi
    done

    hydrate_runtime_arrays
    log "INFO" "[$pipeline] Starting ${label} for $(format_subject_session "$subject" "$session")"
    log "INFO" "[$pipeline] FSF types: ${FSF_TYPES[*]}"

    local fsf_type
    for fsf_type in "${FSF_TYPES[@]}"; do
        local command=$(build_fmri_fsf_command "$subject" "$session" "$fsf_type")
        local fsf_step_id="${step_id}_${fsf_type}"
        local fsf_label="${label} (${fsf_type})"
        if ! run_step "$pipeline" "$fsf_step_id" "$fsf_label" "$command" "$subject" "$session"; then
            return 1
        fi
    done

    log "SUCCESS" "[$pipeline] Completed ${label} for $(format_subject_session "$subject" "$session")"
    return 0
}

# ---------------------------- Pipeline Execution ---------------------------

run_step() {
    local pipeline="$1"
    local step="$2"
    local label="$3"
    local command="$4"
    local subject="$5"
    local session="${6:-}"

    local subject_label=$(format_subject_session "$subject" "$session")
    log "INFO" "[$pipeline] Starting ${label} for ${subject_label}"

    if [ "$DRY_RUN" = true ]; then
        log "DRY-RUN" "[$pipeline] Would execute: $command"
        return 0
    fi

    local temp_log
    temp_log=$(mktemp)
    local status=0
    local error_code=0

    if ! eval "$command" 2>&1 | tee "$temp_log"; then
        status=1
        error_code=$?
    fi

    if grep -qi "error\|exception\|failed" "$temp_log"; then
        status=1
        if [ $error_code -eq 0 ]; then
            error_code=1
        fi
    fi

    if [ $status -eq 0 ]; then
        rm -f "$temp_log"
        log "SUCCESS" "[$pipeline] ${label} completed for ${subject_label}"
        return 0
    else
        local error_msg=$(tail -n 5 "$temp_log")
        rm -f "$temp_log"
        record_error "$pipeline" "$step" "$subject" "$session" "$error_msg" "$error_code"
        log "ERROR" "[$pipeline] ${label} failed for ${subject_label}"
        return 1
    fi
}

prepare_pipeline_subject() {
    local pipeline="$1"
    local subject="$2"
    local session="${3:-}"
    case "$pipeline" in
        fmri)
            prepare_fmri_subject "$subject"
            ;;
        # Additional pipelines (smri, pet, etc.) can be initialised here.
        *)
            :
            ;;
    esac
}

run_pipeline() {
    local pipeline="$1"
    local subject="$2"
    local session="${3:-}"

    hydrate_runtime_arrays

    local subject_label=$(format_subject_session "$subject" "$session")
    log "INFO" "[$pipeline] Running pipeline for ${subject_label}"

    local steps=()
    if [ -n "$PIPELINE_REGISTRY" ]; then
        while IFS='|' read -r reg_pipeline order step_id step_type label handler output_check; do
            if [ "$reg_pipeline" = "$pipeline" ]; then
                steps+=("$order|$step_id|$step_type|$label|$handler|$output_check")
            fi
        done < <(printf '%s\n' "$PIPELINE_REGISTRY")
    fi

    if [ ${#steps[@]} -eq 0 ]; then
        log "WARNING" "[$pipeline] No steps registered; skipping pipeline for ${subject_label}"
        return 0
    fi

    prepare_pipeline_subject "$pipeline" "$subject" "$session"

    local sorted_steps=()
    mapfile -t sorted_steps < <(printf '%s\n' "${steps[@]}" | sort -t'|' -k1,1n)

    local step_line
    for step_line in "${sorted_steps[@]}"; do
        IFS='|' read -r _ step_id step_type label handler output_check <<< "$step_line"

        if [ "$SKIP_EXISTING" = true ] && [ -n "$output_check" ]; then
            if "$output_check" "$subject" "$session"; then
                log "INFO" "[$pipeline] Skipping ${label} for ${subject_label} (already complete)"
                continue
            fi
        fi

        case "$step_type" in
            shell)
                local command=$($handler "$subject" "$session")
                if [ -z "$command" ]; then
                    local msg="Command builder $handler returned an empty command"
                    log "ERROR" "[$pipeline] $msg"
                    record_error "$pipeline" "$step_id" "$subject" "$session" "$msg" 1
                    return 1
                fi
                if ! run_step "$pipeline" "$step_id" "$label" "$command" "$subject" "$session"; then
                    return 1
                fi
                ;;
            custom)
                if ! "$handler" "$pipeline" "$step_id" "$label" "$subject" "$session"; then
                    return 1
                fi
                ;;
            *)
                local msg="Unknown step type '$step_type' for pipeline $pipeline"
                log "ERROR" "[$pipeline] $msg"
                record_error "$pipeline" "$step_id" "$subject" "$session" "$msg" 1
                return 1
                ;;
        esac
    done

    log "SUCCESS" "[$pipeline] Pipeline completed for ${subject_label}"
    return 0
}

process_subject() {
    local subject="$1"
    local session="${2:-}"

    hydrate_runtime_arrays

    local subject_label=$(format_subject_session "$subject" "$session")
    log "INFO" "Processing subject ${subject_label}"

    local pipeline
    for pipeline in "${PIPELINES_TO_RUN[@]}"; do
        if ! run_pipeline "$pipeline" "$subject" "$session"; then
            log "ERROR" "Subject ${subject_label} failed in pipeline $pipeline"
            return 1
        fi
    done

    log "SUCCESS" "Subject ${subject_label} completed across pipelines"
    return 0
}

process_subject_parallel() {
    local subject="$1"
    local session="${2:-}"
    if ! process_subject "$subject" "$session"; then
        return 1
    fi
}

# ---------------------------- Subject Discovery ----------------------------

discover_subjects() {
    local output_file="$1"
    : > "$output_file"

    while IFS= read -r subject_dir; do
        local subject_id
        subject_id=$(basename "$subject_dir")
        if ! subject_in_target_list "$subject_id"; then
            continue
        fi
        if [ -d "$subject_dir/anat" ]; then
            printf '%s\t%s\n' "$subject_id" "" >> "$output_file"
        else
            while IFS= read -r session_dir; do
                local session
                session=$(basename "$session_dir")
                if [ -d "$session_dir/anat" ]; then
                    printf '%s\t%s\n' "$subject_id" "$session" >> "$output_file"
                fi
            done < <(find "$subject_dir" -mindepth 1 -maxdepth 1 -type d)
        fi
    done < <(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name "sub-*")
}

# ---------------------------- Reporting ------------------------------------

generate_report() {
    hydrate_runtime_arrays

    local report_file="$OUTPUT_DIR/processing_report_${CURRENT_DATE}.txt"
    {
        echo "===== Processing Report ====="
        echo "Date: $(date)"
        echo "Input Directory: $INPUT_DIR"
        echo "Output Directory: $OUTPUT_DIR"
        echo "Pipelines Run: ${PIPELINES_TO_RUN[*]}"
        echo "-------------------------"
        echo "Total Subjects Processed: ${total_subjects:-0}"

        if [ -f "$ERROR_SUBJECTS_FILE" ] && [ -s "$ERROR_SUBJECTS_FILE" ]; then
            echo "Failed Subjects Summary:"
            echo "-------------------------"
            local total_errors=$(wc -l < "$ERROR_SUBJECTS_FILE")
            local unique_subjects=$(cut -d'|' -f1 "$ERROR_SUBJECTS_FILE" | sort -u | wc -l)
            echo "Total Errors: $total_errors"
            echo "Unique Subjects with Errors: $unique_subjects"
            echo "Errors by Pipeline and Step:"
            cut -d'|' -f2-3 "$ERROR_SUBJECTS_FILE" | tr '|' ':' | sort | uniq -c
            echo "-------------------------"
            echo "Failed Subjects Details:"
            while IFS='|' read -r subject pipeline step code; do
                echo "  Subject: $subject"
                echo "    - Pipeline: $pipeline"
                echo "    - Step: $step"
                echo "    - Error code: $code"
            done < "$ERROR_SUBJECTS_FILE"
        else
            echo "No errors encountered during processing"
        fi

        echo "-------------------------"
        echo "FSF Types Processed: ${FSF_TYPES[*]}"
        echo "Functional File Pattern: $FUNC_FILE_PATTERN"
        echo "Processing Parameters:"
        echo "  - Threads per subject: $NUM_THREADS"
        echo "  - FWHM: $FWHM"
        echo "  - Sigma: $SIGMA"
        echo "  - TR: $TR"
        echo "  - TE: $TE"
        echo "  - Number of volumes: $N_VOLS"
        echo "  - Timeseries atlas: $TIMESERIES_ATLAS"
    } > "$report_file"

    log "INFO" "Processing report generated: $report_file"
}

# ---------------------------- Main -----------------------------------------

main() {
    if [ -z "$LOG_DIR" ]; then
        LOG_DIR="$OUTPUT_DIR/logs"
    fi
    mkdir -p "$LOG_DIR"

    if [ -z "$ERROR_LOG_DIR" ]; then
        ERROR_LOG_DIR="$OUTPUT_DIR/error_logs"
    fi
    mkdir -p "$ERROR_LOG_DIR"

    ERROR_SUBJECTS_FILE="${ERROR_LOG_DIR}/failed_subjects_${CURRENT_DATE}.txt"
    : > "$ERROR_SUBJECTS_FILE"

    hydrate_runtime_arrays
    initialize_pipelines

    local temp_dir
    temp_dir=$(mktemp -d)
    trap 'if [[ -n "${temp_dir:-}" ]]; then rm -rf "$temp_dir"; fi' EXIT

    local subjects_file="${temp_dir}/subjects_list.txt"

    log "INFO" "Validating environment and parameters..."
    if ! setup_environment || ! validate_parameters; then
        log "ERROR" "Validation failed. Check logs for details."
        exit 1
    fi

    log "INFO" "Setting up directory structure..."
    if ! setup_directories "$OUTPUT_DIR"; then
        log "ERROR" "Failed to setup directories"
        exit 1
    fi

    log "INFO" "Scanning input directory for subjects and sessions..."
    discover_subjects "$subjects_file"

    total_subjects=$(wc -l < "$subjects_file" 2>/dev/null || echo 0)
    log "INFO" "Found $total_subjects subject/session combinations to process"

    if [ "$total_subjects" -eq 0 ]; then
        log "ERROR" "No subjects found in input directory"
        exit 1
    fi

    local total_cores
    total_cores=$(nproc)
    local threads_per_subject=$NUM_THREADS
    local max_parallel_jobs=$(( total_cores / threads_per_subject ))
    if [ "$max_parallel_jobs" -lt 1 ]; then
        max_parallel_jobs=1
        log "WARNING" "Available cores ($total_cores) less than threads per subject ($threads_per_subject). Running single job."
    fi
    log "INFO" "Running with $max_parallel_jobs parallel subjects (Total cores: $total_cores, Threads per subject: $threads_per_subject)"

    PIPELINES_TO_RUN_STRING="${PIPELINES_TO_RUN[*]}"
    FSF_TYPES_STRING="${FSF_TYPES[*]}"
    GENERAL_FLAGS_STRING="${GENERAL_FLAGS[*]:-}"

    export PIPELINES_TO_RUN_STRING FSF_TYPES_STRING GENERAL_FLAGS_STRING
    export PIPELINE_REGISTRY

    export INPUT_DIR OUTPUT_DIR STANDARD_DIR TEMPLATE_DIR TISSUES_DIR RECONALL_DIR FUNC_FILE_PATTERN TARGET_SUBJECTS
    export T1_FILE_PATTERN T2_FILE_PATTERN STRUCT_PREPROC_ENABLED STRUCT_PREPROC_BET_FRACTION
    export FC_STEP0 FC_STEP1 FC_STEP2 FC_STEP3 FC_STEP4 FC_STEP5 FC_STEP6
    export NUM_THREADS FWHM SIGMA HIGHP LOWP TR TE N_VOLS
    export SKIP_EXISTING DRY_RUN VERBOSE LOG_DIR ERROR_LOG_DIR ERROR_SUBJECTS_FILE CURRENT_DATE

    export -f log record_error hydrate_runtime_arrays format_subject_session
    export -f subject_output_dir session_dir_name subject_session_dir subject_anat_dir subject_func_dir subject_in_target_list subject_struct_preproc_dir prepare_fmri_subject prepare_pipeline_subject
    export -f fmri_has_structural_preproc_output fmri_has_recon_output fmri_has_anatomical_output fmri_has_functional_output fmri_has_registration_output fmri_has_segmentation_output fmri_has_timeseries_output
    export -f build_fmri_structural_preproc_command build_fmri_recon_command build_fmri_anatomical_command build_fmri_functional_command build_fmri_registration_command build_fmri_segmentation_command build_fmri_timeseries_command build_fmri_fsf_command
    export -f execute_fmri_fsf_step run_step run_pipeline process_subject process_subject_parallel check_mask_validity

    parallel --jobs "$max_parallel_jobs" --colsep '\t' \
        process_subject_parallel {1} {2} :::: "$subjects_file"

    local failed_count=0
    if [ -s "$ERROR_SUBJECTS_FILE" ]; then
        failed_count=$(cut -d'|' -f1 "$ERROR_SUBJECTS_FILE" | sort -u | wc -l | tr -d ' ')
    fi

    log "INFO" "Generating processing report..."
    generate_report

    if [ "$failed_count" -gt 0 ]; then
        log "WARNING" "Processing completed with $failed_count subject(s) reporting errors"
        exit 1
    fi

    log "SUCCESS" "All processing completed successfully"
    exit 0
}

main "$@"
