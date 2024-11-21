#!/bin/bash

set -euo pipefail

# =============================================================================
# Script: recon_all_parallel.sh
# Description:
#   Parallel processing script for FreeSurfer recon-all analysis
#   Default mode generates brain.mgz through specified processing steps
#   Supports multiple processing modes and configurations
# =============================================================================

# ---------------------------- Configuration ----------------------------------

INPUT_DIR=""
OUTPUT_DIR=""
CORES_PER_SUBJECT=2
LOG_DIR=""
FILE_PATTERN="*T1w.nii*"
SKIP_EXISTING=false
PROCESSING_MODE="default"  # default, minimal, complete, custom
EXPERT_FILE=""
MAX_TIME=""
DRY_RUN=false
VERBOSE=false

# ---------------------------- Usage Function --------------------------------

usage() {
    echo "FreeSurfer Parallel Processing Script"
    echo "Usage: $0 [options]" >&2
    echo ""
    echo "Required Options:"
    echo "  -i    Path to the input data directory"
    echo "  -o    Path to the output directory"
    echo ""
    echo "Optional Arguments:"
    echo "  -c    Number of CPU cores per subject (default: $CORES_PER_SUBJECT)"
    echo "  -p    Input file pattern (default: $FILE_PATTERN)"
    echo "  -m    Processing mode (default: $PROCESSING_MODE)"
    echo "        Options: default (brain.mgz generation)"
    echo "                 minimal (autorecon1 only)"
    echo "                 complete (full recon-all)"
    echo "                 custom (specify steps)"
    echo "  -l    Log directory (default: output_dir/logs)"
    echo "  -e    Expert options file"
    echo "  -t    Maximum time per subject (format: HH:MM:SS)"
    echo "  -x    Skip existing subjects"
    echo "  -d    Dry run (show commands without executing)"
    echo "  -v    Verbose output"
    echo "  -h    Display this help message"
    exit 1
}

# ---------------------------- Argument Parsing ------------------------------

while getopts ":i:o:c:p:m:l:e:t:xdvh" opt; do
    case ${opt} in
        i) INPUT_DIR="$OPTARG";;
        o) OUTPUT_DIR="$OPTARG";;
        c) CORES_PER_SUBJECT="$OPTARG";;
        p) FILE_PATTERN="$OPTARG";;
        m) PROCESSING_MODE="$OPTARG";;
        l) LOG_DIR="$OPTARG";;
        e) EXPERT_FILE="$OPTARG";;
        t) MAX_TIME="$OPTARG";;
        x) SKIP_EXISTING=true;;
        d) DRY_RUN=true;;
        v) VERBOSE=true;;
        h) usage;;
        \?) echo "Invalid Option: -$OPTARG" >&2; usage;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage;;
    esac
done

# ---------------------------- Functions -------------------------------------

log() {
    local level="$1"
    shift
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$level] $*" >&2
    if [ ! -z "$LOG_DIR" ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$level] $*" >> "$LOG_DIR/recon_all_$(date +'%Y%m%d').log"
    fi
}

check_prerequisites() {
    local missing_prereqs=()
    
    # Check for FreeSurfer
    if ! command -v recon-all >/dev/null 2>&1; then
        missing_prereqs+=("FreeSurfer (recon-all)")
    fi
    
    # Check for GNU Parallel
    if ! command -v parallel >/dev/null 2>&1; then
        missing_prereqs+=("GNU Parallel")
    fi
    
    if [ ${#missing_prereqs[@]} -ne 0 ]; then
        log "ERROR" "Missing prerequisites: ${missing_prereqs[*]}"
        exit 1
    fi
}

validate_input() {
    if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
        log "ERROR" "Input and Output directories must be specified."
        usage
    fi

    if [ ! -d "$INPUT_DIR" ]; then
        log "ERROR" "Input directory '$INPUT_DIR' does not exist."
        exit 1
    fi

    if ! [[ "$CORES_PER_SUBJECT" =~ ^[1-9][0-9]*$ ]]; then
        log "ERROR" "Cores per subject must be a positive integer."
        exit 1
    fi

    if [ ! -z "$EXPERT_FILE" ] && [ ! -f "$EXPERT_FILE" ]; then
        log "ERROR" "Expert options file '$EXPERT_FILE' does not exist."
        exit 1
    fi
}

setup_directories() {
    mkdir -p "$OUTPUT_DIR"
    
    if [ -z "$LOG_DIR" ]; then
        LOG_DIR="$OUTPUT_DIR/logs"
    fi
    mkdir -p "$LOG_DIR"
}

run_recon_all_step() {
    local subject_id="$1"
    local output_dir="$2"
    local step="$3"
    local cores="$4"
    local log_file="$5"
    
    local cmd="recon-all -s \"$subject_id\" $step -openmp \"$cores\" -sd \"$output_dir\""
    
    if [ "$VERBOSE" = true ]; then
        log "COMMAND" "$cmd"
    fi
    
    if ! eval "$cmd" >> "$log_file" 2>&1; then
        log "ERROR" "Failed during step: $step"
        return 1
    fi
    return 0
}

process_subject() {
    local subject_id="$1"
    local anat_file="$2"
    local session="${3:-}"
    local subj_sess_id
    
    if [ ! -z "$session" ]; then
        subj_sess_id="${subject_id}_${session}"
    else
        subj_sess_id="${subject_id}"
    fi
    
    local subject_log_file="$LOG_DIR/${subj_sess_id}_$(date +'%Y%m%d').log"
    
    if [ "$SKIP_EXISTING" = true ] && [ -d "$OUTPUT_DIR/$subj_sess_id" ]; then
        if [ -f "$OUTPUT_DIR/$subj_sess_id/mri/brain.mgz" ]; then
            log "INFO" "Subject-Session '$subj_sess_id' already processed. Skipping..."
            return 0
        fi
    fi
    
    log "INFO" "Processing Subject: $subject_id | Session: ${session:-None}"
    log "INFO" "Anatomical File: $anat_file"
    
    if [ "$DRY_RUN" = true ]; then
        log "DRY-RUN" "Would process $subj_sess_id"
        return 0
    fi

    # Define processing steps based on mode
    local steps
    case "$PROCESSING_MODE" in
        "default")
            steps=(
                # "-i \"$anat_file\" -autorecon1"
                # "-gcareg"
                "-canorm"
                "-careg"
                "-calabel"
                "-normalization2"
            )
            ;;
        "minimal")
            steps=("-i \"$anat_file\" -autorecon1")
            ;;
        "complete")
            steps=("-i \"$anat_file\" -all")
            ;;
        "custom")
            if [ ! -z "$EXPERT_FILE" ]; then
                steps=("-i \"$anat_file\" -expert \"$EXPERT_FILE\"")
            else
                log "ERROR" "Custom mode requires expert options file"
                return 1
            fi
            ;;
        *)
            log "ERROR" "Invalid processing mode: $PROCESSING_MODE"
            return 1
            ;;
    esac
    
    # Process each step
    for step in "${steps[@]}"; do
        local cmd_with_timeout="$step"
        if [ ! -z "$MAX_TIME" ]; then
            cmd_with_timeout="timeout $MAX_TIME $step"
        fi
        
        if ! run_recon_all_step "$subj_sess_id" "$OUTPUT_DIR" "$cmd_with_timeout" "$CORES_PER_SUBJECT" "$subject_log_file"; then
            log "ERROR" "Processing failed for subject $subj_sess_id at step: $step"
            return 1
        fi
    done
    
    # Verify brain.mgz was created (for default and complete modes)
    if [[ "$PROCESSING_MODE" != "custom" ]]; then
        if [ ! -f "$OUTPUT_DIR/$subj_sess_id/mri/brain.mgz" ]; then
            log "ERROR" "brain.mgz was not generated for subject $subj_sess_id"
            return 1
        fi
    fi
    
    log "SUCCESS" "Completed processing for $subj_sess_id"
    return 0
}

# ---------------------------- Main Execution ---------------------------------

check_prerequisites
validate_input
setup_directories

# Determine available CPU resources
TOTAL_CORES=$(nproc)
log "INFO" "Total CPU cores available: $TOTAL_CORES"
MAX_JOBS=$(( TOTAL_CORES / CORES_PER_SUBJECT ))
if [ "$MAX_JOBS" -lt 1 ]; then
    MAX_JOBS=1
fi
log "INFO" "Maximum parallel jobs: $MAX_JOBS"

# Create temporary file for subject list
temp_file=$(mktemp)
trap 'rm -f "$temp_file"' EXIT

# Find and process anatomical files
find "$INPUT_DIR" -type f -name "$FILE_PATTERN" | while read -r anat_file; do
    relative_path="${anat_file#"$INPUT_DIR"/}"
    if [ "$VERBOSE" = true ]; then
        log "DEBUG" "Processing path: $relative_path"
    fi
    
    # Extract components from path
    IFS='/' read -r -a path_components <<< "$relative_path"
    
    case ${#path_components[@]} in
        3)  # sub/anat/file.nii.gz
            subject_id="${path_components[0]}"
            session=""
            ;;
        4)  # sub/ses/anat/file.nii.gz
            subject_id="${path_components[0]}"
            session="${path_components[1]}"
            ;;
        *)
            log "WARNING" "Unexpected path structure: $relative_path"
            continue
            ;;
    esac
    
    echo "$subject_id $anat_file $session" >> "$temp_file"
done

total_subjects=$(wc -l < "$temp_file")
log "INFO" "Total subjects to process: $total_subjects"

if [ "$total_subjects" -eq 0 ]; then
    log "ERROR" "No anatomical files found matching pattern: $FILE_PATTERN"
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    log "DRY-RUN" "The following subjects would be processed:"
    cat "$temp_file"
    exit 0
fi

# Process subjects in parallel
export -f process_subject
export -f run_recon_all_step
export -f log
export OUTPUT_DIR LOG_DIR PROCESSING_MODE EXPERT_FILE MAX_TIME CORES_PER_SUBJECT SKIP_EXISTING DRY_RUN VERBOSE

parallel --progress --jobs "$MAX_JOBS" --colsep ' ' process_subject {1} {2} {3} < "$temp_file"

log "INFO" "Processing completed successfully."