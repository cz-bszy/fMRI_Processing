#!/bin/bash

set -euo pipefail

# =============================================================================
# Script: process_pipeline.sh
# Description:
#   Enhanced fMRI processing pipeline with configurable FSF types
# =============================================================================

# ---------------------------- Configuration ----------------------------------

# Directory settings
INPUT_DIR="/mnt/e/ABIDE/Data/Test"          # Raw data directory
OUTPUT_DIR="/mnt/e/ABIDE/Outputs/Test1"     # Output directory
STANDARD_DIR="./standard"                   # Standard brain templates
TEMPLATE_DIR="./template"                   # FSF templates

# Processing parameters
NUM_THREADS=4                               # Number of CPU threads to use
FWHM=6.0                                    # Full Width at Half Maximum for smoothing
SIGMA=2.548                                # Sigma for smoothing (FWHM = 2.355 * SIGMA)
HIGHP=0.1                                  # High-pass filter in Hz
LOWP=0.01                                  # Low-pass filter in Hz
TR=2.0                                     # Repetition Time
TE=30                                      # Echo Time
N_VOLS=200                                 # Number of volumes

FSF_TYPES=("Retain_GRS")          # Default FSF types("NO_GRD" "Retain_GRS")
FILE_PATTERN="*T1w.nii*" 
PROCESSING_MODE="default" 

EXPERT_FILE=""                            # Expert options file for recon-all
MAX_TIME=""                               # Maximum time per subject

# Processing flags
SKIP_EXISTING=true                         # Skip if output exists
DRY_RUN=false                             # Show commands without executing
VERBOSE=true                              # Show detailed output

# Error tracking configuration
ERROR_LOG_DIR=""
ERROR_SUBJECTS_FILE=""
CURRENT_DATE=$(date +"%Y%m%d")
LOG_DIR=""                                 # Will be set to $OUTPUT_DIR/logs



# ---------------------------- Usage Function --------------------------------

usage() {
    echo "Enhanced fMRI Processing Pipeline"
    echo "Usage: $0 [options]" >&2
    echo ""
    echo "Optional Arguments:"
    echo "  -f    FSF types (comma-separated, default: NO_GRD,Retain_GRS)"
    echo "  -s    Skip existing (default: true)"
    echo "  -d    Dry run (default: false)"
    echo "  -v    Verbose output (default: true)"
    echo "  -h    Display this help message"
    exit 1
}

# ---------------------------- Parse Arguments ------------------------------

while getopts "f:sdvh" opt; do
    case ${opt} in
        f)  # Convert comma-separated string to array
            IFS=',' read -r -a FSF_TYPES <<< "$OPTARG"
            ;;
        s) SKIP_EXISTING=true ;;
        d) DRY_RUN=true ;;
        v) VERBOSE=true ;;
        h) usage ;;
        \?) echo "Invalid Option: -$OPTARG" >&2; usage ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
    esac
done

# Validate FSF types
for fsf_type in "${FSF_TYPES[@]}"; do
    if [[ ! "$fsf_type" =~ ^(NO_GRD|Retain_GRS)$ ]]; then
        log "ERROR" "Invalid FSF type: $fsf_type" 
        log "Valid types are: NO_GRD, Retain_GRS"
        exit 1
    fi
done

# Set up error tracking paths after OUTPUT_DIR is confirmed
ERROR_LOG_DIR="${OUTPUT_DIR}/error_logs"
ERROR_SUBJECTS_FILE="${ERROR_LOG_DIR}/failed_subjects.txt"
# ---------------------------- Functions ------------------------------------

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

record_error() {
    local step="$1"
    local subject_id="$2"
    local session="${3:-}"
    local error_msg="$4"
    
    mkdir -p "$ERROR_LOG_DIR"
    
    local error_detail_file="${ERROR_LOG_DIR}/errors_${CURRENT_DATE}.log"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Step ${step} - Subject ${subject_id}${session:+ Session ${session}} - ${error_msg}" >> "$error_detail_file"
    
    # Add to failed subjects list
    if ! grep -q "^${subject_id}${session:+_${session}}$" "$ERROR_SUBJECTS_FILE" 2>/dev/null; then
        echo "${subject_id}${session:+_${session}}" >> "$ERROR_SUBJECTS_FILE"
    fi
}


setup_directories() {
    local base_dir="$1"
    
    log "Setting up directories with proper permissions..."
    
    # Create all required directories
    local dirs=(
        "${base_dir}"
        "${base_dir}/error_logs"
        "${base_dir}/recon_all"
        "${base_dir}/logs"
        "${base_dir}/results"
    )
    
    for dir in "${dirs[@]}"; do
        if ! mkdir -p "$dir"; then
            log "ERROR" "Failed to create directory: $dir"
            return 1
        fi
    done
    
    # Set permissions
    if [ -w "${base_dir}" ]; then
        chmod -R 777 "${base_dir}"
    else
        log "WARNING" "Cannot set permissions. Please run:"
        log "sudo chmod -R 777 ${base_dir}"
    fi
    
    # Verify directories are writable
    for dir in "${dirs[@]}"; do
        if [ ! -w "$dir" ]; then
            log "ERROR" "Directory not writable: $dir"
            return 1
        fi
    done
    
    return 0
}


validate_parameters() {

    for dir in "$INPUT_DIR" "$STANDARD_DIR" "$TEMPLATE_DIR"; do
        if [ ! -d "$dir" ]; then
            log "ERROR" "Directory does not exist: $dir"
            return 1
        fi
    done

    # Check numeric parameters
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
    
    for param in "${num_params[@]}"; do
        local name="${param%%:*}"
        local value="${param#*:}"
        if ! [[ "$value" =~ ^[0-9]+\.?[0-9]*$ ]]; then
            log "ERROR" "Invalid $name: $value. Must be a positive number."
            return 1
        fi
    done

    return 0
}



run_step() {
    local step="$1"
    local cmd="$2"
    local subject_id="$3"
    local session="${4:-}"
    
    log "Starting Step $step for subject $subject_id${session:+ session $session}..."
    
    if [ "$DRY_RUN" = true ]; then
        log "DRY-RUN: Would execute: $cmd"
        return 0
    fi
    
    # Run command and capture output
    local temp_log=$(mktemp)
    if ! eval "$cmd" 2>&1 | tee "$temp_log"; then
        local error_msg=$(tail -n 5 "$temp_log")
        record_error "$step" "$subject_id" "$session" "$error_msg"
        rm "$temp_log"
        log "Step $step failed for subject $subject_id${session:+ session $session}"
        return 1
    fi
    rm "$temp_log"
    
    log "Step $step completed successfully for subject $subject_id${session:+ session $session}"
    return 0
}

process_subject() {
    local subject_id="$1"
    local session="${2:-}"
    
    log "Processing subject $subject_id${session:+ session $session}..."
    
    [ "$DRY_RUN" = "true" ] && DRY_RUN_FLAG="-d" || DRY_RUN_FLAG=""
    [ "$VERBOSE" = "true" ] && VERBOSE_FLAG="-v" || VERBOSE_FLAG=""
    [ "$SKIP_EXISTING" = "true" ] && SKIP_EXISTING_FLAG="-x" || SKIP_EXISTING_FLAG=""
    
    # Step 0: ReconAll Processing
    run_step "0" "bash FC_step0.sh \
        -i \"$INPUT_DIR\" \
        -o \"$OUTPUT_DIR/recon_all\" \
        -c \"$NUM_THREADS\" \
        -p \"$FILE_PATTERN\" \
        ${PROCESSING_MODE:+-m \"$PROCESSING_MODE\"} \
        -l \"$LOG_DIR\" \
        ${EXPERT_FILE:+-e \"$EXPERT_FILE\"} \
        ${MAX_TIME:+-t \"$MAX_TIME\"} \
        $SKIP_EXISTING_FLAG \
        $DRY_RUN_FLAG \
        $VERBOSE_FLAG" "$subject_id" "$session" || return 1
    
    # Step 1: Anatomical Preprocessing
    run_step "1" "bash FC_step1 \
        -i \"$INPUT_DIR\" \
        -o \"$OUTPUT_DIR\" \
        -r \"$OUTPUT_DIR/recon_all\" \
        -c \"$NUM_THREADS\" \
        -l \"$LOG_DIR\" \
        $SKIP_EXISTING_FLAG \
        $DRY_RUN_FLAG \
        $VERBOSE_FLAG" "$subject_id" "$session" || return 1
    
    
    # Step 2: Functional Preprocessing
    run_step "2" "bash FC_step2 \
        -i \"$INPUT_DIR\" \
        -o \"$OUTPUT_DIR\" \
        -n \"$NUM_THREADS\" \
        -w \"$FWHM\" \
        -g \"$SIGMA\" \
        -h \"$HIGHP\" \
        -l \"$LOWP\" \
        -d \"$LOG_DIR\" \
        $SKIP_EXISTING_FLAG \
        $DRY_RUN_FLAG \
        $VERBOSE_FLAG" "$subject_id" "$session" || return 1
    
    # Step 3: Registration
    run_step "3" "bash FC_step3 \
        -i \"$INPUT_DIR\" \
        -o \"$OUTPUT_DIR\" \
        -s \"$STANDARD_DIR\" \
        -n \"$NUM_THREADS\" \
        -l \"$LOG_DIR\" \
        $SKIP_EXISTING_FLAG \
        $DRY_RUN_FLAG \
        $VERBOSE_FLAG" "$subject_id" "$session" || return 1

    # Step 4: Tissue Segmentation
    run_step "4" "bash FC_step4 \
        -i \"$INPUT_DIR\" \
        -o \"$OUTPUT_DIR\" \
        -s \"$STANDARD_DIR\" \
        -n \"$NUM_THREADS\" \
        -g \"$SIGMA\" \
        -l \"$LOG_DIR\" \
        $SKIP_EXISTING_FLAG \
        $DRY_RUN_FLAG \
        $VERBOSE_FLAG" "$subject_id" "$session" || return 1
    
    for fsf_type in "${FSF_TYPES[@]}"; do
        run_step "5-${fsf_type}" "bash FC_step5 \
            -i \"$INPUT_DIR\" \
            -o \"$OUTPUT_DIR\" \
            -t \"$TEMPLATE_DIR\" \
            -r \"$TR\" \
            -e \"$TE\" \
            -v \"$N_VOLS\" \
            -f \"$fsf_type\" \
            $SKIP_EXISTING_FLAG \
            $DRY_RUN_FLAG \
            $VERBOSE_FLAG" "$subject_id" "$session" || return 1
    done

    for fsf_type in "${FSF_TYPES[@]}"; do
        local step_name="6-${fsf_type}"
        run_step "$step_name" "bash FC_step6 \
            -i \"$INPUT_DIR\" \
            -o \"$OUTPUT_DIR\" \
            -f \"$fsf_type\" \
            -l \"$LOG_DIR\" \
            $SKIP_EXISTING_FLAG \
            $DRY_RUN_FLAG \
            $VERBOSE_FLAG" "$subject_id" "$session" || return 1
    done


    log "All steps completed for subject $subject_id${session:+ session $session}"
    return 0
}

# ---------------------------- Main Pipeline --------------------------------
main() {
    # Create and verify output directory structure
    mkdir -p "$OUTPUT_DIR" "$ERROR_LOG_DIR"
    
    # Initialize error log file
    : > "$ERROR_SUBJECTS_FILE"

    if [ -z "$LOG_DIR" ]; then
        LOG_DIR="$OUTPUT_DIR/logs"
    fi
    mkdir -p "$LOG_DIR"
    
    if ! validate_parameters; then
        exit 1
    fi
    # Verify directory permissions
    if [ ! -w "$OUTPUT_DIR" ] || [ ! -w "$ERROR_LOG_DIR" ]; then
        log "ERROR: Output directories not writable"
        log "Please run: sudo chmod -R 777 $OUTPUT_DIR"
        exit 1
    fi
    
    # Create temporary file for subject list
    temp_file=$(mktemp)
    trap 'rm -f "$temp_file"' EXIT
    
    # Find subjects and sessions
    log "Scanning input directory for subjects..."
    while IFS= read -r dir; do
        subject_dir=$(basename "$dir")
        
        if [[ -d "$dir/func" ]]; then
            echo "$subject_dir \"\"" >> "$temp_file"
        else
            while IFS= read -r session_dir; do
                session=$(basename "$session_dir")
                if [[ -d "$session_dir/func" ]]; then
                    echo "$subject_dir $session" >> "$temp_file"
                fi
            done < <(find "$dir" -mindepth 1 -maxdepth 1 -type d)
        fi
    done < <(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d)
    
    total_subjects=$(wc -l < "$temp_file")
    log "Found $total_subjects subjects/sessions to process"
    
    if [ "$total_subjects" -eq 0 ]; then
        log "ERROR: No subjects found"
        exit 1
    fi
    
    # Process each subject
    while IFS= read -r line; do
        read -r subject_id session <<< "$line"
        process_subject "$subject_id" "$session"
    done < "$temp_file"
    
    # Report results
    local error_count=0
    if [ -f "$ERROR_SUBJECTS_FILE" ]; then
        error_count=$(wc -l < "$ERROR_SUBJECTS_FILE")
    fi
    
    if [ "$error_count" -gt 0 ]; then
        log "Processing completed with errors."
        log "Failed subjects ($error_count):"
        cat "$ERROR_SUBJECTS_FILE"
        log "See detailed error logs in: $ERROR_LOG_DIR"
        return 1
    else
        log "All processing steps completed successfully"
        return 0
    fi
}

# ---------------------------- Execute Main --------------------------------

main