#!/bin/bash

set -euo pipefail

# =============================================================================
# Script: process_pipeline.sh
# Description:
#   Enhanced fMRI processing pipeline with configurable FSF types and robust
#   error handling, status tracking, and reporting
# =============================================================================

# ---------------------------- Configuration ----------------------------------

# Directory settings
INPUT_DIR="/mnt/e/ABIDE/Data/Leuven_1"          # Raw data directory
OUTPUT_DIR="/mnt/e/ABIDE/Outputs/Leuven_1"     # Output directory
STANDARD_DIR="./standard"                   # Standard brain templates
TISSUES_DIR="./tissuepriors"
TEMPLATE_DIR="./template"                   # FSF templates

# Processing parameters
NUM_THREADS=6                               # Number of CPU threads to use
FWHM=6.0                                    # Full Width at Half Maximum for smoothing
SIGMA=2.548                                 # Sigma for smoothing (FWHM = 2.355 * SIGMA)
HIGHP=0.1                                   # High-pass filter in Hz
LOWP=0.01                                   # Low-pass filter in Hz
TR=1.6669999999999998                                     # Repetition Time
TE=33                                       # Echo Time
N_VOLS=250                                # Number of volumes

# FSF_TYPES=("NoGRS") "NoGRS" 
FSF_TYPES=("NoGRS" "Retain_GRS")            # Default FSF types "NoGRS" 
FILE_PATTERN="*T1w.nii*"                   # Pattern for anatomical files
PROCESSING_MODE="default"                   # Processing mode for recon-all


# Processing flags
SKIP_EXISTING=true                         # Skip if output exists
DRY_RUN=false                              # Show commands without executing
VERBOSE=false                               # Show detailed output

# Error tracking configuration
ERROR_LOG_DIR="${OUTPUT_DIR}/error_logs"   # Will be created if not exists
ERROR_SUBJECTS_FILE=""                     # Will be set in main
LOG_DIR=""                                 # Will be set to $OUTPUT_DIR/logs
CURRENT_DATE=$(date +"%Y%m%d")

# Step tracking
declare -A STEP_STATUS
STEPS=("recon-all" "anatomical" "functional" "registration" "segmentation" "nuisance" "extraction")

# ---------------------------- Functions ------------------------------------

log() {
    local level="$1"
    shift
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] [$level] $*"
    
    # Only show DEBUG messages if VERBOSE is true
    if [ "$level" = "DEBUG" ] && [ "$VERBOSE" != "true" ]; then
        return 0
    fi
    
    # Color coding for different log levels
    case "$level" in
        "ERROR")   echo -e "\033[31m$message\033[0m" ;;
        "WARNING") echo -e "\033[33m$message\033[0m" ;;
        "SUCCESS") echo -e "\033[32m$message\033[0m" ;;
        "INFO")    echo -e "\033[36m$message\033[0m" ;;
        *)         echo "$message" ;;
    esac
    
    if [ -n "$LOG_DIR" ]; then
        echo "$message" >> "$LOG_DIR/pipeline_${CURRENT_DATE}.log"
    fi
}


record_error() {
    local step="$1"
    local subject_id="$2"
    local session="${3:-}"
    local error_msg="$4"
    local error_code="${5:-1}" 

    mkdir -p "$ERROR_LOG_DIR"

    local error_detail_file="${ERROR_LOG_DIR}/errors_${CURRENT_DATE}.log"
    local error_summary_file="${ERROR_LOG_DIR}/error_summary_${CURRENT_DATE}.log"

    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    local subject_session="${subject_id}${session:+_${session}}"
    local error_entry="[${timestamp}] [ERROR CODE ${error_code}]
    Subject: ${subject_session}
    Step: ${step}
    Error: ${error_msg}
    ----------------------------------------"

    echo "$error_entry" >> "$error_detail_file"

    (
        flock -x 200
        if ! grep -q "^${subject_session}|${step}|${error_code}$" "$ERROR_SUBJECTS_FILE" 2>/dev/null; then
            echo "${subject_session}|${step}|${error_code}" >> "$ERROR_SUBJECTS_FILE"
        fi
    ) 200>"${ERROR_LOG_DIR}/.lock"

    {
        echo "Last Error: ${timestamp}"
        echo "Total Errors: $(wc -l < "$ERROR_SUBJECTS_FILE")"
        echo "Unique Subjects with Errors: $(cut -d'|' -f1 "$ERROR_SUBJECTS_FILE" | sort -u | wc -l)"
    } > "$error_summary_file"
}

setup_environment() {
    local required_software=(
        "recon-all"    # FreeSurfer
        "fslmaths"     # FSL
        "parallel"     # GNU Parallel
        "3dcalc"       # AFNI
        "flirt"        # FSL registration
    )

    for software in "${required_software[@]}"; do
        if ! command -v "$software" >/dev/null 2>&1; then
            log "ERROR" "Required software not found: $software"
            return 1
        fi
    done

    # Check environment variables
    if [ -z "${FREESURFER_HOME:-}" ]; then
        log "ERROR" "FREESURFER_HOME is not set"
        return 1
    fi

    if [ -z "${FSLDIR:-}" ]; then
        log "ERROR" "FSLDIR is not set"
        return 1
    fi

    return 0
}

setup_directories() {
    local base_dir="$1"

    log "INFO" "Setting up directories with proper permissions..."

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

        if [ ! -w "$dir" ]; then
            log "ERROR" "Directory not writable: $dir"
            if ! chmod -R 777 "$dir" 2>/dev/null; then
                log "ERROR" "Failed to set permissions for: $dir"
                return 1
            fi
        fi
    done

    return 0
}

validate_parameters() {
    # Check directories
    for dir in "$INPUT_DIR" "$STANDARD_DIR" "$TEMPLATE_DIR" "$TISSUES_DIR"; do
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

    # Validate FSF types
    for fsf_type in "${FSF_TYPES[@]}"; do
        if [[ ! "$fsf_type" =~ ^(NoGRS|Retain_GRS)$ ]]; then
            log "ERROR" "Invalid FSF type: $fsf_type"
            log "ERROR" "Valid types are: NoGRS, Retain_GRS"
            return 1
        fi
    done

    return 0
}

# ---------------------------- Processing Functions --------------------------

check_previous_steps() {
    local subject_id="$1"
    local step="$2"

    case "$step" in
        0)  # No prerequisites for step 0
            return 0
            ;;
        1)  # Check if ReconAll output exists
            if [ ! -f "$OUTPUT_DIR/recon_all/$subject_id/mri/brain.mgz" ]; then
                log "ERROR" "ReconAll output not found: $OUTPUT_DIR/recon_all/$subject_id/mri/brain.mgz"
                return 1
            fi
            ;;
        2)  # Check anatomical preprocessing
            if [ ! -f "$OUTPUT_DIR/$subject_id/anat/Stru_Brain.nii.gz" ]; then
                log "ERROR" "Anatomical preprocessing must complete before functional preprocessing"
                return 1
            fi
            ;;
        3)  # Check functional preprocessing
            if [ ! -f "$OUTPUT_DIR/$subject_id/func/example_func.nii.gz" ]; then
                log "ERROR" "Functional preprocessing must complete before registration"
                return 1
            fi
            ;;
        4)  # Check registration
            if [ ! -f "$OUTPUT_DIR/$subject_id/func/reg_dir/example_func2standard.nii.gz" ]; then
                log "ERROR" "Registration must complete before tissue segmentation"
                return 1
            fi
            ;;
        5)  # Check tissue segmentation and confirm registration
            if [ ! -f "$OUTPUT_DIR/$subject_id/func/seg/wm_mask.nii.gz" ]; then
                log "ERROR" "Tissue segmentation must complete before nuisance regression"
                return 1
            fi
            if [ ! -f "$OUTPUT_DIR/$subject_id/func/reg_dir/example_func2standard.nii.gz" ]; then
                log "ERROR" "Registration must complete before nuisance regression"
                return 1
            fi
            ;;
        6)  # Check nuisance regression output for specific FSF type
            if [ ! -f "$OUTPUT_DIR/$subject_id/func/rest_res2standard.nii.gz" ]; then
                log "ERROR" "Nuisance regression must complete before results extraction"
                return 1
            fi
            ;;
    esac
    return 0
}

run_step() {
    local step="$1"
    local cmd="$2"
    local subject_id="$3"
    local session="${4:-}"
    local status=0

    log "INFO" "Starting Step $step for subject $subject_id${session:+ session $session}..."

    if [ "$DRY_RUN" = true ]; then
        log "DRY-RUN" "Would execute: $cmd"
        return 0
    fi

    local temp_log=$(mktemp)
    local error_code=0

    if ! eval "$cmd" 2>&1 | tee "$temp_log"; then
        error_code=$?
        status=1
    fi

    if grep -qi "error\|exception\|failed" "$temp_log"; then
        status=1
        error_code=${error_code:-1}
    fi

    if [ $status -eq 0 ]; then
        rm "$temp_log"
        log "SUCCESS" "Step $step completed for subject $subject_id${session:+ session $session}"
        return 0
    else
        local error_msg=$(tail -n 5 "$temp_log")
        record_error "$step" "$subject_id" "$session" "$error_msg" "$error_code"
        rm "$temp_log"
        log "ERROR" "Step $step failed for subject $subject_id${session:+ session $session}"
        return 1
    fi
}

check_mask_validity() {
    local mask_file="$1"
    
    if [ ! -s "$mask_file" ]; then
        return 1
    fi
    
    if ! valid_voxels=$(3dBrickStat -non-zero "$mask_file" 2>/dev/null) || \
       [ -z "$valid_voxels" ] || [ "$valid_voxels" -eq 0 ]; then
        return 1
    fi
    
    return 0
}

process_subject() {
    local subject_id="$1"
    local session="${2:-}"

    log "INFO" "Processing subject $subject_id${session:+ session $session}..."

    GENERAL_FLAGS=()
    if [ "$SKIP_EXISTING" = true ]; then
        GENERAL_FLAGS+=("-x")
    fi
    if [ "$DRY_RUN" = true ]; then
        GENERAL_FLAGS+=("-d")
    fi

    # Get anatomical file path
    local anat_file
    if [ -z "$session" ] || [ "$session" = "\"\"" ]; then
        anat_file=$(find "$INPUT_DIR/$subject_id/anat" -type f -name "*T1w.nii*" | head -n 1)
    else
        anat_file=$(find "$INPUT_DIR/$subject_id/$session/anat" -type f -name "*T1w.nii*" | head -n 1)
    fi

    if [ -z "$anat_file" ]; then
        log "ERROR" "No anatomical file found for subject $subject_id${session:+ session $session}"
        return 1
    fi
    

    # Step 0: ReconAll Processing
    # log "INFO" "Starting ReconAll processing..."
    # run_step "0" "bash FC_step0 \
    #     -i \"$INPUT_DIR\" \
    #     -o \"$OUTPUT_DIR/recon_all\" \
    #     ${session:+-e \"$session\"} \
    #     -c \"$NUM_THREADS\" \
    #     -m \"default\" \
    #     ${GENERAL_FLAGS[*]} \
    #     -v" "$subject_id" "$session" || {  # Assuming FC_step0 supports -v
    #         log "ERROR" "ReconAll processing failed for subject $subject_id"
    #         return 1
    #     }

    # # Verify ReconAll output exists before continuing
    # if [ ! -f "$OUTPUT_DIR/recon_all/$subject_id/mri/brain.mgz" ]; then
    #     log "ERROR" "ReconAll output not found for subject $subject_id"
    #     return 1
    # fi
    
    reconall_dir="/mnt/e/ABIDE/Outputs/recon_all/Leuven_1"

    # Step 1: Check and run Anatomical Preprocessing if needed
    if [ ! -f "$OUTPUT_DIR/$subject_id/anat/Stru_Brain.nii.gz" ]; then
        log "INFO" "Running Step 1: Anatomical Preprocessing..."
        run_step "1" "bash FC_step1 \
            -i \"$INPUT_DIR\" \
            -o \"$OUTPUT_DIR\" \
            -r \"$reconall_dir\" \
            -c \"$NUM_THREADS\" \
            -l \"$LOG_DIR\" \
            ${GENERAL_FLAGS[*]} \
            -v" "$subject_id" "$session" || return 1
    else
        log "INFO" "Skipping Step 1: Anatomical Preprocessing (output exists)"
    fi

    # Step 2: Check and run Functional Preprocessing if needed
    if [ ! -f "$OUTPUT_DIR/$subject_id/func/example_func.nii.gz" ]; then
        log "INFO" "Running Step 2: Functional Preprocessing..."
        run_step "2" "bash FC_step2 \
            -i \"$INPUT_DIR\" \
            -o \"$OUTPUT_DIR\" \
            -n \"$NUM_THREADS\" \
            -w \"$FWHM\" \
            -g \"$SIGMA\" \
            -h \"$HIGHP\" \
            -l \"$LOWP\" \
            -d \"$LOG_DIR\" \
            ${GENERAL_FLAGS[*]} \
            -v" "$subject_id" "$session" || return 1
    else
        log "INFO" "Skipping Step 2: Functional Preprocessing (output exists)"
    fi

    # Step 3: Check and run Registration if needed
    if [ ! -f "$OUTPUT_DIR/$subject_id/func/reg_dir/example_func2standard.nii.gz" ]; then
        log "INFO" "Running Step 3: Registration..."
        run_step "3" "bash FC_step3 \
            -i \"$INPUT_DIR\" \
            -o \"$OUTPUT_DIR\" \
            -s \"$STANDARD_DIR\" \
            -n \"$NUM_THREADS\" \
            -l \"$LOG_DIR\" \
            ${GENERAL_FLAGS[*]} \
            -v" "$subject_id" "$session" || return 1
    else
        log "INFO" "Skipping Step 3: Registration (output exists)"
    fi

    # Step 4: Check and run Tissue Segmentation if needed
    if [ ! -f "$OUTPUT_DIR/$subject_id/func/seg/wm_mask.nii.gz" ]; then
        log "INFO" "Running Step 4: Tissue Segmentation..."
        run_step "4" "bash FC_step4 \
            -i \"$INPUT_DIR\" \
            -o \"$OUTPUT_DIR\" \
            -s \"$TISSUES_DIR\" \
            -n \"$NUM_THREADS\" \
            -g \"$SIGMA\" \
            -l \"$LOG_DIR\" \
            ${GENERAL_FLAGS[*]} \
            -v" "$subject_id" "$session" || return 1
    else
        log "INFO" "Skipping Step 4: Tissue Segmentation (output exists)"
    fi


    for fsf_type in "${FSF_TYPES[@]}"; do
        log "INFO" "Processing FSF type: ${fsf_type}"
        
        # Check required files before processing
        local func_dir="${OUTPUT_DIR}/${subject_id}/func"
        local nuisance_dir="${func_dir}/nuisance"
        
        # Verify tissue masks have valid data
        for mask in "global" "csf" "wm"; do
            local mask_file="${func_dir}/seg/${mask}_mask.nii.gz"
            if [ ! -f "$mask_file" ] || ! check_mask_validity "$mask_file"; then
                log "ERROR" "Invalid or empty ${mask} mask for subject $subject_id${session:+ session $session}"
                return 1
            fi
        done
        
        run_step "processing-${fsf_type}" "bash FC_step5 \
        -i \"$INPUT_DIR\" \
        -o \"$OUTPUT_DIR\" \
        -t \"$TEMPLATE_DIR\" \
        -r \"$TR\" \
        -e \"$TE\" \
        -s \"$N_VOLS\" \
        -f \"${fsf_type}\" \
        -l \"$LOG_DIR\" \
        ${GENERAL_FLAGS[*]} \
        -v" "$subject_id" "$session";
        # log "ERROR" "FSF type ${fsf_type} failed for subject $subject_id${session:+ session $session}"

        
        log "SUCCESS" "Completed FSF type ${fsf_type} for subject $subject_id${session:+ session $session}"
    done

    log "SUCCESS" "All steps completed for subject $subject_id${session:+ session $session}"
    return 0
}

generate_report() {
    local report_file="$OUTPUT_DIR/processing_report_${CURRENT_DATE}.txt"
    {
        echo "===== Processing Report ====="
        echo "Date: $(date)"
        echo "Input Directory: $INPUT_DIR"
        echo "Output Directory: $OUTPUT_DIR"
        echo "-------------------------"
        echo "Total Subjects Processed: $total_subjects"
        
        if [ -f "$ERROR_SUBJECTS_FILE" ] && [ -s "$ERROR_SUBJECTS_FILE" ]; then
            echo "Failed Subjects Summary:"
            echo "-------------------------"
            echo "Total Errors: $(wc -l < "$ERROR_SUBJECTS_FILE")"
            echo "Unique Subjects with Errors: $(cut -d'|' -f1 "$ERROR_SUBJECTS_FILE" | sort -u | wc -l)"
            echo "Errors by Step:"
            echo "-------------------------"
            cut -d'|' -f2 "$ERROR_SUBJECTS_FILE" | sort | uniq -c | while read -r count step; do
                echo "  $step: $count errors"
            done
            echo "-------------------------"
            echo "Failed Subjects Details:"
            while IFS='|' read -r subject step code; do
                echo "  Subject: $subject"
                echo "    - Failed at step: $step"
                echo "    - Error code: $code"
            done < "$ERROR_SUBJECTS_FILE"
        else
            echo "No errors encountered during processing"
        fi
        
        echo "-------------------------"
        echo "FSF Types Processed: ${FSF_TYPES[*]}"
        echo "Processing Parameters:"
        echo "  - Threads: $NUM_THREADS"
        echo "  - FWHM: $FWHM"
        echo "  - Sigma: $SIGMA"
        echo "  - Number Volume: $N_VOLS"
    } > "$report_file"

    log "INFO" "Processing report generated: $report_file"
}

# ---------------------------- Main Pipeline --------------------------------
main() {
    # Set up logging directory
    if [ -z "$LOG_DIR" ]; then
        LOG_DIR="$OUTPUT_DIR/logs"
    fi
    mkdir -p "$LOG_DIR"

    # Initialize error tracking
    ERROR_SUBJECTS_FILE="${ERROR_LOG_DIR}/failed_subjects_${CURRENT_DATE}.txt"
    mkdir -p "$ERROR_LOG_DIR"
    : > "$ERROR_SUBJECTS_FILE"

    # Create temporary directory for temp files
    TEMP_DIR=$(mktemp -d)
    trap 'rm -rf "$TEMP_DIR"' EXIT

    # Define a distinct temporary file for subjects
    subjects_file="${TEMP_DIR}/subjects_list.txt"
    : > "$subjects_file"

    # Validate environment and parameters
    log "INFO" "Validating environment and parameters..."
    if ! setup_environment || ! validate_parameters; then
        log "ERROR" "Validation failed. Check logs for details."
        exit 1
    fi

    # Setup directories
    log "INFO" "Setting up directory structure..."
    if ! setup_directories "$OUTPUT_DIR"; then
        log "ERROR" "Failed to setup directories"
        exit 1
    fi

    log "INFO" "Scanning input directory for subjects..."
    while IFS= read -r dir; do
        subject_dir=$(basename "$dir")
        if [[ -d "$dir/func" ]]; then
            echo "$subject_dir \"\"" >> "$subjects_file"
        else
            while IFS= read -r session_dir; do
                session=$(basename "$session_dir")
                if [[ -d "$session_dir/func" ]]; then
                    echo "$subject_dir $session" >> "$subjects_file"
                fi
            done < <(find "$dir" -mindepth 1 -maxdepth 1 -type d)
        fi
    done < <(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d)

    total_subjects=$(wc -l < "$subjects_file")
    log "INFO" "Found $total_subjects subjects/sessions to process"

    if [ "$total_subjects" -eq 0 ]; then
        log "ERROR" "No subjects found in input directory"
        exit 1
    fi

    # Process subjects
    local failed_count=0
    while IFS= read -r line; do
        read -r subject_id session <<< "$line"
        log "INFO" "Starting pipeline for subject: $subject_id${session:+ session $session}"
        if ! process_subject "$subject_id" "$session"; then
            failed_count=$((failed_count + 1))
            log "ERROR" "Pipeline failed for subject: $subject_id${session:+ session $session}"
        fi
    done < "$subjects_file"

    generate_report

    if [ "$failed_count" -gt 0 ]; then
        log "WARNING" "Processing completed with $failed_count failures"
        exit 1
    else
        log "SUCCESS" "All processing completed successfully"
        exit 0
    fi
}
# ---------------------------- Execute Main --------------------------------
main "$@"
