#!/bin/bash

set -euo pipefail

# =============================================================================
# Script: process_all.sh
# Description:
#   Orchestrates the processing of multiple subjects and sessions.
#   Executes steps: recon_all, Anatproc, fMRI_preproc in sequence for each subject-session pair.
#   Utilizes GNU Parallel for parallel processing based on available CPU cores.
# =============================================================================

# ---------------------------- Configuration ----------------------------------

INPUT_DIR="/mnt/d/Projects/Data_Processing/ABIDE_Longitudinal/ABIDEII-UPSM_Long"
RECON_ALL_DIR="/mnt/d/Projects/Data_Processing/ABIDE_Longitudinal/Outputs/ABIDEII-UPSM_Long/recon_all"
DATA_DIR="/mnt/d/Projects/Data_Processing/ABIDE_Longitudinal/Outputs/ABIDEII-UPSM_Long"
STANDARD_DIR="/mnt/d/Projects/Data_Processing/fMRI_Processing-main/standard"
tissuepriors_dir="/mnt/d/Projects/Data_Processing/fMRI_Processing-main/tissuepriors"
template_dir="/mnt/d/Projects/Data_Processing/fMRI_Processing-main/template"


CORES_PER_TASK=6
SESSIONS=("baseline" "followup_1")
# RECON_SCRIPT="/path/to/recon.sh"       
RECON_ALL_SCRIPT="./FC_step0.sh"
ANATPROC_SCRIPT="./FC_step1"        
FMRI_PREPROC_SCRIPT="./FC_step2" 
REGISTER_SCRIPT="./FC_step3"
Processing_CW="./FC_step4"
NUISANCE_REGRESSION_SCRIPT="./test.sh"

# ---------------------------- fMRI_preproc Configuration ----------------------

FWHM=6
SIGMA=2.54798709
HIGHP=0.1
LOWP=0.005
# TR=2
# TE=30
# N_VOLS=200 
# FSF_TYPE="Retain_GRS" ## or "Retain_GRS" "NO_GRD" 

# ---------------------------- Usage Function --------------------------------

usage() {
    echo "Usage: $0 [options]" >&2
    echo ""
    echo "Options:"
    echo "  -i    Path to the input data directory (default: $INPUT_DIR)"
    echo "  -r    Path to the recon all directory (default: $RECON_ALL_DIR)"
    echo "  -c    Number of CPU cores to allocate per task (default: $CORES_PER_TASK)"
    echo "  -s    Comma-separated list of sessions (default: baseline,followup_1)"
    echo "  -a    Path to recon_all.sh script (default: $RECON_ALL_SCRIPT)"
    echo "  -p    Path to Anatproc.sh script (default: $ANATPROC_SCRIPT)"
    echo "  -q    Path to fMRI_preproc.sh script (default: $FMRI_PREPROC_SCRIPT)"
    echo "  -f    FWHM value for smoothing (default: $FWHM)"
    echo "  -g    Sigma value for smoothing (default: $SIGMA)"
    echo "  -k    High-pass filter frequency for fMRI_preproc.sh (default: $HIGHP)"
    echo "  -l    Low-pass filter frequency for fMRI_preproc.sh (default: $LOWP)"
    echo "  -h    Display this help message"
    exit 1
}

# ---------------------------- Argument Parsing ------------------------------

while getopts ":i:r:c:s:a:p:q:f:g:k:l:h" opt; do
    case ${opt} in
        i) INPUT_DIR="$OPTARG";;
        r) RECON_ALL_DIR="$OPTARG";;
        c) CORES_PER_TASK="$OPTARG";;
        s) IFS=',' read -r -a SESSIONS <<< "$OPTARG";;
        a) RECON_ALL_SCRIPT="$OPTARG";;
        p) ANATPROC_SCRIPT="$OPTARG";;
        q) FMRI_PREPROC_SCRIPT="$OPTARG";;
        f) FWHM="$OPTARG";;
        g) SIGMA="$OPTARG";;
        k) HIGHP="$OPTARG";;
        l) LOWP="$OPTARG";;
        h) usage;;
        \?) echo "Invalid Option: -$OPTARG" >&2; usage;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage;;
    esac
done

# ---------------------------- Validation -------------------------------------

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory '$INPUT_DIR' does not exist." >&2
    exit 1
fi

if [ ${#SESSIONS[@]} -eq 0 ]; then
    echo "ERROR: At least one session must be specified." >&2
    exit 1
fi

if ! [[ "$CORES_PER_TASK" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: Cores per task must be a positive integer." >&2
    exit 1
fi

for script in "$RECON_ALL_SCRIPT" "$ANATPROC_SCRIPT" "$FMRI_PREPROC_SCRIPT" "$REGISTER_SCRIPT"; do
    if [ ! -x "$script" ]; then
        echo "ERROR: Script '$script' does not exist or is not executable." >&2
        exit 1
    fi
done

# Validate fMRI_preproc parameters
if ! [[ "$FWHM" =~ ^[0-9]+(\.[0-9]+)?$ ]] || ! [[ "$SIGMA" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "ERROR: FWHM and sigma must be positive numbers." >&2
    exit 1
fi

if ! [[ "$HIGHP" =~ ^[0-9]+(\.[0-9]+)?$ ]] || ! [[ "$LOWP" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "ERROR: High-pass and low-pass filter frequencies must be positive numbers." >&2
    exit 1
fi

if (( $(echo "$LOWP >= $HIGHP" | bc -l) )); then
    echo "ERROR: Low-pass filter frequency must be less than high-pass filter frequency." >&2
    exit 1
fi

# ---------------------------- Setup ------------------------------------------

MAIN_LOG_DIR="${DATA_DIR}/logs"
mkdir -p "$MAIN_LOG_DIR"


ERROR_LOG="${MAIN_LOG_DIR}/error_log.txt"
> "$ERROR_LOG"

# ---------------------------- Determine CPU Resources -----------------------

TOTAL_CORES=$(nproc)
echo "Total CPU cores available: $TOTAL_CORES" >&2
MAX_JOBS=$(( TOTAL_CORES / CORES_PER_TASK ))
if [ "$MAX_JOBS" -lt 1 ]; then
    MAX_JOBS=1
fi
echo "Cores per task: $CORES_PER_TASK" >&2
echo "Maximum parallel tasks: $MAX_JOBS" >&2

# ---------------------------- Processing Function ---------------------------

process_subject_session() {
    local subject_id="$1"
    local session="$2"

    local subject_session_log_dir="${MAIN_LOG_DIR}/${subject_id}_${session}"
    mkdir -p "$subject_session_log_dir"

    local anatproc_log="${subject_session_log_dir}/Anatproc.log"
    local fmripreproc_log="${subject_session_log_dir}/fMRI_preproc.log"
    local register_log="${subject_session_log_dir}/REGISTER.log"
    local nuisance_log="${subject_session_log_dir}/nuisance_regression.log"

    echo "----------------------------------------" >&2
    echo "Processing Subject: $subject_id | Session: $session" >&2
    echo "Logs are being saved to: $subject_session_log_dir" >&2
    echo "----------------------------------------" >&2

    local structure_dir="${INPUT_DIR}/${subject_id}/${session}/anat_1"
    local functional_dir="${INPUT_DIR}/${subject_id}/${session}/rest_1"

    if [ ! -d "$structure_dir" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Anatomical directory '$structure_dir' does not exist." >&2
        echo "${subject_id},${session}" >> "$ERROR_LOG"
        return 1
    fi

    if [ ! -d "$functional_dir" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Functional directory '$functional_dir' does not exist." >&2
        echo "${subject_id},${session}" >> "$ERROR_LOG"
        return 1
    fi

    ## ------------------------ Step 0: Recon_all ----------------------------

    # Uncomment and modify if recon_all step is needed
    # echo "Starting recon_all for Subject: $subject_id | Session: $session" >&2
    # bash "$RECON_ALL_SCRIPT" \
    #     -s "$subject_id" \
    #     -e "$session" \
    #     -w "$OUTPUT_DIR" \
    #     -o "$subject_output_dir" \
    #     -c "$CORES_PER_TASK"
    #
    # if [ $? -ne 0 ]; then
    #     echo "ERROR: recon_all failed for Subject: $subject_id | Session: $session" >&2
    #     return 1
    # fi
    # echo "Completed recon_all for Subject: $subject_id | Session: $session" >&2


    ## ------------------------ Step 1: Anatproc ------------------------------

    # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Anatproc for Subject: $subject_id | Session: $session" >&2

    # bash "$ANATPROC_SCRIPT" \
    #     -s "$subject_id" \
    #     -e "$session" \
    #     -n "$CORES_PER_TASK" \
    #     -f "$functional_dir" \
    #     -a "$structure_dir" \
    #     -t "$RECON_ALL_DIR" \
    #     -l "$anatproc_log"

    # if [ $? -ne 0 ]; then
    #     echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Anatproc failed for Subject: $subject_id | Session: $session" >&2
    #     echo "${subject_id},${session}" >> "$ERROR_LOG"
    #     return 1
    # fi
    # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Anatproc for Subject: $subject_id | Session: $session" >&2

    ## ------------------------ Step 2: fMRI_preproc ---------------------------

    # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting fMRI_preproc for Subject: $subject_id | Session: $session" >&2

    # bash "$FMRI_PREPROC_SCRIPT" \
    #     -s "$subject_id" \
    #     -n "$CORES_PER_TASK" \
    #     -f "$functional_dir" \
    #     -a "$structure_dir" \
    #     -m "$STANDARD_DIR" \
    #     -l "$fmripreproc_log" \
    #     -f "$FWHM" \
    #     -g "$SIGMA" \
    #     -k "$HIGHP" \
    #     -l "$LOWP"

    # if [ $? -ne 0 ]; then
    #     echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: fMRI_preproc failed for Subject: $subject_id | Session: $session" >&2
    #     echo "${subject_id},${session}" >> "$ERROR_LOG"
    #     return 1
    # fi
    # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed fMRI_preproc for Subject: $subject_id | Session: $session" >&2

    ## ------------------------ Step 3: REGISTER ------------------------------

    # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running REGISTER for Subject: $subject_id | Session: $session" >&2

    # bash "$REGISTER_SCRIPT" \
    #     -s "$subject_id" \
    #     -n "$CORES_PER_TASK" \
    #     -f "$functional_dir" \
    #     -a "$structure_dir" \
    #     -m "$STANDARD_DIR" \
    #     -l "$register_log"

    # if [ $? -ne 0 ]; then
    #     echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: REGISTER failed for Subject: $subject_id | Session: $session" >&2
    #     echo "${subject_id},${session}" >> "$ERROR_LOG"
    #     return 1
    # fi
    # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed REGISTER for Subject: $subject_id | Session: $session" >&2


    # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Anatproc for Subject: $subject_id | Session: $session" >&2

    # bash "$Processing_CW" \
    #     -s "$subject_id" \
    #     -e "$session" \
    #     -n "$CORES_PER_TASK" \
    #     -f "$functional_dir" \
    #     -a "$structure_dir" \
    #     -t "$tissuepriors_dir" \
    #     -g "$SIGMA" \
    #     -l "$anatproc_log"

    # if [ $? -ne 0 ]; then
    #     echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Anatproc failed for Subject: $subject_id | Session: $session" >&2
    #     echo "${subject_id},${session}" >> "$ERROR_LOG"
    #     return 1
    # fi
    # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Anatproc for Subject: $subject_id | Session: $session" >&2

    ## ------------------------ Step 4: Nuisance Regression ----------------------

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Nuisance Regression for Subject: $subject_id | Session: $session" >&2

    local TR=2
    local TE=30
    local N_VOLS=200 
    local FSF_TYPE="Retain_GRS"

    bash "$NUISANCE_REGRESSION_SCRIPT" \
        -s "$subject_id" \
        -n "$CORES_PER_TASK" \
        -w "$functional_dir" \
        -a "$structure_dir" \
        -t "$template_dir" \
        -p "$tissuepriors_dir" \
        -r "$TR" \
        -e "$TE" \
        -v "$N_VOLS" \
        -f "$FSF_TYPE" \
        -l "$nuisance_log"

    if [ $? -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Nuisance Regression failed for Subject: $subject_id | Session: $session" >&2
        echo "${subject_id},${session}" >> "$ERROR_LOG"
        return 1
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Nuisance Regression for Subject: $subject_id | Session: $session" >&2
}


export -f process_subject_session
export INPUT_DIR
export RECON_ALL_DIR
export CORES_PER_TASK
export RECON_ALL_SCRIPT
export ANATPROC_SCRIPT
export NUISANCE_REGRESSION_SCRIPT
export FMRI_PREPROC_SCRIPT
export REGISTER_SCRIPT
export Processing_CW
export FWHM
export SIGMA
export HIGHP
export LOWP
export MAIN_LOG_DIR
export ERROR_LOG
export tissuepriors_dir
export template_dir

# ---------------------------- Main Execution ---------------------------------

tasks=()

for subject_dir in "$INPUT_DIR"/*/; do
    subject_id=$(basename "$subject_dir")
    for session in "${SESSIONS[@]}"; do
        anat_file="${subject_dir}/${session}/anat_1/anat.nii.gz"
        if [ -f "$anat_file" ]; then
            tasks+=("$subject_id" "$session")
            # echo "Added Task - Subject ID: $subject_id, Session: $session" >&2
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: anat.nii.gz not found for Subject: $subject_id | Session: $session. Skipping." >&2
        fi
    done
done

total_tasks=$(( ${#tasks[@]} / 2 ))

task_list=()
for ((i=0; i<${#tasks[@]}; i+=2)); do
    task_list+=("${tasks[i]} ${tasks[i+1]}")
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Constructed Task List: ${#task_list[@]} tasks found." >&2

# ---------------------------- Run Tasks in Parallel ---------------------------


printf "%s\n" "${task_list[@]}" | parallel -j "$MAX_JOBS" --colsep ' ' process_subject_session {1} {2}

# ---------------------------- Summary ------------------------------------------

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All tasks completed. Total tasks processed: $total_tasks" >&2
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Check '$MAIN_LOG_DIR' for individual log files."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Check '$ERROR_LOG' for any errors encountered during processing."

# ---------------------------- Exit ---------------------------------------------

exit 0
