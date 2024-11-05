#!/bin/bash

# =============================================================================
# Script: recon_all_parallel.sh
# Description:
#   Processes T1-weighted anatomical MRI data for multiple subjects and sessions
#   using FreeSurfer's recon-all in parallel, based on available CPU cores.
# =============================================================================

# ---------------------------- Configuration ----------------------------------


INPUT_DIR="/mnt/d/Projects/Data_Processing/ABIDE_Longitudinal/ABIDEII-UPSM_Long"
OUTPUT_DIR="/mnt/d/Projects/Data_Processing/ABIDE_Longitudinal/Outputs/ABIDEII-UPSM_Long/recon_all"
CORES_PER_SUBJECT=3
# ---------------------------- Usage Function --------------------------------

usage() {
    echo "Usage: $0 [-i input_dir] [-o output_dir] [-c cores_per_subject]" >&2
    echo ""
    echo "Options:"
    echo "  -i    Path to the input data directory (default: $INPUT_DIR)"
    echo "  -o    Path to the output directory (default: $OUTPUT_DIR)"
    echo "  -c    Number of CPU cores to allocate per subject (default: $CORES_PER_SUBJECT)"
    echo "  -h    Display this help message"
    exit 1
}

# ---------------------------- Argument Parsing ------------------------------

while getopts ":i:o:c:h" opt; do
    case ${opt} in
        i) INPUT_DIR="$OPTARG";;
        o) OUTPUT_DIR="$OPTARG";;
        c) CORES_PER_SUBJECT="$OPTARG";;
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

if ! [[ "$CORES_PER_SUBJECT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: Cores per subject must be a positive integer." >&2
    exit 1
fi

# ---------------------------- Setup ------------------------------------------

mkdir -p "$OUTPUT_DIR"

# ---------------------------- Determine CPU Resources -----------------------
TOTAL_CORES=$(nproc)
echo "Total CPU cores available: $TOTAL_CORES" >&2
MAX_JOBS=$(( TOTAL_CORES / CORES_PER_SUBJECT ))
if [ "$MAX_JOBS" -lt 1 ]; then
    MAX_JOBS=1
fi
echo "Cores per subject: $CORES_PER_SUBJECT" >&2
echo "Maximum parallel subjects: $MAX_JOBS" >&2

# ---------------------------- Processing Function ---------------------------

process_subject_session() {
    local subject_id="$1"
    local session="$2"
    local anat_file="$3"
    local subj_sess_id="$session"

    echo "----------------------------------------" >&2
    echo "Processing Subject: $subject_id | Session: $session" >&2
    echo "Anatomical File: $anat_file" >&2
    echo "Subject-Session ID: $subj_sess_id" >&2
    echo "Output Directory: $OUTPUT_DIR/$subject_id" >&2
    echo "----------------------------------------" >&2

    local subject_output_dir="$OUTPUT_DIR/$subject_id"
    mkdir -p "$subject_output_dir"

    if [[ -f "$anat_file" ]]; then
        echo "Starting recon-all for $subj_sess_id (Subject: $subject_id)..." >&2
        recon-all -s "$subj_sess_id" -i "$anat_file" -all -openmp "$CORES_PER_SUBJECT" -sd "$subject_output_dir"
        if [ $? -eq 0 ]; then
            echo "Completed recon-all for $subj_sess_id (Subject: $subject_id)." >&2
        else
            echo "ERROR: recon-all failed for $subj_sess_id (Subject: $subject_id)." >&2
        fi
    else
        echo "ERROR: Anatomical file '$anat_file' not found for Subject '$subject_id' Session '$session'." >&2
    fi
}

export -f process_subject_session
export OUTPUT_DIR
export CORES_PER_SUBJECT

# ---------------------------- Main Execution ---------------------------------
parsed_files=0
while IFS= read -r anat_file; do
    relative_path="${anat_file#"$INPUT_DIR"/}"
    echo "Relative path: $relative_path" >&2
    IFS='/' read -r subject_id session subdir anat_filename <<< "$relative_path"
    if [[ -z "$subject_id" || -z "$session" || -z "$subdir" || -z "$anat_filename" ]]; then
        echo "WARNING: Skipping malformed path: $anat_file" >&2
        continue
    fi
    parsed_files=$((parsed_files + 1 ))
    echo "$subject_id $session $anat_file"
done < <(find "$INPUT_DIR" -type f -name "anat.nii.gz") | parallel -j "$MAX_JOBS" --colsep ' ' process_subject_session {1} {2} {3}

# ---------------------------- Summary ------------------------------------------

echo "Total anatomical files found and parsed: $parsed_files" >&2
