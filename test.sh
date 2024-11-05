#!/bin/bash

# =============================================================================
# Script: extract_results.sh
# Description:
#   Extracts preprocessed fMRI data and organizes it into a specified directory structure.
#   The results are saved in 'results/NoGRS' and 'results/GRS' folders, organized by subject ID and session.
# =============================================================================

# ---------------------------- Usage Function --------------------------------

usage() {
    echo "Usage: $0 -s <subj_id> -e <session> -w <working_dir> -f <fsf_type> -r <results_dir> [-h]" >&2
    echo ""
    echo "Options:"
    echo "  -s    Subject ID (required)"
    echo "  -e    Session name (e.g., baseline, followup_1) (required)"
    echo "  -w    Working directory (required)"
    echo "  -f    fsf_type: NoGRS or GRS (required)"
    echo "  -r    results_dir"
    echo "  -h    Display this help message"
    exit 1
}

# ---------------------------- Argument Parsing ------------------------------

# Initialize variables
SUBJ_ID=""
SESSION=""
WORKING_DIR=""
FSF_TYPE=""

while getopts ":s:e:w:f:r:h" opt; do
    case ${opt} in
        s ) SUBJ_ID="$OPTARG" ;;
        e ) SESSION="$OPTARG" ;;
        w ) WORKING_DIR="$OPTARG" ;;
        r ) RESULTS_DIR="$OPTARG" ;;
        f ) FSF_TYPE="$OPTARG" ;;
        h ) usage ;;
        \? ) echo "Invalid Option: -$OPTARG" >&2; usage ;;
        : ) echo "Option -$OPTARG requires an argument." >&2; usage ;;
    esac
done

# ---------------------------- Validation -------------------------------------

if [[ -z "${SUBJ_ID}" || -z "${SESSION}" || -z "${WORKING_DIR}" || -z "${FSF_TYPE}" ]]; then
    echo "ERROR: Missing required arguments." >&2
    usage
fi

if [[ "$FSF_TYPE" != "NoGRS" && "$FSF_TYPE" != "GRS" ]]; then
    echo "ERROR: Invalid fsf_type. Please use NoGRS or GRS." >&2
    exit 1
fi

# ---------------------------- Setup ------------------------------------------

# Define the path to the preprocessed result file
result_file="${WORKING_DIR}/rest_res2standard.nii.gz"

# Define the target directory based on fsf_type, subject ID, and session
target_dir="${RESULTS_DIR}/${FSF_TYPE}"

target_file="${target_dir}/rest_res2standard.nii.gz"

# Create the target directory
mkdir -p "${target_dir}"

# ---------------------------- Copying the File ------------------------------

if [ -f "${result_file}" ]; then
    cp "${result_file}" "${target_file}"
    echo "File ${result_file} copied to ${target_file}"
else
    echo "Source file ${result_file} does not exist!" >&2
    exit 1
fi
