#!/bin/bash

# Set the base directories
BASE_OUTPUT_DIR="/home/zhaochang/Desktop"  # Base directory for all output

# Specific paths
BOLD_DIR="/mnt/hgfs/MDD_01" # Base directory for input .nii.gz files
MASK_DIR="${BASE_OUTPUT_DIR}/brainnetome_regions"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/tem"
AD="${BASE_OUTPUT_DIR}/AD"
SUBJECTS_FILE="${BASE_OUTPUT_DIR}/subj_list.txt"

# Extract subject names and save to a file
ls "${BOLD_DIR}/" | grep -o 'sub-[0-9]\+' > "${SUBJECTS_FILE}"

# Create the output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${AD}"

extract_and_combine_ts() {
    ID=$1
    echo "AD directory: ${AD}" 
    BOLD_FILE="${BOLD_DIR}/${ID}_ses-1_task-rest_run-1_space-MNI152NLin6ASym_reg-defaultNoGSR_desc-preproc_bold.nii.gz"
    COMBINED_FILE="${AD}/${ID}_BN246.txt"
    if [[ -f "${BOLD_FILE}" ]]; then
        # Initialize a string to hold all time series file names
        mask_files=""
        for i in $(seq 1 246); do
            MASK="${MASK_DIR}/${i}.nii.gz"
            if [[ -f "${MASK}" ]]; then
                OUTPUT_TXT="${OUTPUT_DIR}/${ID}_${i}.txt"
                fslmeants -i "${BOLD_FILE}" -o "${OUTPUT_TXT}" -m "${MASK}"
                mask_files+="${OUTPUT_TXT} "
            else
                echo "Mask file not found for ID: ${ID}, mask: ${MASK}"
                return 1
            fi
        done

        # Combine all time series into one file with columns for each region
        paste -d' ' $mask_files > "${COMBINED_FILE}"
        
        # Optionally, remove individual time series files to clean up
        #rm -f $mask_files
    else
        echo "BOLD file not found for ID: ${ID}"
        return 1
    fi
}

export -f extract_and_combine_ts
export BOLD_DIR MASK_DIR OUTPUT_DIR AD

# Read the subject IDs and process
cat "${SUBJECTS_FILE}" | parallel -j "$(nproc --ignore=1)" --bar extract_and_combine_ts {}

zip -r "${AD}/AD.zip" "${AD}"

echo "All done."