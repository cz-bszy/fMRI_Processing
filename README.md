# fMRI Data Processing Pipeline

A comprehensive bash-based pipeline for processing fMRI data, including anatomical preprocessing, functional preprocessing, registration, and nuisance regression.

## Overview

This pipeline integrates multiple processing steps:
1. FreeSurfer's ReconAll Processing (FC_step0)
2. Anatomical Data Processing (FC_step1)
3. Functional Data Preprocessing (FC_step2)
4. Registration (FC_step3)
5. Tissue Segmentation (FC_step4)
6. Nuisance Regression (FC_step5)
7. Results Extraction (FC_step6)

## Prerequisites

- FreeSurfer
- FSL
- AFNI
- GNU Parallel

## Directory Structure

```
input_dir/
├── sub-001/
│   ├── anat/
│   │   └── sub-001_T1w.nii.gz
│   └── func/
│       └── sub-001_task-rest_bold.nii.gz
└── sub-002/
    ├── anat/
    │   └── sub-002_T1w.nii.gz
    └── func/
        └── sub-002_task-rest_bold.nii.gz
```

## Configuration

Key parameters in `main.sh`:
```bash
# Directory settings
INPUT_DIR="/path/to/input"          # Raw data directory
OUTPUT_DIR="/path/to/output"        # Output directory
STANDARD_DIR="./standard"           # Standard brain templates
TEMPLATE_DIR="./template"           # FSF templates

# Processing parameters
NUM_THREADS=4                       # Number of CPU threads
FWHM=6.0                           # Full Width at Half Maximum
SIGMA=2.548                        # Smoothing sigma
TR=2.0                             # Repetition Time
TE=30                              # Echo Time
N_VOLS=200                         # Number of volumes
```

## Usage

### Basic Usage
```bash
bash main.sh
```

### Options
```bash
-f    FSF types (comma-separated, default: NO_GRD,Retain_GRS)
-s    Skip existing (default: true)
-d    Dry run
-v    Verbose output
-h    Display help message
```

### Examples
```bash
# Dry run with verbose output
bash main.sh -d -v

# Process with specific FSF type
bash main.sh -f NO_GRD

# Skip existing processed data
bash main.sh -s
```

## Output Structure

```
output_dir/
├── logs/
│   └── pipeline_YYYYMMDD.log
├── error_logs/
│   └── failed_subjects_YYYYMMDD.txt
├── recon_all/
│   └── [FreeSurfer outputs]
├── sub-001/
│   ├── anat/
│   │   └── [Processed anatomical data]
│   └── func/
│       └── [Processed functional data]
└── results/
    └── [Final outputs]
```

## Error Handling

- Failed subjects are logged in `error_logs/failed_subjects_YYYYMMDD.txt`
- Detailed error messages in `error_logs/errors_YYYYMMDD.log`
- Processing report in `processing_report_YYYYMMDD.txt`

## Notes

1. Ensure all prerequisites are installed and properly configured
2. Set up FreeSurfer environment before running
3. Check disk space requirements
4. Verify input data structure

## Troubleshooting

Common issues:
- `brain.mgz not found`: ReconAll processing failed
- `example_func.nii.gz not found`: Functional preprocessing failed
- Permission errors: Check directory permissions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.