#!/usr/bin/env bash
# A bounded diagnostic run from saved FreeSurfer anatomy and original BOLD.
set -euo pipefail
export FSLDIR=${FSLDIR:-/opt/fsl}
export PATH="${FREESURFER_HOME:-/opt/freesurfer}/bin:$PATH"
export FSLOUTPUTTYPE=NIFTI_GZ
export OMP_NUM_THREADS=4
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
export DROP_VOLUMES=${DROP_VOLUMES:-0}
export STC=required
root=${1:?pilot root}; subject=${2:?subject}; session=${3:?session}
export FS_LICENSE=${FS_LICENSE:-$root/license.txt}
code=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
output="$root/output"; logs="$root/logs/${subject}_${session}"
mkdir -p "$logs"
start=$(date +%s)
start_step=${START_STEP:-1}
[[ "$start_step" =~ ^[1-7]$ ]] || { echo 'START_STEP must be 1..7' >&2; exit 1; }
for step in 1 2 3 4 5 6; do bash -n "$code/FC_step$step"; done
# Steps run as commands, so shell failures cannot be swallowed by an if-function context.
if (( start_step <= 1 )); then
    bash "$code/FC_step1" -i "$root/input" -o "$output" -r "$root/recon" -c 4 -l "$logs" "$subject" "$session" > "$logs/step1.log" 2>&1
fi
if (( start_step <= 2 )); then
    bash "$code/FC_step2" -i "$root/input" -o "$output" -n 4 -w 6 -g 2.548 -h 0.1 -l 0.01 -d "$logs" -p '*_bold.nii.gz' "$subject" "$session" > "$logs/step2.log" 2>&1
fi
if (( start_step <= 3 )); then
    bash "$code/FC_step3" -i "$root/input" -o "$output" -s "$code/standard" -n 4 -l "$logs" "$subject" "$session" > "$logs/step3.log" 2>&1
fi
if (( start_step <= 4 )); then
    bash "$code/FC_step4" -i "$root/input" -o "$output" -s "$code/tissuepriors" -n 4 -g 2.548 -l "$logs" "$subject" "$session" > "$logs/step4.log" 2>&1
fi
func="$output/$subject/$session/func"
tr=$(3dinfo -tr "$func/rest_pp.nii.gz")
nvol=$(3dinfo -nt "$func/rest_pp.nii.gz")
te=$(python3 -c 'import json,sys; print(1000*json.load(open(sys.argv[1]))["EchoTime_s"])' "$func/preprocessing.json")
if (( start_step <= 5 )); then
    for model in NoGRS Retain_GRS; do
        bash "$code/FC_step5" -i "$root/input" -o "$output" -t "$code/template" -r "$tr" -e "$te" -s "$nvol" -f "$model" -n 4 -l "$logs" "$subject" "$session" > "$logs/step5_${model}.log" 2>&1
    done
fi
if (( start_step <= 6 )); then
    bash "$code/FC_step6" -i "$root/input" -o "$output" -T ThomasYeo100 -l "$logs" "$subject" "$session" > "$logs/step6.log" 2>&1
fi
python3 "$code/QC_nor" -p "$output" -o "$root/qc" --scan "${subject}_${session}" --overwrite --jobs 1 > "$logs/qc.log" 2>&1
python3 "$code/check_adni_scan.py" "$root" "$subject" "$session" > "$logs/checks.log" 2>&1
echo "SCAN_FINISHED $subject $session $(( $(date +%s)-start )) seconds"
