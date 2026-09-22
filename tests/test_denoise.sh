#!/bin/bash
# Exercise the actual stage-05 run orchestration and centring helper with
# tiny text arrays standing in for NIfTI files. This does not emulate AFNI's
# projection: it verifies shared-input numerical identity and I/O counts.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
stage="$REPO/stages/05_denoise.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
eval "$(sed -n '/^prepare_centred()/,/^# smooth_series/{ /^# smooth_series/d; p; }' "$stage")"
eval "$(sed -n '/^process_run()/,/^# ----.*main/{ /^# ----.*main/d; p; }' "$stage")"
SUB=sub-test; TEMPLATE_NAME=MNI; MNI_RES=2; SURFACE=yes; KEEP_WORK=no
CENSOR_MODE=NTRP; FILTER_MODE=none; POLORT=2
FILTER_ARGS=(); STRATEGIES=(first second third)
func_dir() { echo "$tmp/func"; }
work_func() { echo "$tmp/work"; }
stage_should_run() { return 0; }
stage_mark_done() { :; }
clear_marker() { :; }
check_inputs() { :; }
run_tr() { echo 2; }
count_censored() { echo 0; }
log() { :; }
dry() { return 1; }
is_yes() { [[ "$1" == yes ]]; }
require_files() { for f in "$@"; do [[ -s "$f" ]]; done; }
nvols() { wc -l < "$1"; }
install_file() { mv "$1" "$2"; }
run() { "$@"; }
guarded() { GUARD_RC=0; ( set -e; "$@" ) || GUARD_RC=$?; }
function 3dTstat {
    echo mean >> "$tmp/calls"
    local out="$3" input="$4"
    awk '{a+=$1;b+=$2} END {print a/NR,b/NR}' "$input" > "$out"
}
function 3dcalc {
    echo centre >> "$tmp/calls"
    local input="$2" mean="$4" out="${10}"
    awk 'NR==FNR {a=$1;b=$2;next} {print $1-a,$2-b}' "$mean" "$input" > "$out"
}
function 3dTproject {
    local input="$2" out="${!#}"
    echo project >> "$tmp/calls"
    cp "$input" "$out"
}
denoise_strategy() {
    local strat="$2" w="$5"
    project_series "$CENTRED_TPL" mask ort censor 0 2 3 "$tmp/${strat}_tpl" "$w" tpl
    project_series "$CENTRED_T1W" mask ort censor 0 2 3 "$tmp/${strat}_t1w" "$w" t1w
}
mkdir -p "$tmp/func"
printf '1 10\n3 20\n5 30\n' > "$tmp/func/run_space-MNI_res-2_desc-preproc_bold.nii.gz"
printf '10 100\n30 200\n50 300\n' > "$tmp/func/run_space-T1w_desc-preproc_bold.nii.gz"
process_run run
[[ "$(grep -c '^mean$' "$tmp/calls")" == 2 ]]
[[ "$(grep -c '^centre$' "$tmp/calls")" == 2 ]]
[[ "$(grep -c '^project$' "$tmp/calls")" == 6 ]]
printf '%s\n' '-2 -10' '0 0' '2 10' > "$tmp/expected_tpl"
printf '%s\n' '-20 -100' '0 0' '20 100' > "$tmp/expected_t1w"
for strat in "${STRATEGIES[@]}"; do
    cmp "$tmp/expected_tpl" "$tmp/${strat}_tpl"
    cmp "$tmp/expected_t1w" "$tmp/${strat}_t1w"
done
[[ -z "$(find "$tmp/work" -name '*centred*' -print)" ]]
echo 'PASS: 3 strategies x 2 spaces share exactly 2 centred inputs; values and cleanup verified'
