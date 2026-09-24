#!/bin/bash
# =============================================================================
# stages/09_group_qc.sh [-c conf]
# Dataset level, no subject argument, run once after every subject's 08_qc:
#   derivatives/group/group_qc.tsv         one row per run, robust-z outlier flags
#   derivatives/group/inclusion.tsv        inclusion decision per run and strategy (EXCLUDE_*)
#   derivatives/group/qcfc_<S>_<A>.tsv     edge-wise QC-FC when >= QCFC_MIN_SUBJECTS runs
#   derivatives/group/qcfc_summary.tsv
#   derivatives/group/group_report.html
# Contract: docs/DESIGN.md section 9 ("Group").
#
# No stage marker: the inputs are every subject's metrics, the step takes
# seconds, and a report that is always regenerated cannot go stale.
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
fp_init "$@"
if [[ ${#FP_ARGS[@]} -gt 0 ]]; then
    die "usage: $0 [-c conf]   (dataset level: no subject argument, got '${FP_ARGS[*]}')"
fi
fp_set_log dataset 09_group_qc

STAGE=09_group_qc
ATLAS_SPACE=MNI152NLin6Asym      # fetch_resources stores the atlases under this label (see 07_timeseries.sh)

# atlas_names : ATLASES plus the names of the name=path pairs of CUSTOM_ATLASES
atlas_names() {
    local names=() pairs=() pair
    read -r -a names <<< "${ATLASES:-}"
    read -r -a pairs <<< "${CUSTOM_ATLASES:-}"
    for pair in "${pairs[@]+"${pairs[@]}"}"; do
        names+=("${pair%%=*}")
    done
    echo "${names[*]+"${names[*]}"}"
}

# custom_atlas_args : one "--atlas-volume name=path" per CUSTOM_ATLASES entry
# (the ATLASES volumes are found through --atlas-dir). Printed one per line.
custom_atlas_args() {
    local pairs=() pair
    read -r -a pairs <<< "${CUSTOM_ATLASES:-}"
    for pair in "${pairs[@]+"${pairs[@]}"}"; do
        [[ "$pair" == *=* ]] || continue
        printf -- '--atlas-volume\n%s\n' "$pair"
    done
}

main() {
    local out_dir="$DERIV_DIR/group" atlas_list extra=() inclusion=()
    atlas_list="$(atlas_names)"
    mapfile -t extra < <(custom_atlas_args)
    mapfile -t inclusion < <(inclusion_args)
    mkdir -p "$out_dir"
    log INFO "group QC over $DERIV_DIR (strategies: ${DENOISE_STRATEGIES:-none}; atlases: ${atlas_list:-none}; QC-FC needs >= $QCFC_MIN_SUBJECTS runs)"

    pyrun group_report --deriv-dir "$DERIV_DIR" --manifest "$MANIFEST" --out-dir "$out_dir" \
        --template "$TEMPLATE_NAME" --strategies "$DENOISE_STRATEGIES" --atlases "$atlas_list" \
        --atlas-dir "$RESOURCE_DIR/atlases" --atlas-space "$ATLAS_SPACE" \
        --qcfc-min-subjects "$QCFC_MIN_SUBJECTS" \
        --qc-fd-mean-warn "$QC_FD_MEAN_WARN" --qc-fd-mean-fail "$QC_FD_MEAN_FAIL" \
        --qc-pct-censored-warn "$QC_PCT_CENSORED_WARN" --qc-pct-censored-fail "$QC_PCT_CENSORED_FAIL" \
        --qc-tsnr-gm-warn "$QC_TSNR_GM_WARN" --qc-tsnr-gm-fail "$QC_TSNR_GM_FAIL" \
        --qc-coreg-dice-warn "$QC_COREG_DICE_WARN" --qc-coreg-dice-fail "$QC_COREG_DICE_FAIL" \
        --qc-norm-dice-warn "$QC_NORM_DICE_WARN" --qc-norm-dice-fail "$QC_NORM_DICE_FAIL" \
        --qc-euler-holes-warn "$QC_EULER_HOLES_WARN" --qc-euler-holes-fail "$QC_EULER_HOLES_FAIL" \
        "${inclusion[@]}" "${extra[@]+"${extra[@]}"}"

    if ! is_yes "${DRY_RUN:-no}"; then
        require_files "$out_dir/group_qc.tsv" "$out_dir/inclusion.tsv" "$out_dir/group_report.html"
    fi
    log OK "$STAGE finished -> $out_dir/group_report.html"
}

main
