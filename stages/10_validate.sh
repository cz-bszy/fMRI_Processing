#!/bin/bash
# =============================================================================
# stages/10_validate.sh - validity of the final ROI time series and
#                         volume-vs-surface comparison (docs/DESIGN.md section 12)
#
# usage: 10_validate.sh [-c dataset.conf] sub-XXXX     per subject, after 07
#        10_validate.sh [-c dataset.conf] --group      dataset level, after all subjects
#
# Reads only the ROI tables of stage 07, <RUN>_desc-censor.1D, the confounds TSV,
# the _denoise.json files and $RESOURCE_DIR/atlases/<A>/ - never 4D data, so it
# is light enough to run anywhere. Per run it writes
#   <RUN>_desc-validation.tsv|.json   one row per stream x strategy x atlas x metric
#   <RUN>_desc-streamcompare.tsv      only when both streams exist
# and with --group: derivatives/group/{validation_long,stream_comparison,
#   strategy_comparison,fc_typicality}.tsv, stream_comparison.png, validation_report.html
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
fp_init "$@"
TARGET="${FP_ARGS[0]:?usage: $0 [-c conf] sub-XXXX | --group}"

STAGE=10_validate

# ----------------------------- helpers ---------------------------------------

# atlas_names : ATLASES plus the names of the name=path pairs of CUSTOM_ATLASES
atlas_names() {
    local names=() pairs=() pair
    read -r -a names <<< "${ATLASES:-}"
    read -r -a pairs <<< "${CUSTOM_ATLASES:-}"
    for pair in "${pairs[@]+"${pairs[@]}"}"; do
        names+=("${pair%%=*}")
    done
    if [[ ${#names[@]} -gt 0 ]]; then
        echo "${names[*]}"
    fi
}

# run_tr RUN : repetition time recorded by stage 03
run_tr() {
    local info
    info="$(func_dir "$SUB")/${1}_desc-prep_info.json"
    if [[ ! -s "$info" ]]; then
        if is_yes "${DRY_RUN:-no}"; then
            echo 0
            return 0
        fi
        die "$1: $info is missing (stage 03 has not run)"
    fi
    json_get "$info" tr ""
}

# ----------------------------- one run ---------------------------------------

process_run() {   # RUN
    local run="$1" fdir tr atlases
    stage_should_run "$STAGE" "$SUB" "$run" --dep "07_timeseries__$run" -- \
        DENOISE_STRATEGIES ATLASES CUSTOM_ATLASES CENSOR_MODE || return 0

    fdir="$(func_dir "$SUB")"
    atlases="$(atlas_names)"
    tr="$(run_tr "$run")"
    [[ "$tr" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] || die "$run: no usable 'tr' in ${run}_desc-prep_info.json (got '$tr')"

    pyrun validate --func-dir "$fdir" --run "$run" --template "$TEMPLATE_NAME" \
        --strategies "$DENOISE_STRATEGIES" --atlases "$atlases" \
        --atlas-dir "$RESOURCE_DIR/atlases" --tr "$tr" --censor-mode "$CENSOR_MODE" \
        --out-tsv "$fdir/${run}_desc-validation.tsv" \
        --out-json "$fdir/${run}_desc-validation.json" \
        --out-compare-tsv "$fdir/${run}_desc-streamcompare.tsv"

    stage_mark_done "$STAGE" "$SUB" "$run"
}

# ----------------------------- subject mode ----------------------------------

subject_main() {
    local rows=() row run_label n_fail=0 rc _s _ses _task _run _group _bold _t1w

    [[ -n "${DENOISE_STRATEGIES// /}" ]] || die "DENOISE_STRATEGIES is empty"
    if [[ -z "$(atlas_names)" ]]; then
        log WARN "ATLASES and CUSTOM_ATLASES are empty: nothing to validate"
        return 0
    fi

    mapfile -t rows < <(manifest_runs "$SUB")
    [[ ${#rows[@]} -gt 0 ]] || die "no runs for $SUB in $MANIFEST"

    for row in "${rows[@]}"; do
        IFS=$'\t' read -r _s _ses _task _run _group _bold _t1w run_label <<< "$row"
        [[ "${run_label:-}" == "$SUB"* ]] || die "manifest run_label '${run_label:-}' does not start with $SUB"
        # Subshell with its own 'set -e': a failing run must not stop the
        # remaining runs, and -e would be ignored inside an 'if' condition.
        set +e
        ( set -e; process_run "$run_label" )
        rc=$?
        set -e
        if [[ $rc -ne 0 ]]; then
            log ERROR "$STAGE failed for $run_label (exit $rc)"
            n_fail=$((n_fail + 1))
        fi
    done

    if [[ $n_fail -gt 0 ]]; then
        die "$STAGE: $n_fail of ${#rows[@]} run(s) failed for $SUB"
    fi
    log OK "$STAGE finished for $SUB (${#rows[@]} run(s))"
}

# ----------------------------- group mode ------------------------------------

group_main() {
    # No skip logic: the step takes seconds and its inputs are every subject's tables.
    pyrun compare_streams --deriv-dir "$DERIV_DIR" --manifest "$MANIFEST" --out-dir "$DERIV_DIR/group"
    log OK "$STAGE --group finished: $DERIV_DIR/group/validation_report.html"
}

# ----------------------------- main ------------------------------------------

case "$TARGET" in
    --group)
        fp_set_log dataset 10_validate_group
        group_main
        ;;
    -*)
        die "unknown option '$TARGET' (usage: $0 [-c conf] sub-XXXX | --group)"
        ;;
    *)
        SUB="$TARGET"
        fp_set_log "$SUB" "$STAGE"
        subject_main
        ;;
esac
