#!/bin/bash
# =============================================================================
# stages/04_confounds.sh [-c conf] sub-XXXX
# Confounds table (fMRIPrep column names), FD, DVARS, aCompCor and the censor
# vector of every run, from the unsmoothed T1w-space BOLD written by stage 03.
# Contract: docs/DESIGN.md sections 6, 7 (04) and 10.
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
fp_init "$@"
SUB="${FP_ARGS[0]:?usage: $0 [-c conf] sub-XXXX}"
fp_set_log "$SUB" 04_confounds

GUARD_RC=0

# guarded CMD ARGS... : run CMD in a subshell in which 'set -e' is effective and
# store its status in GUARD_RC. Calling a function behind 'if' or '||' would
# switch 'set -e' off for everything inside it.
guarded() {
    set +e
    ( set -e; "$@" )
    GUARD_RC=$?
    set -e
    return 0
}

# TR of a run: stage-03 provenance first, NIfTI header as fallback.
run_tr() {   # PREP_INFO_JSON BOLD
    local prep="$1" bold="$2" tr=""
    if [[ -s "$prep" ]]; then
        tr="$(json_get "$prep" tr "")"
    fi
    if [[ -z "$tr" && -s "$bold" ]]; then
        tr="$(img_tr "$bold")"
    fi
    if [[ -z "$tr" ]] && is_yes "$DRY_RUN"; then
        tr=0
    fi
    [[ -n "$tr" ]] || die "cannot determine the TR of $bold"
    echo "$tr"
}

process_run() {   # RUN
    local run_id="$1" fdir pre tr
    fdir="$(func_dir "$SUB")"
    pre="$fdir/$run_id"

    if ! stage_should_run 04_confounds "$SUB" "$run_id" --dep "03_func_prep__$run_id" -- \
            ACOMPCOR_N HIGHPASS_SEC CENSOR_FD CENSOR_PREV CENSOR_NEXT CENSOR_MIN_SEGMENT CENSOR_DVARS; then
        return 0
    fi

    local bold="${pre}_space-T1w_desc-preproc_bold.nii.gz"
    local brain="${pre}_space-T1w_desc-brain_mask.nii.gz"
    local wm="${pre}_space-T1w_label-WM_mask.nii.gz"
    local csf="${pre}_space-T1w_label-CSF_mask.nii.gz"
    local par="${pre}_desc-hmc_motion.par"
    local relrms="${pre}_desc-hmc_relrms.txt"
    local outliers="${pre}_desc-outliers_timeseries.1D"
    local prep="${pre}_desc-prep_info.json"
    local out_tsv="${pre}_desc-confounds_timeseries.tsv"
    local out_json="${pre}_desc-confounds_timeseries.json"
    local out_censor="${pre}_desc-censor.1D"

    if ! is_yes "$DRY_RUN"; then
        require_files "$bold" "$brain" "$wm" "$csf" "$par" "$relrms" "$outliers"
        # forget the previous completion before the table is rebuilt, so a failed
        # rerun cannot leave stage 05 trusting a marker whose outputs are gone
        rm -f "$WORK_DIR/$SUB/.done/04_confounds__${run_id}.hash"
    fi
    tr="$(run_tr "$prep" "$bold")"
    log INFO "$run_id: TR=$tr aCompCor=$ACOMPCOR_N highpass=${HIGHPASS_SEC}s censor FD>$CENSOR_FD prev=$CENSOR_PREV next=$CENSOR_NEXT min_segment=$CENSOR_MIN_SEGMENT std_dvars>$CENSOR_DVARS"

    local censor_prev=no
    if is_yes "$CENSOR_PREV"; then
        censor_prev=yes
    fi

    # a stale table must not survive a failed recomputation
    if ! is_yes "$DRY_RUN"; then
        rm -f "$out_tsv" "$out_json" "$out_censor"
    fi
    pyrun confounds \
        --bold "$bold" --brain-mask "$brain" --wm-mask "$wm" --csf-mask "$csf" \
        --motion-par "$par" --relrms "$relrms" --outliers "$outliers" \
        --tr "$tr" --acompcor-n "$ACOMPCOR_N" --highpass-sec "$HIGHPASS_SEC" \
        --censor-fd "$CENSOR_FD" --censor-prev "$censor_prev" --censor-dvars "$CENSOR_DVARS" \
        --censor-next "$CENSOR_NEXT" --censor-min-segment "$CENSOR_MIN_SEGMENT" \
        --out-tsv "$out_tsv" --out-json "$out_json" --out-censor "$out_censor"

    if ! is_yes "$DRY_RUN"; then
        require_files "$out_tsv" "$out_json" "$out_censor"
    fi
    stage_mark_done 04_confounds "$SUB" "$run_id"
}

main() {
    local runs=() run_id failed=()
    mapfile -t runs < <(manifest_runs "$SUB" | awk -F'\t' '{print $8}')
    [[ ${#runs[@]} -gt 0 ]] || die "no runs for $SUB in $MANIFEST"

    for run_id in "${runs[@]}"; do
        guarded process_run "$run_id"
        if [[ $GUARD_RC -ne 0 ]]; then
            log ERROR "04_confounds failed for $run_id"
            failed+=("$run_id")
        fi
    done

    if [[ ${#failed[@]} -gt 0 ]]; then
        die "04_confounds: ${#failed[@]} of ${#runs[@]} runs failed: ${failed[*]}"
    fi
    log OK "04_confounds finished for $SUB (${#runs[@]} runs)"
}

main
