#!/bin/bash
# =============================================================================
# stages/05_denoise.sh [-c conf] sub-XXXX
# One simultaneous 3dTproject projection (polynomial trend + band filter +
# nuisance regressors + censoring) per run and per denoising strategy, on the
# unsmoothed template-space BOLD (and on the T1w-space BOLD when SURFACE=yes,
# for stage 06). Optional smoothed copy of the template-space result.
# Contract: docs/DESIGN.md sections 6, 7 (05) and 10.
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
fp_init "$@"
SUB="${FP_ARGS[0]:?usage: $0 [-c conf] sub-XXXX}"
fp_set_log "$SUB" 05_denoise

GUARD_RC=0
STRATEGIES=()
FILTER_ARGS=()

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

dry() {
    is_yes "${DRY_RUN:-no}"
}

check_inputs() {
    if dry; then
        return 0
    fi
    require_files "$@"
}

is_positive() {   # VALUE : true for a number > 0
    awk -v v="$1" 'BEGIN { exit !(v + 0 > 0) }'
}

# install_file SRC DEST : the final name only ever appears through a rename
# inside its own directory. work/ and derivatives/ are different filesystems
# in the container (named volume vs bind mount), where a direct mv is a copy
# that an interruption leaves truncated under the final name.
install_file() {
    local src="$1" dst="$2" tmp
    tmp="$(dirname "$dst")/.tmp.$$.$(basename "$dst")"
    run mv -f "$src" "$tmp"
    run mv -f "$tmp" "$dst"
}

# clear_marker STAGE SUB RUN : forget the previous completion before the
# outputs are rebuilt, so a failed rerun cannot leave downstream stages
# trusting a marker whose outputs are gone (layout: DESIGN.md section 4).
clear_marker() {
    rm -f "$WORK_DIR/$2/.done/${1}__${3}.hash"
}

# ----------------------------- configuration ---------------------------------

check_config() {
    local s seen=" "
    read -r -a STRATEGIES <<< "${DENOISE_STRATEGIES:-}"
    [[ ${#STRATEGIES[@]} -gt 0 ]] || die "DENOISE_STRATEGIES is empty"
    local unique=()
    for s in "${STRATEGIES[@]}"; do
        [[ "$s" =~ ^[A-Za-z0-9]+$ ]] || die "strategy name must be alphanumeric (it becomes a desc- label): $s"
        if [[ "$seen" != *" $s "* ]]; then
            unique+=("$s")
            seen+="$s "
        fi
    done
    STRATEGIES=("${unique[@]}")

    case "$CENSOR_MODE" in
        NTRP|KILL|ZERO) ;;
        *) die "CENSOR_MODE must be NTRP, KILL or ZERO: $CENSOR_MODE" ;;
    esac
    [[ "$POLORT" =~ ^[0-9]+$ ]] || die "POLORT must be a non-negative integer: $POLORT"
    [[ "$SMOOTH_FWHM" =~ ^[0-9]*\.?[0-9]+$ ]] || die "SMOOTH_FWHM must be a number (0 = off): $SMOOTH_FWHM"

    # 3dTproject has no high-pass switch: an upper edge far above Nyquist leaves
    # only the low stop band (plus the Nyquist term itself, see strategies.py).
    case "$FILTER_MODE" in
        bandpass) FILTER_ARGS=(-passband "$BAND_LOW_HZ" "$BAND_HIGH_HZ") ;;
        highpass) FILTER_ARGS=(-passband "$BAND_LOW_HZ" 99999) ;;
        none)     FILTER_ARGS=() ;;
    esac
    if [[ "$FILTER_MODE" != none ]]; then
        is_positive "$BAND_LOW_HZ" || die "BAND_LOW_HZ must be > 0 with FILTER_MODE=$FILTER_MODE: $BAND_LOW_HZ"
    fi
    if [[ "$FILTER_MODE" == bandpass ]]; then
        awk -v lo="$BAND_LOW_HZ" -v hi="$BAND_HIGH_HZ" 'BEGIN { exit !(hi + 0 > lo + 0) }' \
            || die "BAND_HIGH_HZ ($BAND_HIGH_HZ) must be above BAND_LOW_HZ ($BAND_LOW_HZ)"
    fi
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
    if [[ -z "$tr" ]] && dry; then
        tr=0
    fi
    [[ -n "$tr" ]] || die "cannot determine the TR of $bold"
    echo "$tr"
}

count_censored() {   # CENSOR_1D : number of 0 entries
    awk 'NF > 0 && $1 + 0 == 0 { n++ } END { print n + 0 }' "$1"
}

# ----------------------------- projection ------------------------------------

# prepare_centred BOLD OUT : one uncompressed copy per run and space, shared
# across strategies. The nuisance design includes an intercept unless polort=-1.
prepare_centred() {
    local bold="$1" out="$2" mean="${2%.nii}_mean.nii"
    run 3dTstat -mean -prefix "$mean" "$bold"
    run 3dcalc -a "$bold" -b "$mean" -expr 'a-b' -datum float -prefix "$out"
    rm -f "$mean"
}

# project_series CENTRED MASK ORT CENSOR N_CENSORED TR N_EXPECTED OUT TMPDIR TAG
project_series() {
    local centred="$1" mask="$2" ort="$3" censor="$4" n_cens="$5" tr="$6" n_expected="$7" out="$8" tmpdir="$9" tag="${10}"
    local proj="$tmpdir/${tag}_projected.nii.gz"
    local n_out
    rm -f "$proj"

    local args=(-input "$centred" -mask "$mask" -ort "$ort" -polort "$POLORT" -dt "$tr")
    args+=(${FILTER_ARGS[@]+"${FILTER_ARGS[@]}"})
    if [[ "$n_cens" -gt 0 ]]; then
        args+=(-censor "$censor" -cenmode "$CENSOR_MODE")
    fi
    run 3dTproject "${args[@]}" -prefix "$proj"

    if dry; then
        return 0
    fi
    require_files "$proj"
    n_out="$(nvols "$proj")"
    if [[ -n "$n_expected" && "$n_out" != "$n_expected" ]]; then
        die "$tag: 3dTproject wrote $n_out volumes, expected $n_expected (CENSOR_MODE=$CENSOR_MODE, $n_cens censored)"
    fi
    install_file "$proj" "$out"
}

# smooth_series IN MASK OUT TMPDIR TAG
smooth_series() {
    local in="$1" mask="$2" out="$3" tmpdir="$4" tag="$5"
    local tmp="$tmpdir/${tag}_smoothed.nii.gz"
    rm -f "$tmp"
    run 3dBlurInMask -input "$in" -FWHM "$SMOOTH_FWHM" -mask "$mask" -quiet -prefix "$tmp"
    if dry; then
        return 0
    fi
    require_files "$tmp"
    install_file "$tmp" "$out"
}

# denoise_strategy RUN STRATEGY TR N_CENSORED TMPDIR   (always called through guarded)
denoise_strategy() {
    local run_id="$1" strat="$2" tr="$3" n_cens="$4" tmpdir="$5"
    local fdir pre tpl work
    fdir="$(func_dir "$SUB")"
    pre="$fdir/$run_id"
    tpl="${pre}_space-${TEMPLATE_NAME}_res-${MNI_RES}"
    work="$(work_func "$SUB" "$run_id")/denoise"

    local confounds="${pre}_desc-confounds_timeseries.tsv"
    local censor="${pre}_desc-censor.1D"
    local bold_tpl="${tpl}_desc-preproc_bold.nii.gz"
    local mask_tpl="${tpl}_desc-brain_mask.nii.gz"
    local bold_t1w="${pre}_space-T1w_desc-preproc_bold.nii.gz"
    local mask_t1w="${pre}_space-T1w_desc-brain_mask.nii.gz"

    local out_1d="${pre}_desc-${strat}_regressors.1D"
    local out_json="${pre}_desc-${strat}_denoise.json"
    local out_tpl="${tpl}_desc-${strat}_bold.nii.gz"
    local out_t1w="$work/${strat}_space-T1w_bold.nii.gz"
    local tmp_1d="$tmpdir/${strat}_regressors.1D"
    local tmp_json="$tmpdir/${strat}_denoise.json"
    local smooth=no label="" out_smooth=""
    if is_positive "$SMOOTH_FWHM"; then
        smooth=yes
        label="sm${SMOOTH_FWHM//./}"
        out_smooth="${tpl}_desc-${strat}${label}_bold.nii.gz"
    fi

    # Nothing of an earlier attempt may survive a failed one; this includes
    # smoothed copies made with another SMOOTH_FWHM and hidden install_file
    # temporaries of an interrupted run.
    if ! dry; then
        rm -f "$out_1d" "$out_json" "$out_tpl" "$out_t1w" "${tpl}_desc-${strat}sm"*"_bold.nii.gz" \
            "$fdir"/.tmp.*."${run_id}"*"_desc-${strat}_"* "$fdir"/.tmp.*."${run_id}"*"_desc-${strat}sm"*
    fi

    local inputs=(--input "bold=$bold_tpl" --input "mask=$mask_tpl")
    if is_yes "$SURFACE"; then
        inputs+=(--input "bold_t1w=$bold_t1w" --input "mask_t1w=$mask_t1w")
    fi
    # ACOMPCOR_N components per tissue exist in the table (stage 04); the
    # aCompCor strategies use all of them, 5 by default
    pyrun strategies \
        --confounds "$confounds" --strategy "$strat" --tr "$tr" \
        --n-volumes-censored-from "$censor" \
        --polort "$POLORT" --filter-mode "$FILTER_MODE" \
        --band-low "$BAND_LOW_HZ" --band-high "$BAND_HIGH_HZ" --min-dof "$MIN_DOF" \
        --acompcor-n "$ACOMPCOR_N" \
        --censor-mode "$CENSOR_MODE" --smooth-fwhm "$SMOOTH_FWHM" \
        "${inputs[@]}" --out-1d "$tmp_1d" --out-json "$tmp_json"

    local n_expected="" n_reg dof
    if ! dry; then
        require_files "$tmp_1d" "$tmp_json"
        n_expected="$(json_get "$tmp_json" n_volumes_out "")"
        n_reg="$(json_get "$tmp_json" n_regressors "?")"
        dof="$(json_get "$tmp_json" dof_remaining "?")"
        if [[ "$(json_get "$tmp_json" low_dof no)" == yes ]]; then
            log WARN "$run_id $strat: $n_reg regressors, only $dof residual degrees of freedom (MIN_DOF=$MIN_DOF)"
        else
            log INFO "$run_id $strat: $n_reg regressors, $dof residual degrees of freedom"
        fi
    fi

    project_series "$CENTRED_TPL" "$mask_tpl" "$tmp_1d" "$censor" "$n_cens" "$tr" "$n_expected" \
        "$out_tpl" "$tmpdir" "${strat}_tpl"
    if is_yes "$SURFACE"; then
        project_series "$CENTRED_T1W" "$mask_t1w" "$tmp_1d" "$censor" "$n_cens" "$tr" "$n_expected" \
            "$out_t1w" "$tmpdir" "${strat}_t1w"
    fi
    if [[ "$smooth" == yes ]]; then
        smooth_series "$out_tpl" "$mask_tpl" "$out_smooth" "$tmpdir" "${strat}_tpl"
    fi

    # regressors and provenance appear only next to a finished series
    if ! dry; then
        install_file "$tmp_1d" "$out_1d"
        install_file "$tmp_json" "$out_json"
    fi
    log OK "$run_id: strategy $strat done"
}

# ----------------------------- one run ---------------------------------------

process_run() {   # RUN
    local run_id="$1" fdir pre tpl work tr n_cens=0 strat tmpdir failed=()
    fdir="$(func_dir "$SUB")"
    pre="$fdir/$run_id"
    tpl="${pre}_space-${TEMPLATE_NAME}_res-${MNI_RES}"

    if ! stage_should_run 05_denoise "$SUB" "$run_id" --dep "04_confounds__$run_id" -- \
            DENOISE_STRATEGIES FILTER_MODE BAND_LOW_HZ BAND_HIGH_HZ POLORT CENSOR_MODE MIN_DOF SMOOTH_FWHM SURFACE; then
        return 0
    fi

    local censor="${pre}_desc-censor.1D"
    check_inputs "${pre}_desc-confounds_timeseries.tsv" "$censor" \
        "${tpl}_desc-preproc_bold.nii.gz" "${tpl}_desc-brain_mask.nii.gz"
    if is_yes "$SURFACE"; then
        check_inputs "${pre}_space-T1w_desc-preproc_bold.nii.gz" "${pre}_space-T1w_desc-brain_mask.nii.gz"
    fi
    if ! dry; then
        clear_marker 05_denoise "$SUB" "$run_id"
    fi

    tr="$(run_tr "${pre}_desc-prep_info.json" "${tpl}_desc-preproc_bold.nii.gz")"
    if [[ -s "$censor" ]]; then
        n_cens="$(count_censored "$censor")"
    fi
    log INFO "$run_id: TR=$tr, $n_cens censored frames ($CENSOR_MODE), filter=$FILTER_MODE ${FILTER_ARGS[*]+${FILTER_ARGS[*]}}, polort=$POLORT, strategies: ${STRATEGIES[*]}"

    work="$(work_func "$SUB" "$run_id")/denoise"
    mkdir -p "$work"
    local shared="$work/centred.$$" CENTRED_TPL CENTRED_T1W
    mkdir -p "$shared"
    CENTRED_TPL="$shared/tpl_centred.nii"
    CENTRED_T1W="$shared/t1w_centred.nii"
    prepare_centred "${tpl}_desc-preproc_bold.nii.gz" "$CENTRED_TPL"
    if is_yes "$SURFACE"; then
        prepare_centred "${pre}_space-T1w_desc-preproc_bold.nii.gz" "$CENTRED_T1W"
    fi
    for strat in "${STRATEGIES[@]}"; do
        tmpdir="$work/tmp_${strat}.$$"
        rm -rf "$tmpdir"
        mkdir -p "$tmpdir"
        guarded denoise_strategy "$run_id" "$strat" "$tr" "$n_cens" "$tmpdir"
        if [[ $GUARD_RC -ne 0 ]]; then
            log ERROR "$run_id: strategy $strat failed"
            failed+=("$strat")
        fi
        # the centred copy is as large as the input series: never leave it behind
        if [[ $GUARD_RC -eq 0 ]] || ! is_yes "$KEEP_WORK"; then
            rm -rf "$tmpdir"
        fi
    done

    rm -rf "$shared"
    if [[ ${#failed[@]} -gt 0 ]]; then
        log ERROR "$run_id: ${#failed[@]} of ${#STRATEGIES[@]} strategies failed: ${failed[*]}"
        return 1
    fi
    stage_mark_done 05_denoise "$SUB" "$run_id"
}

# ----------------------------- main ------------------------------------------

main() {
    local runs=() run_id failed=()
    check_config
    if ! dry; then
        require_cmds 3dTstat 3dcalc 3dTproject 3dBlurInMask fslnvols
        fp_check_mem
    fi
    mapfile -t runs < <(manifest_runs "$SUB" | awk -F'\t' '{print $8}')
    [[ ${#runs[@]} -gt 0 ]] || die "no runs for $SUB in $MANIFEST"

    for run_id in "${runs[@]}"; do
        guarded process_run "$run_id"
        if [[ $GUARD_RC -ne 0 ]]; then
            log ERROR "05_denoise failed for $run_id"
            failed+=("$run_id")
        fi
    done

    if [[ ${#failed[@]} -gt 0 ]]; then
        die "05_denoise: ${#failed[@]} of ${#runs[@]} runs failed: ${failed[*]}"
    fi
    log OK "05_denoise finished for $SUB (${#runs[@]} runs, strategies: ${STRATEGIES[*]})"
}

main
