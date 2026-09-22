#!/bin/bash
# =============================================================================
# stages/08_qc.sh [-c conf] sub-XXXX
# Per run:     <RUN>_desc-qc_metrics.json, <RUN>_atlas-<A>_desc-<S>_roiqc.tsv and
#              the run figures (derivatives/sub-X/figures/<RUN>_*.png)
# Per subject: anatomical figures and derivatives/sub-X.html (always rewritten,
#              so that new stage-10 tables or a re-run of one run show up).
# Contract: docs/DESIGN.md sections 6, 9 and 10.
#
# QC never repairs anything and needs no product unconditionally: what is missing
# becomes n/a in the JSON and "figure not available" in the report.
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
fp_init "$@"
SUB="${FP_ARGS[0]:?usage: $0 [-c conf] sub-XXXX}"
fp_set_log "$SUB" 08_qc

STAGE=08_qc
: "${QC_FWHM:=yes}"          # 3dFWHMx -acf smoothness estimate (about a minute per run)
QC_VARS=(QC_FD_MEAN_WARN QC_FD_MEAN_FAIL QC_PCT_CENSORED_WARN QC_PCT_CENSORED_FAIL QC_TSNR_GM_WARN QC_TSNR_GM_FAIL
         QC_COREG_DICE_WARN QC_COREG_DICE_FAIL QC_NORM_DICE_WARN QC_NORM_DICE_FAIL QC_EULER_HOLES_WARN QC_EULER_HOLES_FAIL
         MIN_DOF MIN_RETAINED_MIN)
ATLAS_LIST=""
GUARD_RC=0

# guarded CMD ARGS... : run CMD in a subshell in which 'set -e' is effective and
# store its status in GUARD_RC (a function called behind 'if' or '||' would run
# with 'set -e' switched off).
guarded() {
    set +e
    ( set -e; "$@" )
    GUARD_RC=$?
    set -e
    return 0
}

# Names of all atlases (ATLASES + the name part of CUSTOM_ATLASES name=path pairs).
atlas_names() {
    local names=() pairs=() pair
    read -r -a names <<< "${ATLASES:-}"
    read -r -a pairs <<< "${CUSTOM_ATLASES:-}"
    for pair in "${pairs[@]+"${pairs[@]}"}"; do
        names+=("${pair%%=*}")
    done
    echo "${names[*]+"${names[*]}"}"
}

# template_or_none KIND : path from template_path, or "none" (no FSL, no file)
template_or_none() {
    local path=""
    path="$(template_path "$1" 2>/dev/null)" || path=""
    if [[ -n "$path" && -s "$path" ]]; then
        echo "$path"
    else
        echo none
    fi
}

# tpl_file RUN SUFFIX : template-space file of a run (res-2, or the BIDS spelling res-02)
tpl_file() {
    local base
    base="$(func_dir "$SUB")/${1}_space-${TEMPLATE_NAME}_res-"
    if [[ ! -e "${base}${MNI_RES}_$2" && -e "${base}0${MNI_RES}_$2" ]]; then
        echo "${base}0${MNI_RES}_$2"
    else
        echo "${base}${MNI_RES}_$2"
    fi
}

# warp_tissue_masks RUN QCDIR : subject GM/WM/CSF masks on the template-space BOLD
# grid. They give the post-denoise tSNR its grey-matter voxels and order the
# "after" carpet; without them QC falls back to the whole brain mask.
warp_tissue_masks() {
    local run="$1" qcdir="$2" adir ref warp affine tissue src out tmp
    adir="$(anat_dir "$SUB")"
    ref="$(tpl_file "$run" boldref.nii.gz)"
    warp="$adir/xfm/T1w_to_MNI_1Warp.nii.gz"
    affine="$adir/xfm/T1w_to_MNI_0GenericAffine.mat"
    if ! command -v antsApplyTransforms >/dev/null 2>&1 || [[ ! -s "$ref" || ! -s "$warp" || ! -s "$affine" ]]; then
        log WARN "$run: no template-space tissue masks (antsApplyTransforms, boldref or T1w->template transform missing)"
        return 0
    fi
    for tissue in GM WM CSF; do
        src="$adir/${SUB}_label-${tissue}_mask.nii.gz"
        out="$qcdir/space-${TEMPLATE_NAME}_label-${tissue}_mask.nii.gz"
        tmp="$qcdir/.tmp$$_label-${tissue}_mask.nii.gz"
        rm -f "$out"
        [[ -s "$src" ]] || continue
        if run antsApplyTransforms -d 3 -i "$src" -r "$ref" -o "$tmp" -n GenericLabel \
                -t "$warp" -t "$affine" -u uchar; then
            is_yes "$DRY_RUN" || mv -f "$tmp" "$out"
        else
            log WARN "$run: warping the $tissue mask failed; QC uses the brain mask instead"
            rm -f "$tmp"
        fi
    done
}

# estimate_fwhm RUN QCDIR : spatial smoothness of the preprocessed series. The ACF
# parameters arrive on stdout, which run() would send to the log - hence no run().
estimate_fwhm() {
    local run="$1" qcdir="$2" fdir bold mask tmp
    fdir="$(func_dir "$SUB")"
    bold="$fdir/${run}_space-T1w_desc-preproc_bold.nii.gz"
    mask="$fdir/${run}_space-T1w_desc-brain_mask.nii.gz"
    tmp="$qcdir/.tmp$$_fwhm_acf.txt"
    rm -f "$qcdir/fwhm_acf.txt"
    is_yes "$QC_FWHM" || return 0
    if ! command -v 3dFWHMx >/dev/null 2>&1 || [[ ! -s "$bold" || ! -s "$mask" ]]; then
        log WARN "$run: 3dFWHMx or its inputs missing, fwhm_acf = n/a"
        return 0
    fi
    log INFO "+ 3dFWHMx -detrend -mask $mask -acf NULL -input $bold"
    is_yes "$DRY_RUN" && return 0
    if 3dFWHMx -detrend -mask "$mask" -acf NULL -input "$bold" > "$tmp" 2>> "$FP_LOGFILE"; then
        mv -f "$tmp" "$qcdir/fwhm_acf.txt"
    else
        log WARN "$run: 3dFWHMx failed, fwhm_acf = n/a"
        rm -f "$tmp"
    fi
}

mask_arg() {   # QCDIR TISSUE -> path or "" (option value)
    local path="$1/space-${TEMPLATE_NAME}_label-${2}_mask.nii.gz"
    if [[ -s "$path" ]]; then echo "$path"; else echo ""; fi
}

process_run() {   # RUN
    local run="$1" fdir adir qcdir out_json tpl_brain tpl_mask opts=()
    # 10_validate is optional (not in STAGES = marker "missing", still a stable hash)
    if ! stage_should_run "$STAGE" "$SUB" "$run" --dep "07_timeseries__$run" --dep "10_validate__$run" -- \
            DENOISE_STRATEGIES ATLASES CUSTOM_ATLASES TEMPLATE_NAME MNI_RES CENSOR_MODE QC_FWHM "${QC_VARS[@]}"; then
        return 0
    fi
    fdir="$(func_dir "$SUB")"
    adir="$(anat_dir "$SUB")"
    qcdir="$(work_func "$SUB" "$run")/qc"
    out_json="$fdir/${run}_desc-qc_metrics.json"
    mkdir -p "$qcdir" "$(fig_dir "$SUB")"
    if ! is_yes "$DRY_RUN"; then
        [[ -d "$fdir" ]] || die "$run: no func derivatives ($fdir)"
        # neither a stale JSON nor the ROI table of a dropped strategy/atlas may survive
        rm -f "$out_json" "$fdir/${run}"_atlas-*_desc-*_roiqc.tsv
    fi

    warp_tissue_masks "$run" "$qcdir"
    estimate_fwhm "$run" "$qcdir"
    [[ -s "$qcdir/fwhm_acf.txt" ]] && opts+=(--fwhm-file "$qcdir/fwhm_acf.txt")
    [[ -n "$(mask_arg "$qcdir" GM)" ]] && opts+=(--tpl-gm-mask "$(mask_arg "$qcdir" GM)")
    [[ -n "$(mask_arg "$qcdir" WM)" ]] && opts+=(--tpl-wm-mask "$(mask_arg "$qcdir" WM)")
    [[ -n "$(mask_arg "$qcdir" CSF)" ]] && opts+=(--tpl-csf-mask "$(mask_arg "$qcdir" CSF)")

    pyrun qc_metrics --func-dir "$fdir" --anat-dir "$adir" --run "$run" \
        --template "$TEMPLATE_NAME" --mni-res "$MNI_RES" \
        --strategies "$DENOISE_STRATEGIES" --atlases "$ATLAS_LIST" --censor-mode "$CENSOR_MODE" \
        --atlas-labels-dir "$RESOURCE_DIR/atlases" --cache-dir "$qcdir" --out-json "$out_json" \
        --qc-fd-mean-warn "$QC_FD_MEAN_WARN" --qc-fd-mean-fail "$QC_FD_MEAN_FAIL" \
        --qc-pct-censored-warn "$QC_PCT_CENSORED_WARN" --qc-pct-censored-fail "$QC_PCT_CENSORED_FAIL" \
        --qc-tsnr-gm-warn "$QC_TSNR_GM_WARN" --qc-tsnr-gm-fail "$QC_TSNR_GM_FAIL" \
        --qc-coreg-dice-warn "$QC_COREG_DICE_WARN" --qc-coreg-dice-fail "$QC_COREG_DICE_FAIL" \
        --qc-norm-dice-warn "$QC_NORM_DICE_WARN" --qc-norm-dice-fail "$QC_NORM_DICE_FAIL" \
        --qc-euler-holes-warn "$QC_EULER_HOLES_WARN" --qc-euler-holes-fail "$QC_EULER_HOLES_FAIL" \
        --min-dof "$MIN_DOF" --min-retained-min "$MIN_RETAINED_MIN" \
        "${opts[@]+"${opts[@]}"}"
    is_yes "$DRY_RUN" || require_files "$out_json"

    tpl_brain="$(template_or_none brain_res)"
    tpl_mask="$(template_or_none mask_res)"
    # figures are a best effort: the metrics above are the product that must exist
    if ! pyrun plots run --func-dir "$fdir" --anat-dir "$adir" --fig-dir "$(fig_dir "$SUB")" --run "$run" \
            --template "$TEMPLATE_NAME" --mni-res "$MNI_RES" \
            --strategies "$DENOISE_STRATEGIES" --atlases "$ATLAS_LIST" --cache-dir "$qcdir" \
            --template-brain "$tpl_brain" --template-mask "$tpl_mask" \
            --atlas-labels-dir "$RESOURCE_DIR/atlases" --resource-dir "$RESOURCE_DIR"; then
        log WARN "$run: figures incomplete (see the log); the report marks the missing ones"
    fi
    stage_mark_done "$STAGE" "$SUB" "$run"
}

anat_figures() {
    local tpl_brain tpl_mask
    # the surface figure reads the GIFTI files of 06_surface (subject level)
    if ! stage_should_run "$STAGE" "$SUB" anat --dep 02_anat_prep --dep 06_surface -- TEMPLATE_NAME SURFACE; then
        return 0
    fi
    mkdir -p "$(fig_dir "$SUB")"
    tpl_brain="$(template_or_none brain)"
    tpl_mask="$(template_or_none mask)"
    pyrun plots anat --anat-dir "$(anat_dir "$SUB")" --fig-dir "$(fig_dir "$SUB")" --subject "$SUB" \
        --template "$TEMPLATE_NAME" --template-brain "$tpl_brain" --template-mask "$tpl_mask"
    stage_mark_done "$STAGE" "$SUB" anat
}

write_report() {   # RUN...
    pyrun report --deriv-dir "$DERIV_DIR" --subject "$SUB" --runs "$*" --template "$TEMPLATE_NAME" \
        --strategies "$DENOISE_STRATEGIES" --atlases "$ATLAS_LIST" --manifest "$MANIFEST" \
        --out "$DERIV_DIR/$SUB.html"
}

main() {
    local runs=() run failed=()
    ATLAS_LIST="$(atlas_names)"
    mapfile -t runs < <(manifest_runs "$SUB" | awk -F'\t' '{print $8}')
    [[ ${#runs[@]} -gt 0 ]] || die "no runs for $SUB in $MANIFEST"

    for run in "${runs[@]}"; do
        guarded process_run "$run"
        if [[ $GUARD_RC -ne 0 ]]; then
            log ERROR "$STAGE failed for $run"
            failed+=("$run")
        fi
    done

    guarded anat_figures
    if [[ $GUARD_RC -ne 0 ]]; then
        log WARN "$SUB: anatomical figures incomplete"
    fi

    # The report is written even when a run failed: it then says which one.
    guarded write_report "${runs[@]}"
    if [[ $GUARD_RC -ne 0 ]]; then
        die "$STAGE: the report of $SUB could not be written"
    fi
    if [[ ${#failed[@]} -gt 0 ]]; then
        die "$STAGE: ${#failed[@]} of ${#runs[@]} run(s) failed: ${failed[*]}"
    fi
    log OK "$STAGE finished for $SUB (${#runs[@]} run(s)) -> $DERIV_DIR/$SUB.html"
}

main
