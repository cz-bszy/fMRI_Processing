#!/bin/bash
# =============================================================================
# stages/03_func_prep.sh [-c conf] sub-XXXX
# Per BOLD run: drop volumes, raw QC, despike, motion ESTIMATION on pre-STC data,
# slice timing, boldref + EPI mask, EPI->T1w coregistration (BBR never trusted
# blindly), then ONE interpolation per volume (HMC o EPI->T1w [o T1w->template])
# onto the T1w-space BOLD grid and the template grid, masks, global scaling,
# transform-chain self-checks and provenance.
# Contract: docs/DESIGN.md sections 6 (func names), 7 ("03"), 10 (prep_info), 11.
#
# Transform chain (all world-coordinate ITK transforms; antsApplyTransforms applies
# the LAST listed transform first):
#   T1w grid:      -t bold2t1_itk.txt -t hmc_i.txt
#   template grid: -t T1w_to_MNI_1Warp -t T1w_to_MNI_0GenericAffine -t bold2t1_itk.txt -t hmc_i.txt
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
fp_init "$@"
SUB="${FP_ARGS[0]:?usage: $0 [-c conf] sub-XXXX}"
STAGE=03_func_prep
fp_set_log "$SUB" "$STAGE"

MIN_VOLUMES=50            # refuse shorter runs (after dropping)
GRID_PAD_MM=10            # margin of the T1w-space BOLD grid around the brain mask
MIN_EPI_MASK_MM3=400000   # a smaller "brain" mask means the skull-strip failed
MIN_COREG_DICE=0.5        # below: transform chain broken or coregistration failed
MIN_HMC_CONSISTENCY=0.9   # corr(mean of resampled series, resampled boldref)
HMC_DIRECTION_MIN_MM=0.5  # direction test only when the worst volume moved this far
HMC_DIRECTION_TOL=0.002   # tolerated loss of similarity caused by applying HMC
HASH_VARS=(DROP_VOLUMES DESPIKE STC STC_INTERP EPI_MASK_METHOD BBR_MAX_DISP_MM BBR_MAX_COST
           FUNC_T1W_RES MNI_RES SCALE_TARGET ANAT_MODE TEMPLATE_NAME MIN_TISSUE_VOX)

ANAT_PRE="$(anat_dir "$SUB")/$SUB"
XFM_AFFINE="$(anat_dir "$SUB")/xfm/T1w_to_MNI_0GenericAffine.mat"
XFM_WARP="$(anat_dir "$SUB")/xfm/T1w_to_MNI_1Warp.nii.gz"
TPL_GRID=""
GUARD_RC=0

# per-run state (process_run runs in a subshell, so nothing leaks between runs)
RUN_ID="" BOLD="" SIDECAR="" W="" PRE="" FINAL_FILES=()
TR="" N_RAW="" N_DROP="" N_VOLS="" NSS=""
DESPIKED=false DESPIKE_FRAC="0" HMC_INPUT="" REF0_IDX="" RESAMPLE_INPUT=""
STC_APPLIED=false STC_REASON="" TZERO="nan" ST_SOURCE="" ST_EVIDENCE=""
EPI_MASK_USED="" COREG_METHOD="" BBR_COST="nan" BBR_DISP="nan" BBR_REJECTED=false
COREG_DICE="nan" HMC_R="nan" HMC_GAIN="nan" HMC_WORST_MM="nan" SCALE=""
WM_N=0 CSF_N=0 GM_N=0 BRAIN_N=0 TISSUE_N=0 TISSUE_THR="" WM_THR="" CSF_THR="" RELAXED=false

# ----------------------------- generic helpers -------------------------------

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

# capture CMD ARGS... : like run(), but stdout goes to the caller: x="$(capture ...)"
capture() {
    local printable rc=0
    printf -v printable '%q ' "$@"
    log INFO "+ ${printable% }"
    "$@" 2>> "$FP_LOGFILE" || rc=$?
    if [[ $rc -ne 0 ]]; then
        log ERROR "command failed (exit $rc): $1"
    fi
    return $rc
}

# run_to FILE CMD ARGS... : like run(), but stdout is the content of FILE
run_to() {
    local out="$1" tmp
    shift
    tmp="$out.tmp$$"
    capture "$@" > "$tmp"
    mv -f "$tmp" "$out"
}

pyval() {
    capture "$PYTHON_BIN" -m fmriproc.prep_utils "$@"
}

is_number() {
    [[ "$1" =~ ^[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$ ]]
}

# num_cmp A OP B (OP: lt le gt ge) : false when an operand is not a number (nan)
num_cmp() {
    if ! is_number "$1" || ! is_number "$3"; then
        return 1
    fi
    awk -v a="$1" -v op="$2" -v b="$3" 'BEGIN {
        a += 0; b += 0
        if (op == "lt") exit !(a < b)
        if (op == "le") exit !(a <= b)
        if (op == "gt") exit !(a > b)
        if (op == "ge") exit !(a >= b)
        exit 2
    }'
}

last_number() {   # last numeric token of a text, or "nan"
    awk '{ for (i = 1; i <= NF; i++) if ($i ~ /^[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?$/) v = $i }
         END { print (v == "" ? "nan" : v) }' <<< "$1"
}

count_rows() {    # data rows of a 1D file
    awk 'NF && $1 !~ /^#/ { n++ } END { print n + 0 }' "$1"
}

# install SRC DST : DST appears only complete (hidden name in the target directory
# first, because work/ and derivatives/ may be different filesystems).
install() {
    local src="$1" dst="$2" tmp
    tmp="$(dirname "$dst")/.tmp.$$.$(basename "$dst")"
    run mv -f "$src" "$tmp"
    run mv -f "$tmp" "$dst"
}

install_copy() {
    local src="$1" dst="$2" tmp
    tmp="$(dirname "$dst")/.tmp.$$.$(basename "$dst")"
    run cp -f "$src" "$tmp"
    run mv -f "$tmp" "$dst"
}

check_license() {
    if [[ -z "${FS_LICENSE:-}" || ! -s "${FS_LICENSE:-}" ]]; then
        die "FreeSurfer license not found (FS_LICENSE='${FS_LICENSE:-}'). Mount license.txt into the container and export FS_LICENSE."
    fi
}

check_config() {
    [[ "$DROP_VOLUMES" =~ ^[0-9]+$ ]] || die "DROP_VOLUMES must be a non-negative integer: $DROP_VOLUMES"
    [[ "$MIN_TISSUE_VOX" =~ ^[0-9]+$ ]] || die "MIN_TISSUE_VOX must be a non-negative integer: $MIN_TISSUE_VOX"
    case "$STC_INTERP" in
        linear|cubic|quintic|heptic|wsinc5|wsinc9|Fourier) ;;
        *) die "STC_INTERP must be one of linear cubic quintic heptic wsinc5 wsinc9 Fourier: $STC_INTERP" ;;
    esac
    case "$EPI_MASK_METHOD" in synthstrip|automask) ;; *) die "EPI_MASK_METHOD must be synthstrip or automask: $EPI_MASK_METHOD" ;; esac
    local v
    for v in FUNC_T1W_RES SCALE_TARGET BBR_MAX_DISP_MM BBR_MAX_COST; do
        num_cmp "${!v}" gt 0 || die "$v must be a positive number: ${!v}"
    done
}

# ----------------------------- transforms ------------------------------------

# boldref-space image -> T1w-space BOLD grid / template grid (no HMC matrix)
to_t1w() {   # IN OUT INTERP
    run antsApplyTransforms -d 3 --float -n "$3" -i "$1" -r "$W/t1w_grid.nii.gz" -o "$2" \
        -t "$W/bold2t1_itk.txt"
}

to_tpl() {   # IN OUT INTERP
    run antsApplyTransforms -d 3 --float -n "$3" -i "$1" -r "$TPL_GRID" -o "$2" \
        -t "$XFM_WARP" -t "$XFM_AFFINE" -t "$W/bold2t1_itk.txt"
}

# T1w-space (anat grid) image -> T1w-space BOLD grid / template grid
anat_to_t1w() {   # IN OUT INTERP
    run antsApplyTransforms -d 3 --float -n "$3" -i "$1" -r "$W/t1w_grid.nii.gz" -o "$2" -t identity
}

anat_to_tpl() {   # IN OUT INTERP
    run antsApplyTransforms -d 3 --float -n "$3" -i "$1" -r "$TPL_GRID" -o "$2" \
        -t "$XFM_WARP" -t "$XFM_AFFINE"
}

binarise() {   # PROBABILITY OUT THRESHOLD
    run fslmaths "$1" -thr "$3" -bin "$2" -odt char
}

# One volume: mcflirt matrix -> ITK, then a single interpolation into each space.
# Runs in child shells started by xargs (exported function + RV_* environment).
resample_volume() {
    local idx="$1"
    local vol="$RV_DIR/vol_${idx}.nii" hmc="$RV_DIR/hmc_${idx}.txt"
    export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 OMP_NUM_THREADS=1
    # source and target of a mcflirt matrix are both the BOLD grid
    wb_command -convert-affine -from-flirt "$RV_MATDIR/MAT_${idx}" "$RV_BOLDREF" "$RV_BOLDREF" -to-itk "$hmc"
    antsApplyTransforms -d 3 --float -n LanczosWindowedSinc -i "$vol" -r "$RV_T1W_GRID" \
        -o "$RV_DIR/t1w_${idx}.nii" -t "$RV_BOLD2T1" -t "$hmc"
    antsApplyTransforms -d 3 --float -n LanczosWindowedSinc -i "$vol" -r "$RV_TPL_GRID" \
        -o "$RV_DIR/mni_${idx}.nii" -t "$RV_WARP" -t "$RV_AFFINE" -t "$RV_BOLD2T1" -t "$hmc"
    [[ -s "$RV_DIR/t1w_${idx}.nii" && -s "$RV_DIR/mni_${idx}.nii" ]]
}

merge_series() {   # PREFIX(t1w|mni) OUT
    local prefix="$1" out="$2" files=() idx
    while read -r idx; do
        [[ -s "$W/vols/${prefix}_${idx}.nii" ]] || die "resampled volume missing: $W/vols/${prefix}_${idx}.nii"
        files+=("$W/vols/${prefix}_${idx}.nii")
    done < "$W/vol_ids.txt"
    [[ ${#files[@]} -eq "$N_VOLS" ]] || die "expected $N_VOLS resampled volumes, found ${#files[@]}"
    # not through run(): the command line lists every volume
    log INFO "+ fslmerge -tr $out <${#files[@]} volumes ${prefix}_*.nii> $TR"
    fslmerge -tr "$out" "${files[@]}" "$TR" >> "$FP_LOGFILE" 2>&1
}

# ----------------------------- steps -----------------------------------------

set_run_paths() {
    local fdir
    fdir="$(func_dir "$SUB")"
    PRE="$fdir/$RUN_ID"
    # private sub-directory: work/sub-X/func/<RUN>/ is shared with stages 05 and 06
    W="$(work_func "$SUB" "$RUN_ID")/$STAGE"
    SIDECAR="$(bold_json "$BOLD")"
    local tpl="space-${TEMPLATE_NAME}_res-${MNI_RES}"
    FINAL_FILES=(
        "${PRE}_desc-hmc_motion.par"
        "${PRE}_desc-hmc_relrms.txt"
        "${PRE}_desc-hmc_absrms.txt"
        "${PRE}_desc-outliers_timeseries.1D"
        "${PRE}_desc-quality_timeseries.1D"
        "${PRE}_from-bold_to-T1w_itk.txt"
        "${PRE}_space-T1w_boldref.nii.gz"
        "${PRE}_space-T1w_desc-brain_mask.nii.gz"
        "${PRE}_space-T1w_desc-preproc_bold.nii.gz"
        "${PRE}_space-T1w_label-WM_mask.nii.gz"
        "${PRE}_space-T1w_label-CSF_mask.nii.gz"
        "${PRE}_space-T1w_label-GM_mask.nii.gz"
        "${PRE}_${tpl}_boldref.nii.gz"
        "${PRE}_${tpl}_desc-brain_mask.nii.gz"
        "${PRE}_${tpl}_desc-preproc_bold.nii.gz"
        "${PRE}_desc-prep_info.json"
    )
}

outputs_complete() {
    local f
    for f in "${FINAL_FILES[@]}"; do
        [[ -s "$f" ]] || return 1
    done
    return 0
}

check_inputs() {
    require_files "$BOLD" "${ANAT_PRE}_desc-preproc_T1w.nii.gz" "${ANAT_PRE}_desc-brain_mask.nii.gz" \
        "${ANAT_PRE}_desc-brain_T1w.nii.gz" "${ANAT_PRE}_label-WM_mask.nii.gz" "${ANAT_PRE}_label-CSF_mask.nii.gz" \
        "${ANAT_PRE}_label-GM_mask.nii.gz" "$XFM_AFFINE" "$XFM_WARP"
    if [[ "$ANAT_MODE" == freesurfer ]]; then
        require_files "$FS_DIR/$SUB/mri/orig.mgz" "$FS_DIR/$SUB/mri/brainmask.mgz" \
            "$FS_DIR/$SUB/mri/aparc+aseg.mgz" "$FS_DIR/$SUB/surf/lh.white" "$FS_DIR/$SUB/surf/rh.white"
    else
        require_files "${ANAT_PRE}_label-WMbbr_mask.nii.gz" "$FSLDIR/etc/flirtsch/bbr.sch"
    fi
}

# 1. float copy without the first volumes; TR; non-steady-state report
step_drop() {
    local drop_json tr_json tr_hdr n_check
    N_RAW="$(nvols "$BOLD")"
    drop_json="$(json_get "$SIDECAR" DropVolumes "")"
    drop_json="${drop_json%.0}"
    if [[ "$drop_json" =~ ^[0-9]+$ ]]; then
        N_DROP="$drop_json"
    else
        N_DROP="$DROP_VOLUMES"
    fi
    N_VOLS=$(( N_RAW - N_DROP ))
    if [[ "$N_VOLS" -lt "$MIN_VOLUMES" ]]; then
        die "$RUN_ID: $N_RAW volumes - $N_DROP dropped = $N_VOLS, fewer than $MIN_VOLUMES: run refused"
    fi
    [[ "$N_VOLS" -le 9999 ]] || die "$RUN_ID: more than 9999 volumes are not supported (fslsplit/mcflirt numbering)"

    tr_json="$(json_get "$SIDECAR" RepetitionTime "")"
    tr_hdr="$(img_tr "$BOLD")"
    if num_cmp "$tr_json" gt 0; then
        TR="$tr_json"
        if is_number "$tr_hdr" && ! awk -v a="$tr_json" -v b="$tr_hdr" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d <= 0.005) }'; then
            log WARN "$RUN_ID: sidecar RepetitionTime $tr_json s and NIfTI header TR $tr_hdr disagree; the sidecar is used"
        fi
    elif num_cmp "$tr_hdr" ge 0.2 && num_cmp "$tr_hdr" le 20; then
        TR="$tr_hdr"
        log WARN "$RUN_ID: no RepetitionTime in $SIDECAR; using the header TR $TR s"
    else
        die "$RUN_ID: no usable TR (sidecar: '${tr_json}', header: '${tr_hdr}')"
    fi
    log INFO "$RUN_ID: TR=$TR s, $N_RAW volumes, dropping $N_DROP -> $N_VOLS"

    run 3dcalc -a "${BOLD}[${N_DROP}..\$]" -expr a -datum float -prefix "$W/bold_dropped.nii.gz"
    n_check="$(nvols "$W/bold_dropped.nii.gz")"
    [[ "$n_check" -eq "$N_VOLS" ]] || die "$RUN_ID: dropped series has $n_check volumes, expected $N_VOLS"

    NSS="$(pyval nss --bold "$BOLD")"
    if [[ "$NSS" -gt "$N_DROP" ]]; then
        log WARN "$RUN_ID: $NSS non-steady-state volumes detected but only $N_DROP dropped (set drop_volumes in the acquisition table)"
    else
        log INFO "$RUN_ID: non-steady-state volumes detected: $NSS (dropped: $N_DROP)"
    fi
}

# 2. QC of the untouched signal: later cleaning must not hide bad volumes
step_raw_qc() {
    local rows
    run_to "$W/outliers.1D" 3dToutcount -automask -fraction -polort 3 -legendre "$W/bold_dropped.nii.gz"
    run_to "$W/quality.1D" 3dTqual -automask "$W/bold_dropped.nii.gz"
    rows="$(count_rows "$W/outliers.1D")"
    [[ "$rows" -eq "$N_VOLS" ]] || die "$RUN_ID: 3dToutcount wrote $rows rows, expected $N_VOLS"
}

# 3. despike
step_despike() {
    if is_yes "$DESPIKE"; then
        run 3dDespike -NEW -nomask -prefix "$W/bold_despiked.nii.gz" "$W/bold_dropped.nii.gz"
        HMC_INPUT="$W/bold_despiked.nii.gz"
        DESPIKED=true
        DESPIKE_FRAC="$(pyval despike-fraction --before "$W/bold_dropped.nii.gz" --after "$HMC_INPUT")"
        log INFO "$RUN_ID: fraction of in-head samples changed by 3dDespike: $DESPIKE_FRAC"
    else
        HMC_INPUT="$W/bold_dropped.nii.gz"
    fi
}

# 4. motion estimate on pre-STC data; only matrices and parameters are kept
step_hmc() {
    local n_mat
    REF0_IDX="$(pyval min-outlier --file "$W/outliers.1D")"
    log INFO "$RUN_ID: initial motion reference = volume $REF0_IDX (lowest outlier fraction)"
    run fslroi "$HMC_INPUT" "$W/ref0.nii.gz" "$REF0_IDX" 1
    # spline: the median of trilinear-resampled volumes would be a blurred BBR source
    run mcflirt -in "$HMC_INPUT" -reffile "$W/ref0.nii.gz" -out "$W/pass1" -spline_final
    run 3dTstat -median -prefix "$W/boldref_median.nii.gz" "$W/pass1.nii.gz"
    # spline overshoot; N4 works on log intensities
    run fslmaths "$W/boldref_median.nii.gz" -thr 0 "$W/boldref.nii.gz" -odt float
    run rm -f "$W/pass1.nii.gz"
    run mcflirt -in "$HMC_INPUT" -reffile "$W/boldref.nii.gz" -out "$W/pass2" -mats -plots -rmsrel -rmsabs
    run rm -f "$W/pass2.nii.gz"
    require_files "$W/pass2.par" "$W/pass2_rel.rms" "$W/pass2_abs.rms"
    n_mat="$(find "$W/pass2.mat" -maxdepth 1 -type f -name 'MAT_*' | wc -l)"
    [[ "$n_mat" -eq "$N_VOLS" ]] || die "$RUN_ID: mcflirt wrote $n_mat matrices, expected $N_VOLS"
    [[ "$(count_rows "$W/pass2.par")" -eq "$N_VOLS" ]] || die "$RUN_ID: pass2.par does not have $N_VOLS rows"
}

# 5. slice timing (only with verified timing; fmriproc.timing decides)
step_stc() {
    local decision
    pyrun timing afni-tpattern --bold "$W/bold_dropped.nii.gz" --json "$SIDECAR" --stc "$STC" \
        --out-1d "$W/slice_timing.1D" --out-json "$W/timing.json"
    require_files "$W/timing.json"
    decision="$(json_get "$W/timing.json" stc skip)"
    STC_REASON="$(json_get "$W/timing.json" reason "")"
    ST_SOURCE="$(json_get "$W/timing.json" source none)"
    ST_EVIDENCE="$(json_get "$W/timing.json" evidence none)"
    if [[ "$decision" == apply ]]; then
        require_files "$W/slice_timing.1D"
        TZERO="$(json_get "$W/timing.json" tzero "")"
        is_number "$TZERO" || die "$RUN_ID: timing.json has no usable tzero"
        run 3dTshift -TR "${TR}s" -tzero "$TZERO" -tpattern "@$W/slice_timing.1D" "-$STC_INTERP" \
            -prefix "$W/bold_stc.nii.gz" "$HMC_INPUT"
        RESAMPLE_INPUT="$W/bold_stc.nii.gz"
        STC_APPLIED=true
    else
        log WARN "$RUN_ID: slice timing correction not applied: $STC_REASON"
        RESAMPLE_INPUT="$HMC_INPUT"
    fi
}

epi_mask_usable() {   # MASK
    local volume
    [[ -s "$1" ]] || return 1
    volume="$(fslstats "$1" -V | awk '{print $2}')"
    num_cmp "$volume" ge "$MIN_EPI_MASK_MM3"
}

# 6. bias-corrected boldref and EPI support mask
step_boldref_mask() {
    run N4BiasFieldCorrection -d 3 -i "$W/boldref.nii.gz" -o "$W/boldref_n4.nii.gz" -s 2 -b "[200]"
    require_files "$W/boldref_n4.nii.gz"
    EPI_MASK_USED="$EPI_MASK_METHOD"
    if [[ "$EPI_MASK_METHOD" == synthstrip ]]; then
        if ! run mri_synthstrip -i "$W/boldref_n4.nii.gz" -m "$W/epimask_raw.nii.gz" \
                || ! epi_mask_usable "$W/epimask_raw.nii.gz"; then
            log WARN "$RUN_ID: mri_synthstrip failed or returned an implausibly small mask: falling back to 3dAutomask"
            EPI_MASK_USED=automask
        fi
    fi
    if [[ "$EPI_MASK_USED" == automask ]]; then
        run rm -f "$W/epimask_raw.nii.gz"
        run 3dAutomask -clfrac 0.4 -dilate 1 -prefix "$W/epimask_raw.nii.gz" "$W/boldref_n4.nii.gz"
    fi
    run fslmaths "$W/epimask_raw.nii.gz" -bin "$W/epimask.nii.gz" -odt char
    run fslmaths "$W/epimask.nii.gz" -dilM "$W/epimask_dil.nii.gz" -odt char
    # field of view of the acquisition (for an overlap measure that ignores truncation)
    run fslmaths "$W/boldref.nii.gz" -mul 0 -add 1 "$W/fov.nii.gz" -odt float
}

bbr_acceptable() {   # uses BBR_COST BBR_DISP
    if ! is_number "$BBR_COST" || ! is_number "$BBR_DISP"; then
        log WARN "$RUN_ID: BBR cost ('$BBR_COST') or displacement ('$BBR_DISP') could not be read: BBR rejected"
        return 1
    fi
    if num_cmp "$BBR_DISP" gt "$BBR_MAX_DISP_MM"; then
        log WARN "$RUN_ID: BBR moved the initialisation by $BBR_DISP mm (> BBR_MAX_DISP_MM=$BBR_MAX_DISP_MM): BBR rejected"
        return 1
    fi
    if num_cmp "$BBR_COST" gt "$BBR_MAX_COST"; then
        log WARN "$RUN_ID: BBR final cost $BBR_COST (> BBR_MAX_COST=$BBR_MAX_COST): BBR rejected"
        return 1
    fi
    return 0
}

coreg_freesurfer() {
    local chosen="$W/coreg.lta" text mincost
    run mri_coreg --s "$SUB" --mov "$W/boldref_n4.nii.gz" --reg "$W/coreg.lta" --dof 6 --threads "$NTHREADS"
    require_files "$W/coreg.lta"
    COREG_METHOD=mri_coreg
    if run bbregister --s "$SUB" --mov "$W/boldref_n4.nii.gz" --init-reg "$W/coreg.lta" --bold \
            --reg "$W/bbr.dat" --lta "$W/bbr.lta" && [[ -s "$W/bbr.lta" ]]; then
        mincost="$W/bbr.dat.mincost"
        [[ -s "$mincost" ]] || mincost="$W/bbr.lta.mincost"
        if [[ -s "$mincost" ]]; then
            BBR_COST="$(awk 'NR == 1 {print $1}' "$mincost")"
        fi
        text="$(capture lta_diff "$W/coreg.lta" "$W/bbr.lta" --dist 4 || true)"
        BBR_DISP="$(last_number "$text")"
        log INFO "$RUN_ID: bbregister cost=$BBR_COST, max displacement vs mri_coreg=$BBR_DISP mm"
        if bbr_acceptable; then
            chosen="$W/bbr.lta"
            COREG_METHOD=bbregister
        else
            BBR_REJECTED=true
        fi
    else
        log WARN "$RUN_ID: bbregister failed: keeping the mri_coreg registration"
        BBR_REJECTED=true
    fi
    # orig.mgz and desc-preproc_T1w share world coordinates, so the RAS2RAS of the
    # LTA is directly the BOLD->T1w world transform
    run lta_convert --inlta "$chosen" --outitk "$W/bold2t1_itk.txt"
}

coreg_synth() {
    local t1brain="${ANAT_PRE}_desc-brain_T1w.nii.gz" wmseg="${ANAT_PRE}_label-WMbbr_mask.nii.gz"
    local chosen="$W/flirt_init.mat" text
    run fslmaths "$W/boldref_n4.nii.gz" -mas "$W/epimask.nii.gz" "$W/boldref_n4_brain.nii.gz"
    # full search: header-derived alignment is never trusted (DESIGN.md section 11)
    run flirt -in "$W/boldref_n4_brain.nii.gz" -ref "$t1brain" -dof 6 -cost corratio \
        -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -omat "$W/flirt_init.mat"
    require_files "$W/flirt_init.mat"
    COREG_METHOD=flirt
    if run flirt -in "$W/boldref_n4.nii.gz" -ref "$t1brain" -dof 6 -cost bbr -wmseg "$wmseg" \
            -init "$W/flirt_init.mat" -schedule "$FSLDIR/etc/flirtsch/bbr.sch" -omat "$W/flirt_bbr.mat" \
            && [[ -s "$W/flirt_bbr.mat" ]]; then
        text="$(capture flirt -in "$W/boldref_n4.nii.gz" -ref "$t1brain" -init "$W/flirt_bbr.mat" -cost bbr \
            -wmseg "$wmseg" -schedule "$FSLDIR/etc/flirtsch/measurecost1.sch" || true)"
        BBR_COST="$(awk 'NR == 1 {print $1}' <<< "$text")"
        text="$(capture rmsdiff "$W/flirt_init.mat" "$W/flirt_bbr.mat" "$W/boldref_n4.nii.gz" || true)"
        BBR_DISP="$(last_number "$text")"
        log INFO "$RUN_ID: FLIRT-BBR cost=$BBR_COST, RMS deviation vs initialisation=$BBR_DISP mm"
        if bbr_acceptable; then
            chosen="$W/flirt_bbr.mat"
            COREG_METHOD=flirt_bbr
        else
            BBR_REJECTED=true
        fi
    else
        log WARN "$RUN_ID: FLIRT-BBR failed: keeping the 6-dof FLIRT registration"
        BBR_REJECTED=true
    fi
    run wb_command -convert-affine -from-flirt "$chosen" "$W/boldref_n4.nii.gz" "$t1brain" -to-itk "$W/bold2t1_itk.txt"
}

# 7. coregistration + first self-check. A wrong transform direction is silent in
# every later step, so the chain is tested before the expensive resampling.
step_coreg() {
    local shape
    if [[ "$ANAT_MODE" == freesurfer ]]; then
        coreg_freesurfer
    else
        coreg_synth
    fi
    require_files "$W/bold2t1_itk.txt"
    log INFO "$RUN_ID: coreg_method=$COREG_METHOD bbr_rejected=$BBR_REJECTED"

    shape="$(pyval make-grid --mask "${ANAT_PRE}_desc-brain_mask.nii.gz" --res "$FUNC_T1W_RES" \
        --pad-mm "$GRID_PAD_MM" --out "$W/t1w_grid.nii.gz")"
    log INFO "$RUN_ID: T1w-space BOLD grid: $shape voxels at $FUNC_T1W_RES mm"

    anat_to_t1w "${ANAT_PRE}_desc-brain_mask.nii.gz" "$W/t1w_anatmask_prob.nii.gz" Linear
    binarise "$W/t1w_anatmask_prob.nii.gz" "$W/t1w_anatmask.nii.gz" 0.5
    to_t1w "$W/epimask.nii.gz" "$W/t1w_epimask_prob.nii.gz" Linear
    binarise "$W/t1w_epimask_prob.nii.gz" "$W/t1w_epimask.nii.gz" 0.5
    to_t1w "$W/fov.nii.gz" "$W/t1w_fov_prob.nii.gz" Linear
    binarise "$W/t1w_fov_prob.nii.gz" "$W/t1w_fov.nii.gz" 0.5

    COREG_DICE="$(pyval dice --a "$W/t1w_epimask.nii.gz" --b "$W/t1w_anatmask.nii.gz" --within "$W/t1w_fov.nii.gz")"
    log INFO "$RUN_ID: coreg_dice (EPI mask vs T1w brain mask inside the EPI field of view) = $COREG_DICE"
    if ! num_cmp "$COREG_DICE" ge "$MIN_COREG_DICE"; then
        die "$RUN_ID: coreg_dice=$COREG_DICE < $MIN_COREG_DICE: transform chain broken or coregistration failed"
    fi
    if num_cmp "$COREG_DICE" lt "$QC_COREG_DICE_FAIL"; then
        log WARN "$RUN_ID: coreg_dice=$COREG_DICE is below QC_COREG_DICE_FAIL=$QC_COREG_DICE_FAIL: inspect the EPI->T1w registration"
    fi
}

# 8 + 9. per volume: mcflirt matrix -> ITK, one LanczosWindowedSinc interpolation
# into the T1w-space BOLD grid and into the template grid
step_resample() {
    local i n_split
    run mkdir -p "$W/vols"
    # uncompressed per-volume files: gzip would dominate the run time
    run env FSLOUTPUTTYPE=NIFTI fslsplit "$RESAMPLE_INPUT" "$W/vols/vol_" -t
    n_split="$(find "$W/vols" -maxdepth 1 -type f -name 'vol_*.nii' | wc -l)"
    [[ "$n_split" -eq "$N_VOLS" ]] || die "$RUN_ID: fslsplit wrote $n_split volumes, expected $N_VOLS"
    for (( i = 0; i < N_VOLS; i++ )); do
        printf '%04d\n' "$i"
    done > "$W/vol_ids.txt"

    export RV_DIR="$W/vols" RV_MATDIR="$W/pass2.mat" RV_BOLDREF="$W/boldref.nii.gz" \
        RV_T1W_GRID="$W/t1w_grid.nii.gz" RV_TPL_GRID="$TPL_GRID" RV_BOLD2T1="$W/bold2t1_itk.txt" \
        RV_WARP="$XFM_WARP" RV_AFFINE="$XFM_AFFINE"
    export -f resample_volume
    log INFO "$RUN_ID: resampling $N_VOLS volumes, $NTHREADS in parallel"
    # xargs exits 123 when any volume failed: the run fails
    run xargs -a "$W/vol_ids.txt" -P "$NTHREADS" -n 1 \
        bash -c 'set -euo pipefail; resample_volume "$1"' resample_volume

    to_t1w "$W/boldref_n4.nii.gz" "$W/t1w_boldref_unscaled.nii.gz" LanczosWindowedSinc
    to_tpl "$W/boldref_n4.nii.gz" "$W/mni_boldref_unscaled.nii.gz" LanczosWindowedSinc

    merge_series t1w "$W/t1w_merged.nii.gz"
    run fslmaths "$W/t1w_merged.nii.gz" -thr 0 "$W/t1w_clipped.nii.gz" -odt float
    run rm -f "$W/t1w_merged.nii.gz"
    run fslmaths "$W/t1w_clipped.nii.gz" -Tmean "$W/t1w_mean_unscaled.nii.gz" -odt float
    run fslmaths "$W/t1w_clipped.nii.gz" -Tmin -bin "$W/t1w_extents.nii.gz" -odt char
}

# make_tissue_mask LABEL THRESHOLD... : first threshold giving >= MIN_TISSUE_VOX
# voxels; sets TISSUE_N / TISSUE_THR and RELAXED when the first one was not enough.
make_tissue_mask() {
    local label="$1" first="$2" thr
    shift
    anat_to_t1w "${ANAT_PRE}_label-${label}_mask.nii.gz" "$W/t1w_label-${label}_prob.nii.gz" Linear
    for thr in "$@"; do
        run rm -f "$W/t1w_label-${label}.nii.gz"
        run fslmaths "$W/t1w_label-${label}_prob.nii.gz" -thr "$thr" -bin -mas "$W/t1w_brainmask.nii.gz" \
            "$W/t1w_label-${label}.nii.gz" -odt char
        TISSUE_N="$(mask_count "$W/t1w_label-${label}.nii.gz")"
        TISSUE_THR="$thr"
        log INFO "$RUN_ID: $label mask at threshold $thr: $TISSUE_N voxels on the BOLD grid"
        if [[ "$TISSUE_N" -ge "$MIN_TISSUE_VOX" ]]; then
            break
        fi
    done
    if [[ "$TISSUE_THR" != "$first" ]]; then
        RELAXED=true
        log WARN "$RUN_ID: $label mask needed a relaxed threshold ($TISSUE_THR instead of $first) to reach MIN_TISSUE_VOX=$MIN_TISSUE_VOX"
    fi
    if [[ "$TISSUE_N" -lt "$MIN_TISSUE_VOX" ]]; then
        log WARN "$RUN_ID: $label mask has only $TISSUE_N voxels even at threshold $TISSUE_THR: its nuisance signal is unreliable"
    fi
}

# 10a. T1w-space masks, scale factor
step_masks_t1w() {
    to_t1w "$W/epimask_dil.nii.gz" "$W/t1w_epimask_dil_prob.nii.gz" Linear
    binarise "$W/t1w_epimask_dil_prob.nii.gz" "$W/t1w_epimask_dil.nii.gz" 0.5
    run fslmaths "$W/t1w_anatmask.nii.gz" -mas "$W/t1w_epimask_dil.nii.gz" -mas "$W/t1w_extents.nii.gz" -bin \
        "$W/t1w_brainmask.nii.gz" -odt char
    BRAIN_N="$(mask_count "$W/t1w_brainmask.nii.gz")"
    [[ "$BRAIN_N" -gt 0 ]] || die "$RUN_ID: the T1w-space brain mask is empty"

    make_tissue_mask WM 0.9 0.7 0.5
    WM_N="$TISSUE_N"; WM_THR="$TISSUE_THR"
    make_tissue_mask CSF 0.9 0.7 0.5
    CSF_N="$TISSUE_N"; CSF_THR="$TISSUE_THR"
    make_tissue_mask GM 0.5
    GM_N="$TISSUE_N"

    SCALE="$(pyval scale-factor --mean "$W/t1w_mean_unscaled.nii.gz" --mask "$W/t1w_brainmask.nii.gz" --target "$SCALE_TARGET")"
    num_cmp "$SCALE" gt 0 || die "$RUN_ID: unusable scale factor '$SCALE'"
    log INFO "$RUN_ID: global scale factor $SCALE (in-brain median of the mean -> $SCALE_TARGET)"
}

# 11. second self-check: the resampled series must reproduce the reference, and
# applying the motion matrices must not make the worst volume less similar to it.
step_check_hmc() {
    local line worst_idx worst_id r_with r_without
    # like for like: the un-bias-corrected reference through the same chain
    to_t1w "$W/boldref.nii.gz" "$W/t1w_boldref_raw.nii.gz" LanczosWindowedSinc
    HMC_R="$(pyval masked-corr --a "$W/t1w_mean_unscaled.nii.gz" --b "$W/t1w_boldref_raw.nii.gz" --mask "$W/t1w_brainmask.nii.gz")"
    log INFO "$RUN_ID: hmc_consistency_r (mean of the resampled series vs resampled boldref) = $HMC_R"
    if ! num_cmp "$HMC_R" ge "$MIN_HMC_CONSISTENCY"; then
        die "$RUN_ID: hmc_consistency_r=$HMC_R < $MIN_HMC_CONSISTENCY: HMC matrices applied in the wrong direction?"
    fi

    line="$(pyval max-index --file "$W/pass2_abs.rms")"
    read -r worst_idx HMC_WORST_MM <<< "$line"
    if ! num_cmp "$HMC_WORST_MM" ge "$HMC_DIRECTION_MIN_MM"; then
        log INFO "$RUN_ID: largest displacement $HMC_WORST_MM mm: too little motion for the HMC direction test"
        return 0
    fi
    worst_id="$(printf '%04d' "$worst_idx")"
    run antsApplyTransforms -d 3 --float -n LanczosWindowedSinc -i "$W/vols/vol_${worst_id}.nii" \
        -r "$W/t1w_grid.nii.gz" -o "$W/t1w_worst_nohmc.nii.gz" -t "$W/bold2t1_itk.txt"
    r_with="$(pyval masked-corr --a "$W/vols/t1w_${worst_id}.nii" --b "$W/t1w_boldref_raw.nii.gz" --mask "$W/t1w_brainmask.nii.gz")"
    r_without="$(pyval masked-corr --a "$W/t1w_worst_nohmc.nii.gz" --b "$W/t1w_boldref_raw.nii.gz" --mask "$W/t1w_brainmask.nii.gz")"
    if is_number "$r_with" && is_number "$r_without"; then
        HMC_GAIN="$(awk -v a="$r_with" -v b="$r_without" 'BEGIN { printf "%.6f", a - b }')"
    fi
    log INFO "$RUN_ID: volume $worst_idx ($HMC_WORST_MM mm): r with HMC=$r_with, without=$r_without, gain=$HMC_GAIN"
    if num_cmp "$HMC_GAIN" lt "-$HMC_DIRECTION_TOL"; then
        die "$RUN_ID: motion correction makes volume $worst_idx LESS similar to the reference (gain $HMC_GAIN): HMC matrices applied in the wrong direction?"
    fi
}

# 9b + 10b. scaled outputs of both spaces, template-space masks
step_finalise_series() {
    run fslmaths "$W/t1w_clipped.nii.gz" -mul "$SCALE" "$W/t1w_bold.nii.gz" -odt float
    run rm -f "$W/t1w_clipped.nii.gz"
    run fslmaths "$W/t1w_boldref_unscaled.nii.gz" -thr 0 -mul "$SCALE" "$W/t1w_boldref.nii.gz" -odt float

    merge_series mni "$W/mni_merged.nii.gz"
    # clip + scale in one pass: the template-space series is the big one
    run fslmaths "$W/mni_merged.nii.gz" -thr 0 -mul "$SCALE" "$W/mni_bold.nii.gz" -odt float
    run rm -f "$W/mni_merged.nii.gz"
    run fslmaths "$W/mni_boldref_unscaled.nii.gz" -thr 0 -mul "$SCALE" "$W/mni_boldref.nii.gz" -odt float

    run fslmaths "$W/mni_bold.nii.gz" -Tmin -bin "$W/mni_extents.nii.gz" -odt char
    anat_to_tpl "${ANAT_PRE}_desc-brain_mask.nii.gz" "$W/mni_anatmask_prob.nii.gz" Linear
    binarise "$W/mni_anatmask_prob.nii.gz" "$W/mni_anatmask.nii.gz" 0.5
    to_tpl "$W/epimask_dil.nii.gz" "$W/mni_epimask_dil_prob.nii.gz" Linear
    binarise "$W/mni_epimask_dil_prob.nii.gz" "$W/mni_epimask_dil.nii.gz" 0.5
    run fslmaths "$W/mni_anatmask.nii.gz" -mas "$W/mni_epimask_dil.nii.gz" -mas "$W/mni_extents.nii.gz" -bin \
        "$W/mni_brainmask.nii.gz" -odt char
    [[ "$(mask_count "$W/mni_brainmask.nii.gz")" -gt 0 ]] || die "$RUN_ID: the template-space brain mask is empty"
}

# 12. provenance + final files
step_write_outputs() {
    local tpl="space-${TEMPLATE_NAME}_res-${MNI_RES}" quality_mean outlier_mean
    quality_mean="$(pyval mean-1d --file "$W/quality.1D")"
    outlier_mean="$(pyval mean-1d --file "$W/outliers.1D")"
    fp_tool_versions "$W/tool_versions.json"
    run "$PYTHON_BIN" -m fmriproc.prep_utils write-info --out "$W/prep_info.json" \
        --versions-json "$W/tool_versions.json" --geometry-from "$W/boldref.nii.gz" \
        --set-str "source=$BOLD" --set-str "run_label=$RUN_ID" \
        --set "tr=$TR" --set "n_volumes_raw=$N_RAW" --set "n_dropped=$N_DROP" --set "n_volumes=$N_VOLS" \
        --set "nss_detected=$NSS" --set "despike=$DESPIKED" --set "despike_fraction=$DESPIKE_FRAC" \
        --set "stc_applied=$STC_APPLIED" --set-str "stc_reason=$STC_REASON" --set-str "stc_interp=$STC_INTERP" \
        --set-str "slice_timing_source=$ST_SOURCE" --set-str "slice_timing_evidence=$ST_EVIDENCE" \
        --set "tzero=$TZERO" \
        --set-str "hmc_reference=median of mcflirt pass 1 (initial reference: volume $REF0_IDX, lowest outlier fraction)" \
        --set-str "coreg_method=$COREG_METHOD" --set "bbr_cost=$BBR_COST" --set "bbr_vs_init_mm=$BBR_DISP" \
        --set "bbr_rejected=$BBR_REJECTED" --set-str "epi_mask_method=$EPI_MASK_USED" \
        --set "scale_factor=$SCALE" --set "func_t1w_res=$FUNC_T1W_RES" --set "mni_res=$MNI_RES" \
        --set-str "template=$TEMPLATE_NAME" \
        --set "wm_mask_voxels=$WM_N" --set "csf_mask_voxels=$CSF_N" --set "gm_mask_voxels=$GM_N" \
        --set "tissue_erosion_relaxed=$RELAXED" --set "coreg_dice=$COREG_DICE" --set "hmc_consistency_r=$HMC_R" \
        --set "hmc_direction_gain=$HMC_GAIN" --set "hmc_max_abs_rms_mm=$HMC_WORST_MM" \
        --set "wm_mask_threshold=$WM_THR" --set "csf_mask_threshold=$CSF_THR" \
        --set "brain_mask_voxels=$BRAIN_N" --set "quality_index_mean=$quality_mean" \
        --set "outlier_frac_mean=$outlier_mean" --set-str "anat_mode=$ANAT_MODE" \
        --set "scale_target=$SCALE_TARGET" --set-str "pipeline_version=$PIPELINE_VERSION"
    require_files "$W/prep_info.json"

    install_copy "$W/pass2.par"        "${PRE}_desc-hmc_motion.par"
    install_copy "$W/pass2_rel.rms"    "${PRE}_desc-hmc_relrms.txt"
    install_copy "$W/pass2_abs.rms"    "${PRE}_desc-hmc_absrms.txt"
    install_copy "$W/outliers.1D"      "${PRE}_desc-outliers_timeseries.1D"
    install_copy "$W/quality.1D"       "${PRE}_desc-quality_timeseries.1D"
    install_copy "$W/bold2t1_itk.txt"  "${PRE}_from-bold_to-T1w_itk.txt"
    install "$W/t1w_boldref.nii.gz"    "${PRE}_space-T1w_boldref.nii.gz"
    install "$W/t1w_brainmask.nii.gz"  "${PRE}_space-T1w_desc-brain_mask.nii.gz"
    install "$W/t1w_label-WM.nii.gz"   "${PRE}_space-T1w_label-WM_mask.nii.gz"
    install "$W/t1w_label-CSF.nii.gz"  "${PRE}_space-T1w_label-CSF_mask.nii.gz"
    install "$W/t1w_label-GM.nii.gz"   "${PRE}_space-T1w_label-GM_mask.nii.gz"
    install "$W/t1w_bold.nii.gz"       "${PRE}_space-T1w_desc-preproc_bold.nii.gz"
    install "$W/mni_boldref.nii.gz"    "${PRE}_${tpl}_boldref.nii.gz"
    install "$W/mni_brainmask.nii.gz"  "${PRE}_${tpl}_desc-brain_mask.nii.gz"
    install "$W/mni_bold.nii.gz"       "${PRE}_${tpl}_desc-preproc_bold.nii.gz"
    # last: its presence means the whole set is in place
    install "$W/prep_info.json"        "${PRE}_desc-prep_info.json"
}

cleanup_work() {
    if is_yes "$KEEP_WORK"; then
        return 0
    fi
    # per-volume temporaries and 4D intermediates; small files stay for debugging
    run rm -rf "$W/vols"
    run rm -f "$W/bold_dropped.nii.gz" "$W/bold_despiked.nii.gz" "$W/bold_stc.nii.gz" \
        "$W/t1w_merged.nii.gz" "$W/t1w_clipped.nii.gz" "$W/mni_merged.nii.gz"
}

# ----------------------------- one run ---------------------------------------

process_run() {   # RUN_LABEL BOLD
    RUN_ID="$1"
    BOLD="$2"
    set_run_paths

    if ! stage_should_run "$STAGE" "$SUB" "$RUN_ID" --dep 02_anat_prep -- "${HASH_VARS[@]}"; then
        if outputs_complete; then
            return 0
        fi
        log WARN "marker of $STAGE $RUN_ID is up to date but outputs are missing: running again"
    fi
    if is_yes "$DRY_RUN"; then
        # every step depends on the files of the previous one: nothing to list per command
        log INFO "DRY_RUN: would preprocess $RUN_ID ($BOLD) in $W"
        return 0
    fi

    check_inputs
    # neither a stale marker nor stale outputs may survive a failed recomputation
    rm -f "$WORK_DIR/$SUB/.done/${STAGE}__${RUN_ID}.hash" "${FINAL_FILES[@]}"
    run rm -rf "$W"
    run mkdir -p "$W" "$(func_dir "$SUB")"

    step_drop
    step_raw_qc
    step_despike
    step_hmc
    step_stc
    step_boldref_mask
    step_coreg
    step_resample
    step_masks_t1w
    step_check_hmc
    step_finalise_series
    step_write_outputs
    cleanup_work
    stage_mark_done "$STAGE" "$SUB" "$RUN_ID"
}

# ----------------------------- main ------------------------------------------

main() {
    local rows=() row run_id bold failed=()
    check_config
    mapfile -t rows < <(manifest_runs "$SUB")
    [[ ${#rows[@]} -gt 0 ]] || die "no runs for $SUB in $MANIFEST"

    if ! is_yes "$DRY_RUN"; then
        require_cmds 3dcalc 3dToutcount 3dTqual 3dDespike 3dTstat 3dTshift 3dAutomask mcflirt fslroi fslsplit \
            fslmerge fslmaths fslstats fslnvols fslval N4BiasFieldCorrection antsApplyTransforms wb_command xargs
        if [[ "$ANAT_MODE" == freesurfer ]]; then
            require_cmds mri_coreg bbregister lta_diff lta_convert
        else
            require_cmds flirt rmsdiff
        fi
        if [[ "$ANAT_MODE" == freesurfer || "$EPI_MASK_METHOD" == synthstrip ]]; then
            check_license
        fi
        if [[ "$EPI_MASK_METHOD" == synthstrip ]]; then
            require_cmds mri_synthstrip
        fi
        fp_check_mem
        TPL_GRID="$(template_path brain_res)"
        require_files "$TPL_GRID"
    fi

    for row in "${rows[@]}"; do
        row="${row%$'\r'}"
        bold="$(awk -F'\t' '{print $6}' <<< "$row")"
        run_id="$(awk -F'\t' '{print $8}' <<< "$row")"
        [[ -n "$run_id" && -n "$bold" ]] || die "malformed manifest row for $SUB: $row"
        guarded process_run "$run_id" "$bold"
        if [[ $GUARD_RC -ne 0 ]]; then
            log ERROR "$STAGE failed for $run_id (exit $GUARD_RC)"
            failed+=("$run_id")
        fi
    done

    if [[ ${#failed[@]} -gt 0 ]]; then
        die "$STAGE: ${#failed[@]} of ${#rows[@]} runs failed: ${failed[*]}"
    fi
    log OK "$STAGE finished for $SUB (${#rows[@]} runs)"
}

main
