#!/bin/bash
# =============================================================================
# 02_anat_prep.sh [-c conf] sub-X
# T1w reference, brain mask, segmentation, tissue masks, T1w->template (ANTs SyN)
# and anatomical QC. Contract: docs/DESIGN.md sections 6, 7 ("02"), 10, 11.
#
# T1w space = FreeSurfer conformed grid (ANAT_MODE=freesurfer) or the raw T1 grid
# (ANAT_MODE=synth). No fslreorient2std anywhere: every transform of the pipeline
# is a world-coordinate ITK/LTA transform, so voxel storage order is irrelevant
# and FreeSurfer volumes/surfaces stay usable without an extra header transform.
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
fp_init "$@"
SUB="${FP_ARGS[0]:?usage: $0 [-c conf] sub-XXXX}"
STAGE=02_anat_prep
fp_set_log "$SUB" "$STAGE"

A="$(anat_dir "$SUB")"
P="$A/$SUB"
W="$(work_anat "$SUB")/$STAGE"
XFM_PREFIX="$W/T1w_to_MNI_"

# aseg / SynthSeg label codes (DESIGN.md section 7)
WM_LABELS=(2 41)
CSF_LABELS=(4 43)
GM_LABELS=(3 42 8 47 10 11 12 13 17 18 26 28 49 50 51 52 53 54 58 60)
# BBR needs every white-matter boundary: aseg splits the corpus callosum
# (251-255) and hypointensities (77) off labels 2/41, and cerebellar WM /
# brainstem border grey matter as well.
WMBBR_LABELS=(2 41 7 46 16 77 251 252 253 254 255)

MIN_WM_VOX_1MM=1000
MIN_CSF_VOX_1MM=50
SYNTHSEG_PAD_MM=15    # margin around the brain box handed to SynthSeg

SEG=""              # label volume on a 1 mm grid (aseg.mgz | SynthSeg output)
SEG_ON_T1_GRID=yes  # no = masks must be resampled to the T1 grid (synth mode)
T1_SOURCE=""
CSF_ERODE_USED="$CSF_ERODE"

FINAL_FILES=(
    "${P}_desc-preproc_T1w.nii.gz"
    "${P}_desc-brain_mask.nii.gz"
    "${P}_desc-brain_T1w.nii.gz"
    "${P}_desc-aseg_dseg.nii.gz"
    "${P}_label-WM_mask.nii.gz"
    "${P}_label-CSF_mask.nii.gz"
    "${P}_label-GM_mask.nii.gz"
    "${P}_label-WMbbr_mask.nii.gz"
    "$A/xfm/T1w_to_MNI_0GenericAffine.mat"
    "$A/xfm/T1w_to_MNI_1Warp.nii.gz"
    "$A/xfm/T1w_to_MNI_1InverseWarp.nii.gz"
    "${P}_space-${TEMPLATE_NAME}_desc-preproc_T1w.nii.gz"
    "${P}_space-${TEMPLATE_NAME}_desc-brain_mask.nii.gz"
    "${P}_desc-anatqc.json"
)

# ----------------------------- helpers ---------------------------------------

outputs_complete() {
    local f
    for f in "${FINAL_FILES[@]}"; do
        [[ -s "$f" ]] || return 1
    done
    return 0
}

check_license() {
    local home="${FREESURFER_HOME:-/opt/freesurfer}"
    # FreeSurfer itself also accepts $FREESURFER_HOME/license.txt and .license
    if [[ -n "${FS_LICENSE:-}" && -s "${FS_LICENSE:-}" ]] || [[ -s "$home/license.txt" || -s "$home/.license" ]]; then
        return 0
    fi
    # SynthStrip/SynthSeg run unlicensed; the classic FreeSurfer binaries may not
    if [[ "$ANAT_MODE" == freesurfer ]]; then
        die "FreeSurfer license not found (FS_LICENSE='${FS_LICENSE:-}'). Mount license.txt into the container and export FS_LICENSE."
    fi
    log WARN "FreeSurfer license not found (FS_LICENSE='${FS_LICENSE:-}'): mri_binarize may refuse to run"
}

# require_inputs FILE... : like require_files, but a dry run only reports
require_inputs() {
    local f
    if ! is_yes "$DRY_RUN"; then
        require_files "$@"
        return 0
    fi
    for f in "$@"; do
        [[ -s "$f" ]] || log WARN "dry run: input does not exist yet: $f"
    done
}

check_config() {
    case "$NORM_QUALITY" in precise|quick) ;; *) die "NORM_QUALITY must be precise or quick: $NORM_QUALITY" ;; esac
    [[ "$WM_ERODE" =~ ^[0-9]+$ ]] || die "WM_ERODE must be a non-negative integer: $WM_ERODE"
    [[ "$CSF_ERODE" =~ ^[0-9]+$ ]] || die "CSF_ERODE must be a non-negative integer: $CSF_ERODE"
    [[ "$ANTS_SEED" =~ ^[0-9]+$ ]] || die "ANTS_SEED must be a non-negative integer: $ANTS_SEED"
    # antsRegistrationSyN*.sh treats -e 0 as "no fixed seed"
    [[ "$ANTS_SEED" != 0 ]] || log WARN "ANTS_SEED=0: the T1w->template registration is not reproducible"
    if [[ "$TEMPLATE_NAME" != MNI152NLin6Asym ]]; then
        log WARN "TEMPLATE_NAME=$TEMPLATE_NAME is only a label: the template files are FSL's MNI152 (MNI152NLin6Asym) from \$FSLDIR/data/standard"
    fi
}

# install_file SRC DST : DST appears only complete (mv to a hidden name in the
# target directory first, because work/ and derivatives/ may be different filesystems).
install_file() {
    local src="$1" dst="$2" tmp
    tmp="$(dirname "$dst")/.tmp.$$.$(basename "$dst")"
    run mv -f "$src" "$tmp"
    run mv -f "$tmp" "$dst"
}

tissue_count() {
    if is_yes "$DRY_RUN"; then
        echo 999999
    else
        mask_count "$1"
    fi
}

# binarize_labels OUT ERODE LABEL... : mask of LABELs of $SEG, eroded ERODE times
# on the 1 mm segmentation grid (iterations ~ millimetres).
binarize_labels() {
    local out="$1" erode="$2"
    shift 2
    local args=(--i "$SEG" --match "$@")
    if [[ "$erode" -gt 0 ]]; then
        args+=(--erode "$erode")
    fi
    run mri_binarize "${args[@]}" --uchar --o "$out"
}

# to_t1_grid IN OUT : binary mask from the segmentation grid to the T1w grid
to_t1_grid() {
    local in="$1" out="$2"
    if is_yes "$SEG_ON_T1_GRID"; then
        run mv -f "$in" "$out"
    else
        run antsApplyTransforms -d 3 -i "$in" -r "$W/T1w.nii.gz" -o "$out" \
            -n GenericLabel -t identity -u uchar
    fi
}

# ----------------------------- T1w reference ---------------------------------

prep_freesurfer() {
    local mri="$FS_DIR/$SUB/mri"
    require_inputs "$mri/nu.mgz" "$mri/brainmask.mgz" "$mri/aseg.mgz"
    T1_SOURCE="$mri/nu.mgz"
    SEG="$mri/aseg.mgz"
    SEG_ON_T1_GRID=yes
    run mri_convert "$mri/nu.mgz" "$W/T1w.nii.gz"
    run mri_convert "$mri/brainmask.mgz" "$W/fs_brainmask.nii.gz"
    # brainmask.mgz is an intensity image with zero-valued CSF pockets inside
    run fslmaths "$W/fs_brainmask.nii.gz" -bin -fillh "$W/brain_mask.nii.gz" -odt char
    run mri_convert -rt nearest "$mri/aseg.mgz" "$W/dseg.nii.gz"
}

# crop_to_mask IMG MASK PAD_MM OUT : crop IMG to the bounding box of MASK
# (same grid) plus PAD_MM on every side. fslroi only moves the origin, so
# world coordinates - and everything later resampled with them - are unchanged.
crop_to_mask() {
    local img="$1" mask="$2" pad_mm="$3" out="$4" i dim pix pad lo hi
    local box=() args=()
    if is_yes "$DRY_RUN"; then
        run fslroi "$img" "$out" 0 -1 0 -1 0 -1
        return 0
    fi
    read -r -a box <<< "$(fslstats "$mask" -w)"   # xmin xsize ymin ysize zmin zsize tmin tsize
    if [[ ${#box[@]} -lt 6 || "${box[1]}" -le 0 || "${box[3]}" -le 0 || "${box[5]}" -le 0 ]]; then
        die "brain mask $mask is empty: skull stripping failed"
    fi
    for i in 0 1 2; do
        dim="$(fslval "$img" "dim$((i + 1))" | tr -d ' ')"
        pix="$(fslval "$img" "pixdim$((i + 1))" | tr -d ' ')"
        pad="$(awk -v mm="$pad_mm" -v d="$pix" 'BEGIN { v = mm / d; printf "%d", (v > int(v)) ? int(v) + 1 : v }')"
        lo=$(( box[2 * i] - pad ))
        hi=$(( box[2 * i] + box[2 * i + 1] + pad ))
        if [[ "$lo" -lt 0 ]]; then lo=0; fi
        if [[ "$hi" -gt "$dim" ]]; then hi="$dim"; fi
        args+=("$lo" "$(( hi - lo ))")
    done
    log INFO "SynthSeg input: brain box + ${pad_mm} mm = ${args[1]}x${args[3]}x${args[5]} voxels"
    run fslroi "$img" "$out" "${args[@]}"
}

prep_synth() {
    local t1
    t1="$(subject_t1w "$SUB" || true)"
    [[ -n "$t1" ]] || die "no T1w listed for $SUB in $MANIFEST"
    require_inputs "$t1"
    T1_SOURCE="$t1"
    run N4BiasFieldCorrection -d 3 -i "$t1" -o "$W/T1w.nii.gz" -s 4 -b "[200]" -c "[50x50x50x50,0.0000001]"
    run mri_synthstrip -i "$W/T1w.nii.gz" -m "$W/synthstrip_mask.nii.gz"
    run fslmaths "$W/synthstrip_mask.nii.gz" -bin -fillh "$W/brain_mask.nii.gz" -odt char
    # SynthSeg always answers on a 1 mm grid: erosion happens there, then the
    # labels and masks are brought to the T1 grid.
    SEG="$W/synthseg_1mm.nii.gz"
    SEG_ON_T1_GRID=no
    # SynthSeg segments the whole field of view it is given: a head FOV padded
    # to 192x256x256 exhausts 12 GB of RAM with --robust. The brain box plus a
    # margin keeps the brain and its surroundings at ~40% of the voxels.
    crop_to_mask "$W/T1w.nii.gz" "$W/brain_mask.nii.gz" "$SYNTHSEG_PAD_MM" "$W/T1w_synthseg_input.nii.gz"
    local flags=()
    read -r -a flags <<< "$SYNTHSEG_FLAGS"
    run mri_synthseg --i "$W/T1w_synthseg_input.nii.gz" --o "$SEG" ${flags[@]+"${flags[@]}"} \
        --threads "$NTHREADS" --cpu --vol "$W/synthseg_volumes.csv" --qc "$W/synthseg_qc.csv"
    run antsApplyTransforms -d 3 -i "$SEG" -r "$W/T1w.nii.gz" -o "$W/dseg.nii.gz" \
        -n GenericLabel -t identity -u short
}

# ----------------------------- tissue masks ----------------------------------

make_wm_mask() {
    local raw="$W/seggrid_label-WM.nii.gz" n
    binarize_labels "$raw" "$WM_ERODE" "${WM_LABELS[@]}"
    n="$(tissue_count "$raw")"
    log INFO "WM mask (erode $WM_ERODE): $n voxels at 1 mm"
    if [[ "$n" -lt "$MIN_WM_VOX_1MM" ]]; then
        die "eroded WM mask has only $n voxels (< $MIN_WM_VOX_1MM): the segmentation of $SUB failed or WM_ERODE=$WM_ERODE is too large"
    fi
    to_t1_grid "$raw" "$W/label-WM.nii.gz"
}

# Small (paediatric) ventricles can vanish under erosion: back off stepwise.
make_csf_mask() {
    local raw="$W/seggrid_label-CSF.nii.gz" erode="$CSF_ERODE" n
    while :; do
        binarize_labels "$raw" "$erode" "${CSF_LABELS[@]}"
        n="$(tissue_count "$raw")"
        log INFO "CSF mask (erode $erode): $n voxels at 1 mm"
        if [[ "$n" -ge "$MIN_CSF_VOX_1MM" || "$erode" -le 0 ]]; then
            break
        fi
        log WARN "eroded CSF mask has only $n voxels (< $MIN_CSF_VOX_1MM): retrying with erode $((erode - 1))"
        erode=$((erode - 1))
    done
    if [[ "$n" -lt "$MIN_CSF_VOX_1MM" ]]; then
        log WARN "lateral-ventricle mask has only $n voxels even without erosion: CSF nuisance signal of $SUB will be unreliable"
    fi
    CSF_ERODE_USED="$erode"
    to_t1_grid "$raw" "$W/label-CSF.nii.gz"
}

make_plain_mask() {   # NAME LABEL...
    local name="$1"
    shift
    binarize_labels "$W/seggrid_label-$name.nii.gz" 0 "$@"
    to_t1_grid "$W/seggrid_label-$name.nii.gz" "$W/label-$name.nii.gz"
}

# ----------------------------- normalisation ---------------------------------

normalise() {
    local script=antsRegistrationSyN.sh
    if [[ "$NORM_QUALITY" == quick ]]; then
        script=antsRegistrationSyNQuick.sh
    fi
    require_cmds "$script"
    # -p f: float precision halves the memory of the 1 mm SyN stage
    run "$script" -d 3 -f "$TPL_BRAIN" -m "$W/brain_T1w.nii.gz" -o "$XFM_PREFIX" \
        -t s -n "$NTHREADS" -p f -e "$ANTS_SEED"
    is_yes "$DRY_RUN" || require_files "${XFM_PREFIX}0GenericAffine.mat" "${XFM_PREFIX}1Warp.nii.gz" \
        "${XFM_PREFIX}1InverseWarp.nii.gz" "${XFM_PREFIX}Warped.nii.gz"
    run antsApplyTransforms -d 3 -i "$W/brain_mask.nii.gz" -r "$TPL_BRAIN" -o "$W/tpl_brain_mask.nii.gz" \
        -n GenericLabel -t "${XFM_PREFIX}1Warp.nii.gz" -t "${XFM_PREFIX}0GenericAffine.mat" -u uchar
    # Determinant (doLog=0) from finite differences (useGeometric=0): signed by
    # construction, so folding shows up as values <= 0. 1Warp lives on the
    # template grid, and so does the Jacobian.
    run CreateJacobianDeterminantImage 3 "${XFM_PREFIX}1Warp.nii.gz" "$W/jacobian.nii.gz" 0 0
}

# ----------------------------- QC --------------------------------------------

# surface_euler HEMI : prints "<euler> <holes>" of ?h.orig.nofix, or "n/a n/a".
# mris_euler_number reports on stderr: "euler # = v-e+f = 2g-2: ... = -116 --> 59 holes"
surface_euler() {
    local surf="$FS_DIR/$SUB/surf/$1.orig.nofix" text
    # holes = (2 - euler) / 2 can be printed negative for a disconnected surface
    local re='= *(-?[0-9]+) *--> *(-?[0-9]+) +holes'
    if [[ "$ANAT_MODE" != freesurfer || ! -f "$surf" ]]; then
        echo "n/a n/a"
        return 0
    fi
    text="$(mris_euler_number "$surf" 2>&1 || true)"
    if [[ "$text" =~ $re ]]; then
        echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
    else
        log WARN "could not parse the mris_euler_number output for $surf"
        echo "n/a n/a"
    fi
}

anat_qc() {
    local euler_lh holes_lh euler_rh holes_rh
    read -r euler_lh holes_lh <<< "$(surface_euler lh)"
    read -r euler_rh holes_rh <<< "$(surface_euler rh)"
    log INFO "Euler number lh=$euler_lh ($holes_lh holes) rh=$euler_rh ($holes_rh holes)"
    pyrun anat_qc \
        --anat-mode "$ANAT_MODE" --norm-quality "$NORM_QUALITY" \
        --brain-mask "$W/brain_mask.nii.gz" \
        --wm-mask "$W/label-WM.nii.gz" --csf-mask "$W/label-CSF.nii.gz" --gm-mask "$W/label-GM.nii.gz" \
        --warped-brain "${XFM_PREFIX}Warped.nii.gz" --warped-mask "$W/tpl_brain_mask.nii.gz" \
        --template-brain "$TPL_BRAIN" --template-mask "$TPL_MASK" \
        --jacobian "$W/jacobian.nii.gz" \
        --euler-lh="$euler_lh" --euler-rh="$euler_rh" --holes-lh="$holes_lh" --holes-rh="$holes_rh" \
        --template-name "$TEMPLATE_NAME" --t1w-source "$T1_SOURCE" \
        --wm-erode "$WM_ERODE" --csf-erode "$CSF_ERODE_USED" --csf-erode-requested "$CSF_ERODE" \
        --pipeline-version "$PIPELINE_VERSION" \
        --out "$W/anatqc.json"
}

report_qc() {
    local json="$W/anatqc.json" dice nonpos holes
    is_yes "$DRY_RUN" && return 0
    dice="$(json_get "$json" norm_dice nan)"
    nonpos="$(json_get "$json" jacobian_nonpos_frac nan)"
    holes="$(json_get "$json" holes_total n/a)"
    log INFO "anat QC: norm_dice=$dice template_corr=$(json_get "$json" template_corr nan) jacobian_nonpos_frac=$nonpos holes_total=$holes"
    if awk -v d="$dice" -v t="$QC_NORM_DICE_FAIL" 'BEGIN { exit !(d == "nan" || d + 0 < t + 0) }'; then
        log WARN "T1w->template Dice $dice is below QC_NORM_DICE_FAIL=$QC_NORM_DICE_FAIL: inspect the normalisation of $SUB"
    fi
    if awk -v f="$nonpos" 'BEGIN { exit !(f != "nan" && f + 0 > 0) }'; then
        log WARN "the T1w->template warp folds inside the template brain (fraction $nonpos)"
    fi
    return 0
}

install_outputs() {
    run mkdir -p "$A/xfm"
    install_file "$W/T1w.nii.gz"        "${P}_desc-preproc_T1w.nii.gz"
    install_file "$W/brain_mask.nii.gz" "${P}_desc-brain_mask.nii.gz"
    install_file "$W/brain_T1w.nii.gz"  "${P}_desc-brain_T1w.nii.gz"
    install_file "$W/dseg.nii.gz"       "${P}_desc-aseg_dseg.nii.gz"
    install_file "$W/label-WM.nii.gz"    "${P}_label-WM_mask.nii.gz"
    install_file "$W/label-CSF.nii.gz"   "${P}_label-CSF_mask.nii.gz"
    install_file "$W/label-GM.nii.gz"    "${P}_label-GM_mask.nii.gz"
    install_file "$W/label-WMbbr.nii.gz" "${P}_label-WMbbr_mask.nii.gz"
    install_file "${XFM_PREFIX}0GenericAffine.mat"   "$A/xfm/T1w_to_MNI_0GenericAffine.mat"
    install_file "${XFM_PREFIX}1Warp.nii.gz"         "$A/xfm/T1w_to_MNI_1Warp.nii.gz"
    install_file "${XFM_PREFIX}1InverseWarp.nii.gz"  "$A/xfm/T1w_to_MNI_1InverseWarp.nii.gz"
    install_file "${XFM_PREFIX}Warped.nii.gz"  "${P}_space-${TEMPLATE_NAME}_desc-preproc_T1w.nii.gz"
    install_file "$W/tpl_brain_mask.nii.gz"    "${P}_space-${TEMPLATE_NAME}_desc-brain_mask.nii.gz"
    # last: its presence means the whole set is in place
    install_file "$W/anatqc.json" "${P}_desc-anatqc.json"
}

cleanup_work() {
    if is_yes "$KEEP_WORK" || is_yes "$DRY_RUN"; then
        return 0
    fi
    # keeps the small text outputs (SynthSeg volumes / QC scores)
    run find "$W" -maxdepth 1 -type f \( -name '*.nii.gz' -o -name '*.mat' \) -delete
}

# ----------------------------- main ------------------------------------------

check_config

DEPS=()
if [[ "$ANAT_MODE" == freesurfer ]]; then
    # a rebuilt reconstruction (new 01 marker) invalidates everything derived from it
    DEPS=(--dep 01_anat_recon)
fi
HASH_VARS=(ANAT_MODE WM_ERODE CSF_ERODE TEMPLATE_NAME NORM_QUALITY ANTS_SEED)
if [[ "$ANAT_MODE" == synth ]]; then
    HASH_VARS+=(SYNTHSEG_FLAGS)
fi
if ! stage_should_run "$STAGE" "$SUB" ${DEPS[@]+"${DEPS[@]}"} -- "${HASH_VARS[@]}"; then
    if outputs_complete; then
        exit 0
    fi
    log WARN "marker of $STAGE is up to date but outputs are missing: running again"
fi

require_cmds mri_binarize fslmaths fslstats antsApplyTransforms CreateJacobianDeterminantImage
if [[ "$ANAT_MODE" == freesurfer ]]; then
    require_cmds mri_convert mris_euler_number
else
    require_cmds N4BiasFieldCorrection mri_synthstrip mri_synthseg fslroi fslval
fi
check_license
is_yes "$DRY_RUN" || fp_check_mem

TPL_BRAIN="$(template_path brain)"
TPL_MASK="$(template_path mask)"
require_inputs "$TPL_BRAIN" "$TPL_MASK"

run rm -rf "$W"
run mkdir -p "$W"
# The QC JSON is installed last and stands for "the set is complete": take it
# away first, so an interrupted rerun never leaves a mix that looks finished.
run rm -f "${P}_desc-anatqc.json"

if [[ "$ANAT_MODE" == freesurfer ]]; then
    prep_freesurfer
else
    prep_synth
fi
run fslmaths "$W/T1w.nii.gz" -mas "$W/brain_mask.nii.gz" "$W/brain_T1w.nii.gz"

make_wm_mask
make_csf_mask
make_plain_mask GM "${GM_LABELS[@]}"
make_plain_mask WMbbr "${WMBBR_LABELS[@]}"

normalise
anat_qc
report_qc
install_outputs
cleanup_work
stage_mark_done "$STAGE" "$SUB"
