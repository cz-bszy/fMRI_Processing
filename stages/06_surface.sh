#!/bin/bash
# =============================================================================
# stages/06_surface.sh - optional fsLR-32k CIFTI branch (docs/SURFACE.md)
#
# usage: 06_surface.sh [-c dataset.conf] sub-XXXX
#
# Subject level (once): FreeSurfer surfaces -> GIFTI in T1w world coordinates,
#   midthickness, native cortex ROI, subject midthickness on the fsLR-32k mesh.
# Run level: HCP goodvoxels on the T1w-space preproc series, ribbon-constrained
#   mapping, dilation, resampling to fsLR-32k, dense time series (cortex from the
#   T1w-space series, subcortex from the 2 mm template-space series) for the
#   preproc series and for every denoising strategy, tSNR dscalar, surface QC.
#
# The BOLD was resampled once into T1w world space by stage 03, so the
# '--to-scanner' surfaces are used as they are: no affine is applied to them.
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
fp_init "$@"
SUB="${FP_ARGS[0]:?usage: $0 [-c conf] sub-XXXX}"
fp_set_log "$SUB" 06_surface

STAGE=06_surface
# HCP Atlas_ROIs.2 and the fsLR meshes belong to this template; template_path()
# serves the FSL MNI152 (= MNI152NLin6Asym) whatever TEMPLATE_NAME says.
CIFTI_SPACE=MNI152NLin6Asym
GOODVOX_SIGMA=5        # mm, neighbourhood of the CoV normalisation (HCP value)
GOODVOX_FACTOR=0.5     # exclude voxels above mean + FACTOR * SD of the modulated CoV (HCP value)
DILATE_MM=10           # fills vertices without goodvoxels (HCP/fMRIPrep value)

if ! is_yes "$SURFACE"; then
    log INFO "SURFACE=$SURFACE: surface branch disabled, nothing to do for $SUB"
    exit 0
fi

ADIR="$(anat_dir "$SUB")"
FDIR="$(func_dir "$SUB")"
AW="$(work_anat "$SUB")/surface"
FS_SUBJ="$FS_DIR/$SUB"
ATLAS_ROIS="$RESOURCE_DIR/hcp/Atlas_ROIs.2.nii.gz"
HEMIS=(L R)
declare -A FS_HEMI=([L]=lh [R]=rh)
declare -A STRUCTURE=([L]=CORTEX_LEFT [R]=CORTEX_RIGHT)
declare -A GIFTI_STRUCTURE=([L]=CortexLeft [R]=CortexRight)
declare -A FSLR_SPHERE=() ATLASROI=()
STRATEGIES=()
SMOOTH_LABEL=""
VOL_ON_GRID=""

# ----------------------------- helpers ---------------------------------------

is_dry() { is_yes "${DRY_RUN:-no}"; }

wb() { run wb_command "$@"; }

# A dry run must be able to print its commands before the upstream stages ran.
check_inputs() {
    if is_dry; then
        return 0
    fi
    require_files "$@"
}

install_file() {   # SRC DEST
    if is_dry; then
        return 0
    fi
    require_files "$1"
    mv -f "$1" "$2"
}

# subject-level files: DESIGN.md names in anat/, the rest is private work
surf_file()  { echo "$ADIR/${SUB}_hemi-${1}_${2}.surf.gii"; }                               # H white|pial|midthickness
fslr_mid()   { echo "$ADIR/${SUB}_hemi-${1}_space-fsLR_den-32k_midthickness.surf.gii"; }    # H
sphere_reg() { echo "$AW/${SUB}_hemi-${1}_desc-reg_sphere.surf.gii"; }                      # H
thickness()  { echo "$AW/${SUB}_hemi-${1}_thickness.shape.gii"; }                           # H
cortex_roi() { echo "$AW/${SUB}_hemi-${1}_desc-cortex_roi.shape.gii"; }                     # H

# tpl_file RUN SUFFIX : template-space file of a run. DESIGN.md writes res-<R>
# with R = MNI_RES; the zero-padded BIDS spelling (res-02) is accepted as well.
tpl_file() {
    local base="$FDIR/${1}_space-${TEMPLATE_NAME}_res-"
    if [[ ! -e "${base}${MNI_RES}_$2" && -e "${base}0${MNI_RES}_$2" ]]; then
        echo "${base}0${MNI_RES}_$2"
    else
        echo "${base}${MNI_RES}_$2"
    fi
}

check_config() {
    [[ "$SURF_SMOOTH_FWHM" =~ ^[0-9]+([.][0-9]+)?$ ]] \
        || die "SURF_SMOOTH_FWHM must be a non-negative number (mm, 0 = off): $SURF_SMOOTH_FWHM"
    read -r -a STRATEGIES <<< "${DENOISE_STRATEGIES:-}"
    [[ ${#STRATEGIES[@]} -gt 0 ]] || die "DENOISE_STRATEGIES is empty"
    if awk -v f="$SURF_SMOOTH_FWHM" 'BEGIN { exit !(f + 0 > 0) }'; then
        # '.' is not allowed inside a BIDS entity value; same spelling as the
        # volume copy of stage 05 (SMOOTH_FWHM=4.5 -> sm45)
        SMOOTH_LABEL="sm${SURF_SMOOTH_FWHM//./}"
    fi
    [[ "$TEMPLATE_NAME" == "$CIFTI_SPACE" ]] \
        || log WARN "TEMPLATE_NAME=$TEMPLATE_NAME but the CIFTI subcortex (Atlas_ROIs.2) is defined in $CIFTI_SPACE: check that this is the same template"
}

check_license() {
    if is_dry; then
        return 0
    fi
    [[ -n "${FS_LICENSE:-}" && -s "${FS_LICENSE:-}" ]] \
        || die "FreeSurfer license not found: set FS_LICENSE or mount it at /opt/freesurfer/license.txt (mris_convert needs it)"
}

# ----------------------------- resources -------------------------------------

resolve_resources() {
    local keys=() paths=() H i roi label
    check_inputs "$ATLAS_ROIS"

    # TemplateFlow file names do not have one entity order (hemi-L_den-32k and
    # den-32k_hemi-L both occur), so the Python side resolves them.
    for H in "${HEMIS[@]}"; do
        keys+=("fslr_fsaverage_sphere_$H")
    done
    if mapfile -t paths < <("$PYTHON_BIN" -m fmriproc.fetch_resources --resource-dir "$RESOURCE_DIR" \
            --locate "${keys[@]}" 2>>"$FP_LOGFILE") && [[ ${#paths[@]} -eq ${#HEMIS[@]} ]]; then
        for i in "${!HEMIS[@]}"; do
            FSLR_SPHERE[${HEMIS[$i]}]="${paths[$i]%$'\r'}"
        done
    elif is_dry; then
        for H in "${HEMIS[@]}"; do
            FSLR_SPHERE[$H]="$RESOURCE_DIR/templateflow/tpl-fsLR/tpl-fsLR_space-fsaverage_hemi-${H}_den-32k_sphere.surf.gii"
        done
    else
        die "fsLR-32k registration spheres not found under $RESOURCE_DIR/templateflow: run stages/fetch_resources.sh (needs network once)"
    fi

    for H in "${HEMIS[@]}"; do
        roi="$RESOURCE_DIR/hcp/$H.atlasroi.32k_fs_LR.shape.gii"
        if [[ ! -s "$roi" ]] && ! is_dry; then
            # same mask, derived from the TemplateFlow medial-wall label
            label="$("$PYTHON_BIN" -m fmriproc.fetch_resources --resource-dir "$RESOURCE_DIR" \
                --locate "fslr_nomedialwall_$H" 2>>"$FP_LOGFILE")" \
                || die "fsLR medial-wall ROI missing ($roi) and no TemplateFlow desc-nomedialwall label either: run stages/fetch_resources.sh"
            roi="$AW/$H.atlasroi.32k_fs_LR.shape.gii"
            log WARN "HCP atlasroi for hemisphere $H is missing: deriving it from $label"
            mkdir -p "$AW"
            pyrun surface_utils label-to-roi "$label" "$roi" --structure "${GIFTI_STRUCTURE[$H]}"
        fi
        ATLASROI[$H]="$roi"
        log INFO "hemisphere $H: fsLR sphere=${FSLR_SPHERE[$H]} atlasroi=$roi"
    done
}

# ----------------------------- subject level ---------------------------------

subject_outputs() {
    local H
    for H in "${HEMIS[@]}"; do
        surf_file "$H" white
        surf_file "$H" pial
        surf_file "$H" midthickness
        fslr_mid "$H"
        sphere_reg "$H"
        thickness "$H"
        cortex_roi "$H"
    done
}

subject_complete() {
    local f
    while IFS= read -r f; do
        [[ -s "$f" ]] || return 1
    done < <(subject_outputs)
    return 0
}

# convert_hemisphere H BUILD_DIR : everything of one hemisphere, under private names
convert_hemisphere() {
    local H="$1" b="$2" hemi struct white pial mid sphere thick roi mid32k
    hemi="${FS_HEMI[$H]}"
    struct="${STRUCTURE[$H]}"
    white="$b/$H.white.surf.gii"
    pial="$b/$H.pial.surf.gii"
    mid="$b/$H.midthickness.surf.gii"
    sphere="$b/$H.sphere.reg.surf.gii"
    thick="$b/$H.thickness.shape.gii"
    roi="$b/$H.roi.shape.gii"
    mid32k="$b/$H.midthickness.32k_fs_LR.surf.gii"

    check_inputs "$FS_SUBJ/surf/$hemi.white" "$FS_SUBJ/surf/$hemi.pial" \
        "$FS_SUBJ/surf/$hemi.sphere.reg" "$FS_SUBJ/surf/$hemi.thickness"

    # --to-scanner adds c_ras: without it the surfaces sit in tkr coordinates,
    # displaced from the T1w/BOLD world space, and the ribbon mapping silently
    # samples the wrong tissue. Spheres have no anatomical coordinates: never
    # convert them. Output paths must be absolute (mris_convert otherwise writes
    # next to the input surface).
    run mris_convert --to-scanner "$FS_SUBJ/surf/$hemi.white" "$white"
    run mris_convert --to-scanner "$FS_SUBJ/surf/$hemi.pial" "$pial"
    run mris_convert "$FS_SUBJ/surf/$hemi.sphere.reg" "$sphere"
    # mris_convert -c prepends the hemisphere to an output name that does not
    # start with lh./rh. (L.thickness.shape.gii -> lh.L.thickness.shape.gii):
    # write under the FreeSurfer-style name, then rename.
    run mris_convert -c "$FS_SUBJ/surf/$hemi.thickness" "$FS_SUBJ/surf/$hemi.white" \
        "$(dirname "$thick")/$hemi.thickness.shape.gii"
    run mv -f "$(dirname "$thick")/$hemi.thickness.shape.gii" "$thick"
    wb -set-structure "$white" "$struct" -surface-type ANATOMICAL -surface-secondary-type GRAY_WHITE
    wb -set-structure "$pial" "$struct" -surface-type ANATOMICAL -surface-secondary-type PIAL
    wb -set-structure "$sphere" "$struct" -surface-type SPHERICAL
    wb -set-structure "$thick" "$struct"

    wb -surface-average "$mid" -surf "$white" -surf "$pial"
    wb -set-structure "$mid" "$struct" -surface-type ANATOMICAL -surface-secondary-type MIDTHICKNESS

    # medial wall = thickness 0
    wb -metric-math 'thickness > 0' "$roi" -var thickness "$thick"
    wb -metric-fill-holes "$mid" "$roi" "$roi"
    wb -metric-remove-islands "$mid" "$roi" "$roi"
    wb -set-structure "$roi" "$struct"

    # sphere.reg is in register with fsaverage; the space-fsaverage fsLR sphere
    # is the fsLR mesh expressed in fsaverage space (HCP resample_fsaverage recipe)
    wb -surface-resample "$mid" "$sphere" "${FSLR_SPHERE[$H]}" BARYCENTRIC "$mid32k"
    wb -set-structure "$mid32k" "$struct" -surface-type ANATOMICAL -surface-secondary-type MIDTHICKNESS
}

install_hemisphere() {   # H BUILD_DIR
    local H="$1" b="$2"
    install_file "$b/$H.white.surf.gii" "$(surf_file "$H" white)"
    install_file "$b/$H.pial.surf.gii" "$(surf_file "$H" pial)"
    install_file "$b/$H.midthickness.surf.gii" "$(surf_file "$H" midthickness)"
    install_file "$b/$H.midthickness.32k_fs_LR.surf.gii" "$(fslr_mid "$H")"
    install_file "$b/$H.sphere.reg.surf.gii" "$(sphere_reg "$H")"
    install_file "$b/$H.thickness.shape.gii" "$(thickness "$H")"
    install_file "$b/$H.roi.shape.gii" "$(cortex_roi "$H")"
}

prepare_subject() {
    local H build="$AW/build"
    # marker 06_surface (no run): a rebuilt reconstruction invalidates the
    # surfaces, and new surfaces invalidate every run (--dep in process_run)
    if ! stage_should_run "$STAGE" "$SUB" --dep 01_anat_recon -- ANAT_MODE; then
        if subject_complete; then
            return 0
        fi
        log WARN "marker of $STAGE is up to date but subject-level surfaces are missing: converting again"
    fi

    require_cmds mris_convert
    check_license
    [[ -d "$FS_SUBJ/surf" ]] || is_dry || die "no FreeSurfer reconstruction for $SUB in $FS_DIR (run stage 01_anat_recon)"
    run rm -rf "$build"
    run mkdir -p "$build" "$ADIR"
    for H in "${HEMIS[@]}"; do
        convert_hemisphere "$H" "$build"
    done
    # nothing is installed before both hemispheres succeeded
    for H in "${HEMIS[@]}"; do
        install_hemisphere "$H" "$build"
    done
    run rm -rf "$build"
    stage_mark_done "$STAGE" "$SUB"
}

# ----------------------------- run level: goodvoxels -------------------------

# make_goodvoxels BOLD WORKDIR : HCP RibbonVolumeToSurfaceMapping rule on the BOLD
# grid. Needs the scaled, NON-demeaned series (residuals have no mean).
# Writes ribbon_only, mask (voxels with data) and goodvoxels into WORKDIR.
make_goodvoxels() {
    local bold="$1" w="$2" H n_ribbon="" cov_mean="" mod_mean="" mod_sd="" upper=""

    run fslmaths "$bold" -Tmean "$w/mean.nii.gz" -odt float
    run fslmaths "$bold" -Tstd "$w/std.nii.gz" -odt float
    run fslmaths "$w/mean.nii.gz" -bin "$w/mask.nii.gz"

    for H in "${HEMIS[@]}"; do
        # The mean image defines the output grid, so the ribbon is on the BOLD
        # grid by construction. Signed distance: negative inside the surface.
        wb -create-signed-distance-volume "$(surf_file "$H" white)" "$w/mean.nii.gz" "$w/$H.white_dist.nii.gz"
        wb -create-signed-distance-volume "$(surf_file "$H" pial)" "$w/mean.nii.gz" "$w/$H.pial_dist.nii.gz"
        run fslmaths "$w/$H.white_dist.nii.gz" -thr 0 -bin "$w/$H.white_out.nii.gz"
        run fslmaths "$w/$H.pial_dist.nii.gz" -uthr 0 -abs -bin "$w/$H.pial_in.nii.gz"
        run fslmaths "$w/$H.pial_in.nii.gz" -mas "$w/$H.white_out.nii.gz" "$w/$H.ribbon.nii.gz"
    done
    run fslmaths "$w/L.ribbon.nii.gz" -add "$w/R.ribbon.nii.gz" -bin "$w/ribbon_only.nii.gz"

    run fslmaths "$w/std.nii.gz" -div "$w/mean.nii.gz" "$w/cov.nii.gz"
    run fslmaths "$w/cov.nii.gz" -mas "$w/ribbon_only.nii.gz" "$w/cov_ribbon.nii.gz"

    if is_dry; then
        cov_mean=1
    else
        n_ribbon="$(mask_count "$w/cov_ribbon.nii.gz")"
        [[ "$n_ribbon" -gt 0 ]] \
            || die "no BOLD signal inside the cortical ribbon: surfaces and T1w-space BOLD do not overlap (check the coregistration of stage 03 and the surfaces)"
        cov_mean="$(fslstats "$w/cov_ribbon.nii.gz" -M)"
        cov_mean="${cov_mean// /}"
        log INFO "goodvoxels: $n_ribbon ribbon voxels with signal, mean CoV $cov_mean"
    fi

    run fslmaths "$w/cov_ribbon.nii.gz" -div "$cov_mean" "$w/cov_ribbon_norm.nii.gz"
    run fslmaths "$w/cov_ribbon_norm.nii.gz" -bin -s "$GOODVOX_SIGMA" "$w/smooth_norm.nii.gz"
    run fslmaths "$w/cov_ribbon_norm.nii.gz" -s "$GOODVOX_SIGMA" -div "$w/smooth_norm.nii.gz" -dilD \
        "$w/cov_ribbon_norm_s.nii.gz"
    run fslmaths "$w/cov.nii.gz" -div "$cov_mean" -div "$w/cov_ribbon_norm_s.nii.gz" "$w/cov_norm_modulate.nii.gz"
    run fslmaths "$w/cov_norm_modulate.nii.gz" -mas "$w/ribbon_only.nii.gz" "$w/cov_norm_modulate_ribbon.nii.gz"

    if is_dry; then
        upper=1
    else
        mod_mean="$(fslstats "$w/cov_norm_modulate_ribbon.nii.gz" -M)"
        mod_sd="$(fslstats "$w/cov_norm_modulate_ribbon.nii.gz" -S)"
        mod_mean="${mod_mean// /}"
        mod_sd="${mod_sd// /}"
        upper="$(awk -v m="$mod_mean" -v s="$mod_sd" -v k="$GOODVOX_FACTOR" 'BEGIN { printf "%.8f", m + k * s }')"
        log INFO "goodvoxels: locally normalised CoV mean $mod_mean sd $mod_sd -> excluded above $upper"
    fi
    # goodvoxels = voxels with data minus the noisy (vessel/edge/partial-volume) ones
    run fslmaths "$w/cov_norm_modulate.nii.gz" -thr "$upper" -bin -sub "$w/mask.nii.gz" -mul -1 "$w/goodvoxels.nii.gz"
}

# ----------------------------- run level: mapping ----------------------------

# map_series BOLD TAG WORKDIR SUBDIV VOXEL_ROI QC : T1w-space series ->
# WORKDIR/TAG_hemi-{L,R}.32k.func.gii. QC=yes keeps the ROI of fsLR vertices that
# received data (it depends only on geometry and goodvoxels: once per run).
map_series() {
    local bold="$1" tag="$2" w="$3" subdiv="$4" voxel_roi="$5" qc="$6" H native bad out white pial mid roi
    local resample_opts=()
    for H in "${HEMIS[@]}"; do
        native="$w/${tag}_hemi-$H.native.func.gii"
        bad="$w/${tag}_hemi-$H.badvert.shape.gii"
        out="$w/${tag}_hemi-$H.32k.func.gii"
        white="$(surf_file "$H" white)"
        pial="$(surf_file "$H" pial)"
        mid="$(surf_file "$H" midthickness)"
        roi="$(cortex_roi "$H")"
        resample_opts=()
        if [[ "$qc" == yes ]]; then
            resample_opts=(-valid-roi-out "$w/${tag}_hemi-$H.valid32k.shape.gii")
        fi
        wb -volume-to-surface-mapping "$bold" "$mid" "$native" \
            -ribbon-constrained "$white" "$pial" \
            -volume-roi "$voxel_roi" -voxel-subdiv "$subdiv" \
            -bad-vertices-out "$bad"
        # Denoised series are zero-mean, so "value == 0" is no criterion for a
        # vertex without data: the vertices to fill are named explicitly.
        wb -metric-dilate "$native" "$mid" "$DILATE_MM" "$native" -bad-vertex-roi "$bad" -nearest
        wb -metric-mask "$native" "$roi" "$native"
        wb -metric-resample "$native" "$(sphere_reg "$H")" "${FSLR_SPHERE[$H]}" ADAP_BARY_AREA "$out" \
            -area-surfs "$mid" "$(fslr_mid "$H")" -current-roi "$roi" \
            ${resample_opts[@]+"${resample_opts[@]}"}
        wb -metric-mask "$out" "${ATLASROI[$H]}" "$out"
        wb -set-structure "$out" "${STRUCTURE[$H]}"
        run rm -f "$native"
    done
}

# make_sampled_mask OUT WORKDIR : 1 on the grayordinates whose value comes from
# their own data. Vertices without goodvoxels in their ribbon (badvert of the
# preproc mapping) only receive a neighbour's copy through the dilation, and an
# fsLR vertex inherits the area-weighted share of such native vertices: stage 07
# does not count them as covered (principle 8). The subcortex is never dilated.
make_sampled_mask() {
    local out="$1" w="$2" H native fslr tmp="$2/sampled_mask.dscalar.nii"
    for H in "${HEMIS[@]}"; do
        native="$w/sampled_hemi-$H.native.shape.gii"
        fslr="$w/sampled_hemi-$H.32k.shape.gii"
        wb -metric-math "(roi > 0) * (bad == 0)" "$native" \
            -var roi "$(cortex_roi "$H")" -var bad "$w/preproc_hemi-$H.badvert.shape.gii"
        wb -metric-resample "$native" "$(sphere_reg "$H")" "${FSLR_SPHERE[$H]}" ADAP_BARY_AREA "$fslr" \
            -area-surfs "$(surf_file "$H" midthickness)" "$(fslr_mid "$H")" -current-roi "$(cortex_roi "$H")"
        wb -metric-math "x >= 0.5" "$fslr" -var x "$fslr"
        wb -set-structure "$fslr" "${STRUCTURE[$H]}"
    done
    run fslmaths "$ATLAS_ROIS" -bin "$w/sampled_subcortex.nii.gz"
    wb -cifti-create-dense-scalar "$tmp" -volume "$w/sampled_subcortex.nii.gz" "$ATLAS_ROIS" \
        -left-metric "$w/sampled_hemi-L.32k.shape.gii" -roi-left "${ATLASROI[L]}" \
        -right-metric "$w/sampled_hemi-R.32k.shape.gii" -roi-right "${ATLASROI[R]}"
    install_file "$tmp" "$out"
}

# volume_on_atlas_grid VOLUME TAG WORKDIR : sets VOL_ON_GRID to a series on the
# exact grid of Atlas_ROIs.2 (wb_command refuses any other volume space).
volume_on_atlas_grid() {
    local vol="$1" tag="$2" w="$3" relation=""
    VOL_ON_GRID="$vol"
    if is_dry && [[ ! -s "$vol" ]]; then
        return 0
    fi
    relation="$("$PYTHON_BIN" -m fmriproc.surface_utils check-grid "$vol" "$ATLAS_ROIS" 2>>"$FP_LOGFILE")" || true
    case "$relation" in
        same) ;;
        reordered)
            # same voxel centres, other storage order (a tool rewrote the axes):
            # enclosing-voxel resampling is then an exact re-indexing
            log WARN "$tag: voxel storage order differs from Atlas_ROIs.2, re-indexing with -volume-resample ENCLOSING_VOXEL"
            wb -volume-resample "$vol" "$ATLAS_ROIS" ENCLOSING_VOXEL "$w/${tag}_subcortex.nii.gz"
            VOL_ON_GRID="$w/${tag}_subcortex.nii.gz"
            ;;
        *)
            die "$vol is not on the grid of $ATLAS_ROIS (FSL MNI152 2 mm, 91x109x91; details in $FP_LOGFILE). The CIFTI subcortex needs MNI_RES=2 and TEMPLATE_NAME=$CIFTI_SPACE."
            ;;
    esac
}

# make_dtseries BOLD_T1W BOLD_TPL TAG OUT WORKDIR SUBDIV TR QC
make_dtseries() {
    local bold_t1="$1" bold_tpl="$2" tag="$3" out="$4" w="$5" subdiv="$6" tr="$7" qc="$8"
    local n_cortex n_sub voxel_roi="$5/goodvoxels.nii.gz" tmp="$5/${3}_bold.dtseries.nii"
    check_inputs "$bold_t1" "$bold_tpl"
    if ! is_dry; then
        n_cortex="$(nvols "$bold_t1")"
        n_sub="$(nvols "$bold_tpl")"
        [[ "$n_cortex" == "$n_sub" ]] \
            || die "$tag: $n_cortex volumes in $bold_t1 but $n_sub in $bold_tpl (stage 05 must denoise both spaces with the same censoring)"
    fi
    if [[ "$tag" != preproc ]]; then
        # If stage 05 restricted 3dTproject to a mask, the voxels outside it are
        # constant and would dilute the ribbon average: they are no goodvoxels.
        voxel_roi="$w/${tag}_goodvoxels.nii.gz"
        run fslmaths "$bold_t1" -Tstd -bin -mul "$w/goodvoxels.nii.gz" "$voxel_roi"
    fi
    volume_on_atlas_grid "$bold_tpl" "$tag" "$w"
    map_series "$bold_t1" "$tag" "$w" "$subdiv" "$voxel_roi" "$qc"
    wb -cifti-create-dense-timeseries "$tmp" \
        -volume "$VOL_ON_GRID" "$ATLAS_ROIS" \
        -left-metric "$w/${tag}_hemi-L.32k.func.gii" -roi-left "${ATLASROI[L]}" \
        -right-metric "$w/${tag}_hemi-R.32k.func.gii" -roi-right "${ATLASROI[R]}" \
        -timestep "$tr"
    run rm -f "$w/${tag}_hemi-L.32k.func.gii" "$w/${tag}_hemi-R.32k.func.gii" "$w/${tag}_subcortex.nii.gz"
    install_file "$tmp" "$out"
}

smooth_dtseries() {   # IN OUT WORKDIR TAG
    local in="$1" out="$2" tmp="$3/${4}_bold.dtseries.nii"
    # geodesic on the subject's own fsLR midthickness, parcel-constrained in the
    # subcortex; zeros (outside the field of view) are missing data, not signal
    wb -cifti-smoothing "$in" "$SURF_SMOOTH_FWHM" "$SURF_SMOOTH_FWHM" COLUMN "$tmp" -fwhm \
        -left-surface "$(fslr_mid L)" -right-surface "$(fslr_mid R)" \
        -fix-zeros-volume -fix-zeros-surface
    install_file "$tmp" "$out"
}

run_tr() {   # PREP_INFO BOLD
    local tr=""
    if [[ -s "$1" ]]; then
        tr="$(json_get "$1" tr "")"
    fi
    if [[ -z "$tr" && -s "$2" ]]; then
        tr="$(img_tr "$2")"
    fi
    if [[ -z "$tr" ]] && is_dry; then
        tr=0
    fi
    [[ -n "$tr" ]] || die "cannot determine the TR of $2"
    echo "$tr"
}

voxel_subdiv() {   # PREP_INFO
    local n=""
    if [[ -s "$1" ]]; then
        n="$("$PYTHON_BIN" -m fmriproc.surface_utils voxel-subdiv --prep-info "$1" 2>>"$FP_LOGFILE")" || n=""
    fi
    if [[ ! "$n" =~ ^[0-9]+$ ]]; then
        # the finer subdivision is always safe, it only costs a few seconds
        log WARN "voxel_size not available in $1: using -voxel-subdiv 7"
        n=7
    fi
    echo "$n"
}

# ----------------------------- one run ---------------------------------------

run_complete() {   # RUN
    local pre="$FDIR/$1" strat
    [[ -s "${pre}_space-fsLR_den-91k_desc-preproc_bold.dtseries.nii" \
        && -s "${pre}_space-fsLR_den-91k_desc-preproc_tsnr.dscalar.nii" \
        && -s "${pre}_space-fsLR_den-91k_desc-sampled_mask.dscalar.nii" \
        && -s "${pre}_desc-surfqc.json" ]] || return 1
    for strat in "${STRATEGIES[@]}"; do
        [[ -s "${pre}_space-fsLR_den-91k_desc-${strat}_bold.dtseries.nii" ]] || return 1
        if [[ -n "$SMOOTH_LABEL" ]]; then
            [[ -s "${pre}_space-fsLR_den-91k_desc-${strat}${SMOOTH_LABEL}_bold.dtseries.nii" ]] || return 1
        fi
    done
    return 0
}

process_run() {   # RUN
    local run_id="$1" w pre prep bold_t1 bold_tpl tr subdiv strat dt dt_pre tsnr qc tmp t1_series
    pre="$FDIR/$run_id"
    if ! stage_should_run "$STAGE" "$SUB" "$run_id" --dep "05_denoise__$run_id" --dep "$STAGE" -- \
            SURF_SMOOTH_FWHM DENOISE_STRATEGIES; then
        if run_complete "$run_id"; then
            return 0
        fi
        log WARN "$run_id: marker of $STAGE is up to date but outputs are missing: running again"
    fi

    w="$(work_func "$SUB" "$run_id")/surface"
    prep="${pre}_desc-prep_info.json"
    bold_t1="${pre}_space-T1w_desc-preproc_bold.nii.gz"
    bold_tpl="$(tpl_file "$run_id" desc-preproc_bold.nii.gz)"
    dt_pre="${pre}_space-fsLR_den-91k_desc-preproc_bold.dtseries.nii"
    tsnr="${pre}_space-fsLR_den-91k_desc-preproc_tsnr.dscalar.nii"
    qc="${pre}_desc-surfqc.json"
    check_inputs "$bold_t1" "$bold_tpl"

    tr="$(run_tr "$prep" "$bold_t1")"
    subdiv="$(voxel_subdiv "$prep")"
    log INFO "$run_id: TR=$tr voxel-subdiv=$subdiv strategies='${STRATEGIES[*]}' smoothing=${SMOOTH_LABEL:-off}"

    # stale results must not survive a failed recomputation (the space-fsLR_atlas-*
    # tables belong to stage 07 and follow through its --dep on this stage)
    run rm -rf "$w"
    run mkdir -p "$w"
    run rm -f "$qc" "${pre}_space-fsLR_den-91k_"*

    make_goodvoxels "$bold_t1" "$w"

    make_dtseries "$bold_t1" "$bold_tpl" preproc "$dt_pre" "$w" "$subdiv" "$tr" yes
    make_sampled_mask "${pre}_space-fsLR_den-91k_desc-sampled_mask.dscalar.nii" "$w"
    tmp="$w/preproc_tsnr.dscalar.nii"
    wb -cifti-reduce "$dt_pre" TSNR "$tmp"
    install_file "$tmp" "$tsnr"

    for strat in "${STRATEGIES[@]}"; do
        dt="${pre}_space-fsLR_den-91k_desc-${strat}_bold.dtseries.nii"
        # 05 -> 06 hand-over file (DESIGN.md section 10); it lives in work/ and may have been cleaned
        t1_series="$(work_func "$SUB" "$run_id")/denoise/${strat}_space-T1w_bold.nii.gz"
        if ! is_dry && [[ ! -s "$t1_series" ]]; then
            die "$run_id: $t1_series is missing: rerun stage 05 with SURFACE=yes (FORCE=yes STAGES=denoise)"
        fi
        make_dtseries "$t1_series" "$(tpl_file "$run_id" "desc-${strat}_bold.nii.gz")" \
            "$strat" "$dt" "$w" "$subdiv" "$tr" no
        if [[ -n "$SMOOTH_LABEL" ]]; then
            smooth_dtseries "$dt" "${pre}_space-fsLR_den-91k_desc-${strat}${SMOOTH_LABEL}_bold.dtseries.nii" \
                "$w" "${strat}${SMOOTH_LABEL}"
        fi
    done

    pyrun surface_utils surfqc \
        --badvert "$w/preproc_hemi-L.badvert.shape.gii" "$w/preproc_hemi-R.badvert.shape.gii" \
        --roi "$(cortex_roi L)" "$(cortex_roi R)" \
        --ribbon "$w/ribbon_only.nii.gz" --goodvoxels "$w/goodvoxels.nii.gz" --mask "$w/mask.nii.gz" \
        --tsnr "$tsnr" \
        --valid "$w/preproc_hemi-L.valid32k.shape.gii" "$w/preproc_hemi-R.valid32k.shape.gii" \
        --atlasroi "${ATLASROI[L]}" "${ATLASROI[R]}" \
        --voxel-subdiv "$subdiv" --strategies "${STRATEGIES[@]}" --smooth-fwhm "$SURF_SMOOTH_FWHM" \
        --out "$qc"
    check_inputs "$qc"
    if ! is_dry; then
        log INFO "$run_id: surface QC: tsnr_cortex_median=$(json_get "$qc" tsnr_cortex_median n/a) pct_badvertices=$(json_get "$qc" pct_badvertices n/a) pct_goodvoxels_excluded=$(json_get "$qc" pct_goodvoxels_excluded n/a)"
    fi

    if ! is_yes "$KEEP_WORK"; then
        run rm -rf "$w"
    fi
    stage_mark_done "$STAGE" "$SUB" "$run_id"
}

# ----------------------------- main ------------------------------------------

main() {
    local rows=() row run_label n_fail=0 rc _s _ses _task _run _group _bold _t1w

    check_config
    require_cmds wb_command fslmaths fslstats fslnvols fslval
    is_dry || fp_check_mem
    run mkdir -p "$AW" "$FDIR"
    resolve_resources
    prepare_subject

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
    log OK "$STAGE finished for $SUB (${#rows[@]} run(s), ${#STRATEGIES[@]} strategy(ies))"
}

main
