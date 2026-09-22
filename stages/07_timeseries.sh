#!/bin/bash
# =============================================================================
# stages/07_timeseries.sh - atlas time series + FC (volume and CIFTI)
#
# usage: 07_timeseries.sh [-c dataset.conf] sub-XXXX
#
# Per run, atlas and denoising strategy (always from the UNSMOOTHED series):
#   <RUN>_space-<TPL>_atlas-<A>_desc-<S>_timeseries.tsv|.json, _coverage.tsv,
#   _connectivity.tsv, plus <RUN>_space-<TPL>_atlas-<A>_desc-preproc_timeseries.*
#   (pre-denoise ROI means, needed by QC/validation for the ROI tSNR), and with
#   SURFACE=yes, for every atlas that has a dlabel, the same set as
#   <RUN>_space-fsLR_atlas-<A>_desc-{<S>,preproc}_* (columns = dlabel parcels in
#   ascending label-key order, under the dlabel names).
#
# An atlas must already be in the template space of the pipeline: a volume in a
# different MNI template has the same kind of grid and cannot be detected here.
# Only the grid (resolution, field of view) is adapted, by label resampling.
# CUSTOM_ATLASES paths must not contain spaces.
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
fp_init "$@"
SUB="${FP_ARGS[0]:?usage: $0 [-c conf] sub-XXXX}"
fp_set_log "$SUB" 07_timeseries

STAGE=07_timeseries
# fetch_resources stores the atlases under this space label; template_path()
# always serves the FSL MNI152 (= MNI152NLin6Asym) whatever TEMPLATE_NAME says.
ATLAS_SPACE=MNI152NLin6Asym

ATLAS_NAMES=()
ATLAS_VOLS=()
ATLAS_LABELS=()
ATLAS_DLABELS=()
STRATEGIES=()
ATLAS_ON_GRID=""

# ----------------------------- atlas resolution ------------------------------

add_atlas() {   # NAME VOLUME LABELS|none DLABEL|none
    local name="$1" known
    [[ "$name" =~ ^[A-Za-z0-9_]+$ ]] \
        || die "atlas name '$name': only letters, digits and '_' are allowed (it becomes the atlas-<name> entity)"
    for known in "${ATLAS_NAMES[@]+"${ATLAS_NAMES[@]}"}"; do
        [[ "$known" != "$name" ]] || die "atlas '$name' is listed twice (ATLASES / CUSTOM_ATLASES)"
    done
    ATLAS_NAMES+=("$name")
    ATLAS_VOLS+=("$2")
    ATLAS_LABELS+=("$3")
    ATLAS_DLABELS+=("$4")
    log INFO "atlas $name: volume=$2 labels=$3 dlabel=$4"
}

resolve_atlases() {
    local names=() pairs=() name pair path dir vol labels dlabel stem
    read -r -a names <<< "${ATLASES:-}"
    read -r -a pairs <<< "${CUSTOM_ATLASES:-}"

    for name in "${names[@]+"${names[@]}"}"; do
        dir="$RESOURCE_DIR/atlases/$name"
        vol="$dir/${name}_space-${ATLAS_SPACE}_res-02_dseg.nii.gz"
        [[ -s "$vol" ]] \
            || die "atlas '$name' not found ($vol): run stages/fetch_resources.sh first, or remove it from ATLASES"
        labels="$dir/labels.tsv"
        dlabel="$dir/$name.dlabel.nii"
        if [[ ! -s "$labels" ]]; then
            log WARN "atlas $name has no labels.tsv: columns will be named roi_<index>"
            labels=none
        fi
        [[ -s "$dlabel" ]] || dlabel=none
        add_atlas "$name" "$vol" "$labels" "$dlabel"
    done

    for pair in "${pairs[@]+"${pairs[@]}"}"; do
        [[ "$pair" == *=* ]] || die "CUSTOM_ATLASES entries must look like name=/path/atlas.nii.gz, got '$pair'"
        name="${pair%%=*}"
        path="${pair#*=}"
        [[ -s "$path" ]] || die "custom atlas '$name' not found: $path"
        case "$path" in
            *.nii.gz) stem="${path%.nii.gz}" ;;
            *.nii)    stem="${path%.nii}" ;;
            *) die "custom atlas '$name' must be a .nii or .nii.gz volume: $path" ;;
        esac
        labels="$stem.tsv"
        [[ -s "$labels" ]] || labels=none
        add_atlas "$name" "$path" "$labels" none
    done
}

# ----------------------------- helpers ---------------------------------------

# A dry run must be able to print its commands before the upstream stages ran.
check_inputs() {
    if is_yes "${DRY_RUN:-no}"; then
        return 0
    fi
    require_files "$@"
}

# tpl_file RUN SUFFIX : template-space file of a run. DESIGN.md writes res-<R>
# with R = MNI_RES; the zero-padded BIDS spelling (res-02) is accepted as well.
tpl_file() {
    local base
    base="$(func_dir "$SUB")/${1}_space-${TEMPLATE_NAME}_res-"
    if [[ ! -e "${base}${MNI_RES}_$2" && -e "${base}0${MNI_RES}_$2" ]]; then
        echo "${base}0${MNI_RES}_$2"
    else
        echo "${base}${MNI_RES}_$2"
    fi
}

# atlas_to_grid NAME VOLUME REFERENCE WORKDIR : sets ATLAS_ON_GRID to a label
# volume on the grid of REFERENCE (the volume itself when the grids agree).
atlas_to_grid() {
    local name="$1" vol="$2" ref="$3" work="$4" verdict out tmp
    ATLAS_ON_GRID="$vol"
    if is_yes "${DRY_RUN:-no}" && [[ ! -s "$ref" ]]; then
        return 0
    fi
    verdict="$("$PYTHON_BIN" -m fmriproc.timeseries gridcheck --image "$vol" --reference "$ref")" \
        || die "cannot compare the grids of $vol and $ref"
    verdict="${verdict%$'\r'}"
    case "$verdict" in
        same) return 0 ;;
        different) ;;
        *) die "unexpected answer of the grid check for $vol: '$verdict'" ;;
    esac
    require_cmds antsApplyTransforms
    out="$work/atlas_${name}.nii.gz"
    tmp="$work/.tmp$$_atlas_${name}.nii.gz"
    log INFO "atlas $name: grid differs from the BOLD grid, resampling labels (GenericLabel)"
    # "identity" is the keyword nipype/fMRIPrep pass; the ANTs help only says that
    # the identity is always on the transform stack, hence the form without -t.
    if ! run antsApplyTransforms -d 3 -i "$vol" -r "$ref" -o "$tmp" -n GenericLabel -t identity -u int; then
        log WARN "antsApplyTransforms rejected '-t identity'; retrying without a transform"
        run antsApplyTransforms -d 3 -i "$vol" -r "$ref" -o "$tmp" -n GenericLabel -u int
    fi
    if ! is_yes "${DRY_RUN:-no}"; then
        mv -f "$tmp" "$out"
    fi
    ATLAS_ON_GRID="$out"
}

# volume_series BOLD STRATEGY NAME ATLAS LABELS MASK CENSOR PREFIX [extra args]
volume_series() {
    local bold="$1" strat="$2" name="$3" atlas="$4" labels="$5" mask="$6" censor="$7" prefix="$8"
    shift 8
    check_inputs "$bold"
    pyrun timeseries volume --bold "$bold" --atlas "$atlas" --labels "$labels" --mask "$mask" \
        --censor "$censor" --censor-mode "$CENSOR_MODE" --min-coverage "$MIN_ROI_COVERAGE" \
        --atlas-name "$name" --strategy "$strat" --out-prefix "$prefix" "$@"
}

# cifti_series RUN DESC NAME DLABEL LABELS CENSOR WORKDIR [extra args]
#   DESC = a denoising strategy, or "preproc" for the pre-denoise dense series.
cifti_series() {
    local run="$1" strat="$2" name="$3" dlabel="$4" labels="$5" censor="$6" work="$7" dt pt tmp
    shift 7
    dt="$(func_dir "$SUB")/${run}_space-fsLR_den-91k_desc-${strat}_bold.dtseries.nii"
    check_inputs "$dt"
    pt="$work/${name}_desc-${strat}.ptseries.nii"
    tmp="$work/.tmp$$_${name}_desc-${strat}.ptseries.nii"
    # Without -legacy-mode a parcel vertex that is missing from the dense series
    # is an error; legacy mode uses the overlap and drops empty parcels (the
    # Python step restores them as n/a columns from the dlabel label table).
    if ! run wb_command -cifti-parcellate "$dt" "$dlabel" COLUMN "$tmp" -method MEAN; then
        log WARN "$run $name $strat: -cifti-parcellate failed, retrying with -legacy-mode"
        run wb_command -cifti-parcellate "$dt" "$dlabel" COLUMN "$tmp" -method MEAN -legacy-mode
    fi
    if ! is_yes "${DRY_RUN:-no}"; then
        mv -f "$tmp" "$pt"
    fi
    # --dtseries: the coverage rule of the volume stream (principle 8) also holds
    # for parcels that are partly outside the field of view of the EPI.
    pyrun timeseries cifti --ptseries "$pt" --dlabel "$dlabel" --dtseries "$dt" --labels "$labels" \
        --censor "$censor" --censor-mode "$CENSOR_MODE" --min-coverage "$MIN_ROI_COVERAGE" \
        --atlas-name "$name" --strategy "$strat" \
        --out-prefix "$(func_dir "$SUB")/${run}_space-fsLR_atlas-${name}_desc-${strat}" "$@"
}

# ----------------------------- one run ---------------------------------------

process_run() {   # RUN
    local run="$1" fdir rundir work boldref mask preproc censor bold prefix i name labels dlabel strat
    stage_should_run "$STAGE" "$SUB" "$run" --dep "05_denoise__$run" --dep "06_surface__$run" -- \
        ATLASES CUSTOM_ATLASES MIN_ROI_COVERAGE DENOISE_STRATEGIES SURFACE || return 0

    fdir="$(func_dir "$SUB")"
    rundir="$(work_func "$SUB" "$run")"
    work="$rundir/timeseries"
    mkdir -p "$work"
    boldref="$(tpl_file "$run" boldref.nii.gz)"
    mask="$(tpl_file "$run" desc-brain_mask.nii.gz)"
    preproc="$(tpl_file "$run" desc-preproc_bold.nii.gz)"
    censor="$fdir/${run}_desc-censor.1D"
    check_inputs "$boldref" "$mask" "$preproc" "$censor"

    for i in "${!ATLAS_NAMES[@]}"; do
        name="${ATLAS_NAMES[$i]}"
        labels="${ATLAS_LABELS[$i]}"
        dlabel="${ATLAS_DLABELS[$i]}"
        prefix="$fdir/${run}_space-${TEMPLATE_NAME}_atlas-${name}"
        atlas_to_grid "$name" "${ATLAS_VOLS[$i]}" "$boldref" "$rundir"

        volume_series "$preproc" preproc "$name" "$ATLAS_ON_GRID" "$labels" "$mask" "$censor" \
            "${prefix}_desc-preproc" --no-connectivity
        for strat in "${STRATEGIES[@]}"; do
            bold="$(tpl_file "$run" "desc-${strat}_bold.nii.gz")"
            volume_series "$bold" "$strat" "$name" "$ATLAS_ON_GRID" "$labels" "$mask" "$censor" \
                "${prefix}_desc-${strat}"
        done

        if ! is_yes "$SURFACE"; then
            continue
        fi
        if [[ "$dlabel" == none ]]; then
            log INFO "$run: atlas $name has no dlabel, no CIFTI time series"
            continue
        fi
        cifti_series "$run" preproc "$name" "$dlabel" "$labels" "$censor" "$work" --no-connectivity
        for strat in "${STRATEGIES[@]}"; do
            cifti_series "$run" "$strat" "$name" "$dlabel" "$labels" "$censor" "$work"
        done
    done

    stage_mark_done "$STAGE" "$SUB" "$run"
}

# ----------------------------- main ------------------------------------------

main() {
    local rows=() row run_label n_fail=0 rc _s _ses _task _run _group _bold _t1w

    read -r -a STRATEGIES <<< "${DENOISE_STRATEGIES:-}"
    [[ ${#STRATEGIES[@]} -gt 0 ]] || die "DENOISE_STRATEGIES is empty"
    [[ "$TEMPLATE_NAME" == "$ATLAS_SPACE" ]] \
        || log WARN "TEMPLATE_NAME=$TEMPLATE_NAME but the atlases are in $ATLAS_SPACE: check that this is the same template"
    resolve_atlases
    if [[ ${#ATLAS_NAMES[@]} -eq 0 ]]; then
        log WARN "ATLASES and CUSTOM_ATLASES are empty: nothing to extract"
        return 0
    fi
    if is_yes "$SURFACE"; then
        require_cmds wb_command
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
    log OK "$STAGE finished for $SUB (${#rows[@]} run(s), ${#ATLAS_NAMES[@]} atlas(es), ${#STRATEGIES[@]} strategy(ies))"
}

main
