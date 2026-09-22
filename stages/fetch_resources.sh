#!/bin/bash
# =============================================================================
# stages/fetch_resources.sh - one-time download of templates and atlases
#
# usage: fetch_resources.sh [-c dataset.conf] [--check] [extra fetch_resources.py options]
#
# Dataset level, run once, needs network access (not needed with --check):
#   $RESOURCE_DIR/templateflow/   fsLR-32k meshes, Schaefer volumes (TEMPLATEFLOW_HOME)
#   $RESOURCE_DIR/hcp/            Atlas_ROIs.2.nii.gz, {L,R}.atlasroi.32k_fs_LR.shape.gii
#   $RESOURCE_DIR/atlases/<A>/    <A>_space-MNI152NLin6Asym_res-02_dseg.nii.gz, labels.tsv, <A>.dlabel.nii
# for every Schaefer2018_<N>Parcels_<M>Networks name in ATLASES. The fsLR/HCP
# files and the dlabels are skipped when SURFACE != yes.
#
# --check only reports what is missing: exit 0 = complete, 1 = incomplete.
# Extra options are handed to the Python module, e.g. --no-templateflow-api,
# --retries N, --timeout S, --github-raw URL, --templateflow-s3 URL (mirrors).
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
fp_init "$@"

FP_LOGFILE="$LOG_DIR/fetch_resources.log"
export FP_LOGFILE

main() {
    local check=no arg extra=() atlases=() surface=no cmd=()
    for arg in ${FP_ARGS[@]+"${FP_ARGS[@]}"}; do
        if [[ "$arg" == --check ]]; then
            check=yes
        else
            extra+=("$arg")
        fi
    done
    read -r -a atlases <<< "${ATLASES:-}"
    if is_yes "$SURFACE"; then
        surface=yes
    fi

    [[ -x "$PYTHON_BIN" ]] || command -v "$PYTHON_BIN" >/dev/null 2>&1 \
        || die "PYTHON_BIN not found: $PYTHON_BIN"
    cmd=("$PYTHON_BIN" -m fmriproc.fetch_resources --resource-dir "$RESOURCE_DIR" --surface "$surface")
    # --atlases with an empty list is valid: only the surface assets are handled
    cmd+=(--atlases ${atlases[@]+"${atlases[@]}"})
    cmd+=(${extra[@]+"${extra[@]}"})

    if [[ "$check" == yes ]]; then
        # Not through run(): exit 1 is an answer here, not a failure, and the
        # list of missing files belongs on the terminal of the caller.
        log INFO "checking resources in $RESOURCE_DIR (surface=$surface, atlases: ${atlases[*]:-none})"
        local rc=0
        "${cmd[@]}" --check || rc=$?
        return "$rc"
    fi

    log INFO "===== fetch_resources | pipeline $PIPELINE_VERSION | $(date) ====="
    log INFO "resource dir $RESOURCE_DIR (surface=$surface, atlases: ${atlases[*]:-none}); network access is needed"
    run mkdir -p "$RESOURCE_DIR/templateflow" "$RESOURCE_DIR/hcp" "$RESOURCE_DIR/atlases"
    run "${cmd[@]}"
    log OK "resources complete in $RESOURCE_DIR"
}

main
