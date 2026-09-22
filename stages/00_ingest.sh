#!/bin/bash
# =============================================================================
# stages/00_ingest.sh [-c conf] [--overwrite]
# Dataset-level stage (not per subject): raw layout (dpabi | bids) -> rawdata/
# with header-normalised images, sidecars from the acquisition table,
# manifest.tsv, participants.tsv and ingest_report.tsv.
# Contract: docs/DESIGN.md sections 4, 5 and 7 (00 ingest).
#
# No stage hash: ingest is idempotent and cheap. Image copies whose header is
# already the wanted one are kept (FORCE=yes or --overwrite rewrites them);
# sidecars and the three tables are rewritten on every call so that a changed
# acquisition table always takes effect.
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
fp_init "$@"
fp_set_log "dataset" 00_ingest

OVERWRITE=no

parse_stage_args() {
    local arg
    for arg in "$@"; do
        case "$arg" in
            --overwrite) OVERWRITE=yes ;;
            *) die "usage: $(basename "$0") [-c conf] [--overwrite]   (unexpected argument: $arg)" ;;
        esac
    done
    if is_yes "${FORCE:-no}"; then
        OVERWRITE=yes
    fi
}

# absolute_path FILE : the file's absolute path (the Python module must not
# depend on the working directory of this script)
absolute_path() {
    echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

check_settings() {
    require_cmds "$PYTHON_BIN"
    [[ -n "$INPUT_DIR" ]] || die "INPUT_DIR is not set (dataset conf or environment)"
    [[ -d "$INPUT_DIR" ]] || die "INPUT_DIR is not a directory: $INPUT_DIR"
    INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
    case "$INPUT_LAYOUT" in
        bids|dpabi) ;;
        *) die "INPUT_LAYOUT must be bids or dpabi: $INPUT_LAYOUT" ;;
    esac
    [[ "$DROP_VOLUMES" =~ ^[0-9]+$ ]] || die "DROP_VOLUMES must be a non-negative integer: $DROP_VOLUMES"
    [[ -n "$TASK_NAME" ]] || die "TASK_NAME is empty"
    if [[ -n "$ACQ_TABLE" ]]; then
        require_files "$ACQ_TABLE"
        ACQ_TABLE="$(absolute_path "$ACQ_TABLE")"
    else
        log WARN "ACQ_TABLE is not set: slice timing correction is possible only for runs with a SliceTiming sidecar"
    fi
    if [[ -n "$SUBJECT_LIST" ]]; then
        require_files "$SUBJECT_LIST"
        SUBJECT_LIST="$(absolute_path "$SUBJECT_LIST")"
    fi
}

# Every config value the Python module needs is passed explicitly: the module
# never reads the bash configuration.
build_ingest_args() {
    INGEST_ARGS=(
        --input-dir "$INPUT_DIR"
        --layout "$INPUT_LAYOUT"
        --raw-dir "$RAW_DIR"
        --task "$TASK_NAME"
        --bids-func-glob "$BIDS_FUNC_GLOB"
        --bids-t1-glob "$BIDS_T1_GLOB"
        --dpabi-func-dir "$DPABI_FUNC_DIR"
        --dpabi-t1-dir "$DPABI_T1_DIR"
        --drop-volumes "$DROP_VOLUMES"
    )
    if [[ -n "$ACQ_TABLE" ]]; then
        INGEST_ARGS+=(--acq-table "$ACQ_TABLE")
    fi
    if [[ -n "$SUBJECT_LIST" ]]; then
        INGEST_ARGS+=(--subject-list "$SUBJECT_LIST")
    fi
    if is_yes "$OVERWRITE"; then
        INGEST_ARGS+=(--overwrite)
    fi
}

report_summary() {
    local report="$RAW_DIR/ingest_report.tsv" n_runs n_err line
    require_files "$MANIFEST" "$report" "$RAW_DIR/participants.tsv"
    n_runs="$(awk 'NR > 1' "$MANIFEST" | wc -l | tr -d ' ')"
    n_err="$(awk -F'\t' 'NR > 1 && $5 == "error"' "$report" | wc -l | tr -d ' ')"
    if [[ "$n_err" -gt 0 ]]; then
        log WARN "$n_err run(s) rejected (column 'warnings' of $report):"
        while IFS= read -r line; do
            log WARN "  $line"
        done < <(awk -F'\t' 'NR > 1 && $5 == "error" {print $3 ": " $22}' "$report")
    fi
    log OK "ingest: $n_runs run(s) in $MANIFEST"
}

parse_stage_args "${FP_ARGS[@]+"${FP_ARGS[@]}"}"
check_settings
build_ingest_args
pyrun ingest "${INGEST_ARGS[@]}"
if is_yes "${DRY_RUN:-no}"; then
    log INFO "dry run: nothing was written"
else
    report_summary
fi
