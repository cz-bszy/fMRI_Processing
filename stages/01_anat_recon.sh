#!/bin/bash
# =============================================================================
# 01_anat_recon.sh [-c conf] sub-X
# FreeSurfer recon-all (ANAT_MODE=freesurfer): new run, resume, or reuse of an
# existing reconstruction. ANAT_MODE=synth has no surfaces: nothing to do.
# Contract: docs/DESIGN.md sections 4, 7 ("01 anat_recon").
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
fp_init "$@"
SUB="${FP_ARGS[0]:?usage: $0 [-c conf] sub-XXXX}"
STAGE=01_anat_recon
fp_set_log "$SUB" "$STAGE"

SUBJ_DIR="$FS_DIR/$SUB"

recon_complete() {
    [[ -f "$SUBJ_DIR/scripts/recon-all.done" \
        && -e "$SUBJ_DIR/surf/lh.pial" && -e "$SUBJ_DIR/surf/rh.pial" \
        && -s "$SUBJ_DIR/mri/aseg.mgz" ]]
}

recon_is_running() {
    compgen -G "$SUBJ_DIR/scripts/IsRunning*" >/dev/null
}

# An existing directory can only be resumed when the T1 import had finished.
recon_has_input() {
    [[ -s "$SUBJ_DIR/mri/orig/001.mgz" ]]
}

check_license() {
    local home="${FREESURFER_HOME:-/opt/freesurfer}"
    # FreeSurfer itself also accepts $FREESURFER_HOME/license.txt and .license
    if [[ -n "${FS_LICENSE:-}" && -s "${FS_LICENSE:-}" ]] || [[ -s "$home/license.txt" || -s "$home/.license" ]]; then
        return 0
    fi
    die "FreeSurfer license not found (FS_LICENSE='${FS_LICENSE:-}'). Mount license.txt into the container and export FS_LICENSE."
}

# recon-all creates symlinks (fsaverage, ?h.white.H, ...). Windows bind mounts
# often refuse them, which would only surface hours into the run.
check_filesystem() {
    local fstype probe="$FS_DIR/.fp_symlink_probe.$$"
    fstype="$(stat -f -c %T "$FS_DIR" 2>/dev/null || echo unknown)"
    case "$fstype" in
        9p|v9fs|fuse*|drvfs|cifs|smb*|ntfs*|vboxsf|virtiofs)
            log WARN "##############################################################"
            log WARN "FS_DIR=$FS_DIR is on a '$fstype' filesystem (Windows/host bind mount)."
            log WARN "recon-all needs symlinks and does heavy small-file I/O: expect failures"
            log WARN "or a 5-20x slowdown. Put FS_DIR on a Docker named volume or a Linux path."
            log WARN "##############################################################"
            ;;
        *) log INFO "FS_DIR filesystem type: $fstype" ;;
    esac
    is_yes "$DRY_RUN" && return 0
    rm -f "$probe"
    if ln -s "$FS_DIR" "$probe" 2>/dev/null && [[ -L "$probe" ]]; then
        rm -f "$probe"
        return 0
    fi
    rm -f "$probe" 2>/dev/null || true
    die "cannot create symlinks in FS_DIR=$FS_DIR (filesystem: $fstype); recon-all would fail after hours. Use a Docker named volume or a Linux filesystem for FS_DIR."
}

# Subjects running in parallel (N_JOBS > 1) would race to create this link.
ensure_fsaverage() {
    local src="${FREESURFER_HOME:-/opt/freesurfer}/subjects/fsaverage" link="$FS_DIR/fsaverage"
    is_yes "$DRY_RUN" && return 0
    if [[ -e "$link" || ! -d "$src" ]]; then
        return 0
    fi
    ln -s "$src" "$link" 2>/dev/null || true
    [[ -e "$link" ]] || log WARN "could not link fsaverage into $FS_DIR (recon-all will try again)"
}

# recon-all refuses inputs whose field of view exceeds 256 mm unless -cw256.
fov_exceeds_256() {
    local t1="$1" i n d
    for i in 1 2 3; do
        n="$(fslval "$t1" "dim$i" | tr -d ' ')"
        d="$(fslval "$t1" "pixdim$i" | tr -d ' ')"
        if awk -v n="$n" -v d="$d" 'BEGIN { if (d < 0) d = -d; exit !(n * d > 256.0) }'; then
            log INFO "T1 FOV along axis $i = ${n} x ${d} mm exceeds 256 mm"
            return 0
        fi
    done
    return 1
}

mark_reused_recon() {
    local marker="$WORK_DIR/$SUB/.done/$STAGE.hash"
    # Only computes FP_STAGE_HASH here: the decision is file based (FORCE and
    # SKIP_EXISTING=no must not rebuild a finished reconstruction).
    FORCE=yes stage_should_run "$STAGE" "$SUB" -- RECON_FLAGS || true
    if [[ ! -f "$marker" ]]; then
        log INFO "adopting existing complete recon-all output: $SUBJ_DIR"
        stage_mark_done "$STAGE" "$SUB"
    elif [[ "$(head -n 1 "$marker")" != "$FP_STAGE_HASH" ]]; then
        # Not re-marked on purpose: a new marker would make every downstream
        # stage rerun although the reconstruction itself did not change.
        log WARN "existing recon-all output of $SUB is reused although RECON_FLAGS, the pipeline version or this script changed. A finished FreeSurfer run is never rebuilt automatically: delete $SUBJ_DIR to rebuild it."
    fi
}

# ----------------------------------------------------------------------------

if [[ "$ANAT_MODE" == synth ]]; then
    log INFO "ANAT_MODE=synth: no FreeSurfer reconstruction (surface branch unavailable)"
    exit 0
fi

if recon_complete; then
    log INFO "recon-all complete for $SUB (recon-all.done, lh/rh.pial, aseg.mgz)"
    mark_reused_recon
    exit 0
fi

if recon_is_running; then
    die "$SUBJ_DIR/scripts/IsRunning* exists: another recon-all owns this subject, or a previous run was killed. Check that no recon-all is running, remove the IsRunning file by hand, then rerun (nothing was deleted)."
fi

require_cmds recon-all fslval awk stat
check_license
fp_check_mem
check_filesystem
ensure_fsaverage

# Only to set FP_STAGE_HASH for stage_mark_done: the decision to run is file
# based, so FORCE=yes keeps the helper from logging a misleading "skip".
FORCE=yes stage_should_run "$STAGE" "$SUB" -- RECON_FLAGS || true

EXTRA_FLAGS=()
if [[ -n "${RECON_FLAGS// /}" ]]; then
    read -r -a EXTRA_FLAGS <<< "$RECON_FLAGS"
fi

T1="$(subject_t1w "$SUB" || true)"

MODE=new
if [[ -d "$SUBJ_DIR" ]]; then
    if recon_has_input; then
        MODE=resume
    else
        # Import never finished: neither resumable nor restartable with -i in place.
        stale="${SUBJ_DIR}.incomplete_$(date +%Y%m%d%H%M%S)"
        log WARN "$SUBJ_DIR has no mri/orig/001.mgz: moving it aside to $stale and starting from scratch"
        run mv "$SUBJ_DIR" "$stale"
    fi
fi

if [[ "$MODE" == resume ]]; then
    # -all starts over from the imported mri/orig/001.mgz; nothing is deleted first
    log INFO "incomplete reconstruction without IsRunning file: rerunning recon-all in place (no re-import)"
    CMD=(recon-all -sd "$FS_DIR" -s "$SUB" -all -parallel -threads "$NTHREADS" -no-isrunning)
else
    [[ -n "$T1" ]] || die "no T1w listed for $SUB in $MANIFEST"
    require_files "$T1"
    CMD=(recon-all -sd "$FS_DIR" -s "$SUB" -i "$T1" -all -parallel -threads "$NTHREADS")
fi

if [[ " ${RECON_FLAGS} " != *" -cw256 "* && -n "$T1" && -s "$T1" ]] && fov_exceeds_256 "$T1"; then
    log WARN "adding -cw256 (T1 field of view > 256 mm)"
    CMD+=(-cw256)
fi
if [[ ${#EXTRA_FLAGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_FLAGS[@]}")
fi

log INFO "recon-all mode: $MODE (threads: $NTHREADS; expect several hours)"
run "${CMD[@]}" || die "recon-all failed for $SUB; see $SUBJ_DIR/scripts/recon-all.log (rerunning this stage reruns recon-all in the same directory)"

if is_yes "$DRY_RUN"; then
    exit 0
fi
if ! recon_complete; then
    die "recon-all returned 0 but the output is incomplete (need scripts/recon-all.done, surf/lh.pial, surf/rh.pial, mri/aseg.mgz); see $SUBJ_DIR/scripts/recon-all.log"
fi
stage_mark_done "$STAGE" "$SUB"
