#!/bin/bash
# =============================================================================
# docker/run_docker.sh - run fMRI_Processing v2 inside the container
#                        (Linux, macOS, Git Bash on Windows)
#
# Same behaviour as docker/run_docker.ps1: repository, data and license are
# mounted read-only, the output directory read-write, and the FreeSurfer, work
# and resource directories live in Docker NAMED volumes (Windows/macOS bind
# mounts are slow and cannot hold the symbolic links that recon-all creates).
# =============================================================================
set -euo pipefail

# Git Bash rewrites arguments that look like POSIX paths (/out -> C:/Program
# Files/Git/out) before docker.exe sees them; this switches the rewriting off.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIN_MEMORY_GB=6

DATA=""
OUT=""
CONFIG="config/datasets/abide_test.conf"
LICENSE=""
IMAGE="zhaochang07/myubuntu:neuro-v2"
VOLUME_PREFIX="fmriproc"
CPUS=""
MEMORY=""
MODE=run
PRINT_ONLY=no
ENV_ARGS=()
PIPELINE_ARGS=()

usage() {
    cat <<'EOF'
usage: docker/run_docker.sh --data DIR --out DIR [options] [--] [run_pipeline.sh arguments]

  --data DIR             host directory with the raw dataset      -> /data (read-only)
  --out DIR              host output directory (created)          -> /out
  --config FILE          dataset conf relative to the repository
                         (default config/datasets/abide_test.conf); an absolute path
                         outside the repository is mounted with its directory at /config
  --license FILE         FreeSurfer license (default: $FS_LICENSE, $FREESURFER_HOME/license.txt,
                         ~/license.txt, or the Windows desktop)
  --image NAME           container image (default zhaochang07/myubuntu:neuro-v2)
  --volume-prefix P      prefix of the named volumes P_freesurfer, P_work, P_resources
                         (default fmriproc; use one prefix per dataset)
  --cpus N | --memory M  docker resource limits, e.g. --cpus 8 --memory 16g
  --min-memory-gb N      warn when the Docker engine has less RAM than this (default 6)
  --env KEY=VALUE        configuration override, may be repeated (e.g. --env SURFACE=no)
  --shell                interactive login shell with the same mounts
  --export-freesurfer    copy the P_freesurfer volume to <out>/freesurfer_export
                         (symbolic links are replaced by the files they point to)
  --print                print the docker command and exit
  -h, --help

Everything that is not listed above (or that follows '--') is passed to
run_pipeline.sh, e.g.  -s "sub-A sub-B" --stages "ingest anat_recon" --jobs 2 --force
Paths given to run_pipeline.sh options are container paths (/data, /out, /opt/fmriproc).
EOF
}

fail() {
    echo "run_docker.sh: $*" >&2
    exit 2
}

need_value() {   # OPTION N_REMAINING
    [[ "$2" -ge 2 ]] || fail "option $1 needs a value"
}

parse_cli() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --data)              need_value "$1" $#; DATA="$2"; shift 2 ;;
            --out)               need_value "$1" $#; OUT="$2"; shift 2 ;;
            --config)            need_value "$1" $#; CONFIG="$2"; shift 2 ;;
            --license)           need_value "$1" $#; LICENSE="$2"; shift 2 ;;
            --image)             need_value "$1" $#; IMAGE="$2"; shift 2 ;;
            --volume-prefix)     need_value "$1" $#; VOLUME_PREFIX="$2"; shift 2 ;;
            --cpus)              need_value "$1" $#; CPUS="$2"; shift 2 ;;
            --memory)            need_value "$1" $#; MEMORY="$2"; shift 2 ;;
            --min-memory-gb)     need_value "$1" $#
                                 [[ "$2" =~ ^[0-9]+$ ]] || fail "--min-memory-gb expects an integer, got: $2"
                                 MIN_MEMORY_GB="$2"; shift 2 ;;
            --env)               need_value "$1" $#
                                 [[ "$2" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || fail "--env expects KEY=VALUE, got: $2"
                                 ENV_ARGS+=(-e "$2"); shift 2 ;;
            --shell)             MODE=shell; shift ;;
            --export-freesurfer) MODE=export; shift ;;
            --print)             PRINT_ONLY=yes; shift ;;
            -h|--help)           usage; exit 0 ;;
            --)                  shift; PIPELINE_ARGS+=("$@"); break ;;
            *)                   PIPELINE_ARGS+=("$1"); shift ;;
        esac
    done
}

is_windows() {
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) return 0 ;;
        *) return 1 ;;
    esac
}

# host_path PATH : absolute path in the form the Docker engine of this host expects
host_path() {
    local path="$1" abs
    if [[ -d "$path" ]]; then
        abs="$(cd "$path" && pwd)"
    else
        abs="$(cd "$(dirname "$path")" && pwd)/$(basename "$path")"
    fi
    if is_windows; then
        cygpath -m "$abs"
    else
        echo "$abs"
    fi
}

check_mount_source() {   # PATH
    [[ "$1" != *,* ]] || fail "paths must not contain a comma (docker --mount syntax): $1"
}

default_license() {
    local candidate
    for candidate in "${FS_LICENSE:-}" "${FREESURFER_HOME:-/nonexistent}/license.txt" "$HOME/license.txt" \
                     "${USERPROFILE:-/nonexistent}/Desktop/license.txt"; do
        if [[ -n "$candidate" && -f "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 0
}

check_line_endings() {
    local name
    for name in run_pipeline.sh lib/common.sh; do
        if [[ -f "$REPO_DIR/$name" ]] && grep -q $'\r' "$REPO_DIR/$name"; then
            echo "WARNING: $name has CRLF line endings; bash will fail inside the container. Re-checkout with LF (README: troubleshooting)." >&2
            return 0
        fi
    done
}

check_docker_memory() {
    local info bytes gb
    command -v docker >/dev/null 2>&1 || fail "docker was not found on PATH"
    # A stopped engine may still answer with exit status 0, MemTotal 0 and no server version.
    info="$(docker info --format '{{.MemTotal}}|{{.ServerVersion}}' 2>/dev/null | tail -n 1)" || info=""
    [[ "$info" =~ ^[1-9][0-9]*\|.+$ ]] || fail "the Docker engine does not answer ('docker info') - start Docker first"
    bytes="${info%%|*}"
    gb=$(( bytes / 1024 / 1024 / 1024 ))
    echo "Docker engine memory: ~${gb} GB" >&2
    if [[ "$gb" -lt "$MIN_MEMORY_GB" ]]; then
        echo "WARNING: Docker sees only ~${gb} GB RAM (threshold --min-memory-gb ${MIN_MEMORY_GB}); recon-all, ANTs SyN and SynthSeg --robust want 8-16 GB." >&2
        echo "         Docker Desktop (Hyper-V): Settings > Resources > Advanced > Memory, Apply & restart." >&2
        echo "         Docker Desktop (WSL2): 'memory=16GB' under [wsl2] in %USERPROFILE%\\.wslconfig, 'wsl --shutdown', restart Docker Desktop." >&2
        echo "         The pipeline refuses heavy stages below MIN_MEM_GB of the dataset conf." >&2
    fi
}

# bash_word TEXT : TEXT single-quoted for the 'bash -lc' command string (no
# backslash escapes: they do not survive the Windows command line of docker.exe)
bash_word() {
    local escaped="'\\''"
    printf "'%s'" "${1//\'/$escaped}"
}

run_docker() {
    local printable
    printf -v printable '%q ' docker "$@"
    echo >&2
    echo "${printable% }" >&2
    echo >&2
    if [[ "$PRINT_ONLY" == yes ]]; then
        exit 0
    fi
    # Git Bash (mintty) has no console TTY for 'docker run -it': winpty provides one.
    if [[ "$MODE" == shell ]] && is_windows && command -v winpty >/dev/null 2>&1; then
        exec winpty docker "$@"
    fi
    exec docker "$@"
}

container_config() {   # sets CONF_PATH and CONF_MOUNT
    local abs rel
    CONF_MOUNT=""
    # a Windows path (E:\x\y.conf) must become POSIX before dirname/basename see it
    if [[ "$CONFIG" =~ ^[A-Za-z]:[\\/] ]] && is_windows; then
        CONFIG="$(cygpath -u "$CONFIG")"
    fi
    if [[ "$CONFIG" == /* ]]; then
        [[ -f "$CONFIG" ]] || fail "config not found: $CONFIG"
        abs="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
        if [[ "$abs" == "$REPO_DIR"/* ]]; then
            CONF_PATH="/opt/fmriproc/${abs#"$REPO_DIR"/}"
        else
            CONF_PATH="/config/$(basename "$abs")"
            CONF_MOUNT="$(host_path "$(dirname "$abs")")"
            check_mount_source "$CONF_MOUNT"
        fi
        return 0
    fi
    rel="${CONFIG#./}"
    [[ -f "$REPO_DIR/$rel" ]] || fail "config not found in the repository: $rel (give a path relative to $REPO_DIR)"
    CONF_PATH="/opt/fmriproc/$rel"
}

export_freesurfer() {
    local out script
    [[ -n "$OUT" ]] || fail "--out is required"
    [[ "$PRINT_ONLY" == yes ]] || mkdir -p "$OUT"
    out="$(host_path "$OUT")"
    check_mount_source "$out"
    [[ "$PRINT_ONLY" == yes ]] || check_docker_memory
    script='set -o pipefail; mkdir -p /to/freesurfer_export && '
    script+='tar -C /from --exclude=./fsaverage -chf - . | '
    script+='tar -C /to/freesurfer_export --no-same-owner --no-same-permissions -xf - && '
    script+='echo exported: $(ls /to/freesurfer_export | wc -l) entries in freesurfer_export'
    echo "Copying volume ${VOLUME_PREFIX}_freesurfer to $out/freesurfer_export" >&2
    run_docker run --rm \
        --mount "type=volume,source=${VOLUME_PREFIX}_freesurfer,target=/from,readonly" \
        --mount "type=bind,source=$out,target=/to" \
        "$IMAGE" bash -lc "$script"
}

main() {
    local repo out data license command word args=()
    parse_cli "$@"
    if [[ "$MODE" == export ]]; then
        export_freesurfer
    fi

    [[ -n "$OUT" ]] || fail "--out is required (see --help)"
    [[ "$PRINT_ONLY" == yes ]] || mkdir -p "$OUT"
    [[ -d "$OUT" || "$PRINT_ONLY" == yes ]] || fail "cannot create the output directory: $OUT"
    if [[ -d "$OUT" ]]; then out="$(host_path "$OUT")"; else out="$OUT"; fi
    repo="$(host_path "$REPO_DIR")"
    [[ -n "$LICENSE" ]] || LICENSE="$(default_license)"
    [[ -n "$LICENSE" && -f "$LICENSE" ]] || fail "FreeSurfer license not found (use --license FILE)"
    license="$(host_path "$LICENSE")"
    container_config
    check_mount_source "$repo"
    check_mount_source "$out"
    check_mount_source "$license"

    args=(run --rm --init)
    if [[ "$MODE" == shell ]]; then args+=(-it); fi
    if [[ -n "$CPUS" ]]; then args+=(--cpus "$CPUS"); fi
    if [[ -n "$MEMORY" ]]; then args+=(--memory "$MEMORY"); fi
    args+=(--mount "type=bind,source=$repo,target=/opt/fmriproc,readonly")
    if [[ -n "$DATA" || "$MODE" != shell ]]; then
        [[ -n "$DATA" ]] || fail "--data is required (see --help)"
        [[ -d "$DATA" ]] || fail "data directory not found: $DATA"
        data="$(host_path "$DATA")"
        check_mount_source "$data"
        args+=(--mount "type=bind,source=$data,target=/data,readonly")
    fi
    args+=(--mount "type=bind,source=$out,target=/out"
           --mount "type=bind,source=$license,target=/opt/freesurfer/license.txt,readonly"
           --mount "type=volume,source=${VOLUME_PREFIX}_freesurfer,target=/out/freesurfer"
           --mount "type=volume,source=${VOLUME_PREFIX}_work,target=/out/work"
           --mount "type=volume,source=${VOLUME_PREFIX}_resources,target=/out/resources")
    if [[ -n "$CONF_MOUNT" ]]; then
        args+=(--mount "type=bind,source=$CONF_MOUNT,target=/config,readonly")
    fi
    if [[ ${#ENV_ARGS[@]} -gt 0 ]]; then args+=("${ENV_ARGS[@]}"); fi
    args+=(-e "FMRIPROC_CONFIG=$CONF_PATH" -e FS_LICENSE=/opt/freesurfer/license.txt)

    check_line_endings
    [[ "$PRINT_ONLY" == yes ]] || check_docker_memory

    if [[ "$MODE" == shell ]]; then
        echo "Interactive shell; the pipeline is /opt/fmriproc/run_pipeline.sh (FMRIPROC_CONFIG is set)." >&2
        run_docker "${args[@]}" "$IMAGE" bash -l
    fi

    command="bash /opt/fmriproc/run_pipeline.sh -c $(bash_word "$CONF_PATH")"
    if [[ ${#PIPELINE_ARGS[@]} -gt 0 ]]; then
        for word in "${PIPELINE_ARGS[@]}"; do
            command+=" $(bash_word "$word")"
        done
    fi
    run_docker "${args[@]}" "$IMAGE" bash -lc "$command"
}

main "$@"
