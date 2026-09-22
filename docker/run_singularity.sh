#!/usr/bin/env bash
# Run the same Docker-derived environment as an immutable local SIF.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
DATA= OUT= IMAGE= LICENSE=${FS_LICENSE:-} WORK= RESOURCES= FREESURFER=
CONFIG=config/datasets/abide_test.conf
RUNTIME=singularity
PRINT=no
SHELL_MODE=no
ENV_PAIRS=()
PIPELINE_ARGS=()
fail() { echo "run_singularity.sh: $*" >&2; exit 2; }
usage() {
    cat <<'EOF'
usage: docker/run_singularity.sh --image FILE.sif --data DIR --out DIR --license FILE [options] [-- pipeline arguments]
  --config FILE       repository-relative or absolute dataset config
  --work DIR         writable work directory (default OUT/work)
  --resources DIR    prepared resource directory (default OUT/resources)
  --freesurfer DIR   writable FreeSurfer directory (default OUT/freesurfer)
  --runtime NAME     singularity (default), apptainer, or executable path
  --env KEY=VALUE    explicit configuration override; repeatable
  --shell            interactive login shell with identical mounts
  --print            print only; no runtime invocation or directory creation
Only an existing local image is accepted. This script does not download or submit jobs.
EOF
}
while (($#)); do
    case "$1" in
        --image|--data|--out|--license|--work|--resources|--freesurfer|--config|--runtime|--env)
            (($# >= 2)) || fail "$1 needs a value"
            option=$1; value=$2; shift 2
            case "$option" in
                --image) IMAGE=$value;; --data) DATA=$value;; --out) OUT=$value;;
                --license) LICENSE=$value;; --work) WORK=$value;; --resources) RESOURCES=$value;;
                --freesurfer) FREESURFER=$value;; --config) CONFIG=$value;; --runtime) RUNTIME=$value;;
                --env) [[ $value =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || fail '--env requires KEY=VALUE'; ENV_PAIRS+=("$value");;
            esac;;
        --print) PRINT=yes; shift;; --shell) SHELL_MODE=yes; shift;;
        -h|--help) usage; exit 0;;
        --) shift; PIPELINE_ARGS+=("$@"); break;;
        *) PIPELINE_ARGS+=("$1"); shift;;
    esac
done
[[ -f $IMAGE ]] || fail '--image must name an existing local SIF file'
[[ -d $DATA ]] || fail '--data must name an existing directory'
[[ -n $OUT ]] || fail '--out is required'
[[ -f $LICENSE ]] || fail '--license must name an existing file'
# realpath -m resolves existing symlinks as well as not-yet-created output paths.
canonical() {
    local path
    path=$(realpath -m -- "$1")
    [[ $path != *:* && $path != *,* && $path != *$'\n'* ]] || fail "unsupported bind-path delimiter: $path"
    printf '%s' "$path"
}
DATA=$(canonical "$DATA"); OUT=$(canonical "$OUT"); IMAGE=$(canonical "$IMAGE"); LICENSE=$(canonical "$LICENSE")
WORK=$(canonical "${WORK:-$OUT/work}")
RESOURCES=$(canonical "${RESOURCES:-$OUT/resources}")
FREESURFER=$(canonical "${FREESURFER:-$OUT/freesurfer}")
[[ $CONFIG == /* ]] || CONFIG="$REPO/$CONFIG"
[[ -f $CONFIG ]] || fail "config not found: $CONFIG"
CONFIG=$(canonical "$CONFIG")
# Reject writable aliases/parents/children of raw inputs or the code checkout.
for writable in "$OUT" "$WORK" "$RESOURCES" "$FREESURFER"; do
    for readonly in "$DATA" "$REPO"; do
        [[ $writable != "$readonly" && $writable != "$readonly/"* && $readonly != "$writable/"* ]] ||
            fail "writable directory overlaps read-only input: $writable and $readonly"
    done
done
for pair in "WORK:$WORK" "RESOURCES:$RESOURCES" "FREESURFER:$FREESURFER"; do
    # Distinct caches must not overwrite each other's contents.
    name=${pair%%:*}; path=${pair#*:}
    [[ $path != "$OUT" && $OUT != "$path/"* ]] || fail "$name must not equal or contain the output directory"
    for other in "$WORK" "$RESOURCES" "$FREESURFER"; do
        [[ $path != "$other/"* ]] || fail "$name overlaps another cache directory"
    done
done
[[ $WORK != "$RESOURCES" && $WORK != "$FREESURFER" && $RESOURCES != "$FREESURFER" ]] || fail 'cache directories must be distinct'
args=(exec --cleanenv --containall --pwd /opt/fmriproc
      --bind "$REPO:/opt/fmriproc:ro" --bind "$DATA:/data:ro"
      --bind "$OUT:/out:rw" --bind "$WORK:/out/work:rw"
      --bind "$RESOURCES:/out/resources:rw" --bind "$FREESURFER:/out/freesurfer:rw"
      --bind "$LICENSE:/opt/freesurfer/license.txt:ro"
      --bind "$CONFIG:/config/dataset.conf:ro"
      --bind "$WORK/container-tmp:/tmp:rw")
# env(1) inside the image avoids Singularity's comma splitting and shell evaluation
# of user values. Host module/Python settings are excluded by --cleanenv.
container_env=(HOME=/out/work/container-home TMPDIR=/tmp MPLCONFIGDIR=/out/work/matplotlib
               INPUT_DIR=/data OUT_DIR=/out FS_DIR=/out/freesurfer RESOURCE_DIR=/out/resources
               FS_LICENSE=/opt/freesurfer/license.txt FMRIPROC_CONFIG=/config/dataset.conf)
for name in SLURM_CPUS_PER_TASK SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU CPU_BUDGET MEMORY_BUDGET_GB; do
    [[ -z ${!name:-} ]] || container_env+=("$name=${!name}")
done
container_env+=("${ENV_PAIRS[@]}")
args+=("$IMAGE" env "${container_env[@]}")
if [[ $SHELL_MODE == yes ]]; then
    args+=(bash -l)
else
    args+=(bash -lc 'exec bash /opt/fmriproc/run_pipeline.sh -c /config/dataset.conf "$@"' fmriproc "${PIPELINE_ARGS[@]}")
fi
printf '%q ' "$RUNTIME" "${args[@]}"; printf '\n'
[[ $PRINT != yes ]] || exit 0
command -v "$RUNTIME" >/dev/null || fail "runtime not found: $RUNTIME"
mkdir -p "$OUT" "$WORK" "$RESOURCES" "$FREESURFER" "$WORK/container-tmp" "$WORK/container-home" "$WORK/matplotlib"
exec "$RUNTIME" "${args[@]}"
