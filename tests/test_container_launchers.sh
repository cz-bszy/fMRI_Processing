#!/usr/bin/env bash
# Argument-level checks; no Docker/Singularity service, image build, or data processing.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
mkdir -p "$tmp/data" "$tmp/bin"
touch "$tmp/image.sif" "$tmp/license.txt"
cat > "$tmp/bin/runtime" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$CAPTURE"
exit "${STUB_EXIT:-0}"
EOF
chmod +x "$tmp/bin/runtime"
runner=(bash "$REPO/docker/run_singularity.sh" --image "$tmp/image.sif" --data "$tmp/data"
        --out "$tmp/output with spaces" --license "$tmp/license.txt" --runtime "$tmp/bin/runtime")
"${runner[@]}" --print -- --subjects-file '/data/list with spaces.txt' >/dev/null
[[ ! -e "$tmp/output with spaces" ]] || { echo 'print created directories'; exit 1; }
export CAPTURE="$tmp/captured"
SLURM_CPUS_PER_TASK=96 "${runner[@]}" --env 'VALUE=a,b $literal' -- --subjects-file '/data/list with spaces.txt' >/dev/null
for expected in --cleanenv --containall 'SLURM_CPUS_PER_TASK=96' 'VALUE=a,b $literal' '/data/list with spaces.txt'; do
    grep -Fx -- "$expected" "$CAPTURE" >/dev/null
done
grep -F -- "$tmp/data:/data:ro" "$CAPTURE" >/dev/null
grep -F -- "$tmp/license.txt:/opt/freesurfer/license.txt:ro" "$CAPTURE" >/dev/null
[[ -d "$tmp/output with spaces/work/container-home" ]]
set +e
STUB_EXIT=17 "${runner[@]}" >/dev/null
status=$?
set -e
[[ $status == 17 ]] || { echo "exit code not preserved: $status"; exit 1; }
if "${runner[@]}" --out "$tmp/data" --print >/dev/null 2>&1; then
    echo 'overlapping input/output accepted'; exit 1
fi
if "${runner[@]}" --image docker://example/image --print >/dev/null 2>&1; then
    echo 'remote image accepted'; exit 1
fi
echo 'PASS: Singularity launcher mounts, literal arguments, print-only, resource environment, exit status, input protection'
cat > "$tmp/bin/neurodocker" <<'EOF'
#!/usr/bin/env bash
if [[ $1 == --version ]]; then echo 'neurodocker, version 2.1.2'; exit 0; fi
printf '%s\n' "$@" > "$CAPTURE"
echo 'FROM ubuntu:22.04'
EOF
chmod +x "$tmp/bin/neurodocker"
NEURODOCKER="$tmp/bin/neurodocker" bash "$REPO/docker/generate_runtime.sh" > "$tmp/Dockerfile"
for expected in version=7.4.1 version=2.6.2 version=6.0.7.22 version=py311_25.7.0-2 env_name=neuro; do
    grep -Fx -- "$expected" "$CAPTURE" >/dev/null
done
grep -F 'ARG AFNI_SHA256' "$tmp/Dockerfile" >/dev/null
grep -F 'sha256sum -c -' "$tmp/Dockerfile" >/dev/null
grep -F '3dTproject -help' "$tmp/Dockerfile" >/dev/null
echo 'PASS: clean runtime generator fixed versions and checksum/build verification fragments (stub generator)'
