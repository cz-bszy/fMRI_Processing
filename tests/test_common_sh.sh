#!/bin/bash
# =============================================================================
# tests/test_common_sh.sh - bash unit checks that need no neuroimaging tools
#   part 1: lib/common.sh (is_yes, fp_init precedence, run(), stage markers,
#           manifest helpers)
#   part 2: run_pipeline.sh driven with fake stage scripts (FMRIPROC_STAGE_DIR):
#           stage order, failure isolation, status table, exit status
# Only bash, coreutils, awk, sed and md5sum are required.
# usage: bash tests/test_common_sh.sh        (exit 1 when a check fails)
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMON="$REPO/lib/common.sh"
PIPELINE="$REPO/run_pipeline.sh"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/fmriproc_test.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

N_PASS=0
N_FAIL=0

# The pipeline must see the same configuration on every machine.
CONFIG_VARS=(INPUT_DIR OUT_DIR FS_DIR RESOURCE_DIR PYTHON_BIN INPUT_LAYOUT ACQ_TABLE SUBJECT_LIST
    NTHREADS N_JOBS MIN_MEM_GB SKIP_EXISTING FORCE DRY_RUN STAGES ANAT_MODE SURFACE MNI_RES STC
    FILTER_MODE POLORT ATLASES CUSTOM_ATLASES EPI_MASK_METHOD FMRIPROC_CONFIG FMRIPROC_STAGE_DIR
    FP_LOGFILE FP_STATUS_DIR FP_RUN_ID FP_PIPELINE_LOG VERBOSE CPU_BUDGET MEMORY_BUDGET_GB
    SLURM_CPUS_PER_TASK SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU)
unset "${CONFIG_VARS[@]}" 2>/dev/null || true

pass() {
    N_PASS=$(( N_PASS + 1 ))
    echo "ok    $1"
}

fail() {
    N_FAIL=$(( N_FAIL + 1 ))
    echo "FAIL  $1" >&2
}

# check NAME CMD... : CMD must succeed
check() {
    local name="$1"
    shift
    if "$@" >/dev/null 2>&1; then pass "$name"; else fail "$name"; fi
}

# check_not NAME CMD... : CMD must fail
check_not() {
    local name="$1"
    shift
    if "$@" >/dev/null 2>&1; then fail "$name"; else pass "$name"; fi
}

# check_eq NAME EXPECTED ACTUAL
check_eq() {
    if [[ "$2" == "$3" ]]; then
        pass "$1"
    else
        fail "$1 (expected '$2', got '$3')"
    fi
}

# check_status NAME EXPECTED CMD... : exit status of CMD
check_status() {
    local name="$1" expected="$2" rc=0
    shift 2
    "$@" >/dev/null 2>&1 || rc=$?
    check_eq "$name" "$expected" "$rc"
}

# in_common SNIPPET [ARGS...] : evaluate SNIPPET in a fresh shell that sourced
# lib/common.sh ($0 = this file, which stage_should_run hashes as "the script")
in_common() {
    local snippet="$1"
    shift
    bash -c "set -euo pipefail; source '$COMMON'; $snippet" "${BASH_SOURCE[0]}" "$@"
}

write_conf() {   # FILE OUT_DIR [extra lines...]
    local file="$1" out="$2"
    shift 2
    {
        echo ": \"\${OUT_DIR:=$out}\""
        echo ": \"\${INPUT_DIR:=$TMP/input}\""
        echo ': "${MIN_MEM_GB:=0}"'
        if [[ $# -gt 0 ]]; then printf '%s\n' "$@"; fi
    } > "$file"
}

# ============================================================================
# part 1: lib/common.sh
# ============================================================================

test_is_yes() {
    local v
    for v in yes YES true True 1 on On; do
        check "is_yes $v" in_common 'is_yes "$1"' "$v"
    done
    for v in no false 0 off "" maybe; do
        check_not "is_yes '$v' is false" in_common 'is_yes "$1"' "$v"
    done
}

test_fp_init_precedence() {
    local conf="$TMP/prec.conf" out="$TMP/prec_out"
    write_conf "$conf" "$out" ': "${NTHREADS:=3}"' ': "${SURFACE:=no}"'
    check_eq "dataset conf overrides default.conf" 3 "$(in_common 'fp_init -c "$1" 2>/dev/null; echo "$NTHREADS"' "$conf")"
    check_eq "environment overrides dataset conf" 7 "$(NTHREADS=7 in_common 'fp_init -c "$1"; echo "$NTHREADS"' "$conf")"
    check_eq "default.conf fills the rest" 2 "$(in_common 'fp_init -c "$1"; echo "$POLORT"' "$conf")"
    check_eq "FMRIPROC_CONFIG replaces -c" 3 "$(FMRIPROC_CONFIG="$conf" in_common 'fp_init; echo "$NTHREADS"')"
    check_eq "remaining arguments reach FP_ARGS" "sub-01|--flag" \
        "$(in_common 'fp_init sub-01 -c "$1" --flag; IFS="|"; echo "${FP_ARGS[*]}"' "$conf")"
    check_eq "FMRIPROC_CONFIG is exported as an absolute path" "$conf" \
        "$(cd "$TMP" && in_common 'fp_init -c prec.conf; bash -c "echo \$FMRIPROC_CONFIG"')"
    check_eq "derived directories" "$out/rawdata/manifest.tsv|$out/freesurfer|$out/freesurfer" \
        "$(in_common 'fp_init -c "$1"; echo "$MANIFEST|$FS_DIR|$SUBJECTS_DIR"' "$conf")"
    check "directories are created" test -d "$out/work" -a -d "$out/derivatives" -a -d "$out/logs"
    check_eq "OMP_NUM_THREADS follows NTHREADS" 5 \
        "$(NTHREADS=5 in_common 'fp_init -c "$1"; bash -c "echo \$OMP_NUM_THREADS"' "$conf")"
    check_eq "PYTHONPATH starts with the package directory" "$REPO/py" \
        "$(in_common 'fp_init -c "$1"; echo "${PYTHONPATH%%:*}"' "$conf")"
}

test_fp_init_validation() {
    local conf="$TMP/val.conf"
    write_conf "$conf" "$TMP/val_out" ': "${SURFACE:=no}"'
    check_status "missing config file is fatal" 1 in_common 'fp_init -c "$1"' "$TMP/does_not_exist.conf"
    check_status "OUT_DIR is mandatory" 1 in_common 'fp_init'
    check_status "NTHREADS must be a positive integer" 1 env NTHREADS=0 bash -c "source '$COMMON'; fp_init -c '$conf'"
    check_status "unknown ANAT_MODE is fatal" 1 env ANAT_MODE=fast bash -c "source '$COMMON'; fp_init -c '$conf'"
    check_status "SURFACE=yes needs ANAT_MODE=freesurfer" 1 env SURFACE=yes ANAT_MODE=synth bash -c "source '$COMMON'; fp_init -c '$conf'"
    check_status "SURFACE=yes needs MNI_RES=2" 1 env SURFACE=yes MNI_RES=3 bash -c "source '$COMMON'; fp_init -c '$conf'"
    check_status "SURFACE=yes with freesurfer and 2 mm is accepted" 0 env SURFACE=yes MNI_RES=2 bash -c "source '$COMMON'; fp_init -c '$conf'"
}

test_run_and_require() {
    local conf="$TMP/run.conf" out="$TMP/run_out" logfile
    write_conf "$conf" "$out"
    logfile="$out/logs/sub-01/teststage.log"
    check_eq "run() returns the exit status of the command" 3 \
        "$(in_common 'fp_init -c "$1"; rc=0; run bash -c "exit 3" 2>/dev/null || rc=$?; echo $rc' "$conf")"
    in_common 'fp_init -c "$1"; fp_set_log sub-01 teststage; run echo hello-from-run' "$conf" >"$TMP/run_stdout.txt" 2>/dev/null
    check "command output goes to the stage log" grep -q '^hello-from-run$' "$logfile"
    check "the command line is logged" grep -q '+ echo hello-from-run' "$logfile"
    check_eq "run() keeps stdout clean" "" "$(cat "$TMP/run_stdout.txt")"
    in_common 'fp_init -c "$1"; DRY_RUN=yes; run touch "$2"' "$conf" "$TMP/dry_run_marker" 2>/dev/null
    check_not "DRY_RUN=yes does not execute" test -e "$TMP/dry_run_marker"
    check_status "require_cmds accepts existing commands" 0 in_common 'require_cmds bash awk'
    check_status "require_cmds dies on a missing command" 1 in_common 'require_cmds bash no_such_command_xyz'
    : > "$TMP/empty_file"
    check_status "require_files dies on an empty file" 1 in_common 'require_files "$1"' "$TMP/empty_file"
    check_status "require_files accepts a non-empty file" 0 in_common 'require_files "$1"' "$conf"
    check_eq "DEBUG lines need VERBOSE=yes" "" "$(in_common 'log DEBUG hidden' 2>&1)"
    check "DEBUG lines with VERBOSE=yes" bash -c "VERBOSE=yes bash -c 'source \"$COMMON\"; log DEBUG shown' 2>&1 | grep -q shown"
}

test_stage_markers() {
    local conf="$TMP/stage.conf" out="$TMP/stage_out" init
    write_conf "$conf" "$out"
    init='fp_init -c "$1" 2>/dev/null; exec 2>/dev/null; MYVAR="${MYVAR:-a}"; '

    check_status "first call: stage must run" 0 in_common "$init"'stage_should_run st sub-01 -- MYVAR' "$conf"
    check_status "no marker without stage_mark_done" 0 in_common "$init"'stage_should_run st sub-01 -- MYVAR' "$conf"
    in_common "$init"'stage_should_run st sub-01 -- MYVAR; stage_mark_done st sub-01' "$conf"
    check "marker file name" test -s "$out/work/sub-01/.done/st.hash"
    check_status "unchanged hash: skipped" 1 in_common "$init"'stage_should_run st sub-01 -- MYVAR' "$conf"
    check_status "changed parameter: run again" 0 env MYVAR=b bash -c "set -euo pipefail; source '$COMMON'; $init"'stage_should_run st sub-01 -- MYVAR' "${BASH_SOURCE[0]}" "$conf"
    check_status "unlisted parameter does not matter" 1 env OTHER=zzz bash -c "set -euo pipefail; source '$COMMON'; $init"'stage_should_run st sub-01 -- MYVAR' "${BASH_SOURCE[0]}" "$conf"
    check_status "FORCE=yes: run again" 0 env FORCE=yes bash -c "set -euo pipefail; source '$COMMON'; $init"'stage_should_run st sub-01 -- MYVAR' "${BASH_SOURCE[0]}" "$conf"
    check_status "SKIP_EXISTING=no: run again" 0 env SKIP_EXISTING=no bash -c "set -euo pipefail; source '$COMMON'; $init"'stage_should_run st sub-01 -- MYVAR' "${BASH_SOURCE[0]}" "$conf"
    check_status "other subject has its own marker" 0 in_common "$init"'stage_should_run st sub-02 -- MYVAR' "$conf"

    # per-run markers
    in_common "$init"'stage_should_run fn sub-01 sub-01_task-rest -- MYVAR; stage_mark_done fn sub-01 sub-01_task-rest' "$conf"
    check "per-run marker file name" test -s "$out/work/sub-01/.done/fn__sub-01_task-rest.hash"
    check_status "per-run marker: skipped" 1 in_common "$init"'stage_should_run fn sub-01 sub-01_task-rest -- MYVAR' "$conf"
    check_status "another run of the subject must run" 0 in_common "$init"'stage_should_run fn sub-01 sub-01_task-rest_run-2 -- MYVAR' "$conf"

    # dependency invalidation: redoing the upstream stage changes its marker
    in_common "$init"'stage_should_run up sub-01 -- MYVAR; stage_mark_done up sub-01' "$conf"
    in_common "$init"'stage_should_run down sub-01 --dep up -- MYVAR; stage_mark_done down sub-01' "$conf"
    check_status "downstream up to date" 1 in_common "$init"'stage_should_run down sub-01 --dep up -- MYVAR' "$conf"
    sleep 1
    FORCE=yes in_common "$init"'stage_should_run up sub-01 -- MYVAR; stage_mark_done up sub-01' "$conf"
    check_status "upstream redone: downstream must run" 0 in_common "$init"'stage_should_run down sub-01 --dep up -- MYVAR' "$conf"
    check_status "missing upstream marker: must run" 0 in_common "$init"'stage_should_run down sub-01 --dep never_ran -- MYVAR' "$conf"

    # location independence: the same code elsewhere (a frozen copy in
    # logs/code_<run>) and values containing $REPO_DIR must give the same hash,
    # otherwise every frozen run would redo every stage
    local copy="$TMP/relocated" snippet h1 h2
    mkdir -p "$copy"
    cp -R "$(dirname "$COMMON")" "$copy/lib"
    cp -R "$(dirname "$COMMON")/../config" "$copy/config"
    cp -R "$(dirname "$COMMON")/../py" "$copy/py"
    snippet='fp_init -c "$1" 2>/dev/null; exec 2>/dev/null; MYVAR="$REPO_DIR/parcellations/a.nii"; stage_should_run 03_func_prep sub-01 -- MYVAR || true; echo "$FP_STAGE_HASH"'
    h1="$(bash -c "set -euo pipefail; source '$COMMON'; $snippet" "${BASH_SOURCE[0]}" "$conf")"
    h2="$(bash -c "set -euo pipefail; source '$copy/lib/common.sh'; $snippet" "${BASH_SOURCE[0]}" "$conf")"
    check "stage hash computed" test -n "$h1"
    check_eq "stage hash does not depend on the code location" "$h1" "$h2"

    in_common "$init"'DRY_RUN=yes; stage_should_run dry sub-01 -- MYVAR; stage_mark_done dry sub-01' "$conf"
    check_not "DRY_RUN=yes writes no marker" test -e "$out/work/sub-01/.done/dry.hash"
    check_status "stage_should_run rejects unknown options" 1 in_common "$init"'stage_should_run st sub-01 --bogus x -- MYVAR' "$conf"
}

test_allocation_limits() {
    local root="$TMP/cgroup"
    mkdir -p "$root"
    printf '200000 100000\n' > "$root/cpu.max"
    printf '8589934592\n' > "$root/memory.max"
    printf 'MemTotal:       67108864 kB\n' > "$root/meminfo"
    check_eq "CPU quota beats host capacity" 2 \
        "$(in_common 'nproc() { echo 64; }; fp_cpu_limit "$1"' "$root")"
    check_eq "Slurm CPU allocation is a cap" 1 \
        "$(in_common 'nproc() { echo 64; }; SLURM_CPUS_PER_TASK=1; fp_cpu_limit "$1"' "$root")"
    check_eq "explicit CPU budget cannot raise cgroup quota" 2 \
        "$(in_common 'nproc() { echo 64; }; CPU_BUDGET=16; fp_cpu_limit "$1"' "$root")"
    check_eq "memory quota beats host capacity" 8388608 \
        "$(in_common 'fp_mem_limit_kb "$1" "$1/meminfo"' "$root")"
    check_eq "Slurm node memory is a cap" 4194304 \
        "$(in_common 'SLURM_MEM_PER_NODE=4096; fp_mem_limit_kb "$1" "$1/meminfo"' "$root")"
    check_eq "Slurm per-CPU memory uses allocated CPUs" 2097152 \
        "$(in_common 'SLURM_MEM_PER_CPU=1024; SLURM_CPUS_PER_TASK=2; fp_mem_limit_kb "$1" "$1/meminfo"' "$root")"
    printf 'max 100000\n' > "$root/cpu.max"
    check_eq "per-worker OMP setting is not the CPU allocation" 8 \
        "$(in_common 'nproc() { echo "${OMP_NUM_THREADS:-64}"; }; OMP_NUM_THREADS=4; SLURM_CPUS_PER_TASK=8; fp_cpu_limit "$1"' "$root")"
    check_eq "explicit memory budget cannot raise detected limits" 8388608 \
        "$(in_common 'MEMORY_BUDGET_GB=32; fp_mem_limit_kb "$1" "$1/meminfo"' "$root")"
    check_status "parallel workers cannot exceed CPU allocation" 1 in_common \
        'fp_cpu_limit() { echo 4; }; fp_mem_limit_kb() { echo 16777216; }; N_JOBS=2; NTHREADS=4; MIN_MEM_GB=6; fp_check_resources'
    check_status "parallel workers cannot exceed memory allocation" 1 in_common \
        'fp_cpu_limit() { echo 8; }; fp_mem_limit_kb() { echo 8388608; }; N_JOBS=2; NTHREADS=4; MIN_MEM_GB=6; fp_check_resources'
}

test_python_source_marker() {
    local conf="$TMP/source.conf" fixture="$TMP/source_repo"
    write_conf "$conf" "$TMP/source_out"
    mkdir -p "$fixture/py/fmriproc" "$fixture/lib"
    cp "$COMMON" "$fixture/lib/common.sh"
    printf '# original statistic\n' > "$fixture/py/fmriproc/strategies.py"
    printf '# unchanged shared helper\n' > "$fixture/py/fmriproc/utils.py"
    local init='fp_init -c "$1"; REPO_DIR="$2"; '
    in_common "$init"'stage_should_run 05_denoise sub-01 --; stage_mark_done 05_denoise sub-01' "$conf" "$fixture" 2>/dev/null
    check_status "unchanged Python source permits reuse" 1 in_common "$init"'stage_should_run 05_denoise sub-01 --' "$conf" "$fixture"
    printf '# fixed statistic\n' > "$fixture/py/fmriproc/strategies.py"
    check_status "changed Python computation invalidates stage" 0 in_common "$init"'stage_should_run 05_denoise sub-01 --' "$conf" "$fixture"
}

write_fake_manifest() {   # FILE
    {
        printf 'subject\tsession\ttask\trun\tgroup\tbold\tt1w\trun_label\n'
        printf 'sub-01\t-\trest\t-\tNYU\t/raw/sub-01/func/sub-01_task-rest_bold.nii.gz\t/raw/sub-01/anat/sub-01_T1w.nii.gz\tsub-01_task-rest\n'
        printf 'sub-02\tses-1\trest\t1\tSDSU\t/raw/sub-02/ses-1/func/a_bold.nii.gz\t-\tsub-02_ses-1_task-rest_run-1\n'
        printf 'sub-02\tses-1\trest\t2\tSDSU\t/raw/sub-02/ses-1/func/b_bold.nii.gz\t/raw/sub-02/ses-1/anat/sub-02_ses-1_T1w.nii.gz\tsub-02_ses-1_task-rest_run-2\n'
        printf 'sub-021\t-\trest\t-\t-\t/raw/sub-021/func/c_bold.nii.gz\t/raw/sub-021/anat/sub-021_T1w.nii.gz\tsub-021_task-rest\n'
    } > "$1"
}

test_manifest() {
    local conf="$TMP/mani.conf" out="$TMP/mani_out"
    write_conf "$conf" "$out"
    mkdir -p "$out/rawdata"
    write_fake_manifest "$out/rawdata/manifest.tsv"
    check_eq "manifest_subjects is unique and sorted" "sub-01 sub-02 sub-021" \
        "$(in_common 'fp_init -c "$1"; manifest_subjects | tr "\n" " " | sed "s/ $//"' "$conf")"
    check_eq "manifest_runs matches the whole id (sub-02 is not sub-021)" 2 \
        "$(in_common 'fp_init -c "$1"; manifest_runs sub-02 | wc -l | tr -d " "' "$conf")"
    check_eq "manifest_runs keeps the eight columns" "sub-02_ses-1_task-rest_run-2" \
        "$(in_common 'fp_init -c "$1"; manifest_runs sub-02 | awk -F"\t" "NR == 2 {print \$8}"' "$conf")"
    check_eq "manifest_runs of an unknown subject is empty" "" \
        "$(in_common 'fp_init -c "$1"; manifest_runs sub-99' "$conf")"
    check_eq "subject_t1w" "/raw/sub-01/anat/sub-01_T1w.nii.gz" \
        "$(in_common 'fp_init -c "$1"; subject_t1w sub-01' "$conf")"
    check_eq "subject_t1w skips rows without a T1w" "/raw/sub-02/ses-1/anat/sub-02_ses-1_T1w.nii.gz" \
        "$(in_common 'fp_init -c "$1"; subject_t1w sub-02' "$conf")"
    check_eq "naming helpers" "$out/derivatives/sub-01/anat|$out/derivatives/sub-01/func|$out/derivatives/sub-01/figures|$out/work/sub-01/anat|$out/work/sub-01/func/R" \
        "$(in_common 'fp_init -c "$1"; echo "$(anat_dir sub-01)|$(func_dir sub-01)|$(fig_dir sub-01)|$(work_anat sub-01)|$(work_func sub-01 R)"' "$conf")"
    check_eq "bold_json" "/x/sub-01_task-rest_bold.json" "$(in_common 'bold_json /x/sub-01_task-rest_bold.nii.gz')"
    rm -f "$out/rawdata/manifest.tsv"
    check_status "manifest_runs dies without a manifest" 1 in_common 'fp_init -c "$1"; manifest_runs sub-01' "$conf"
}

test_json_get() {
    local py="" candidate
    # 'python3' may be a non-functional Windows store alias: probe before use
    for candidate in "${PYTHON_BIN:-}" python3 python; do
        if [[ -n "$candidate" ]] && "$candidate" -c 'import json' >/dev/null 2>&1; then
            py="$candidate"
            break
        fi
    done
    if [[ -z "$py" ]]; then
        echo "skip  json_get (no python on this machine)"
        return 0
    fi
    printf '{"tr": 2.0, "stc_applied": true, "name": "x y", "list": [1, 2], "empty": []}\n' > "$TMP/j.json"
    check_eq "json_get number" "2.0" "$(PYTHON_BIN="$py" in_common 'json_get "$1" tr' "$TMP/j.json")"
    check_eq "json_get bool prints yes/no" "yes" "$(PYTHON_BIN="$py" in_common 'json_get "$1" stc_applied' "$TMP/j.json")"
    check_eq "json_get default for a missing key" "dflt" "$(PYTHON_BIN="$py" in_common 'json_get "$1" nokey dflt' "$TMP/j.json")"
    check_eq "json_get default for a missing file" "dflt" "$(PYTHON_BIN="$py" in_common 'json_get "$1" tr dflt' "$TMP/none.json")"
    check_status "json_has_list: non-empty list" 0 env PYTHON_BIN="$py" bash -c "source '$COMMON'; json_has_list '$TMP/j.json' list"
    check_status "json_has_list: empty list" 1 env PYTHON_BIN="$py" bash -c "source '$COMMON'; json_has_list '$TMP/j.json' empty"
}

# ============================================================================
# part 2: run_pipeline.sh with fake stage scripts
# ============================================================================

# Every fake stage appends "<script> <args without -c conf>|DRY_RUN|FORCE" to
# $OUT_DIR/calls.txt. 00_ingest.sh writes the manifest, 04_confounds.sh fails
# for sub-B, PYTHON_BIN is a stub so that the pre-flight needs no real python.
make_fake_stages() {   # DIR
    local dir="$1" name
    mkdir -p "$dir"
    for name in 00_ingest 01_anat_recon 02_anat_prep 03_func_prep 04_confounds 05_denoise 06_surface \
                07_timeseries 08_qc 09_group_qc 10_validate fetch_resources; do
        cat > "$dir/$name.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
name="$(basename "$0")"
[[ "$1" == -c ]] && shift 2
echo "$name $*|${DRY_RUN:-unset}|${FORCE:-unset}" >> "$OUT_DIR/calls.txt"
echo "$(cd "$(dirname "$0")" && pwd)|${FMRIPROC_CONFIG:-unset}" >> "$OUT_DIR/stage_dirs.txt"
echo "fake $name $*"
if [[ "$name" == 00_ingest.sh ]]; then
    mkdir -p "$OUT_DIR/rawdata"
    {
        printf 'subject\tsession\ttask\trun\tgroup\tbold\tt1w\trun_label\n'
        for s in sub-A sub-B sub-C; do
            printf '%s\t-\trest\t-\tSITE\t/raw/%s_bold.nii.gz\t/raw/%s_T1w.nii.gz\t%s_task-rest\n' "$s" "$s" "$s" "$s"
        done
    } > "$OUT_DIR/rawdata/manifest.tsv"
fi
if [[ "$name" == 04_confounds.sh && "${1:-}" == sub-B ]]; then
    echo "simulated failure" >&2
    exit 3
fi
if [[ "$name" == fetch_resources.sh && "${1:-}" == --check ]]; then
    [[ -e "$OUT_DIR/resources/complete" ]] || exit 1
fi
if [[ "$name" == fetch_resources.sh && $# -eq 0 ]]; then
    mkdir -p "$OUT_DIR/resources"
    : > "$OUT_DIR/resources/complete"
fi
exit 0
EOF
    done
    printf '#!/bin/bash\nexit 0\n' > "$dir/fakepython"
    chmod +x "$dir/fakepython" "$dir"/*.sh
}

# pipeline OUT_DIR ARGS... : run the orchestrator against the fake stages
pipeline() {
    local out="$1"
    shift
    # ':=' cannot express an empty value (default.conf would refill it): one blank = no atlas
    write_conf "$out.conf" "$out" ': "${SURFACE:=no}"' ': "${ATLASES:= }"'
    env FMRIPROC_STAGE_DIR="$TMP/fake_stages" PYTHON_BIN="$TMP/fake_stages/fakepython" \
        bash "$PIPELINE" -c "$out.conf" "$@"
}

# status_of FILE SUBJECT STAGE
status_of() {
    awk -F'\t' -v s="$2" -v g="$3" 'NR > 1 && $1 == s && $2 == g {print $3}' "$1"
}

latest_status() {   # OUT_DIR
    # C collation: in UTF-8 locales sort ignores punctuation and would put
    # status_<stamp>.tsv after status_<stamp>_1.tsv (same-second runs)
    # shellcheck disable=SC2012
    ls -1 "$1"/logs/status_*.tsv | LC_ALL=C sort | tail -n 1
}

test_pipeline_failure_isolation() {
    local out="$TMP/pipe1" rc=0 status calls
    pipeline "$out" --stages "ingest confounds surface timeseries validate qc" >"$out.stdout" 2>"$out.stderr" || rc=$?
    check_eq "exit status 1 when one subject failed" 1 "$rc"
    status="$(latest_status "$out")"
    calls="$out/calls.txt"
    check_eq "status table header" "subject stage status seconds log note" "$(head -n 1 "$status" | tr '\t' ' ')"
    check_eq "dataset ingest ok" ok "$(status_of "$status" - ingest)"
    check_eq "sub-A confounds ok" ok "$(status_of "$status" sub-A confounds)"
    check_eq "sub-A qc ok" ok "$(status_of "$status" sub-A qc)"
    check_eq "sub-B confounds failed" failed "$(status_of "$status" sub-B confounds)"
    check_eq "sub-B timeseries skipped" skipped "$(status_of "$status" sub-B timeseries)"
    check_eq "sub-B validate skipped" skipped "$(status_of "$status" sub-B validate)"
    check_eq "sub-B qc skipped" skipped "$(status_of "$status" sub-B qc)"
    check_eq "sub-C processed after the failure of sub-B" ok "$(status_of "$status" sub-C qc)"
    check_eq "surface skipped when SURFACE=no" skipped "$(status_of "$status" sub-A surface)"
    check_not "06_surface.sh is not called when SURFACE=no" grep -q '^06_surface.sh' "$calls"
    check_not "stages after the failure are not called for sub-B" grep -q '^0[78]_.* sub-B' "$calls"
    check_eq "per-subject order: confounds, timeseries, validate, qc" \
        "04_confounds.sh 07_timeseries.sh 10_validate.sh 08_qc.sh" \
        "$(grep ' sub-A|' "$calls" | cut -d' ' -f1 | tr '\n' ' ' | sed 's/ $//')"
    check_eq "group validation recorded" ok "$(status_of "$status" - validate_group)"
    check_eq "group_qc runs because qc was selected" ok "$(status_of "$status" - group_qc)"
    check_eq "dataset stages come last, validate --group before group_qc" \
        "10_validate.sh --group,09_group_qc.sh " \
        "$(tail -n 2 "$calls" | cut -d'|' -f1 | tr '\n' ',' | sed 's/,$//')"
    check_eq "ingest is called first, once, without arguments" "00_ingest.sh " "$(head -n 1 "$calls" | cut -d'|' -f1)"
    check "summary names the failed stage" grep -q 'FAILED  sub-B  confounds' "$out.stdout"
    check "tool_versions.json is written" test -s "$out/logs/tool_versions.json"
    check "pipeline log keeps the stage output" grep -q '\[sub-B\] simulated failure' "$(ls -1 "$out"/logs/pipeline_*.log | tail -n 1)"
    check_not "status fragments are removed" ls -d "$out"/logs/status_*.d
    check_not "no automatic fetch when ATLASES is blank and SURFACE=no" grep -q '^fetch_resources.sh' "$calls"
    check "the calls were recorded" test -s "$calls"
}

test_pipeline_selection_and_parallel() {
    local out="$TMP/pipe2" rc=0 status
    pipeline "$out" --stages ingest >/dev/null 2>&1 || rc=$?
    check_eq "ingest only: exit status 0" 0 "$rc"
    check_eq "--list-subjects prints the manifest subjects" "sub-A sub-B sub-C" \
        "$(pipeline "$out" --list-subjects 2>/dev/null | tr '\n' ' ' | sed 's/ $//')"
    check_eq "--list-subjects intersects with -s (prefix optional, commas allowed)" "sub-A sub-C" \
        "$(pipeline "$out" --list-subjects -s "A,sub-C" -s Z 2>/dev/null | tr '\n' ' ' | sed 's/ $//')"
    printf '# comment\nC\n\nsub-A  # trailing\n' > "$TMP/subjects.txt"
    check_eq "--subjects-file" "sub-A sub-C" \
        "$(pipeline "$out" --list-subjects --subjects-file "$TMP/subjects.txt" 2>/dev/null | tr '\n' ' ' | sed 's/ $//')"
    check_eq "SUBJECT_LIST from the configuration/environment" "sub-C" \
        "$(printf 'C\n' > "$TMP/list2.txt"; SUBJECT_LIST="$TMP/list2.txt" pipeline "$out" --list-subjects 2>/dev/null | tr '\n' ' ' | sed 's/ $//')"

    sleep 1
    rm -f "$out/calls.txt"
    rc=0
    pipeline "$out" --stages "confounds qc" -s "A C" --jobs 2 --force >/dev/null 2>&1 || rc=$?
    check_eq "parallel run of the good subjects: exit status 0" 0 "$rc"
    status="$(latest_status "$out")"
    check_eq "parallel: sub-A ok" ok "$(status_of "$status" sub-A qc)"
    check_eq "parallel: sub-C ok" ok "$(status_of "$status" sub-C qc)"
    check_eq "parallel: sub-B not processed" "" "$(status_of "$status" sub-B qc)"
    check_not "ingest is not repeated when it is not selected" grep -q '^00_ingest.sh' "$out/calls.txt"
    check_eq "--force is exported to the stages" yes "$(grep '^04_confounds.sh sub-A|' "$out/calls.txt" | cut -d'|' -f3)"
    check_eq "--jobs reaches the configuration" 1 "$(grep -c 'N_JOBS=2 ' "$(ls -1 "$out"/logs/pipeline_*.log | tail -n 1)")"

    sleep 1
    rc=0
    pipeline "$out" --stages "confounds" -s "A B Z" --jobs 3 >/dev/null 2>&1 || rc=$?
    check_eq "parallel run with a failing and an unknown subject: exit status 1" 1 "$rc"
    status="$(latest_status "$out")"
    check_eq "parallel: failure recorded" failed "$(status_of "$status" sub-B confounds)"
    check_eq "parallel: other subject unaffected" ok "$(status_of "$status" sub-A confounds)"
    check_eq "unknown subject recorded as failed" failed "$(status_of "$status" sub-Z ingest)"
    check_eq "no group stage when neither qc nor validate is selected" "" "$(status_of "$status" - group_qc)"
}

test_pipeline_cli() {
    local out="$TMP/pipe3" rc=0 status
    check_status "unknown option: usage error" 2 pipeline "$out" --bogus
    check_status "option without value: usage error" 2 pipeline "$out" --stages
    check_status "missing configuration: usage error" 2 bash "$PIPELINE" --list-subjects
    check_status "unknown stage is fatal" 1 pipeline "$out" --stages "ingest nonsense"
    check_status "--help" 0 bash "$PIPELINE" --help
    check_status "no manifest and no ingest: failure, not a crash" 1 pipeline "$out" --stages confounds
    status="$(latest_status "$out")"
    check_eq "missing manifest is recorded" failed "$(status_of "$status" - subjects)"

    out="$TMP/pipe3_dry"
    rc=0
    pipeline "$out" --stages "ingest confounds" --dry-run -s A >/dev/null 2>&1 || rc=$?
    check_eq "dry run: exit status 0" 0 "$rc"
    check_eq "--dry-run is exported to the stages" yes "$(grep '^00_ingest.sh' "$out/calls.txt" | cut -d'|' -f2)"
    check_not "dry run writes no tool_versions.json" test -e "$out/logs/tool_versions.json"
}

test_pipeline_auto_fetch() {
    local out="$TMP/pipe4" rc=0 status
    write_conf "$out.conf" "$out" ': "${SURFACE:=no}"' ': "${ATLASES:=Schaefer2018_100Parcels_7Networks}"'
    env FMRIPROC_STAGE_DIR="$TMP/fake_stages" PYTHON_BIN="$TMP/fake_stages/fakepython" \
        bash "$PIPELINE" -c "$out.conf" --stages "ingest timeseries" -s A >/dev/null 2>&1 || rc=$?
    check_eq "auto fetch: exit status 0" 0 "$rc"
    status="$(latest_status "$out")"
    check_eq "fetch ran because --check failed" ok "$(status_of "$status" - fetch)"
    check_eq "check precedes the download" "fetch_resources.sh --check,fetch_resources.sh " \
        "$(grep '^fetch_resources.sh' "$out/calls.txt" | cut -d'|' -f1 | tr '\n' ',' | sed 's/,$//')"
    sleep 1
    rm -f "$out/calls.txt"
    env FMRIPROC_STAGE_DIR="$TMP/fake_stages" PYTHON_BIN="$TMP/fake_stages/fakepython" \
        bash "$PIPELINE" -c "$out.conf" --stages "timeseries" -s A >/dev/null 2>&1 || rc=$?
    check_eq "complete resources: only the check runs" "fetch_resources.sh --check" \
        "$(grep '^fetch_resources.sh' "$out/calls.txt" | cut -d'|' -f1 | tr '\n' ',' | sed 's/,$//')"
}

# Every run executes a frozen copy of the code (logs/code_<run>), so editing the
# repository during a run can neither corrupt the running stage (bash reads
# scripts incrementally) nor mix code versions; FREEZE_CODE=no runs the repository.
test_pipeline_frozen_code() {
    local out="$TMP/pipe_frozen" rc=0 code_dir used
    pipeline "$out" --stages "ingest confounds" -s A >/dev/null 2>&1 || rc=$?
    check_eq "frozen code: exit status 0" 0 "$rc"
    # shellcheck disable=SC2012
    code_dir="$(ls -d "$out"/logs/code_* 2>/dev/null | head -n 1)"
    check "frozen code: copy in logs/code_<run>" test -n "$code_dir"
    check "frozen code: run_pipeline.sh copied" test -s "$code_dir/run_pipeline.sh"
    # Linux md5sum writes '<hash>  ./file', Git Bash '<hash> *./file'
    check "frozen code: md5 manifest" grep -Eq '[ *]\./stages/04_confounds\.sh$' "$code_dir/code_manifest.md5"
    check "frozen code: external dataset conf copied" test -s "$code_dir/config/external/pipe_frozen.conf"
    used="$(cut -d'|' -f1 "$out/stage_dirs.txt" | sort -u)"
    check_eq "frozen code: every stage ran from the copy" "$(cd "$code_dir/stages" && pwd)" "$used"
    used="$(cut -d'|' -f2 "$out/stage_dirs.txt" | sort -u)"
    check_eq "frozen code: stages read the frozen conf" "$code_dir/config/external/pipe_frozen.conf" "$used"
    rm -f "$out/stage_dirs.txt"
    rc=0
    FREEZE_CODE=no pipeline "$out" --stages confounds -s A >/dev/null 2>&1 || rc=$?
    check_eq "FREEZE_CODE=no: exit status 0" 0 "$rc"
    used="$(cut -d'|' -f1 "$out/stage_dirs.txt" | sort -u)"
    check_eq "FREEZE_CODE=no: stages ran from the repository" "$(cd "$TMP/fake_stages" && pwd)" "$used"
}

# A stage script that does not exist (e.g. a module not yet written) must be
# recorded as failed, not crash the orchestrator or pass silently.
test_pipeline_missing_script() {
    local out="$TMP/pipe5" rc=0 status
    pipeline "$out" --stages ingest >/dev/null 2>&1 || rc=$?
    mv "$TMP/fake_stages/09_group_qc.sh" "$TMP/fake_stages/09_group_qc.sh.off"
    rc=0
    pipeline "$out" --stages "confounds group_qc" -s A >/dev/null 2>&1 || rc=$?
    mv "$TMP/fake_stages/09_group_qc.sh.off" "$TMP/fake_stages/09_group_qc.sh"
    check_eq "missing stage script: exit status 1" 1 "$rc"
    status="$(latest_status "$out")"
    check_eq "missing stage script: recorded as failed" failed "$(status_of "$status" - group_qc)"
    check_eq "missing stage script: the subject stages still ran" ok "$(status_of "$status" sub-A confounds)"
    check "missing stage script: exit status 127 in the note" grep -q $'group_qc	failed	.*exit status 127' "$status"
}

main() {
    test_is_yes
    test_fp_init_precedence
    test_fp_init_validation
    test_run_and_require
    test_stage_markers
    test_allocation_limits
    test_python_source_marker
    test_manifest
    test_json_get
    if [[ -f "$PIPELINE" ]]; then
        make_fake_stages "$TMP/fake_stages"
        test_pipeline_failure_isolation
        test_pipeline_selection_and_parallel
        test_pipeline_cli
        test_pipeline_auto_fetch
        test_pipeline_missing_script
        test_pipeline_frozen_code
    fi
    echo
    echo "$N_PASS passed, $N_FAIL failed"
    [[ "$N_FAIL" -eq 0 ]]
}

main "$@"
