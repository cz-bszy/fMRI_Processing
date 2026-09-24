#!/bin/bash
# =============================================================================
# tests/run_tests.sh - every check that runs without neuroimaging tools
#   1. bash -n          run_pipeline.sh lib/*.sh stages/*.sh docker/*.sh tests/*.sh
#   2. shellcheck       same files, warnings only (skipped when not installed)
#   3. CRLF check       no '\r' in any script or configuration file
#   4. bash unit tests  tests/test_common_sh.sh (lib/common.sh + run_pipeline.sh)
#   5. python tests     python -m unittest discover -s tests -p 'test_*.py'
#                       with PYTHONPATH=py, interpreter $PYTHON_BIN when it exists, else the
#                       PYTHON_BIN default of config/default.conf, else 'python'/'python3'
#                       (the first one with the full stack wins)
# usage: bash tests/run_tests.sh [--no-python] [--no-bash]   (exit 1 on any failure)
# Host (Git Bash, python 3.11 with numpy/scipy/pandas/nibabel/nilearn/sklearn/
# matplotlib/jinja2) or container: cd <repo> && bash tests/run_tests.sh
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

RUN_PYTHON=yes
RUN_BASH=yes
for arg in "$@"; do
    case "$arg" in
        --no-python) RUN_PYTHON=no ;;
        --no-bash)   RUN_BASH=no ;;
        -h|--help)   sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "run_tests.sh: unknown option $arg" >&2; exit 2 ;;
    esac
done

N_FAIL=0
step_ok()   { echo "ok    $1"; }
step_fail() { echo "FAIL  $1" >&2; N_FAIL=$(( N_FAIL + 1 )); }

shell_files() {
    local f
    for f in run_pipeline.sh lib/*.sh stages/*.sh docker/*.sh tests/*.sh; do
        [[ -f "$f" ]] && echo "$f"
    done
}

conf_files() {
    local f
    for f in config/default.conf config/datasets/* .gitattributes .gitignore docker/*.ps1; do
        [[ -f "$f" ]] && echo "$f"
    done
}

# ----------------------------- 1. syntax --------------------------------------
echo "== bash -n"
while IFS= read -r f; do
    if bash -n "$f" 2>"$REPO/.run_tests_err.$$"; then
        step_ok "bash -n $f"
    else
        step_fail "bash -n $f: $(tr '\n' ' ' < "$REPO/.run_tests_err.$$")"
    fi
done < <(shell_files)
rm -f "$REPO/.run_tests_err.$$"

# ----------------------------- 2. shellcheck (advisory) -----------------------
echo "== shellcheck"
if command -v shellcheck >/dev/null 2>&1; then
    # SC1090/SC1091: sourced files are computed at run time; SC2312: style only
    mapfile -t files < <(shell_files)
    if shellcheck -S warning -e SC1090,SC1091,SC2312 "${files[@]}"; then
        step_ok "shellcheck: no warnings"
    else
        echo "shellcheck reported warnings (advisory, not a failure)" >&2
    fi
else
    echo "skip  shellcheck (not installed)"
fi

# ----------------------------- 3. line endings --------------------------------
echo "== line endings"
crlf=()
while IFS= read -r f; do
    # tr, not grep: Git Bash's grep reads files in text mode and never sees '\r'
    if [[ "$(tr -cd '\r' < "$f" | wc -c)" -gt 0 ]]; then crlf+=("$f"); fi
done < <({ shell_files; conf_files; ls py/fmriproc/*.py tests/*.py 2>/dev/null; } | sort -u)
if [[ ${#crlf[@]} -eq 0 ]]; then
    step_ok "no CRLF line endings"
else
    step_fail "CRLF line endings in: ${crlf[*]} (git config core.autocrlf false; re-checkout)"
fi

# ----------------------------- 4. bash unit tests -----------------------------
if [[ "$RUN_BASH" == yes ]]; then
    echo "== tests/test_common_sh.sh"
    if bash tests/test_common_sh.sh; then
        step_ok "tests/test_common_sh.sh"
    else
        step_fail "tests/test_common_sh.sh"
    fi
    for focused_test in tests/test_denoise.sh tests/test_container_launchers.sh; do
        if bash "$focused_test"; then
            step_ok "$focused_test"
        else
            step_fail "$focused_test"
        fi
    done
fi

# ----------------------------- 5. python unit tests ---------------------------
if [[ "$RUN_PYTHON" == yes ]]; then
    echo "== python unittest"
    # The interpreter of the pipeline first: $PYTHON_BIN, else the default of config/default.conf.
    # A bare 'python' may be another one (in the container: FSL's, without nilearn).
    default_py="$(sed -n 's/^: "${PYTHON_BIN:=\([^}]*\)}".*/\1/p' "$REPO/config/default.conf" | head -n 1)"
    PY="" PY_PARTIAL=""
    for candidate in "${PYTHON_BIN:-}" "$default_py" python python3; do
        [[ -n "$candidate" ]] && command -v "$candidate" >/dev/null 2>&1 || continue
        if "$candidate" -c 'import numpy, scipy, pandas, nibabel, nilearn' >/dev/null 2>&1; then
            PY="$candidate"
            break
        fi
        # 'python3' on Windows may be a Microsoft Store stub: it must at least import numpy
        if [[ -z "$PY_PARTIAL" ]] && "$candidate" -c 'import numpy' >/dev/null 2>&1; then
            PY_PARTIAL="$candidate"
        fi
    done
    if [[ -z "$PY" && -n "$PY_PARTIAL" ]]; then
        PY="$PY_PARTIAL"
        echo "note: $PY lacks part of numpy/scipy/pandas/nibabel/nilearn: the tests needing it are skipped"
    fi
    if [[ -z "$PY" ]]; then
        step_fail "no usable python (PYTHON_BIN='${PYTHON_BIN:-}'): set PYTHON_BIN or install numpy/scipy/pandas/nibabel/nilearn"
    else
        echo "interpreter: $PY ($("$PY" --version 2>&1))"
        if PYTHONPATH="$REPO/py${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
                "$PY" -m unittest discover -s tests -p 'test_*.py'; then
            step_ok "python unittest"
        else
            step_fail "python unittest"
        fi
    fi
fi

echo
if [[ "$N_FAIL" -eq 0 ]]; then
    echo "run_tests.sh: all checks passed"
    exit 0
fi
echo "run_tests.sh: $N_FAIL check(s) failed" >&2
exit 1
