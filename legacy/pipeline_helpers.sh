#!/usr/bin/env bash

# Each invocation owns one run. A broad pattern must not silently select run 1.
find_functional_input() {
    local search_dir="$1"
    local pattern_string="${2:-*_bold.nii*}"
    local pattern
    local -a patterns candidates
    IFS=',' read -r -a patterns <<< "$pattern_string"
    for pattern in "${patterns[@]}"; do
        pattern="${pattern#${pattern%%[![:space:]]*}}"
        pattern="${pattern%${pattern##*[![:space:]]}}"
        [[ -n "$pattern" ]] || continue
        mapfile -t candidates < <(find "$search_dir" -maxdepth 1 -type f -name "$pattern" | sort)
        if (( ${#candidates[@]} > 1 )); then
            printf 'Ambiguous BOLD input in %s: narrow the run pattern (%s).\n' "$search_dir" "$pattern" >&2
            return 1
        fi
        if (( ${#candidates[@]} == 1 )); then
            printf '%s' "${candidates[0]}"
            return 0
        fi
    done
    return 1
}
