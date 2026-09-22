"""Stage 05 helper: one named denoising strategy -> regressors.1D + degrees of freedom.

Picks the strategy's columns from the stage-04 confounds table, centres and
scales them in float64 (the column space is unchanged, the conditioning of the
3dTproject projection improves), drops constant and duplicate columns, and
reports joint-design rank and the separate AFNI nominal-column gate.
The algebraic residual dimension is not an effective sample size.

Exit codes: 0 ok (low DOF is only flagged), 1 bad input, 2 unknown strategy.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fmriproc.utils import _atomic_write, read_1d, read_tsv, write_json

MOTION6 = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
EXPANSIONS = ["", "_derivative1", "_power2", "_derivative1_power2"]

# name -> (motion parameters, mean signals, expand the mean signals, aCompCor)
STRATEGIES: dict[str, dict[str, Any]] = {
    "wmcsf24": {"motion": 24, "signals": ["white_matter", "csf"], "expand": False, "compcor": False,
                "text": "24 motion parameters + mean WM + mean CSF"},
    "wmcsf24gsr": {"motion": 24, "signals": ["white_matter", "csf", "global_signal"], "expand": False, "compcor": False,
                   "text": "24 motion parameters + mean WM + mean CSF + global signal"},
    "36p": {"motion": 24, "signals": ["white_matter", "csf", "global_signal"], "expand": True, "compcor": False,
            "text": "24 motion parameters + WM, CSF and global signal with derivatives and squares"},
    "acompcor": {"motion": 12, "signals": [], "expand": False, "compcor": True,
                 "text": "12 motion parameters + WM and CSF aCompCor components"},
    "acompcorgsr": {"motion": 12, "signals": ["global_signal"], "expand": False, "compcor": True,
                    "text": "12 motion parameters + WM and CSF aCompCor components + global signal"},
    "legacy8": {"motion": 6, "signals": ["white_matter", "csf"], "expand": False, "compcor": False,
                "text": "6 motion parameters + mean WM + mean CSF"},
    "legacy9gsr": {"motion": 6, "signals": ["white_matter", "csf", "global_signal"], "expand": False, "compcor": False,
                   "text": "6 motion parameters + mean WM + mean CSF + global signal"},
}

# 3dTproject turns "-passband lo hi" into the stop bands 0..lo-EPS and hi+EPS..inf
AFNI_BAND_EPS = 1e-4
HIGHPASS_TOP_HZ = 99999.0


def _warn(message: str, collected: list[str] | None = None) -> None:
    print(f"[strategies] WARNING: {message}", file=sys.stderr)
    if collected is not None:
        collected.append(message)


# ----------------------------------------------------------------------------
# column selection
# ----------------------------------------------------------------------------

def motion_columns(n_params: int) -> list[str]:
    suffixes = {6: EXPANSIONS[:1], 12: EXPANSIONS[:2], 24: EXPANSIONS}[n_params]
    return [f"{name}{suffix}" for suffix in suffixes for name in MOTION6]


def strategy_columns(name: str, acompcor_n: int = 5) -> list[str]:
    """Nominal column list of a strategy (before looking at the actual table)."""
    spec = STRATEGIES[name]
    columns = motion_columns(spec["motion"])
    for signal in spec["signals"]:
        columns += [f"{signal}{suffix}" for suffix in (EXPANSIONS if spec["expand"] else EXPANSIONS[:1])]
    if spec["compcor"]:
        for prefix in ("w", "c"):
            columns += [f"{prefix}_comp_cor_{i:02d}" for i in range(acompcor_n)]
    return columns


def cosine_columns_needed(
    available: list[str], n_volumes: int, tr: float, filter_mode: str, band_low: float
) -> list[str]:
    """Keep the complete DCT basis removed before aCompCor PCA.

    A finite-run DCT is not the Fourier basis used by AFNI, even at the same
    nominal cutoff. Joint-design rank accounts for actual dependencies.
    """
    return sorted(c for c in available if c.startswith("cosine_") and c[7:].isdigit())


def select_columns(
    name: str,
    frame: pd.DataFrame,
    tr: float,
    filter_mode: str,
    band_low: float,
    acompcor_n: int = 5,
    warnings: list[str] | None = None,
) -> list[str]:
    """Columns of `frame` used by strategy `name`. Missing aCompCor components
    (small tissue mask) are tolerated with a warning; anything else is an error."""
    available = list(frame.columns)
    columns = []
    missing = []
    for column in strategy_columns(name, acompcor_n):
        if column in frame.columns:
            columns.append(column)
        elif "_comp_cor_" in column:
            _warn(f"{name}: aCompCor component {column} is not in the confounds table", warnings)
        else:
            missing.append(column)
    if missing:
        raise ValueError(f"strategy {name}: confound columns missing: {' '.join(missing)}")
    if STRATEGIES[name]["compcor"]:
        if not any("_comp_cor_" in c for c in columns):
            raise ValueError(f"strategy {name}: the confounds table holds no aCompCor components")
        cosines = cosine_columns_needed(available, len(frame), tr, filter_mode, band_low)
        columns += cosines
    return columns


# ----------------------------------------------------------------------------
# conditioning
# ----------------------------------------------------------------------------

def prepare_regressors(
    frame: pd.DataFrame, columns: list[str], keep: np.ndarray | None = None, has_intercept: bool = True
) -> tuple[np.ndarray, list[str], dict[str, str]]:
    """Centre and scale to unit SD (float64). Mean and SD are taken over the
    supplied fit rows, but every row is written. NTRP uses all rows;
    ZERO/KILL use retained rows.

    Returns (matrix (T, n_kept), kept column names, {dropped column: reason}).
    """
    raw = frame[columns].to_numpy(dtype=np.float64)
    bad = [c for c, ok in zip(columns, np.isfinite(raw).all(axis=0)) if not ok]
    if bad:
        raise ValueError(f"non-finite values in confound columns: {' '.join(bad)}")
    keep = np.ones(len(frame), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    if keep.size != raw.shape[0]:
        raise ValueError(f"censor vector has {keep.size} rows, confounds table has {raw.shape[0]}")
    if keep.sum() < 2:
        raise ValueError("fewer than 2 fit rows")
    ref = raw[keep]
    # With no intercept, AFNI's full-run demeaning must be preserved: a
    # column constant on kept rows can still contribute a nonzero offset.
    mean = ref.mean(axis=0) if has_intercept else raw.mean(axis=0)
    sd = np.sqrt(np.mean((ref - mean) ** 2, axis=0))
    varying = sd > 1e-9 * np.abs(ref).max(axis=0)
    dropped = {c: "constant over fit rows" if has_intercept else "zero after full-run demeaning"
               for c, ok in zip(columns, varying) if not ok}

    safe_sd = np.where(varying, sd, 1.0)
    scaled = (raw - mean) / safe_sd
    corr = (scaled[keep].T @ scaled[keep]) / keep.sum()
    kept: list[int] = []
    for j in range(len(columns)):
        if not varying[j]:
            continue
        twin = next((i for i in kept if abs(corr[i, j]) > 1.0 - 1e-9), None)
        if twin is None:
            kept.append(j)
        else:
            dropped[columns[j]] = f"duplicate of {columns[twin]}"
    if not kept:
        raise ValueError("no varying regressors left")
    return scaled[:, kept], [columns[j] for j in kept], dropped


# ----------------------------------------------------------------------------
# degrees of freedom
# ----------------------------------------------------------------------------

def stopbands(filter_mode: str, band_low: float, band_high: float) -> list[tuple[float, float]]:
    if filter_mode == "none":
        return []
    f32 = np.float32
    eps = f32(AFNI_BAND_EPS)
    low = max(f32(band_low), f32(0.0))
    top = f32(band_high if filter_mode == "bandpass" else HIGHPASS_TOP_HZ)
    return [(0.0, float(low - eps)), (float(top + eps), 999999.9)]


def bandpass_frequencies(n_volumes: int, tr: float, filter_mode: str, band_low: float, band_high: float) -> np.ndarray:
    """Frequency indices for sine/cosine regressors 3dTproject builds for the band filter.

    Mirrors 3dTproject.c: Fourier frequencies j / (N * TR), j = 1..N/2, of the
    FULL run (censoring removes rows, not columns); a stop band a..b marks
    j = rint(a/df + 1/6) .. rint(b/df - 1/6); two regressors per marked frequency,
    one at Nyquist when N is even; j = 0 belongs to polort.
    Roughly N * (1 - 2 * TR * (f_hi - f_lo)) for a band-pass, 2 * TR * f_lo * N
    for a high-pass. "-passband lo 99999" clips its upper stop band to the
    Nyquist frequency, which is therefore removed as well (1 or 2 regressors).
    """
    bands = stopbands(filter_mode, band_low, band_high)
    if not bands or n_volumes < 2:
        return np.array([], dtype=int)
    # single precision on purpose: the C code rounds float32 quotients, and a band
    # edge that falls half-way between two Fourier frequencies must round the same way
    f32 = np.float32
    df = f32(1.0) / (f32(n_volumes) * f32(tr))
    sixth = f32(0.1666666) * df
    n_freq = n_volumes // 2
    f_top = (f32(n_freq) + f32(0.1)) * df
    marked = np.zeros(n_freq + 1, dtype=bool)
    for low, high in bands:
        low32 = min(max(f32(low), f32(0.0)), f_top)
        high32 = min(max(f32(high), f32(0.0)), f_top)
        j_low = max(int(np.rint(f32(low32 + sixth) / df)), 0)
        j_high = min(int(np.rint(f32(high32 - sixth) / df)), n_freq)
        if j_high >= j_low:
            marked[j_low: j_high + 1] = True
    marked[0] = False
    return np.flatnonzero(marked)


def bandpass_cost(n_volumes: int, tr: float, filter_mode: str, band_low: float, band_high: float) -> int:
    frequencies = bandpass_frequencies(n_volumes, tr, filter_mode, band_low, band_high)
    return 2 * len(frequencies) - int(n_volumes % 2 == 0 and n_volumes // 2 in frequencies)


def joint_design(matrix: np.ndarray, polort: int, tr: float, filter_mode: str,
                 band_low: float, band_high: float) -> np.ndarray:
    """Single-run AFNI fixed design, before deleting ZERO/KILL rows.

    3dTproject.c: Legendre at 2*t/(N-1)-1; Fourier j/N; no Nyquist sine;
    external ort columns demeaned over the FULL run before row censoring.
    Values are float32 as in AFNI. This is for accounting, not a replacement
    for AFNI's regularized pseudo-inverse (PSINV_EPS=1e-6).
    Source: https://github.com/afni/afni/blob/master/src/3dTproject.c
    """
    n = len(matrix)
    degree = max(polort, 0) if filter_mode != "none" else polort
    columns = []
    if degree >= 0:
        columns.extend(np.polynomial.legendre.legvander(np.linspace(-1, 1, n), degree).T)
    time = np.arange(n, dtype=np.float32)
    for j in bandpass_frequencies(n, tr, filter_mode, band_low, band_high):
        phase = np.float32(2.0 * np.pi * j / n) * time
        columns.append(np.cos(phase))
        if not (n % 2 == 0 and j == n // 2):
            columns.append(np.sin(phase))
    nuisance = np.asarray(matrix, dtype=np.float32)
    # Match vector_demean's sequential float summation.
    means = np.cumsum(nuisance, axis=0, dtype=np.float32)[-1] / np.float32(n)
    columns.extend((nuisance - means).T)
    return np.column_stack(columns).astype(np.float32) if columns else np.empty((n, 0), np.float32)


def design_accounting(matrix: np.ndarray, keep: np.ndarray, censor_mode: str,
                      polort: int, tr: float, filter_mode: str,
                      band_low: float, band_high: float, min_dof: float) -> dict[str, Any]:
    """Numerical design rank and AFNI's separate pre-SVD column-count gate.

    Rank is evaluated after unit-L2 column scaling, at float32 resolution
    (max(shape)*eps32*smax), not interpreted as independent sample size.
    AFNI uses a ridge-like SVD inverse, so this algebraic dimension is also
    not the effective degrees of freedom of its regularized smoother.
    """
    design = joint_design(matrix, polort, tr, filter_mode, band_low, band_high)
    fit = design if censor_mode == "NTRP" else design[keep]
    norms = np.linalg.norm(fit.astype(np.float64), axis=0)
    scaled = fit[:, norms > 0].astype(np.float64) / norms[norms > 0]
    singular = np.linalg.svd(scaled, compute_uv=False)
    tol = max(fit.shape) * np.finfo(np.float32).eps * (float(singular[0]) if singular.size else 0.0)
    rank = int(np.sum(singular > tol))
    remaining = int(len(fit) - rank)
    retained = int(keep.sum())
    nominal_dof = retained - design.shape[1]
    return {
        "n_volumes": len(matrix), "n_censored": int((~keep).sum()), "n_retained": retained,
        "retained_observations": retained, "fit_rows": len(fit),
        "design_columns": design.shape[1], "design_rank": rank,
        "design_rank_tolerance": tol,
        "design_rank_method": "unit-L2 columns; SVD tolerance max(shape)*float32_eps*smax",
        "algebraic_dof": remaining, "dof_remaining": remaining,
        "dof_definition": "fit_rows minus numerical joint-design rank; not effective sample size or regularized-smoother DOF",
        "afni_nominal_dof": nominal_dof,
        "afni_model_feasible": bool(retained >= 9 and nominal_dof > 0),
        "dof_bandpass_cost": bandpass_cost(len(matrix), tr, filter_mode, band_low, band_high),
        "low_dof": bool(remaining < min_dof),
    }


def dof_accounting(
    n_volumes: int,
    n_censored: int,
    n_regressors: int,
    polort: int,
    tr: float,
    filter_mode: str,
    band_low: float,
    band_high: float,
    min_dof: float,
) -> dict[str, Any]:
    """Legacy nominal column-budget estimate; NOT joint rank or NTRP fit DOF.

    Retained for callers of the old helper. CLI output uses design_accounting.
    """
    n_retained = n_volumes - n_censored
    cost = bandpass_cost(n_volumes, tr, filter_mode, band_low, band_high)
    # a stop band that starts at 0 Hz makes 3dTproject raise polort -1 to 0
    n_polort = max(polort + 1, 0 if filter_mode == "none" else 1)
    remaining = n_retained - n_regressors - n_polort - cost
    return {
        "n_volumes": n_volumes,
        "n_censored": n_censored,
        "n_retained": n_retained,
        "dof_bandpass_cost": cost,
        "dof_remaining": remaining,
        "low_dof": bool(remaining < min_dof),
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def read_censor(path: Path | None, n_volumes: int) -> np.ndarray:
    if path is None:
        return np.ones(n_volumes, dtype=bool)
    values = read_1d(path).ravel()
    if values.size != n_volumes:
        raise ValueError(f"{path}: {values.size} rows, confounds table has {n_volumes}")
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError(f"{path}: censor values must be 0 or 1")
    return values == 1.0


def _key_value(text: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=PATH: {text}")
    return key, value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confounds", required=True, type=Path, help="<RUN>_desc-confounds_timeseries.tsv")
    parser.add_argument("--strategy", required=True, help=" | ".join(STRATEGIES))
    parser.add_argument("--tr", required=True, type=float)
    parser.add_argument("--n-volumes-censored-from", type=Path, default=None, metavar="CENSOR_1D",
                        help="<RUN>_desc-censor.1D (1 = keep, 0 = censored)")
    parser.add_argument("--polort", type=int, default=2)
    parser.add_argument("--filter-mode", choices=["bandpass", "highpass", "none"], default="bandpass")
    parser.add_argument("--band-low", type=float, default=0.01)
    parser.add_argument("--band-high", type=float, default=0.1)
    parser.add_argument("--min-dof", type=float, default=15)
    parser.add_argument("--acompcor-n", type=int, default=5, help="aCompCor components per tissue")
    parser.add_argument("--censor-mode", choices=["NTRP", "KILL", "ZERO"], default="NTRP", help="recorded; decides n_volumes_out")
    parser.add_argument("--smooth-fwhm", type=float, default=0.0, help="recorded only")
    parser.add_argument("--input", action="append", type=_key_value, default=[], metavar="KEY=PATH",
                        help="recorded in the JSON 'inputs' object (repeatable)")
    parser.add_argument("--out-1d", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    return parser


def run(args: argparse.Namespace) -> None:
    if args.tr <= 0:
        raise ValueError(f"--tr must be positive: {args.tr}")
    if args.polort < -1:
        raise ValueError(f"--polort must be >= -1: {args.polort}")
    if args.filter_mode != "none" and args.band_low < 0:
        raise ValueError("--band-low must not be negative")
    if args.filter_mode == "bandpass" and not args.band_low < args.band_high:
        raise ValueError(f"band-pass needs band-low < band-high: {args.band_low} {args.band_high}")

    notes: list[str] = []
    frame = read_tsv(args.confounds)
    n_volumes = len(frame)
    keep = read_censor(args.n_volumes_censored_from, n_volumes)
    n_censored = int((~keep).sum())

    nyquist = 0.5 / args.tr
    if args.filter_mode == "bandpass" and args.band_high >= nyquist:
        _warn(f"band-high {args.band_high} Hz is not below Nyquist ({nyquist:.4g} Hz): effectively a high-pass", notes)

    columns = select_columns(args.strategy, frame, args.tr, args.filter_mode, args.band_low, args.acompcor_n, notes)
    fit_keep = np.ones(n_volumes, dtype=bool) if args.censor_mode == "NTRP" else keep
    matrix, kept, dropped = prepare_regressors(frame, columns, fit_keep,
                                              has_intercept=args.polort >= 0 or args.filter_mode != "none")
    for column, reason in dropped.items():
        _warn(f"{args.strategy}: column {column} dropped ({reason})", notes)
    rank = int(np.linalg.matrix_rank(matrix[fit_keep]))
    if rank < len(kept):
        # Joint rank below counts dependencies; AFNI still gates nominal column count.
        _warn(f"{args.strategy}: regressors are collinear (rank {rank} of {len(kept)} columns)", notes)

    dof = design_accounting(matrix, keep, args.censor_mode, args.polort, args.tr,
                            args.filter_mode, args.band_low, args.band_high, args.min_dof)
    if not dof["afni_model_feasible"]:
        raise ValueError(f"{args.strategy}: AFNI requires >=9 retained observations and fewer nominal "
                         f"design columns than retained observations; got {dof['design_columns']} columns "
                         f"and {dof['n_retained']} retained observations (mode={args.censor_mode}). "
                         "Joint rank does not override AFNI's pre-SVD gate.")
    if dof["low_dof"]:
        _warn(f"{args.strategy}: {dof['dof_remaining']} algebraic residual degrees of freedom "
              f"(< {args.min_dof:g}); this is not effective sample size", notes)

    if args.filter_mode == "bandpass":
        band: list[float | None] = [args.band_low, args.band_high]
    elif args.filter_mode == "highpass":
        band = [args.band_low, None]
    else:
        band = [None, None]
    killed = args.censor_mode == "KILL" and n_censored > 0
    inputs = {"confounds": str(args.confounds)}
    if args.n_volumes_censored_from is not None:
        inputs["censor"] = str(args.n_volumes_censored_from)
    inputs.update(dict(args.input))

    buffer = io.StringIO()
    np.savetxt(buffer, matrix, fmt="%.10g", delimiter=" ", newline="\n")
    _atomic_write(Path(args.out_1d), buffer.getvalue())
    write_json(args.out_json, {
        **dof,
        "strategy": args.strategy,
        "description": STRATEGIES[args.strategy]["text"],
        "columns": kept,
        "n_regressors": len(kept),
        "regressor_rank": rank,
        "dropped_columns": dropped,
        "polort": args.polort,
        "filter_mode": args.filter_mode,
        "band": band,
        "tr": args.tr,
        "censor_mode": args.censor_mode,
        "n_volumes_out": dof["n_retained"] if killed else n_volumes,
        "min_dof": args.min_dof,
        "smooth_fwhm": args.smooth_fwhm,
        "scaling": f"centred with intercept-aware full-run AFNI demeaning; unit RMS over {'all frames (NTRP)' if args.censor_mode == 'NTRP' else 'retained frames'} (float64)",
        "warnings": notes,
        "inputs": inputs,
    })
    print(
        f"[strategies] {args.strategy}: {len(kept)} regressors, {dof['n_retained']}/{n_volumes} frames retained, "
        f"band filter cost {dof['dof_bandpass_cost']}, residual DOF {dof['dof_remaining']}",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.strategy not in STRATEGIES:
        print(f"[strategies] ERROR: unknown strategy '{args.strategy}'; valid: {' '.join(STRATEGIES)}", file=sys.stderr)
        return 2
    try:
        run(args)
    except (OSError, ValueError) as error:
        print(f"[strategies] ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
