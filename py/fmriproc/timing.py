"""Slice timing: DPABI order codes, acquisition table lookup, AFNI @file export.

Principle (docs/DESIGN.md section 1.1): slice order is never guessed. Timing comes
from a sidecar ``SliceTiming`` or from the dataset acquisition table; anything
that cannot be verified against the image (slice count, TR) means "skip".

CLI used by stage 03::

    python -m fmriproc.timing afni-tpattern --bold B.nii.gz --json B.json \
        --stc auto|require|off --out-1d slice_timing.1D --out-json timing.json
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import pandas as pd

from fmriproc.utils import read_json, write_json

ORDER_CODES = ("SA", "SD", "IA", "IA2", "ID", "ID2")
FILE_PREFIX = "file:"
DEFAULT_GROUP = "*"
TR_TOLERANCE_S = 1e-3          # header vs declared TR: more than 1 ms apart = conflict
TR_PLAUSIBLE_S = (0.1, 30.0)   # anything else in pixdim4 is treated as garbage
_MISSING = {"", "-", "n/a", "na", "nan", "none"}
_PE_DIRECTIONS = {"i", "i-", "j", "j-", "k", "k-"}


class TimingError(ValueError):
    """Slice timing information is unusable (the caller decides: skip or fail)."""


# ----------------------------------------------------------------------------
# order codes -> slice times
# ----------------------------------------------------------------------------

def slice_order_vector(code: str, n_slices: int) -> list[int]:
    """1-based slice numbers in temporal acquisition order (DPABI definitions).

    SA 1,2,..N | SD N,..,1 | IA 1,3,..,2,4,.. | IA2 2,4,..,1,3,..
    ID N,N-2,..,N-1,N-3,.. | ID2 N-1,N-3,..,N,N-2,..
    """
    code = str(code).strip().upper()
    n = int(n_slices)
    if n < 1:
        raise TimingError(f"n_slices must be positive: {n_slices}")
    if code == "SA":
        order = list(range(1, n + 1))
    elif code == "SD":
        order = list(range(n, 0, -1))
    elif code == "IA":
        order = list(range(1, n + 1, 2)) + list(range(2, n + 1, 2))
    elif code == "IA2":
        order = list(range(2, n + 1, 2)) + list(range(1, n + 1, 2))
    elif code == "ID":
        order = list(range(n, 0, -2)) + list(range(n - 1, 0, -2))
    elif code == "ID2":
        order = list(range(n - 1, 0, -2)) + list(range(n, 0, -2))
    else:
        raise TimingError(f"unknown slice order code '{code}' (expected one of {' '.join(ORDER_CODES)})")
    return order


def slice_times_from_order(order: Sequence[int], tr: float) -> np.ndarray:
    """Per-slice-index acquisition times (s): the slice at rank r gets r * TR / n."""
    values = np.asarray(order)
    n = values.size
    if values.ndim != 1 or n == 0:
        raise TimingError("slice order must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(values.astype(float))) or not np.all(values == np.rint(values)):
        raise TimingError("slice order must contain integers")
    values = values.astype(int)
    if sorted(values.tolist()) != list(range(1, n + 1)):
        raise TimingError(f"slice order is not a permutation of 1..{n}")
    tr = float(tr)
    if not np.isfinite(tr) or tr <= 0:
        raise TimingError(f"TR must be positive: {tr}")
    times = np.empty(n, dtype=np.float64)
    times[values - 1] = np.arange(n, dtype=np.float64) * tr / n
    return times


def load_slice_times_file(spec: str, base_dir: str | Path | None = None) -> np.ndarray:
    """'file:<path>' (or a bare path): one slice time in seconds per line/field.

    A relative path is resolved against `base_dir` (the acquisition table folder).
    """
    text = str(spec).strip()
    if text.lower().startswith(FILE_PREFIX):
        text = text[len(FILE_PREFIX):].strip()
    if not text:
        raise TimingError("empty slice timing file specification")
    path = Path(text)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    if not path.is_file():
        raise TimingError(f"slice timing file not found: {path}")
    try:
        values = np.loadtxt(path, dtype=float, comments="#", ndmin=1)
    except ValueError as err:
        raise TimingError(f"slice timing file is not numeric: {path} ({err})") from err
    return np.atleast_1d(values).ravel()


def validate_slice_timing(times: Any, n_slices: int, tr: float) -> np.ndarray:
    """One finite time per slice within [0, TR) seconds; returns a float array."""
    try:
        values = np.asarray(times, dtype=np.float64)
    except (TypeError, ValueError) as err:
        raise TimingError(f"SliceTiming is not numeric: {err}") from err
    if values.ndim != 1:
        raise TimingError("SliceTiming must be a flat list")
    if values.size != int(n_slices):
        raise TimingError(f"SliceTiming has {values.size} values but the image has {int(n_slices)} slices")
    if not np.all(np.isfinite(values)):
        raise TimingError("SliceTiming contains non-finite values")
    tr = float(tr)
    if not np.isfinite(tr) or tr <= 0:
        raise TimingError(f"TR must be positive: {tr}")
    if np.any(values < 0) or np.any(values >= tr):
        raise TimingError(
            f"SliceTiming must lie within [0, TR={tr:g}) seconds (found {values.min():g} .. {values.max():g}; "
            "milliseconds?)"
        )
    return values


def is_file_spec(slice_order: str) -> bool:
    return str(slice_order).strip().lower().startswith(FILE_PREFIX)


# ----------------------------------------------------------------------------
# TR in the NIfTI header
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class HeaderTR:
    raw: float              # pixdim[4] as stored
    units: str              # 'sec' | 'msec' | 'usec' | 'unknown' ...
    literal_s: float        # raw converted with the declared units only
    seconds: float | None   # usable TR in seconds, None when the header value is garbage
    note: str               # non-empty when the value was reinterpreted or rejected


def _plausible(value: float) -> bool:
    return bool(np.isfinite(value)) and TR_PLAUSIBLE_S[0] <= value <= TR_PLAUSIBLE_S[1]


def header_tr(header: Any) -> HeaderTR:
    """TR from pixdim[4] honouring xyzt_units (legacy headers mix s and ms)."""
    raw = float(header["pixdim"][4])
    try:
        units = str(header.get_xyzt_units()[1])
    except Exception:  # noqa: BLE001 - header classes without units
        units = "unknown"
    factor = {"sec": 1.0, "msec": 1e-3, "usec": 1e-6}.get(units, 1.0)
    literal = raw * factor if np.isfinite(raw) else float("nan")
    if _plausible(literal):
        note = "time units unknown, pixdim4 read as seconds" if units not in ("sec", "msec", "usec") else ""
        return HeaderTR(raw, units, literal, literal, note)
    # A TR of 2000 "s" or 2 "ms" is a unit error, not an acquisition.
    for other, other_factor in (("msec", 1e-3), ("sec", 1.0), ("usec", 1e-6)):
        if other_factor != factor and np.isfinite(raw) and _plausible(raw * other_factor):
            note = f"pixdim4={raw:g} with units '{units}' is implausible, read as {other}"
            return HeaderTR(raw, units, literal, raw * other_factor, note)
    return HeaderTR(raw, units, literal, None, f"header TR unusable (pixdim4={raw:g}, units '{units}')")


# ----------------------------------------------------------------------------
# acquisition table
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class AcqRow:
    group: str
    tr: float | None = None
    n_slices: int | None = None
    slice_order: str = "unknown"      # order code, 'file:<path>' or 'unknown'
    stc: str = "skip"                 # 'apply' | 'skip'
    drop_volumes: int | None = None
    pe_dir: str | None = None
    evidence: str = ""
    note: str = ""
    matched: str = "exact"            # 'exact' | 'default' (filled by lookup_acq)


def _cell(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in _MISSING else text


def _cell_float(value: Any, what: str) -> float | None:
    text = _cell(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError as err:
        raise TimingError(f"acquisition table: {what} is not a number: '{text}'") from err
    if not np.isfinite(number):
        raise TimingError(f"acquisition table: {what} is not finite: '{text}'")
    return number


def _cell_int(value: Any, what: str) -> int | None:
    number = _cell_float(value, what)
    if number is None:
        return None
    if number != int(number) or number < 0:
        raise TimingError(f"acquisition table: {what} must be a non-negative integer: '{value}'")
    return int(number)


def load_acq_table(path: str | Path) -> dict[str, AcqRow]:
    """Acquisition table (docs/DESIGN.md section 5) -> {group: AcqRow}."""
    path = Path(path)
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame.columns]
    if "group" not in frame.columns:
        raise TimingError(f"acquisition table has no 'group' column: {path}")
    rows: dict[str, AcqRow] = {}
    for record in frame.to_dict(orient="records"):
        group = str(record.get("group", "")).strip()
        if not group or group.startswith("#"):
            continue
        if group in rows:
            raise TimingError(f"acquisition table lists group '{group}' twice: {path}")
        what = f"group {group}"
        order = _cell(record.get("slice_order")) or "unknown"
        if not is_file_spec(order):
            order = order.upper() if order.upper() in ORDER_CODES else order.lower()
            if order not in ORDER_CODES and order != "unknown":
                raise TimingError(f"acquisition table: {what}: unknown slice_order '{order}'")
        stc = (_cell(record.get("stc")) or "skip").lower()
        if stc not in ("apply", "skip"):
            raise TimingError(f"acquisition table: {what}: stc must be apply or skip, not '{stc}'")
        pe_dir = _cell(record.get("pe_dir")) or None
        if pe_dir is not None and pe_dir not in _PE_DIRECTIONS:
            raise TimingError(f"acquisition table: {what}: pe_dir must be one of {sorted(_PE_DIRECTIONS)}")
        n_slices = _cell_int(record.get("n_slices"), f"{what} n_slices")
        rows[group] = AcqRow(
            group=group,
            tr=_cell_float(record.get("tr"), f"{what} tr"),
            n_slices=n_slices if n_slices else None,
            slice_order=order,
            stc=stc,
            drop_volumes=_cell_int(record.get("drop_volumes"), f"{what} drop_volumes"),
            pe_dir=pe_dir,
            evidence=_cell(record.get("evidence")),
            note=_cell(record.get("note")),
        )
    return rows


def lookup_acq(table: dict[str, AcqRow] | None, group: str) -> AcqRow | None:
    """Exact group match, else the '*' default row, else None."""
    if not table:
        return None
    row = table.get(str(group))
    if row is not None:
        return row
    default = table.get(DEFAULT_GROUP)
    if default is None:
        return None
    return replace(default, matched="default")


@dataclass(frozen=True)
class TimingDecision:
    stc: str                          # 'apply' | 'skip'
    reason: str
    slice_times: np.ndarray | None    # per slice index, seconds (only when stc == 'apply')
    slice_order: str
    evidence: str
    source: str


def table_timing(
    row: AcqRow | None,
    n_slices: int,
    tr: float,
    table_path: str | Path | None = None,
    tr_conflict: bool = False,
) -> TimingDecision:
    """Decide STC for one run from its acquisition table row. Never raises:
    every problem is a documented 'skip'."""
    table_name = Path(table_path).name if table_path else "acquisition_table"
    base_dir = Path(table_path).parent if table_path else None
    if row is None:
        return TimingDecision("skip", "no acquisition table row for this group", None, "unknown", "", "none")
    source = f"acquisition_table:{table_name}[group={row.group}]"

    def skip(reason: str) -> TimingDecision:
        return TimingDecision("skip", reason, None, row.slice_order, row.evidence, source)

    if row.slice_order == "unknown":
        return skip("slice order unknown in the acquisition table")
    if row.stc != "apply":
        return skip("acquisition table says stc=skip")
    if tr_conflict:
        return skip("TR of the acquisition table and of the NIfTI header disagree")
    if row.tr is not None and abs(row.tr - float(tr)) > TR_TOLERANCE_S:
        return skip(f"table tr={row.tr:g} s but the run uses TR={float(tr):g} s")
    if row.n_slices is not None and row.n_slices != int(n_slices):
        return skip(f"table n_slices={row.n_slices} but the image has {int(n_slices)} slices")
    try:
        if is_file_spec(row.slice_order):
            times = load_slice_times_file(row.slice_order, base_dir)
        else:
            times = slice_times_from_order(slice_order_vector(row.slice_order, n_slices), tr)
        times = validate_slice_timing(times, n_slices, tr)
    except TimingError as err:
        return skip(str(err))
    return TimingDecision("apply", "timing adopted from the acquisition table; numeric checks passed", times, row.slice_order,
                          row.evidence, source)


# ----------------------------------------------------------------------------
# CLI: sidecar SliceTiming -> AFNI '-tpattern @file'
# ----------------------------------------------------------------------------

def afni_slice_times(times: Any, n_slices: int, tr: float, direction: str = "k") -> np.ndarray:
    """Validated per-slice times in stored slice-index order along k.

    BIDS 'k-' means the first SliceTiming entry belongs to the last slice, so the
    vector is reversed (same as legacy/prepare_bold.py). i/j slice axes are
    refused: 3dTshift shifts along k only and metadata is never silently reordered.
    """
    direction = str(direction or "k").strip()
    if direction not in ("k", "k-"):
        raise TimingError(f"SliceEncodingDirection '{direction}' is not supported (3dTshift needs slice axis k)")
    values = validate_slice_timing(times, n_slices, tr)
    return values[::-1].copy() if direction == "k-" else values


def _resolve_tr(meta: dict, header: Any) -> tuple[float | None, str | None]:
    """(TR, problem). Sidecar RepetitionTime wins; a usable header TR must agree."""
    from_header = header_tr(header)
    declared = meta.get("RepetitionTime")
    if declared is None:
        if from_header.seconds is None:
            return None, f"no RepetitionTime in the sidecar and {from_header.note}"
        return from_header.seconds, None
    try:
        declared = float(declared)
    except (TypeError, ValueError):
        return None, f"RepetitionTime is not a number: {declared!r}"
    if not np.isfinite(declared) or declared <= 0:
        return None, f"RepetitionTime must be positive: {declared}"
    if from_header.seconds is not None and abs(from_header.seconds - declared) > TR_TOLERANCE_S:
        return declared, f"sidecar TR {declared:g} s and NIfTI header TR {from_header.seconds:g} s disagree"
    return declared, None


def afni_tpattern(bold: Path, sidecar: Path, stc: str, out_1d: Path, out_json: Path) -> int:
    try:
        img = nib.load(str(bold))
    except Exception as err:  # noqa: BLE001 - any unreadable image is the same user-facing error
        print(f"ERROR: cannot read {bold}: {err}", file=sys.stderr)
        return 1
    shape = img.shape
    if len(shape) != 4:
        print(f"ERROR: expected a 4D BOLD image, got shape {shape}: {bold}", file=sys.stderr)
        return 1
    n_slices = int(shape[2])
    meta = read_json(sidecar, default={})
    tr, tr_problem = _resolve_tr(meta, img.header)
    if tr is None:
        print(f"ERROR: {tr_problem} ({bold})", file=sys.stderr)
        return 1

    result: dict[str, Any] = {
        "stc": "skip",
        "reason": "",
        "tzero": 0.0,
        "n_slices": n_slices,
        "tr": tr,
        "source": str(meta.get("SliceTimingSource") or (str(sidecar) if sidecar.is_file() else "none")),
        "evidence": str(meta.get("SliceTimingEvidence") or ("sidecar" if "SliceTiming" in meta else "none")),
    }

    times: np.ndarray | None = None
    problem: str | None = None
    if stc == "off":
        result["reason"] = "STC=off"
    else:
        if not sidecar.is_file():
            problem = f"sidecar not found: {sidecar}"
        elif meta.get("SliceTiming") is None:
            why = meta.get("SliceTimingSkipReason")
            problem = "no SliceTiming in the sidecar" + (f" ({why})" if why else "")
        elif tr_problem is not None:
            problem = tr_problem
        else:
            try:
                times = afni_slice_times(meta["SliceTiming"], n_slices, tr, meta.get("SliceEncodingDirection", "k"))
            except TimingError as err:
                problem = str(err)
        if problem is not None:
            if stc == "require":
                print(f"ERROR: STC=require but no valid slice timing for {bold}: {problem}", file=sys.stderr)
                return 1
            print(f"WARNING: slice timing correction skipped for {bold}: {problem}", file=sys.stderr)
            result["reason"] = problem

    if times is None:
        # A stale @file from an earlier run must not survive a 'skip' decision.
        out_1d.unlink(missing_ok=True)
    else:
        out_1d.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_1d.with_name(f".tmp_{out_1d.name}")
        np.savetxt(tmp, times[None, :], fmt="%.10g")
        tmp.replace(out_1d)
        result["stc"] = "apply"
        result["reason"] = "slice timing numeric checks passed"
        result["tzero"] = float((times.min() + times.max()) / 2.0)
    write_json(out_json, result)
    print(f"stc={result['stc']} reason={result['reason']} tzero={result['tzero']:g} n_slices={n_slices} tr={tr:g}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fmriproc.timing", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    tp = sub.add_parser("afni-tpattern", help="validate sidecar SliceTiming and write the 3dTshift @file")
    tp.add_argument("--bold", type=Path, required=True)
    tp.add_argument("--json", type=Path, required=True, help="BIDS sidecar of the BOLD run")
    tp.add_argument("--stc", choices=["auto", "require", "off"], default="auto")
    tp.add_argument("--out-1d", type=Path, required=True, help="one row of per-slice times (s), slice index order")
    tp.add_argument("--out-json", type=Path, required=True)
    od = sub.add_parser("order", help="print the slice order vector and slice times of an order code")
    od.add_argument("--code", required=True, help=" | ".join(ORDER_CODES))
    od.add_argument("--n-slices", type=int, required=True)
    od.add_argument("--tr", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "afni-tpattern":
        return afni_tpattern(args.bold, args.json, args.stc, args.out_1d, args.out_json)
    try:
        order = slice_order_vector(args.code, args.n_slices)
        times = slice_times_from_order(order, args.tr)
    except TimingError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print("order (1-based, temporal):", " ".join(str(v) for v in order))
    print("times (s, slice index order):", " ".join(f"{t:.6g}" for t in times))
    return 0


if __name__ == "__main__":
    sys.exit(main())
