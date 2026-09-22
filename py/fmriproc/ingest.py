"""Stage 00 ingest: raw layout -> rawdata/ (BIDS-like) + sidecars + manifest.

Discovers BOLD runs (``dpabi`` or ``bids`` layout), validates the NIfTI headers,
writes header-normalised copies (voxel data are streamed byte for byte, only the
348-byte NIfTI-1 header is replaced), writes sidecars from the acquisition table
and the three dataset tables ``manifest.tsv``, ``participants.tsv`` and
``ingest_report.tsv`` (docs/DESIGN.md sections 4, 5 and 7 "00 ingest").

A run that cannot be used is a row with status ``error`` in ``ingest_report.tsv``
and is left out of the manifest; the program fails only when no run is valid.
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import shutil
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.affines import obliquity

from fmriproc import timing
from fmriproc.utils import read_json, write_json, write_tsv

NIFTI1_HEADER_BYTES = 348
MIN_VOLUMES_DEFAULT = 30
SHORT_FOV_Z_MM = 110.0
QS_FORM_TOLERANCE_MM = 0.1     # qform/sform disagreement (at the volume corners) worth reporting
VALID_XFORM_CODES = (1, 2)     # scanner / aligned; 3 (talairach) makes AFNI label data +tlrc
GROUP_NONE = "-"

MANIFEST_COLUMNS = ["subject", "session", "task", "run", "group", "bold", "t1w", "run_label"]
REPORT_COLUMNS = [
    "subject", "session", "run_label", "group", "status", "dim", "zooms", "tr_header", "tr_used",
    "n_volumes", "dtype", "orientation", "qform_code", "sform_code", "obliquity_deg", "fov_z_mm",
    "stc_decision", "stc_reason", "slice_order", "evidence", "drop_volumes", "warnings",
    "header_changes", "source_bold", "source_t1w",
]
# bids layout: participants.tsv columns that may name the acquisition group (first match wins).
# The BIDS 'group' column is a diagnosis, not a site, and is deliberately not used.
PARTICIPANT_GROUP_COLUMNS = ("acq_group", "site", "site_id")


class IngestError(Exception):
    """This run (or image) cannot be used; reported as status 'error'."""


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def info(message: str) -> None:
    print(message, flush=True)


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


def is_nifti_name(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def nifti_extension(name: str) -> str:
    return ".nii.gz" if name.lower().endswith(".gz") else ".nii"


def nifti_stem(name: str) -> str:
    return name[: -len(nifti_extension(name))]


def clean_label(text: str) -> str:
    """BIDS labels are alphanumeric."""
    return re.sub(r"[^A-Za-z0-9]", "", str(text))


def normalise_subject_id(name: str) -> str:
    """Folder name or list entry -> 'sub-<alnum>'."""
    label = clean_label(re.sub(r"^sub[-_]", "", str(name).strip(), flags=re.IGNORECASE))
    if not label:
        raise IngestError(f"cannot derive a subject id from '{name}'")
    return f"sub-{label}"


def read_subject_list(path: Path) -> set[str]:
    """One or more ids per line (comma/blank separated, '#' comments), with or without 'sub-'."""
    wanted: set[str] = set()
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            for token in re.split(r"[,\s]+", line.split("#", 1)[0]):
                if token:
                    wanted.add(normalise_subject_id(token))
    return wanted


def abspath(path: Path) -> str:
    # Not Path.resolve(): a rawdata symlink must stay a rawdata path in the manifest.
    return os.path.abspath(str(path))


# ----------------------------------------------------------------------------
# NIfTI-1 header access and byte-exact copies
# ----------------------------------------------------------------------------

def _open_image(path: Path, mode: str) -> Any:
    if str(path).lower().endswith(".gz"):
        if "w" in mode:
            return gzip.open(path, mode, compresslevel=1)
        return gzip.open(path, mode)
    return open(path, mode)


def read_header_block(path: Path) -> bytes:
    try:
        with _open_image(path, "rb") as fh:
            block = fh.read(NIFTI1_HEADER_BYTES)
    except (OSError, EOFError, zlib.error) as err:
        raise IngestError(f"cannot read {path.name}: {err}") from err
    if len(block) < NIFTI1_HEADER_BYTES:
        raise IngestError(f"{path.name} is shorter than a NIfTI-1 header")
    return block


def read_header(path: Path) -> nib.Nifti1Header:
    """Header exactly as stored. nib.load() is not used because it resets
    scl_slope/scl_inter in the image header, which a byte copy must keep."""
    block = read_header_block(path)
    try:
        header = nib.Nifti1Header(binaryblock=block, check=False)
    except Exception as err:  # noqa: BLE001 - nibabel raises several header error types
        raise IngestError(f"{path.name} has no valid NIfTI-1 header: {err}") from err
    magic = bytes(header["magic"].item())
    if magic != b"n+1":
        raise IngestError(f"{path.name} is not a single-file NIfTI-1 image (magic {magic!r})")
    return header


def expected_tail_bytes(header: nib.Nifti1Header) -> int:
    """Bytes that must follow the 348-byte header: extensions + voxel data."""
    try:
        itemsize = int(header.get_data_dtype().itemsize)
    except Exception as err:  # noqa: BLE001
        raise IngestError(f"unsupported NIfTI data type: {err}") from err
    n_voxels = int(np.prod([int(v) for v in header.get_data_shape()], dtype=np.int64))
    offset = max(int(header["vox_offset"]), NIFTI1_HEADER_BYTES + 4)
    return offset - NIFTI1_HEADER_BYTES + n_voxels * itemsize


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": abspath(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def copy_record_path(target: Path) -> Path:
    return target.with_name(target.name + ".ingest.json")


def record_copy(source: Path, target: Path) -> None:
    write_json(copy_record_path(target), {"source": file_identity(source), "target": file_identity(target)})


def target_is_current(target: Path, header: nib.Nifti1Header, source: Path) -> bool:
    """Reuse only with matching source/target stat identities and expected header.

    This checks ordinary source replacement, not adversarial preservation of file
    size and timestamps. Old outputs without an identity record are recopied.
    """
    try:
        record = read_json(copy_record_path(target))
        return (isinstance(record, dict)
                and record.get("source") == file_identity(source)
                and record.get("target") == file_identity(target)
                and read_header_block(target) == header.binaryblock)
    except (OSError, ValueError, IngestError):
        return False


def copy_with_header(source: Path, target: Path, header: nib.Nifti1Header) -> None:
    """Stream `source` to `target`, replacing only the NIfTI-1 header.

    Everything after byte 348 (extensions, voxel data, scaling) is copied
    unchanged, so the data cannot be altered and memory use stays constant. A
    truncated source is detected by counting the streamed bytes.
    """
    expected = expected_tail_bytes(header)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".tmp{os.getpid()}_{target.name}")
    copied = 0
    try:
        with _open_image(source, "rb") as fin, _open_image(tmp, "wb") as fout:
            fin.read(NIFTI1_HEADER_BYTES)
            fout.write(header.binaryblock)
            while True:
                chunk = fin.read(1 << 22)
                if not chunk:
                    break
                fout.write(chunk)
                copied += len(chunk)
        if copied < expected:
            raise IngestError(f"{source.name} is truncated: {copied} data bytes, header promises {expected}")
        os.replace(tmp, target)
    except EOFError as err:
        raise IngestError(f"{source.name} is truncated (compressed stream ends early): {err}") from err
    except (OSError, zlib.error) as err:
        raise IngestError(f"cannot copy {source.name}: {err}") from err
    finally:
        if tmp.exists():
            tmp.unlink()


def link_or_copy(source: Path, target: Path) -> str:
    """Symlink (hard link, then copy as fall-backs: Windows, odd filesystems)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    try:
        os.symlink(abspath(source), target)
        return "symlink"
    except (OSError, NotImplementedError):
        pass
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        pass
    tmp = target.with_name(f".tmp{os.getpid()}_{target.name}")
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return "copy"


def remove_sibling(target: Path) -> None:
    """<stem>.nii and <stem>.nii.gz share one sidecar: keep only the current one."""
    stem = nifti_stem(target.name)
    other = target.with_name(stem + (".nii" if target.name.lower().endswith(".gz") else ".nii.gz"))
    if other.is_symlink() or other.exists():
        other.unlink()


# ----------------------------------------------------------------------------
# header description and normalisation
# ----------------------------------------------------------------------------

def describe_header(header: nib.Nifti1Header) -> dict[str, Any]:
    shape = tuple(int(v) for v in header.get_data_shape())
    zooms = tuple(float(v) for v in header.get_zooms()[:3])
    try:
        dtype = header.get_data_dtype().name
    except Exception as err:  # noqa: BLE001 - unknown datatype code
        raise IngestError(f"unsupported NIfTI data type: {err}") from err
    out: dict[str, Any] = {
        "shape": shape,
        "dim": "x".join(str(v) for v in shape),
        "zooms": "x".join(f"{v:.4g}" for v in zooms),
        "zooms_tuple": zooms,
        "n_volumes": shape[3] if len(shape) >= 4 else 1,
        "dtype": dtype,
        "qform_code": int(header["qform_code"]),
        "sform_code": int(header["sform_code"]),
        "orientation": None,
        "obliquity_deg": None,
        "fov_z_mm": shape[2] * zooms[2] if len(shape) >= 3 and len(zooms) >= 3 else None,
    }
    try:
        affine = header.get_best_affine()
        out["orientation"] = "".join(code or "?" for code in nib.aff2axcodes(affine))
        out["obliquity_deg"] = float(np.degrees(np.max(obliquity(affine))))
    except Exception:  # noqa: BLE001 - a broken affine is reported by normalise_orientation
        pass
    return out


def _corner_distance(shape: Sequence[int], a: np.ndarray, b: np.ndarray) -> float:
    """Largest displacement (mm) between two voxel->world affines over the volume corners."""
    ext = [max(int(v) - 1, 0) for v in list(shape[:3]) + [1, 1, 1]][:3]
    corners = np.array([[i, j, k, 1.0] for i in (0, ext[0]) for j in (0, ext[1]) for k in (0, ext[2])])
    delta = (corners @ a.T - corners @ b.T)[:, :3]
    return float(np.sqrt((delta ** 2).sum(axis=1)).max())


def normalise_orientation(header: nib.Nifti1Header) -> tuple[list[str], list[str]]:
    """Make qform and sform usable and equal in every tool. Returns (changes, warnings).

    Tools disagree on which form to read when both are set (AFNI honours
    AFNI_NIFTI_PRIORITY, FSL/nibabel/Workbench take the sform first), so a header
    in which only one is set, or in which they differ, gives a different geometry
    per tool. The sform (nibabel's 'best affine') is kept as the truth; the voxel
    data and that affine are never changed.
    """
    changes: list[str] = []
    warnings: list[str] = []
    qcode, scode = int(header["qform_code"]), int(header["sform_code"])
    if qcode <= 0 and scode <= 0:
        raise IngestError("qform_code and sform_code are both 0: the image has no orientation information")

    def valid(code: int) -> int:
        return code if code in VALID_XFORM_CODES else 1

    try:
        if scode <= 0:
            header.set_sform(header.get_qform(), code=valid(qcode))
            changes.append("sform copied from qform")
        elif qcode <= 0:
            header.set_qform(header.get_sform(), code=valid(scode))
            changes.append("qform set from sform")
        else:
            distance = _corner_distance(header.get_data_shape(), header.get_qform(), header.get_sform())
            if distance > QS_FORM_TOLERANCE_MM:
                header.set_qform(header.get_sform(), code=qcode)
                warnings.append(f"qform and sform differed by {distance:.2f} mm; qform replaced by sform")
                changes.append("qform set from sform")
    except IngestError:
        raise
    except Exception as err:  # noqa: BLE001 - invalid quaternion / singular affine
        raise IngestError(f"unusable orientation information: {err}") from err

    for key, code in (("qform_code", qcode), ("sform_code", scode)):
        if code > 0 and code not in VALID_XFORM_CODES:
            header[key] = 1
            changes.append(f"{key} {code} -> 1")
    return changes, warnings


@dataclass
class TRDecision:
    tr_used: float
    tr_header: float | None        # pixdim4 converted with the declared units (as found)
    conflict: bool = False         # declared and header TR disagree -> STC skipped
    essential_change: bool = False # the source header does not give this TR to every tool
    changes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def set_time_units_seconds(header: nib.Nifti1Header) -> None:
    """xyzt_units := <spatial code as found> | seconds. Direct bit surgery because
    nibabel's set_xyzt_units() raises on undefined spatial codes (4..7)."""
    xyz_code = int(header["xyzt_units"]) & 7
    header["xyzt_units"] = (xyz_code if xyz_code <= 3 else 0) + 8


def resolve_tr(header: nib.Nifti1Header, declared: float | None, declared_by: str) -> TRDecision:
    """Choose the TR of a run and store it in seconds in `header` (modified in place).

    declared: TR from the sidecar or the acquisition table (None if neither).
    A header TR of 0, 1 or garbage is a converter placeholder, not a measurement.
    """
    found = timing.header_tr(header)
    tr_header = found.literal_s if np.isfinite(found.literal_s) else None
    placeholder = found.seconds is None or abs(found.raw - 1.0) < 1e-6
    out = TRDecision(tr_used=float("nan"), tr_header=tr_header)

    if declared is not None and (not np.isfinite(declared) or declared <= 0):
        raise IngestError(f"TR of the {declared_by} must be positive: {declared}")

    if declared is None:
        if found.seconds is None:
            raise IngestError(f"TR unknown: {found.note}; no sidecar or acquisition table TR")
        out.tr_used = found.seconds
        store = found.seconds
        if found.note:
            out.warnings.append(found.note)
        if placeholder:
            out.warnings.append("header TR is 1 s, a common placeholder, and no table TR confirms it")
    elif found.seconds is not None and abs(found.seconds - declared) <= timing.TR_TOLERANCE_S:
        out.tr_used = declared
        store = found.seconds
    elif placeholder:
        out.tr_used = declared
        store = declared
        out.warnings.append(
            f"header TR unusable (pixdim4={found.raw:g}, units '{found.units}'); "
            f"replaced by the {declared_by} TR {declared:g} s"
        )
    else:
        out.conflict = True
        # A run-level sidecar comes from this run's DICOMs; a site table is only
        # a generalisation, so the run's own header is preferred over it.
        out.tr_used = declared if declared_by == "sidecar" else found.seconds
        store = found.seconds
        out.warnings.append(
            f"TR conflict: {declared_by} {declared:g} s vs header {found.seconds:g} s; "
            f"using {out.tr_used:g} s, STC skipped"
        )

    # FSL reads pixdim4 literally, AFNI honours the units: both must see `store`.
    # Unknown units with a correct value are harmless (cosmetic change only).
    if not np.isfinite(found.raw) or abs(found.raw - store) > 1e-6 or abs(found.literal_s - store) > 1e-6:
        out.essential_change = True
        header["pixdim"][4] = store
        out.changes.append(f"pixdim4 {found.raw:g} ({found.units}) -> {store:g} s")
    if found.units != "sec":
        set_time_units_seconds(header)
        if not out.essential_change:
            out.changes.append(f"time units '{found.units}' -> sec")
    return out


# ----------------------------------------------------------------------------
# discovery
# ----------------------------------------------------------------------------

@dataclass
class Candidate:
    subject: str
    session: str = "-"
    task: str = "rest"
    run: str = "-"
    group: str = GROUP_NONE
    bold: Path | None = None
    t1w: Path | None = None
    t1w_session: str = "-"
    sidecars: list[Path] = field(default_factory=list)   # low -> high priority
    error: str | None = None
    origin: str = ""                                      # folder shown when there is no BOLD

    @property
    def run_label(self) -> str:
        parts = [self.subject]
        if self.session != "-":
            parts.append(f"ses-{self.session}")
        parts.append(f"task-{self.task}")
        if self.run != "-":
            parts.append(f"run-{self.run}")
        return "_".join(parts)


def _nifti_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and is_nifti_name(p.name) and not p.name.startswith("."))


def _exactly_one(folder: Path, what: str) -> Path:
    files = _nifti_files(folder)
    if not files:
        raise IngestError(f"no {what} image (.nii/.nii.gz) in {folder}")
    if len(files) > 1:
        names = ", ".join(p.name for p in files[:4])
        raise IngestError(f"{len(files)} {what} images in {folder} ({names}): exactly one is expected")
    return files[0]


def dpabi_sites(input_dir: Path, func_dir: str) -> list[tuple[str, Path]]:
    if (input_dir / func_dir).is_dir():
        return [(GROUP_NONE, input_dir)]
    return [(p.name, p) for p in sorted(input_dir.iterdir()) if p.is_dir() and (p / func_dir).is_dir()]


def discover_dpabi(input_dir: Path, func_dir: str, t1_dir: str, task: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    for group, site_dir in dpabi_sites(input_dir, func_dir):
        for folder in sorted(p for p in (site_dir / func_dir).iterdir() if p.is_dir()):
            try:
                subject = normalise_subject_id(folder.name)
            except IngestError as err:
                warn(f"{folder}: {err}; folder ignored")
                continue
            cand = Candidate(subject=subject, task=task, group=group, origin=str(folder))
            try:
                cand.bold = _exactly_one(folder, "functional")
                sidecar = cand.bold.with_name(nifti_stem(cand.bold.name) + ".json")
                cand.sidecars = [sidecar] if sidecar.is_file() else []
                cand.t1w = _exactly_one(site_dir / t1_dir / folder.name, "T1")
            except IngestError as err:
                cand.error = str(err)
            candidates.append(cand)
    return candidates


def _entity(name: str, key: str) -> str | None:
    match = re.search(rf"(?:^|_){key}-([A-Za-z0-9]+)", name)
    return match.group(1) if match else None


def bids_groups(input_dir: Path) -> dict[str, str]:
    """participant -> acquisition group from participants.tsv (acq_group/site/site_id)."""
    path = input_dir / "participants.tsv"
    if not path.is_file():
        return {}
    try:
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (OSError, ValueError) as err:
        warn(f"cannot read {path}: {err}")
        return {}
    lower = {str(c).strip().lower(): c for c in frame.columns}
    column = next((lower[c] for c in PARTICIPANT_GROUP_COLUMNS if c in lower), None)
    if column is None or "participant_id" not in lower:
        return {}
    out: dict[str, str] = {}
    for pid, value in zip(frame[lower["participant_id"]], frame[column]):
        value = str(value).strip()
        if value and value.lower() not in ("n/a", "na", "nan"):
            try:
                out[normalise_subject_id(pid)] = value
            except IngestError:
                continue
    return out


def bids_t1(sub_dir: Path, session_dir: str | None, pattern: str) -> Path | None:
    """T1w of the same session, else of any session of the subject (sorted, first)."""
    folders = [sub_dir / session_dir / "anat"] if session_dir else []
    folders += [sub_dir / "anat"] + sorted(sub_dir.glob("ses-*/anat"))
    for folder in folders:
        if folder.is_dir():
            files = sorted(p for p in folder.glob(pattern) if p.is_file() and is_nifti_name(p.name))
            if files:
                return files[0]
    return None


def bids_sidecar_chain(input_dir: Path, bold: Path, sub_name: str, session_dir: str | None, task_raw: str) -> list[Path]:
    """Inheritance, low -> high priority: dataset root, subject, session, same stem."""
    chain = [input_dir / f"task-{task_raw}_bold.json", input_dir / sub_name / f"{sub_name}_task-{task_raw}_bold.json"]
    if session_dir:
        chain.append(input_dir / sub_name / session_dir / f"{sub_name}_{session_dir}_task-{task_raw}_bold.json")
    chain.append(bold.with_name(nifti_stem(bold.name) + ".json"))
    return [p for p in chain if p.is_file()]


def discover_bids(input_dir: Path, func_glob: str, t1_glob: str, default_task: str) -> list[Candidate]:
    groups = bids_groups(input_dir)
    candidates: list[Candidate] = []
    for sub_dir in sorted(p for p in input_dir.glob("sub-*") if p.is_dir()):
        try:
            subject = normalise_subject_id(sub_dir.name)
        except IngestError as err:
            warn(f"{sub_dir}: {err}; folder ignored")
            continue
        func_dirs = [(None, sub_dir / "func")] + [(p.name, p / "func") for p in sorted(sub_dir.glob("ses-*"))]
        for session_dir, func_dir in func_dirs:
            if not func_dir.is_dir():
                continue
            for bold in sorted(p for p in func_dir.glob(func_glob) if p.is_file() and is_nifti_name(p.name)):
                task_raw = _entity(bold.name, "task") or default_task
                cand = Candidate(
                    subject=subject,
                    session=clean_label(session_dir[4:]) if session_dir else "-",
                    task=clean_label(task_raw) or default_task,
                    run=_entity(bold.name, "run") or "-",
                    group=groups.get(subject, GROUP_NONE),
                    bold=bold,
                    origin=str(func_dir),
                    sidecars=bids_sidecar_chain(input_dir, bold, sub_dir.name, session_dir, task_raw),
                )
                cand.t1w = bids_t1(sub_dir, session_dir, t1_glob)
                if cand.t1w is None:
                    cand.error = f"no T1w image matching '{t1_glob}' for {sub_dir.name}"
                else:
                    cand.t1w_session = clean_label(_entity(cand.t1w.name, "ses") or "") or "-"
                candidates.append(cand)
    return candidates


def flag_duplicates(candidates: list[Candidate]) -> None:
    """<RUN> must be unique: it names every derivative of the run."""
    seen: dict[str, Candidate] = {}
    for cand in candidates:
        first = seen.setdefault(cand.run_label, cand)
        if first is not cand and cand.error is None:
            cand.error = (f"run label {cand.run_label} already used by {first.bold or first.origin} "
                          "(same subject id in two places, or entities other than sub/ses/task/run)")


# ----------------------------------------------------------------------------
# per-run processing
# ----------------------------------------------------------------------------

@dataclass
class Settings:
    layout: str
    raw_dir: Path
    drop_volumes: int
    min_volumes: int
    overwrite: bool
    table: dict[str, timing.AcqRow] | None
    table_path: Path | None


def check_bold_geometry(desc: dict[str, Any], min_volumes: int) -> None:
    shape = desc["shape"]
    if len(shape) != 4:
        raise IngestError(f"BOLD is not 4D (dim {desc['dim']})")
    if shape[3] < min_volumes:
        raise IngestError(f"only {shape[3]} volumes (fewer than {min_volumes})")
    if min(shape[:3]) < 2 or not all(np.isfinite(z) and z > 0 for z in desc["zooms_tuple"]):
        raise IngestError(f"implausible geometry (dim {desc['dim']}, zooms {desc['zooms']})")


def load_sidecars(paths: Sequence[Path], warnings: list[str]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        try:
            content = read_json(path)
        except (OSError, ValueError) as err:
            warnings.append(f"sidecar {path.name} unreadable ({err})")
            continue
        if isinstance(content, dict):
            merged.update(content)
    return merged


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def decide_timing(
    meta: dict[str, Any],
    row: timing.AcqRow | None,
    header: nib.Nifti1Header,
    desc: dict[str, Any],
    tr: TRDecision,
    settings: Settings,
) -> tuple[timing.TimingDecision, dict[str, Any]]:
    """STC decision of one run and the sidecar keys it adds (table fills gaps only)."""
    n_slices = desc["shape"][2]
    direction = str(meta.get("SliceEncodingDirection") or "k").strip()

    if meta.get("SliceTiming") is not None:
        source = str(meta.get("SliceTimingSource") or "sidecar")
        evidence = str(meta.get("SliceTimingEvidence") or "sidecar")
        if tr.conflict:
            return timing.TimingDecision("skip", "TR of the sidecar and of the NIfTI header disagree", None,
                                         "sidecar", evidence, source), {}
        try:
            timing.afni_slice_times(meta["SliceTiming"], n_slices, tr.tr_used, direction)
        except timing.TimingError as err:
            return timing.TimingDecision("skip", f"sidecar SliceTiming rejected: {err}", None, "sidecar",
                                         evidence, source), {}
        return timing.TimingDecision("apply", "SliceTiming from the sidecar", None, "sidecar", evidence, source), {}

    decision = timing.table_timing(row, n_slices, tr.tr_used, settings.table_path, tr_conflict=tr.conflict)
    if decision.stc != "apply":
        return decision, {}

    def skip(reason: str) -> tuple[timing.TimingDecision, dict[str, Any]]:
        return timing.TimingDecision("skip", reason, None, decision.slice_order, decision.evidence,
                                     decision.source), {}

    # 3dTshift shifts along k only, and the table order codes count stored slices
    # for volumes stored inferior -> superior along k (how every ABIDE/ADNI site
    # table was derived); a 'file:' row gives explicit times per stored slice and
    # is therefore exempt from the orientation rule.
    slice_dim = header.get_dim_info()[2]
    if slice_dim is not None and slice_dim != 2:
        return skip(f"NIfTI dim_info says the slice axis is {'ijk'[slice_dim]}, not k")
    if direction not in ("k", "k-"):
        return skip(f"SliceEncodingDirection '{direction}' is not supported (slice axis must be k)")
    orientation = desc["orientation"] or "???"
    if orientation[2] != "S" and not timing.is_file_spec(decision.slice_order):
        return skip(f"third voxel axis points to '{orientation[2]}' but table order codes assume slices stored "
                    "inferior->superior (give explicit per-slice times with file:<path>)")

    times = decision.slice_times
    assert times is not None
    # The stage-03 reader reverses the vector for 'k-'; store it so that it comes out right.
    stored = times[::-1] if direction == "k-" else times
    added = {
        "SliceTiming": [round(float(t), 6) for t in stored],
        "SliceEncodingDirection": direction,
        "SliceTimingSource": decision.source,
        "SliceTimingEvidence": decision.evidence,
    }
    return decision, added


class T1Cache:
    """T1w images are shared by the runs of a subject: validate/copy each one once."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._done: dict[Path, tuple[str | None, str | None]] = {}

    def resolve(self, cand: Candidate) -> str:
        assert cand.t1w is not None
        if cand.t1w not in self._done:
            try:
                self._done[cand.t1w] = (self._ingest(cand), None)
            except IngestError as err:
                self._done[cand.t1w] = (None, f"T1w {cand.t1w.name}: {err}")
        path, error = self._done[cand.t1w]
        if error is not None or path is None:
            raise IngestError(error or "T1w unusable")
        return path

    def _ingest(self, cand: Candidate) -> str:
        assert cand.t1w is not None
        header = read_header(cand.t1w)
        shape = [int(v) for v in header.get_data_shape()]
        if len(shape) < 3 or min(shape[:3]) < 2 or int(np.prod(shape[3:])) != 1:
            raise IngestError(f"not a 3D image (dim {'x'.join(map(str, shape))})")
        changes, warnings = normalise_orientation(header)
        for message in warnings:
            warn(f"{cand.subject} T1w: {message}")
        if self.settings.layout == "bids" and not changes:
            return abspath(cand.t1w)
        name = cand.subject + (f"_ses-{cand.t1w_session}" if cand.t1w_session != "-" else "") + "_T1w.nii.gz"
        folder = self.settings.raw_dir / cand.subject
        if cand.t1w_session != "-":
            folder = folder / f"ses-{cand.t1w_session}"
        target = folder / "anat" / name
        if self.settings.overwrite or not target_is_current(target, header, cand.t1w):
            copy_with_header(cand.t1w, target, header)
            record_copy(cand.t1w, target)
            info(f"{cand.subject}: T1w -> {target}" + (f" [{'; '.join(changes)}]" if changes else ""))
        return abspath(target)


def new_report_row(cand: Candidate) -> dict[str, Any]:
    row: dict[str, Any] = {key: None for key in REPORT_COLUMNS}
    row.update(subject=cand.subject, session=cand.session, run_label=cand.run_label, group=cand.group,
               status="error", source_bold=str(cand.bold) if cand.bold else cand.origin,
               source_t1w=str(cand.t1w) if cand.t1w else None)
    return row


def declared_tr(meta: dict[str, Any], acq: timing.AcqRow | None) -> tuple[float | None, str]:
    """TR stated outside the image: sidecar first, acquisition table second."""
    value = meta.get("RepetitionTime")
    if value is not None:
        number = _float_or_none(value)
        if number is None:
            raise IngestError(f"sidecar RepetitionTime is not a number: {value!r}")
        return number, "sidecar"
    if acq is not None and acq.tr is not None:
        return acq.tr, "acquisition table"
    return None, "none"


def table_fill(meta: dict[str, Any], acq: timing.AcqRow | None, timing_keys: dict[str, Any]) -> dict[str, Any]:
    """Keys the acquisition table adds; an existing sidecar value always wins.

    DropVolumes is written only when the table states it, so that the DROP_VOLUMES
    setting stays effective in stage 03 for every other run.
    """
    filled = dict(timing_keys)
    if acq is not None and acq.pe_dir and "PhaseEncodingDirection" not in meta:
        filled["PhaseEncodingDirection"] = acq.pe_dir
    if acq is not None and acq.drop_volumes is not None and "DropVolumes" not in meta:
        filled["DropVolumes"] = acq.drop_volumes
    return filled


def build_sidecar(meta: dict[str, Any], filled: dict[str, Any], tr_used: float, cand: Candidate,
                  decision: timing.TimingDecision) -> dict[str, Any]:
    sidecar = {**meta, **filled}
    sidecar["RepetitionTime"] = tr_used
    sidecar.setdefault("TaskName", cand.task)
    sidecar.setdefault("SliceEncodingDirection", "k")
    sidecar["AcquisitionGroup"] = cand.group
    if decision.stc == "apply":
        sidecar.pop("SliceTimingSkipReason", None)
        sidecar.setdefault("SliceTimingSource", decision.source)
        sidecar.setdefault("SliceTimingEvidence", decision.evidence)
    else:
        # SliceTiming is written only when it is applied: a rejected or empty
        # vector would make stage 03 re-decide on its own. The dataset's own
        # sidecar is never modified, only this rawdata copy.
        sidecar.pop("SliceTiming", None)
        sidecar["SliceTimingSkipReason"] = decision.reason
        sidecar["SliceTimingSource"] = decision.source
        sidecar["SliceTimingEvidence"] = decision.evidence
    return sidecar


def write_rawdata_bold(cand: Candidate, settings: Settings, header: nib.Nifti1Header,
                       source_header: nib.Nifti1Header, copy: bool) -> Path:
    """Normalised copy (`copy`) or a link to the untouched original; returns the rawdata path."""
    assert cand.bold is not None
    folder = settings.raw_dir / cand.subject
    if cand.session != "-":
        folder = folder / f"ses-{cand.session}"
    if copy:
        target = folder / "func" / f"{cand.run_label}_bold.nii.gz"
        if settings.overwrite or not target_is_current(target, header, cand.bold):
            copy_with_header(cand.bold, target, header)
            record_copy(cand.bold, target)
            info(f"{cand.run_label}: BOLD -> {target}")
    else:
        target = folder / "func" / f"{cand.run_label}_bold{nifti_extension(cand.bold.name)}"
        if settings.overwrite or not target_is_current(target, source_header, cand.bold):
            how = link_or_copy(cand.bold, target)
            record_copy(cand.bold, target)
            info(f"{cand.run_label}: BOLD {how} -> {target}")
    remove_sibling(target)
    return target


def run_warnings(cand: Candidate, desc: dict[str, Any], drop: int, acq: timing.AcqRow | None,
                 min_volumes: int) -> list[str]:
    out: list[str] = []
    if desc["fov_z_mm"] is not None and desc["fov_z_mm"] < SHORT_FOV_Z_MM:
        out.append(f"short z-FOV ({desc['fov_z_mm']:.0f} mm)")
    if desc["n_volumes"] - drop < min_volumes:
        out.append(f"only {desc['n_volumes'] - drop} volumes left after dropping {drop}")
    if acq is not None and acq.matched == "default" and cand.group != GROUP_NONE:
        out.append(f"group '{cand.group}' is not in the acquisition table (default row used)")
    return out


def ingest_run(cand: Candidate, settings: Settings, t1_cache: T1Cache, row: dict[str, Any],
               warnings: list[str]) -> dict[str, str]:
    """Validate one run, write its rawdata files, fill `row`; returns the manifest row.

    Everything is validated before anything is written, so an error row never
    leaves a plausible-looking BOLD copy behind.
    """
    if cand.error is not None or cand.bold is None or cand.t1w is None:
        raise IngestError(cand.error or "incomplete run")

    header = read_header(cand.bold)
    source_header = header.copy()
    desc = describe_header(header)
    row.update({key: desc[key] for key in ("dim", "zooms", "n_volumes", "dtype", "orientation", "qform_code",
                                           "sform_code", "obliquity_deg", "fov_z_mm")})
    check_bold_geometry(desc, settings.min_volumes)
    orient_changes, orient_warnings = normalise_orientation(header)
    warnings.extend(orient_warnings)

    meta = load_sidecars(cand.sidecars, warnings)
    same_stem = cand.bold.with_name(nifti_stem(cand.bold.name) + ".json")
    inherited = meta != (load_sidecars([same_stem], []) if same_stem in cand.sidecars else {})
    acq = timing.lookup_acq(settings.table, cand.group)

    tr = resolve_tr(header, *declared_tr(meta, acq))
    warnings.extend(tr.warnings)
    row.update(tr_header=tr.tr_header, tr_used=tr.tr_used)

    decision, timing_keys = decide_timing(meta, acq, header, desc, tr, settings)
    row.update(stc_decision=decision.stc, stc_reason=decision.reason, slice_order=decision.slice_order,
               evidence=decision.evidence or None)
    timing_offered = meta.get("SliceTiming") is not None or (
        acq is not None and acq.stc == "apply" and acq.slice_order != "unknown")
    if timing_offered and decision.stc == "skip":
        warnings.append(f"STC skipped: {decision.reason}")

    filled = table_fill(meta, acq, timing_keys)
    drop = _float_or_none({**meta, **filled}.get("DropVolumes"))
    drop = settings.drop_volumes if drop is None or drop < 0 else int(drop)
    row.update(drop_volumes=drop)
    warnings.extend(run_warnings(cand, desc, drop, acq, settings.min_volumes))

    t1_path = t1_cache.resolve(cand)

    # Header edits that only restate the units do not justify copying a BIDS image.
    needs_copy = settings.layout == "dpabi" or tr.essential_change or bool(orient_changes)
    if needs_copy or inherited or filled:
        target = write_rawdata_bold(cand, settings, header, source_header, needs_copy)
        write_json(target.with_name(nifti_stem(target.name) + ".json"),
                   build_sidecar(meta, filled, tr.tr_used, cand, decision))
        bold_path = abspath(target)
    else:
        bold_path = abspath(cand.bold)
    changes = orient_changes + tr.changes if needs_copy else []

    row.update(header_changes="; ".join(changes) or None)
    return {"subject": cand.subject, "session": cand.session, "task": cand.task, "run": cand.run,
            "group": cand.group, "bold": bold_path, "t1w": t1_path, "run_label": cand.run_label}


def process_candidate(cand: Candidate, settings: Settings, t1_cache: T1Cache) -> tuple[dict[str, Any], dict[str, str] | None]:
    row = new_report_row(cand)
    warnings: list[str] = []
    manifest_row: dict[str, str] | None = None
    try:
        manifest_row = ingest_run(cand, settings, t1_cache, row, warnings)
        row["status"] = "ok"
        info(f"{cand.run_label}: ok (group {cand.group}, TR {row['tr_used']:g} s, {row['dim']}, "
             f"STC {row['stc_decision']}: {row['stc_reason']})")
    except IngestError as err:
        warnings.insert(0, f"ERROR: {err}")
        print(f"ERROR: {cand.run_label}: {err}", file=sys.stderr, flush=True)
    for message in warnings:
        if not message.startswith("ERROR"):
            warn(f"{cand.run_label}: {message}")
    row["warnings"] = re.sub(r"[\t\r\n]+", " ", "; ".join(warnings)) or None
    return row, manifest_row


# ----------------------------------------------------------------------------
# dataset level
# ----------------------------------------------------------------------------

def write_tables(raw_dir: Path, report: list[dict[str, Any]], manifest: list[dict[str, str]]) -> None:
    groups: dict[str, str] = {}
    for row in manifest:
        previous = groups.setdefault(row["subject"], row["group"])
        if previous != row["group"]:
            raise IngestError(f"conflicting acquisition groups for {row['subject']}: {previous}, {row['group']}")
    participants = pd.DataFrame(sorted(groups.items()), columns=["participant_id", "acq_group"])
    write_tsv(raw_dir / "ingest_report.tsv", pd.DataFrame(report, columns=REPORT_COLUMNS))
    write_tsv(raw_dir / "manifest.tsv", pd.DataFrame(manifest, columns=MANIFEST_COLUMNS))
    write_tsv(raw_dir / "participants.tsv", participants)


def run_ingest(args: argparse.Namespace) -> int:
    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        return 2
    task = clean_label(args.task) or "rest"
    try:
        table = timing.load_acq_table(args.acq_table) if args.acq_table else None
        wanted = read_subject_list(args.subject_list) if args.subject_list else None
    except (OSError, ValueError, IngestError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2

    if args.layout == "dpabi":
        candidates = discover_dpabi(input_dir, args.dpabi_func_dir, args.dpabi_t1_dir, task)
    else:
        candidates = discover_bids(input_dir, args.bids_func_glob, args.bids_t1_glob, task)
    info(f"{len(candidates)} run(s) found in {input_dir} (layout {args.layout})")
    if wanted is not None:
        found = {c.subject for c in candidates}
        for missing in sorted(wanted - found):
            warn(f"subject list: {missing} not found in {input_dir}")
        candidates = [c for c in candidates if c.subject in wanted]
        info(f"{len(candidates)} run(s) kept by the subject list")
    candidates.sort(key=lambda c: (c.subject, c.session, c.run_label))
    flag_duplicates(candidates)

    settings = Settings(layout=args.layout, raw_dir=args.raw_dir, drop_volumes=args.drop_volumes,
                        min_volumes=args.min_volumes, overwrite=args.overwrite, table=table,
                        table_path=args.acq_table)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    t1_cache = T1Cache(settings)
    report: list[dict[str, Any]] = []
    manifest: list[dict[str, str]] = []
    for cand in candidates:
        row, manifest_row = process_candidate(cand, settings, t1_cache)
        report.append(row)
        if manifest_row is not None:
            manifest.append(manifest_row)
    write_tables(args.raw_dir, report, manifest)

    n_error = len(report) - len(manifest)
    n_stc = sum(1 for row in report if row["status"] == "ok" and row["stc_decision"] == "apply")
    info(f"ingest: {len(manifest)} valid run(s), {n_error} error(s), STC timing for {n_stc} run(s); "
         f"see {args.raw_dir / 'ingest_report.tsv'}")
    if not manifest:
        print("ERROR: no valid run found", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fmriproc.ingest", description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--layout", choices=["dpabi", "bids"], required=True)
    parser.add_argument("--raw-dir", type=Path, required=True, help="$OUT_DIR/rawdata")
    parser.add_argument("--acq-table", type=Path, default=None, help="acquisition table TSV (DESIGN.md section 5)")
    parser.add_argument("--task", default="rest", help="task label of dpabi runs; bids fall-back")
    parser.add_argument("--bids-func-glob", default="*task-rest*_bold.nii*")
    parser.add_argument("--bids-t1-glob", default="*_T1w.nii*")
    parser.add_argument("--dpabi-func-dir", default="FunImg")
    parser.add_argument("--dpabi-t1-dir", default="T1Img")
    parser.add_argument("--subject-list", type=Path, default=None, help="restrict to these subject ids")
    parser.add_argument("--drop-volumes", type=int, default=0,
                        help="default reported when neither sidecar nor table gives DropVolumes")
    parser.add_argument("--min-volumes", type=int, default=MIN_VOLUMES_DEFAULT)
    parser.add_argument("--overwrite", action="store_true", help="rewrite image copies that already exist")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.drop_volumes < 0:
        print("ERROR: --drop-volumes must not be negative", file=sys.stderr)
        return 2
    return run_ingest(args)


if __name__ == "__main__":
    sys.exit(main())
