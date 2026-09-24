"""Tests of fmriproc.ingest on synthetic DPABI and BIDS trees (tiny NIfTIs)."""
from __future__ import annotations

import contextlib
import gzip
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from fmriproc import ingest
from fmriproc.timing import slice_order_vector, slice_times_from_order

ACQ_TABLE = (
    "group\ttr\tn_slices\tslice_order\tstc\tdrop_volumes\tpe_dir\tevidence\tnote\n"
    "SITE_A\t2\t5\tIA\tapply\t4\tj-\tA:protocol\t-\n"
    "*\t-\t-\tunknown\tskip\t2\t-\tC:no_information\tdefault row\n"
)
AFFINE = np.array([[-3.0, 0.0, 0.0, 90.0], [0.0, 3.0, 0.0, -126.0], [0.0, 0.0, 4.0, -72.0], [0.0, 0.0, 0.0, 1.0]])
IA5 = slice_times_from_order(slice_order_vector("IA", 5), 2.0)


def _read_bytes(path: Path) -> bytes:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as fh:
        return fh.read()


def _write_bytes(path: Path, payload: bytes) -> None:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "wb") as fh:
        fh.write(payload)


def patch_header(path: Path, **fields: float) -> None:
    """Edit raw header fields of a saved file (nibabel refuses to save some of these states)."""
    payload = _read_bytes(path)
    header = nib.Nifti1Header(binaryblock=payload[:348], check=False)
    for key, value in fields.items():
        if key == "pixdim4":
            header["pixdim"][4] = value
        else:
            header[key] = value
    _write_bytes(path, header.binaryblock + payload[348:])


def make_image(path: Path, shape: tuple[int, ...], dtype: type = np.int16, tr: float = 2.0,
               t_units: str = "sec", scaled: bool = False, seed: int = 0, affine: np.ndarray = AFFINE,
               **header_fields: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 1000, size=shape).astype(dtype)
    img = nib.Nifti1Image(data, affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    if len(shape) == 4:
        img.header.set_xyzt_units("mm", t_units)
    if scaled:
        img.header.set_slope_inter(2.0, 10.0)
    nib.save(img, str(path))
    fields = dict(header_fields)
    if len(shape) == 4:
        fields.setdefault("pixdim4", tr)
    if fields:
        patch_header(path, **fields)


def raw_header(path: Path) -> nib.Nifti1Header:
    return nib.Nifti1Header(binaryblock=_read_bytes(path)[:348], check=False)


def run_ingest(*argv: str) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = ingest.main(list(argv))
    return code, out.getvalue() + err.getvalue()


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


class TreeCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.input = self.root / "input"
        self.raw = self.root / "out" / "rawdata"
        self.table = self.root / "acq.tsv"
        self.table.write_text(ACQ_TABLE, encoding="utf-8")

    def report(self) -> pd.DataFrame:
        return read_table(self.raw / "ingest_report.tsv").set_index("subject")

    def manifest(self) -> pd.DataFrame:
        return read_table(self.raw / "manifest.tsv").set_index("subject")

    def sidecar(self, bold: str) -> dict:
        return json.loads(Path(bold.replace(".nii.gz", ".json").replace(".nii", ".json")).read_text(encoding="utf-8"))


class DpabiTree(TreeCase):
    def setUp(self) -> None:
        super().setUp()
        a, b = self.input / "SITE_A", self.input / "SITE_B"
        # sform_code 0, scaled int16
        make_image(a / "FunImg" / "sub-0001" / "rest.nii.gz", (6, 6, 5, 32), scaled=True, sform_code=0)
        make_image(a / "T1Img" / "sub-0001" / "anat.nii.gz", (8, 8, 8))
        # two functional images
        make_image(a / "FunImg" / "sub-0002" / "rest.nii.gz", (6, 6, 5, 32))
        make_image(a / "FunImg" / "sub-0002" / "rest_old.nii", (6, 6, 5, 32))
        make_image(a / "T1Img" / "sub-0002" / "anat.nii.gz", (8, 8, 8))
        # placeholder header TR, TR conflict, slice count mismatch
        make_image(a / "FunImg" / "sub-0009" / "rest.nii.gz", (6, 6, 5, 32), tr=1.0)
        make_image(a / "T1Img" / "sub-0009" / "anat.nii.gz", (8, 8, 8))
        make_image(a / "FunImg" / "sub-0010" / "rest.nii.gz", (6, 6, 5, 32), tr=2.5)
        make_image(a / "T1Img" / "sub-0010" / "anat.nii.gz", (8, 8, 8))
        make_image(a / "FunImg" / "sub-0011" / "rest.nii.gz", (6, 6, 7, 32))
        make_image(a / "T1Img" / "sub-0011" / "anat.nii.gz", (8, 8, 8))
        # codes 3 (talairach), TR in milliseconds, uncompressed, folder without 'sub-', site not in the table
        make_image(b / "FunImg" / "0003" / "rest.nii", (6, 6, 5, 40), dtype=np.float32, tr=2000.0, t_units="msec",
                   qform_code=3, sform_code=3)
        make_image(b / "T1Img" / "0003" / "mprage.nii.gz", (8, 8, 8), qform_code=3, sform_code=3)
        # missing T1, 3D BOLD, too short, no orientation
        make_image(b / "FunImg" / "sub-0004" / "rest.nii.gz", (6, 6, 5, 32))
        make_image(b / "FunImg" / "sub-0005" / "rest.nii.gz", (6, 6, 5))
        make_image(b / "T1Img" / "sub-0005" / "mprage.nii.gz", (8, 8, 8))
        make_image(b / "FunImg" / "sub-0006" / "rest.nii.gz", (6, 6, 5, 10))
        make_image(b / "T1Img" / "sub-0006" / "mprage.nii.gz", (8, 8, 8))
        make_image(b / "FunImg" / "sub-0007" / "rest.nii.gz", (6, 6, 5, 32), qform_code=0, sform_code=0)
        make_image(b / "T1Img" / "sub-0007" / "mprage.nii.gz", (8, 8, 8))
        (self.input / "notes").mkdir()                          # not a site: no FunImg inside

    def ingest(self, *extra: str) -> tuple[int, str]:
        return run_ingest("--input-dir", str(self.input), "--layout", "dpabi", "--raw-dir", str(self.raw),
                          "--acq-table", str(self.table), "--task", "rest", "--drop-volumes", "3", *extra)

    def test_full_tree(self) -> None:
        code, log = self.ingest()
        self.assertEqual(code, 0, log)
        report, manifest = self.report(), self.manifest()

        self.assertEqual(sorted(manifest.index), ["sub-0001", "sub-0003", "sub-0009", "sub-0010", "sub-0011"])
        self.assertEqual(list(read_table(self.raw / "manifest.tsv").columns), ingest.MANIFEST_COLUMNS)
        self.assertEqual(list(read_table(self.raw / "ingest_report.tsv").columns)[:22], ingest.REPORT_COLUMNS[:22])
        status = report["status"].to_dict()
        for sub in ("sub-0002", "sub-0004", "sub-0005", "sub-0006", "sub-0007"):
            self.assertEqual(status[sub], "error", sub)
            self.assertFalse((self.raw / sub / "func").exists(), sub)
        self.assertIn("2 functional images", report.loc["sub-0002", "warnings"])
        self.assertIn("no T1 image", report.loc["sub-0004", "warnings"])
        self.assertIn("not 4D", report.loc["sub-0005", "warnings"])
        self.assertIn("only 10 volumes", report.loc["sub-0006", "warnings"])
        self.assertIn("both 0", report.loc["sub-0007", "warnings"])

        participants = read_table(self.raw / "participants.tsv")
        self.assertEqual(list(participants.columns), ["participant_id", "acq_group"])
        self.assertEqual(dict(zip(participants["participant_id"], participants["acq_group"]))["sub-0003"], "SITE_B")

        row = manifest.loc["sub-0001"]
        self.assertEqual((row["session"], row["task"], row["run"], row["group"], row["run_label"]),
                         ("-", "rest", "-", "SITE_A", "sub-0001_task-rest"))
        self.assertEqual(Path(row["bold"]), self.raw / "sub-0001" / "func" / "sub-0001_task-rest_bold.nii.gz")
        self.assertEqual(Path(row["t1w"]), self.raw / "sub-0001" / "anat" / "sub-0001_T1w.nii.gz")
        self.assertTrue(os.path.isabs(row["bold"]) and Path(row["bold"]).is_file() and Path(row["t1w"]).is_file())

    def test_sform_from_qform_and_untouched_data(self) -> None:
        self.assertEqual(self.ingest()[0], 0)
        source = self.input / "SITE_A" / "FunImg" / "sub-0001" / "rest.nii.gz"
        copy = Path(self.manifest().loc["sub-0001", "bold"])
        self.assertEqual(int(raw_header(source)["sform_code"]), 0)
        header = raw_header(copy)
        self.assertEqual((int(header["qform_code"]), int(header["sform_code"])), (1, 1))
        np.testing.assert_allclose(header.get_sform(), AFFINE, atol=1e-4)
        np.testing.assert_allclose(nib.load(str(copy)).affine, nib.load(str(source)).affine, atol=1e-4)
        self.assertEqual(_read_bytes(copy)[348:], _read_bytes(source)[348:])      # voxel bytes identical
        self.assertEqual((float(header["scl_slope"]), float(header["scl_inter"])), (2.0, 10.0))
        a, b = nib.load(str(source)), nib.load(str(copy))
        self.assertEqual(b.get_data_dtype(), np.dtype(np.int16))
        np.testing.assert_array_equal(np.asanyarray(a.dataobj), np.asanyarray(b.dataobj))

        row = self.report().loc["sub-0001"]
        self.assertEqual((row["qform_code"], row["sform_code"]), ("1", "0"))        # as found in the source
        self.assertIn("sform copied from qform", row["header_changes"])
        self.assertEqual((row["dim"], row["n_volumes"], row["dtype"], row["orientation"]),
                         ("6x6x5x32", "32", "int16", "LAS"))
        self.assertEqual(float(row["fov_z_mm"]), 20.0)
        self.assertIn("short z-FOV", row["warnings"])
        self.assertAlmostEqual(float(row["obliquity_deg"]), 0.0, places=3)

    def test_sidecar_from_table(self) -> None:
        self.assertEqual(self.ingest()[0], 0)
        meta = self.sidecar(self.manifest().loc["sub-0001", "bold"])
        self.assertEqual(meta["RepetitionTime"], 2.0)
        np.testing.assert_allclose(meta["SliceTiming"], IA5, atol=1e-6)
        self.assertEqual(meta["SliceEncodingDirection"], "k")
        self.assertEqual(meta["PhaseEncodingDirection"], "j-")
        self.assertEqual(meta["DropVolumes"], 4)
        self.assertEqual(meta["AcquisitionGroup"], "SITE_A")
        self.assertEqual(meta["SliceTimingEvidence"], "A:protocol")
        self.assertIn("acq.tsv", meta["SliceTimingSource"])
        self.assertNotIn("SliceTimingSkipReason", meta)
        row = self.report().loc["sub-0001"]
        self.assertEqual((row["stc_decision"], row["slice_order"], row["evidence"], row["drop_volumes"]),
                         ("apply", "IA", "A:protocol", "4"))

    def test_code_3_and_millisecond_tr(self) -> None:
        self.assertEqual(self.ingest()[0], 0)
        manifest, report = self.manifest(), self.report()
        bold = Path(manifest.loc["sub-0003", "bold"])
        self.assertEqual(bold.name, "sub-0003_task-rest_bold.nii.gz")               # .nii source, .nii.gz copy
        header = raw_header(bold)
        self.assertEqual((int(header["qform_code"]), int(header["sform_code"])), (1, 1))
        self.assertAlmostEqual(float(header["pixdim"][4]), 2.0)
        self.assertEqual(header.get_xyzt_units(), ("mm", "sec"))
        self.assertEqual(nib.load(str(bold)).get_data_dtype(), np.dtype(np.float32))
        t1 = raw_header(Path(manifest.loc["sub-0003", "t1w"]))
        self.assertEqual((int(t1["qform_code"]), int(t1["sform_code"])), (1, 1))

        row = report.loc["sub-0003"]
        self.assertEqual((row["group"], row["stc_decision"], row["drop_volumes"]), ("SITE_B", "skip", "2"))
        self.assertEqual((float(row["tr_header"]), float(row["tr_used"])), (2.0, 2.0))
        self.assertIn("qform_code 3 -> 1", row["header_changes"])
        self.assertIn("pixdim4 2000", row["header_changes"])
        self.assertIn("not in the acquisition table", row["warnings"])
        meta = self.sidecar(str(bold))
        self.assertNotIn("SliceTiming", meta)
        self.assertTrue(meta["SliceTimingSkipReason"])
        self.assertEqual((meta["RepetitionTime"], meta["DropVolumes"], meta["AcquisitionGroup"]), (2.0, 2, "SITE_B"))

    def test_tr_placeholder_conflict_and_slice_count(self) -> None:
        self.assertEqual(self.ingest()[0], 0)
        manifest, report = self.manifest(), self.report()

        placeholder = report.loc["sub-0009"]
        self.assertEqual((float(placeholder["tr_header"]), float(placeholder["tr_used"])), (1.0, 2.0))
        self.assertEqual(placeholder["stc_decision"], "apply")
        self.assertIn("replaced by the acquisition table TR", placeholder["warnings"])
        self.assertAlmostEqual(float(raw_header(Path(manifest.loc["sub-0009", "bold"]))["pixdim"][4]), 2.0)

        conflict = report.loc["sub-0010"]
        self.assertEqual((float(conflict["tr_used"]), conflict["stc_decision"]), (2.5, "skip"))
        self.assertIn("TR conflict", conflict["warnings"])
        self.assertAlmostEqual(float(raw_header(Path(manifest.loc["sub-0010", "bold"]))["pixdim"][4]), 2.5)
        meta = self.sidecar(manifest.loc["sub-0010", "bold"])
        self.assertEqual(meta["RepetitionTime"], 2.5)
        self.assertNotIn("SliceTiming", meta)

        mismatch = report.loc["sub-0011"]
        self.assertEqual(mismatch["stc_decision"], "skip")
        self.assertIn("n_slices", mismatch["stc_reason"])
        self.assertIn("STC skipped", mismatch["warnings"])
        self.assertNotIn("SliceTiming", self.sidecar(manifest.loc["sub-0011", "bold"]))

    def test_idempotent_copies_and_overwrite(self) -> None:
        self.assertEqual(self.ingest()[0], 0)
        bold = Path(self.manifest().loc["sub-0001", "bold"])
        good = _read_bytes(bold)
        _write_bytes(bold, good[:348] + good[348:400])           # same header, damaged data
        self.assertEqual(self.ingest()[0], 0)
        self.assertEqual(_read_bytes(bold), good)               # target identity changed: repair copy
        self.assertEqual(self.ingest("--overwrite")[0], 0)
        self.assertEqual(_read_bytes(bold), good)
        patch_header(bold, pixdim4=9.0)                          # header no longer the wanted one
        self.assertEqual(self.ingest()[0], 0)
        self.assertEqual(_read_bytes(bold), good)
        self.assertEqual([p.name for p in bold.parent.iterdir() if p.name.startswith(".tmp")], [])

    def test_truncated_source(self) -> None:
        source = self.input / "SITE_A" / "FunImg" / "sub-0001" / "rest.nii.gz"
        payload = _read_bytes(source)
        _write_bytes(source, payload[: len(payload) // 2])
        self.assertEqual(self.ingest()[0], 0)
        row = self.report().loc["sub-0001"]
        self.assertEqual(row["status"], "error")
        self.assertIn("truncated", row["warnings"])
        self.assertFalse((self.raw / "sub-0001" / "func" / "sub-0001_task-rest_bold.nii.gz").exists())

    def test_subject_list(self) -> None:
        listing = self.root / "subjects.txt"
        listing.write_text("0003   # without prefix\nsub-0001, sub-9999\n", encoding="utf-8")
        code, log = self.ingest("--subject-list", str(listing))
        self.assertEqual(code, 0, log)
        self.assertEqual(sorted(self.manifest().index), ["sub-0001", "sub-0003"])
        self.assertEqual(sorted(self.report().index), ["sub-0001", "sub-0003"])
        self.assertIn("sub-9999 not found", log)

    def test_no_valid_run_fails(self) -> None:
        listing = self.root / "subjects.txt"
        listing.write_text("sub-0002\nsub-0004\n", encoding="utf-8")
        code, _ = self.ingest("--subject-list", str(listing))
        self.assertEqual(code, 1)
        self.assertEqual(len(read_table(self.raw / "manifest.tsv")), 0)
        self.assertEqual(len(read_table(self.raw / "ingest_report.tsv")), 2)

    def test_duplicate_subject_in_two_sites(self) -> None:
        make_image(self.input / "SITE_B" / "FunImg" / "sub_0001" / "rest.nii.gz", (6, 6, 5, 32))
        make_image(self.input / "SITE_B" / "T1Img" / "sub_0001" / "mprage.nii.gz", (8, 8, 8))
        self.assertEqual(self.ingest()[0], 0)
        rows = read_table(self.raw / "ingest_report.tsv")
        rows = rows[rows["subject"] == "sub-0001"]
        self.assertEqual(sorted(rows["status"]), ["error", "ok"])
        self.assertEqual(self.manifest().loc["sub-0001", "group"], "SITE_A")


class DpabiWithoutSiteLevel(TreeCase):
    def test_group_is_dash_and_default_row_applies(self) -> None:
        make_image(self.input / "FunImg" / "S01" / "rest.nii.gz", (6, 6, 5, 32))
        make_image(self.input / "T1Img" / "S01" / "t1.nii", (8, 8, 8))
        code, log = run_ingest("--input-dir", str(self.input), "--layout", "dpabi", "--raw-dir", str(self.raw),
                               "--acq-table", str(self.table), "--drop-volumes", "3")
        self.assertEqual(code, 0, log)
        row = self.manifest().loc["sub-S01"]
        self.assertEqual((row["group"], row["run_label"]), ("-", "sub-S01_task-rest"))
        report = self.report().loc["sub-S01"]
        self.assertEqual((report["stc_decision"], report["drop_volumes"]), ("skip", "2"))
        self.assertEqual(report["warnings"], "short z-FOV (20 mm)")

    def test_without_table_the_default_drop_is_reported_not_written(self) -> None:
        make_image(self.input / "FunImg" / "S01" / "rest.nii.gz", (6, 6, 5, 32))
        make_image(self.input / "T1Img" / "S01" / "t1.nii", (8, 8, 8))
        code, log = run_ingest("--input-dir", str(self.input), "--layout", "dpabi", "--raw-dir", str(self.raw),
                               "--drop-volumes", "3")
        self.assertEqual(code, 0, log)
        self.assertEqual(self.report().loc["sub-S01", "drop_volumes"], "3")
        meta = self.sidecar(self.manifest().loc["sub-S01", "bold"])
        self.assertNotIn("DropVolumes", meta)                    # stage 03 falls back to DROP_VOLUMES
        self.assertEqual(meta["RepetitionTime"], 2.0)


class BidsTree(TreeCase):
    def setUp(self) -> None:
        super().setUp()
        root = self.input
        root.mkdir(parents=True)
        self.inherited = {"RepetitionTime": 2.0, "SliceTiming": [round(float(t), 6) for t in IA5], "Manufacturer": "X"}
        (root / "task-rest_bold.json").write_text(json.dumps(self.inherited), encoding="utf-8")
        (root / "participants.tsv").write_text(
            "participant_id\tgroup\tsite\nsub-01\tASD\tSITE_A\nsub-02\tTDC\tSITE_A\nsub-03\tASD\tSITE_A\n",
            encoding="utf-8")
        # sub-01: sidecar only by inheritance
        self.bold1 = root / "sub-01" / "func" / "sub-01_task-rest_bold.nii.gz"
        make_image(self.bold1, (6, 6, 5, 32))
        make_image(root / "sub-01" / "anat" / "sub-01_T1w.nii.gz", (8, 8, 8))
        # sub-02: complete same-stem sidecar, sessions, T1w only in another session and with sform_code 0
        self.bold2 = root / "sub-02" / "ses-1" / "func" / "sub-02_ses-1_task-rest_run-01_bold.nii.gz"
        make_image(self.bold2, (6, 6, 5, 32))
        self.bold2.with_name("sub-02_ses-1_task-rest_run-01_bold.json").write_text(
            json.dumps({**self.inherited, "EchoTime": 0.03}), encoding="utf-8")
        make_image(root / "sub-02" / "ses-2" / "anat" / "sub-02_ses-2_T1w.nii.gz", (8, 8, 8), sform_code=0)
        # sub-03: uncompressed, sform_code 0, slice axis k-, no SliceTiming of its own
        self.bold3 = root / "sub-03" / "func" / "sub-03_task-rest_bold.nii"
        make_image(self.bold3, (6, 6, 5, 32), sform_code=0)
        self.bold3.with_name("sub-03_task-rest_bold.json").write_text(
            json.dumps({"RepetitionTime": 2.0, "SliceTiming": None, "SliceEncodingDirection": "k-"}), encoding="utf-8")
        make_image(root / "sub-03" / "anat" / "sub-03_T1w.nii.gz", (8, 8, 8))
        # sub-04: no T1w; sub-05: another task (not matched by the glob)
        make_image(root / "sub-04" / "func" / "sub-04_task-rest_bold.nii.gz", (6, 6, 5, 32))
        make_image(root / "sub-05" / "func" / "sub-05_task-nback_bold.nii.gz", (6, 6, 5, 32))
        make_image(root / "sub-05" / "anat" / "sub-05_T1w.nii.gz", (8, 8, 8))

    def ingest(self, *extra: str) -> tuple[int, str]:
        return run_ingest("--input-dir", str(self.input), "--layout", "bids", "--raw-dir", str(self.raw),
                          "--bids-func-glob", "*task-rest*_bold.nii*", "--bids-t1-glob", "*_T1w.nii*",
                          "--drop-volumes", "3", *extra)

    def test_without_table(self) -> None:
        code, log = self.ingest()
        self.assertEqual(code, 0, log)
        manifest, report = self.manifest(), self.report()
        self.assertEqual(sorted(manifest.index), ["sub-01", "sub-02", "sub-03"])
        self.assertEqual(report.loc["sub-04", "status"], "error")
        self.assertIn("no T1w", report.loc["sub-04", "warnings"])
        self.assertNotIn("sub-05", report.index)

        # inherited sidecar: merged JSON next to a link/copy in rawdata, image itself untouched
        row = manifest.loc["sub-01"]
        bold = Path(row["bold"])
        self.assertEqual(bold, self.raw / "sub-01" / "func" / "sub-01_task-rest_bold.nii.gz")
        self.assertEqual(_read_bytes(bold), _read_bytes(self.bold1))
        self.assertEqual(Path(row["t1w"]), self.input / "sub-01" / "anat" / "sub-01_T1w.nii.gz")
        meta = self.sidecar(row["bold"])
        self.assertEqual(meta["SliceTiming"], self.inherited["SliceTiming"])
        self.assertEqual((meta["Manufacturer"], meta["AcquisitionGroup"], meta["RepetitionTime"]), ("X", "SITE_A", 2.0))
        self.assertNotIn("DropVolumes", meta)
        self.assertEqual((report.loc["sub-01", "stc_decision"], report.loc["sub-01", "slice_order"],
                          report.loc["sub-01", "drop_volumes"], report.loc["sub-01", "group"]),
                         ("apply", "sidecar", "3", "SITE_A"))
        self.assertEqual(report.loc["sub-01", "header_changes"], "n/a")

        # complete sidecar and clean header: the manifest points at the original
        row = manifest.loc["sub-02"]
        self.assertEqual((row["session"], row["run"], row["run_label"]), ("1", "01", "sub-02_ses-1_task-rest_run-01"))
        self.assertEqual(Path(row["bold"]), self.bold2)
        self.assertFalse((self.raw / "sub-02" / "ses-1").exists())
        t1 = Path(row["t1w"])                                    # other session, normalised copy
        self.assertEqual(t1, self.raw / "sub-02" / "ses-2" / "anat" / "sub-02_ses-2_T1w.nii.gz")
        self.assertEqual(int(raw_header(t1)["sform_code"]), 1)

        # header needs normalising: copy; no timing anywhere: skip
        row = manifest.loc["sub-03"]
        bold = Path(row["bold"])
        self.assertEqual(bold, self.raw / "sub-03" / "func" / "sub-03_task-rest_bold.nii.gz")
        self.assertEqual(int(raw_header(bold)["sform_code"]), 1)
        self.assertEqual(_read_bytes(bold)[348:], _read_bytes(self.bold3)[348:])
        self.assertEqual(report.loc["sub-03", "stc_decision"], "skip")

    def test_table_fills_gaps_only(self) -> None:
        code, log = self.ingest("--acq-table", str(self.table))
        self.assertEqual(code, 0, log)
        manifest, report = self.manifest(), self.report()

        # sub-03: SliceTiming from the table, stored reversed because the sidecar says k-
        meta = self.sidecar(manifest.loc["sub-03", "bold"])
        self.assertEqual(meta["SliceEncodingDirection"], "k-")
        np.testing.assert_allclose(meta["SliceTiming"], IA5[::-1], atol=1e-6)
        self.assertEqual((meta["PhaseEncodingDirection"], meta["DropVolumes"]), ("j-", 4))
        self.assertEqual((report.loc["sub-03", "stc_decision"], report.loc["sub-03", "slice_order"]), ("apply", "IA"))

        # sub-02: own timing wins, the table only adds what is missing -> sidecar completed in rawdata
        row = manifest.loc["sub-02"]
        bold = Path(row["bold"])
        self.assertEqual(bold.parent, self.raw / "sub-02" / "ses-1" / "func")
        self.assertEqual(_read_bytes(bold), _read_bytes(self.bold2))
        meta = self.sidecar(row["bold"])
        self.assertEqual(meta["SliceTiming"], self.inherited["SliceTiming"])
        self.assertEqual((meta["EchoTime"], meta["DropVolumes"], meta["PhaseEncodingDirection"]), (0.03, 4, "j-"))
        self.assertEqual(report.loc["sub-02", "slice_order"], "sidecar")

    def test_invalid_sidecar_timing_is_skipped(self) -> None:
        self.bold2.with_name("sub-02_ses-1_task-rest_run-01_bold.json").write_text(
            json.dumps({"RepetitionTime": 2.0, "SliceTiming": [0, 400, 800, 1200, 1600]}), encoding="utf-8")
        self.assertEqual(self.ingest()[0], 0)
        row = self.report().loc["sub-02"]
        self.assertEqual((row["status"], row["stc_decision"]), ("ok", "skip"))
        self.assertIn("rejected", row["stc_reason"])
        self.assertIn("STC skipped", row["warnings"])
        meta = self.sidecar(self.manifest().loc["sub-02", "bold"])          # rawdata sidecar: SliceTiming only when applied
        self.assertNotIn("SliceTiming", meta)
        self.assertIn("rejected", meta["SliceTimingSkipReason"])
        original = json.loads(self.bold2.with_name("sub-02_ses-1_task-rest_run-01_bold.json").read_text(encoding="utf-8"))
        self.assertEqual(len(original["SliceTiming"]), 5)                     # the dataset itself is never edited


class OrientationGuard(TreeCase):
    """Table order codes count stored slices inferior->superior; explicit file: times are exempt."""

    def setUp(self) -> None:
        super().setUp()
        self.table.write_text(ACQ_TABLE + "SITE_F\t2\t5\tfile:times.txt\tapply\t4\t-\tA:measured\t-\n",
                              encoding="utf-8")
        (self.root / "times.txt").write_text("0.0\n0.8\n1.6\n0.4\n1.2\n", encoding="utf-8")
        flipped = AFFINE.copy()
        flipped[2, 2] = -4.0                                     # k axis points inferior (LAI)
        make_image(self.input / "SITE_A" / "FunImg" / "sub-01" / "rest.nii.gz", (6, 6, 5, 32), affine=flipped)
        make_image(self.input / "SITE_A" / "T1Img" / "sub-01" / "anat.nii.gz", (8, 8, 8))
        make_image(self.input / "SITE_F" / "FunImg" / "sub-02" / "rest.nii.gz", (6, 6, 5, 32), affine=flipped)
        make_image(self.input / "SITE_F" / "T1Img" / "sub-02" / "anat.nii.gz", (8, 8, 8))

    def test_order_code_skipped_file_times_applied(self) -> None:
        code, log = run_ingest("--input-dir", str(self.input), "--layout", "dpabi", "--raw-dir", str(self.raw),
                               "--acq-table", str(self.table), "--drop-volumes", "3")
        self.assertEqual(code, 0, log)
        report, manifest = self.report(), self.manifest()
        self.assertEqual(report.loc["sub-01", "orientation"], "LAI")
        self.assertEqual(report.loc["sub-01", "stc_decision"], "skip")
        self.assertIn("inferior->superior", report.loc["sub-01", "stc_reason"])
        self.assertIn("STC skipped", report.loc["sub-01", "warnings"])
        self.assertNotIn("SliceTiming", self.sidecar(manifest.loc["sub-01", "bold"]))
        self.assertEqual(report.loc["sub-02", "stc_decision"], "apply")
        np.testing.assert_allclose(self.sidecar(manifest.loc["sub-02", "bold"])["SliceTiming"],
                                   [0.0, 0.8, 1.6, 0.4, 1.2])


class HeaderOddities(TreeCase):
    def setUp(self) -> None:
        super().setUp()
        site = self.input / "SITE_A"
        # undefined spatial unit code (6) together with a millisecond TR
        make_image(site / "FunImg" / "sub-01" / "rest.nii.gz", (6, 6, 5, 32), tr=2000.0, xyzt_units=6 + 16)
        make_image(site / "T1Img" / "sub-01" / "anat.nii.gz", (8, 8, 8))
        # a 4D image offered as T1
        make_image(site / "FunImg" / "sub-02" / "rest.nii.gz", (6, 6, 5, 32))
        make_image(site / "T1Img" / "sub-02" / "anat.nii.gz", (8, 8, 8, 2))

    def test_garbage_units_and_4d_t1(self) -> None:
        code, log = run_ingest("--input-dir", str(self.input), "--layout", "dpabi", "--raw-dir", str(self.raw),
                               "--acq-table", str(self.table), "--drop-volumes", "3")
        self.assertEqual(code, 0, log)
        report, manifest = self.report(), self.manifest()
        row = report.loc["sub-01"]
        self.assertEqual((row["status"], float(row["tr_header"]), float(row["tr_used"]), row["stc_decision"]),
                         ("ok", 2000.0, 2.0, "apply"))
        header = raw_header(Path(manifest.loc["sub-01", "bold"]))
        self.assertAlmostEqual(float(header["pixdim"][4]), 2.0)
        self.assertEqual(header.get_xyzt_units(), ("unknown", "sec"))
        self.assertEqual(report.loc["sub-02", "status"], "error")
        self.assertIn("not a 3D image", report.loc["sub-02", "warnings"])
        self.assertNotIn("sub-02", manifest.index)


AFNI_XML = (b'<?xml version="1.0" ?>\n<AFNI_attributes self_idcode="XYZ">\n'
            b'<AFNI_atr atr_name="TAXIS_FLOATS" ni_type="float" ni_dimen="5">0 1 0 0 0</AFNI_atr>\n'
            b'</AFNI_attributes>\n')


def make_afni_image(path: Path, shape: tuple[int, ...], tr: float = 2.0) -> np.ndarray:
    """int16 image with an AFNI extension (ecode 4) claiming TR 1 s, like ABIDEII-GU_1."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(7).integers(0, 1000, size=shape).astype(np.int16)
    img = nib.Nifti1Image(data, AFFINE)
    img.set_qform(AFFINE, code=1)
    img.set_sform(AFFINE, code=1)
    if len(shape) == 4:
        img.header.set_xyzt_units("mm", "sec")
        img.header["pixdim"][4] = tr
    img.header.extensions.append(nib.nifti1.Nifti1Extension(4, AFNI_XML))
    nib.save(img, str(path))
    return data


class AfniExtensionRegression(TreeCase):
    """AFNI reads TR/view from an AFNI extension instead of the NIfTI header, so a
    stale one (ABIDEII-GU_1: TR 1 s against a 2 s header) made stage 03 skip
    verified slice timing. The rawdata copies must not carry it."""

    def assert_clean_copy(self, path: Path, data: np.ndarray) -> None:
        self.assertEqual(int(raw_header(path)["vox_offset"]), 352)
        img = nib.load(str(path))
        self.assertEqual(len(img.header.extensions), 0)
        np.testing.assert_array_equal(np.asanyarray(img.dataobj), data)

    def test_dpabi_bold_and_t1_lose_the_extension(self) -> None:
        bold = make_afni_image(self.input / "SITE_A" / "FunImg" / "sub-0001" / "rest.nii.gz", (6, 6, 5, 32))
        t1 = make_afni_image(self.input / "SITE_A" / "T1Img" / "sub-0001" / "anat.nii.gz", (8, 8, 8))
        self.assertTrue(ingest.afni_extension_present(self.input / "SITE_A" / "FunImg" / "sub-0001" / "rest.nii.gz"))
        code, out = run_ingest("--input-dir", str(self.input), "--layout", "dpabi", "--raw-dir", str(self.raw),
                               "--acq-table", str(self.table), "--task", "rest")
        self.assertEqual(code, 0, out)
        row = self.manifest().loc["sub-0001"]
        self.assert_clean_copy(Path(row["bold"]), bold)
        self.assert_clean_copy(Path(row["t1w"]), t1)
        self.assertAlmostEqual(float(raw_header(Path(row["bold"]))["pixdim"][4]), 2.0)
        self.assertIn("AFNI NIfTI extension removed", self.report().loc["sub-0001", "header_changes"])
        self.assertEqual(self.report().loc["sub-0001", "stc_decision"], "apply")

    def test_bids_image_with_extension_is_copied_not_linked(self) -> None:
        root = self.input
        bold = make_afni_image(root / "sub-01" / "func" / "sub-01_task-rest_bold.nii.gz", (6, 6, 5, 32))
        (root / "sub-01" / "func" / "sub-01_task-rest_bold.json").write_text(
            json.dumps({"RepetitionTime": 2.0}), encoding="utf-8")
        make_image(root / "sub-01" / "anat" / "sub-01_T1w.nii.gz", (8, 8, 8))
        code, out = run_ingest("--input-dir", str(root), "--layout", "bids", "--raw-dir", str(self.raw),
                               "--task", "rest")
        self.assertEqual(code, 0, out)
        target = Path(self.manifest().loc["sub-01", "bold"])
        self.assertTrue(str(target).startswith(str(self.raw)), target)
        self.assertFalse(target.is_symlink())
        self.assert_clean_copy(target, bold)
        # rerun: the copy is current, nothing is rewritten
        before = target.stat().st_mtime_ns
        code, out = run_ingest("--input-dir", str(root), "--layout", "bids", "--raw-dir", str(self.raw),
                               "--task", "rest")
        self.assertEqual(code, 0, out)
        self.assertEqual(target.stat().st_mtime_ns, before)


class SmallHelpers(unittest.TestCase):
    def test_time_units_rewrite_keeps_valid_spatial_code(self) -> None:
        header = nib.Nifti1Header()
        header["xyzt_units"] = 2 + 16                            # mm, msec
        ingest.set_time_units_seconds(header)
        self.assertEqual(header.get_xyzt_units(), ("mm", "sec"))
        header["xyzt_units"] = 6 + 24                            # undefined spatial code, usec
        ingest.set_time_units_seconds(header)
        self.assertEqual(header.get_xyzt_units(), ("unknown", "sec"))

    def test_subject_ids(self) -> None:
        self.assertEqual(ingest.normalise_subject_id("sub-0029864"), "sub-0029864")
        self.assertEqual(ingest.normalise_subject_id("0050952"), "sub-0050952")
        self.assertEqual(ingest.normalise_subject_id("SUB_ab-12.x"), "sub-ab12x")
        self.assertEqual(ingest.normalise_subject_id("Subject01"), "sub-Subject01")
        with self.assertRaises(ingest.IngestError):
            ingest.normalise_subject_id("sub-__")

    def test_qform_replaced_when_it_disagrees_with_sform(self) -> None:
        header = nib.Nifti1Header()
        header.set_data_shape((6, 6, 5, 32))
        shifted = AFFINE.copy()
        shifted[:3, 3] += 5.0
        header.set_qform(shifted, code=1)
        header.set_sform(AFFINE, code=1)
        changes, warnings = ingest.normalise_orientation(header)
        self.assertEqual(changes, ["qform set from sform"])
        self.assertEqual(len(warnings), 1)
        np.testing.assert_allclose(header.get_qform(), AFFINE, atol=1e-4)

    def test_consistent_header_is_left_alone(self) -> None:
        header = nib.Nifti1Header()
        header.set_data_shape((6, 6, 5, 32))
        header.set_qform(AFFINE, code=1)
        header.set_sform(AFFINE, code=2)
        before = header.binaryblock
        self.assertEqual(ingest.normalise_orientation(header), ([], []))
        self.assertEqual(header.binaryblock, before)



class InputIdentityRegression(unittest.TestCase):
    def test_same_header_changed_voxels_invalidates_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target = root / "source.nii", root / "target.nii"
            make_image(source, (3, 3, 5, 30), seed=1)
            header = ingest.read_header(source)
            ingest.copy_with_header(source, target, header)
            ingest.record_copy(source, target)
            self.assertTrue(ingest.target_is_current(target, header, source))
            old = source.stat()
            make_image(source, (3, 3, 5, 30), seed=2)
            os.utime(source, ns=(old.st_atime_ns, old.st_mtime_ns + 1000000000))
            self.assertEqual(header.binaryblock, ingest.read_header(source).binaryblock)
            self.assertFalse(ingest.target_is_current(target, header, source))
            ingest.copy_with_header(source, target, header)
            ingest.record_copy(source, target)
            np.testing.assert_array_equal(nib.load(target).get_fdata(), nib.load(source).get_fdata())
            self.assertTrue(ingest.target_is_current(target, header, source))
            ingest.copy_record_path(target).unlink()
            self.assertFalse(ingest.target_is_current(target, header, source))

    def test_dpabi_discovers_run_sidecar(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bold = root / "SITE" / "FunImg" / "sub-01" / "rest.nii"
            make_image(bold, (3, 3, 5, 30))
            make_image(root / "SITE" / "T1Img" / "sub-01" / "anat.nii", (3, 3, 5))
            sidecar = bold.with_suffix(".json")
            sidecar.write_text(json.dumps({"RepetitionTime": 2, "SliceTiming": IA5.tolist()}))
            candidates = ingest.discover_dpabi(root, "FunImg", "T1Img", "rest")
            self.assertEqual(candidates[0].sidecars, [sidecar])
            meta = ingest.load_sidecars(candidates[0].sidecars, [])
            header = ingest.read_header(bold)
            settings = ingest.Settings("dpabi", root / "out", 4, 30, False, None, None)
            decision, _ = ingest.decide_timing(meta, None, header, ingest.describe_header(header),
                                             ingest.TRDecision(2, 2), settings)
            self.assertEqual(decision.stc, "apply")

    def test_participant_site_roundtrip_and_conflict(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ingest.write_tables(root, [], [{"subject": "sub-01", "group": "SITE"}])
            self.assertEqual(ingest.bids_groups(root), {"sub-01": "SITE"})
            with self.assertRaisesRegex(ingest.IngestError, "conflicting acquisition"):
                ingest.write_tables(root, [], [{"subject": "sub-01", "group": "A"},
                                               {"subject": "sub-01", "group": "B"}])


if __name__ == "__main__":
    unittest.main()
