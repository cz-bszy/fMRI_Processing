"""Tests of fmriproc.timing: order codes, slice times, acquisition table, stage-03 CLI."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from fmriproc import timing
from fmriproc.timing import (
    TimingError,
    afni_slice_times,
    header_tr,
    load_acq_table,
    load_slice_times_file,
    lookup_acq,
    slice_order_vector,
    slice_times_from_order,
    table_timing,
    validate_slice_timing,
)

# slice_order_vector_matlab of E:\ASD\test_abide_ASD\dpabi_parameters_by_subject.csv
CSV_VECTORS = {
    ("ID", 37): "37 35 33 31 29 27 25 23 21 19 17 15 13 11 9 7 5 3 1 "
                "36 34 32 30 28 26 24 22 20 18 16 14 12 10 8 6 4 2",
    ("IA", 33): "1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32",
    ("IA", 42): "1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 "
                "2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42",
    ("IA2", 34): "2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33",
    ("IA2", 36): "2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 "
                 "1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35",
    ("SA", 38): " ".join(str(v) for v in range(1, 39)),
}

ACQ_HEADER = "group\ttr\tn_slices\tslice_order\tstc\tdrop_volumes\tpe_dir\tevidence\tnote\n"


def write_bold(path: Path, shape=(4, 4, 6, 8), tr: float = 2.0, units: str | None = "sec") -> None:
    img = nib.Nifti1Image(np.zeros(shape, dtype=np.int16), np.diag([3.0, 3.0, 4.0, 1.0]))
    img.header["pixdim"][4] = tr
    if units is not None:
        img.header.set_xyzt_units("mm", units)
    nib.save(img, str(path))


def quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return func(*args, **kwargs)


class OrderCodes(unittest.TestCase):
    def test_every_code_odd_and_even(self) -> None:
        expected = {
            ("SA", 5): [1, 2, 3, 4, 5], ("SA", 6): [1, 2, 3, 4, 5, 6],
            ("SD", 5): [5, 4, 3, 2, 1], ("SD", 6): [6, 5, 4, 3, 2, 1],
            ("IA", 5): [1, 3, 5, 2, 4], ("IA", 6): [1, 3, 5, 2, 4, 6],
            ("IA2", 5): [2, 4, 1, 3, 5], ("IA2", 6): [2, 4, 6, 1, 3, 5],
            ("ID", 5): [5, 3, 1, 4, 2], ("ID", 6): [6, 4, 2, 5, 3, 1],
            ("ID2", 5): [4, 2, 5, 3, 1], ("ID2", 6): [5, 3, 1, 6, 4, 2],
        }
        for (code, n), order in expected.items():
            with self.subTest(code=code, n=n):
                self.assertEqual(slice_order_vector(code, n), order)

    def test_every_code_is_a_permutation(self) -> None:
        for code in timing.ORDER_CODES:
            for n in (1, 2, 3, 33, 34, 40, 43):
                with self.subTest(code=code, n=n):
                    self.assertEqual(sorted(slice_order_vector(code, n)), list(range(1, n + 1)))

    def test_known_answers_from_the_dpabi_csv(self) -> None:
        for (code, n), text in CSV_VECTORS.items():
            with self.subTest(code=code, n=n):
                self.assertEqual(slice_order_vector(code, n), [int(v) for v in text.split()])

    def test_case_and_blanks_are_tolerated(self) -> None:
        self.assertEqual(slice_order_vector(" ia2 ", 4), [2, 4, 1, 3])

    def test_unknown_code_and_bad_count(self) -> None:
        with self.assertRaises(TimingError):
            slice_order_vector("unknown", 10)
        with self.assertRaises(TimingError):
            slice_order_vector("SA", 0)


class SliceTimes(unittest.TestCase):
    def test_rank_times(self) -> None:
        times = slice_times_from_order([1, 3, 5, 2, 4], 2.0)
        np.testing.assert_allclose(times, [0.0, 1.2, 0.4, 1.6, 0.8])

    def test_descending_interleaved_37(self) -> None:
        times = slice_times_from_order(slice_order_vector("ID", 37), 2.0)
        self.assertAlmostEqual(times[36], 0.0)                  # slice 37 is acquired first
        self.assertAlmostEqual(times[0], 18 * 2.0 / 37)         # slice 1 closes the odd pass
        self.assertAlmostEqual(times[35], 19 * 2.0 / 37)        # slice 36 opens the even pass
        self.assertAlmostEqual(times.max(), 2.0 - 2.0 / 37)     # TA = TR - TR/n
        validate_slice_timing(times, 37, 2.0)

    def test_not_a_permutation(self) -> None:
        for order in ([1, 2, 2], [0, 1, 2], [1, 2, 4], [], [1.5, 2, 3]):
            with self.subTest(order=order), self.assertRaises(TimingError):
                slice_times_from_order(order, 2.0)

    def test_bad_tr(self) -> None:
        for tr in (0, -1, float("nan")):
            with self.subTest(tr=tr), self.assertRaises(TimingError):
                slice_times_from_order([1, 2, 3], tr)

    def test_validate_rejects(self) -> None:
        good = [0.0, 0.5, 1.0, 1.5]
        np.testing.assert_allclose(validate_slice_timing(good, 4, 2.0), good)
        cases = {
            "wrong length": (good, 5, 2.0),
            "equal to TR": ([0.0, 0.5, 1.0, 2.0], 4, 2.0),
            "milliseconds": ([0, 500, 1000, 1500], 4, 2.0),
            "negative": ([-0.1, 0.5, 1.0, 1.5], 4, 2.0),
            "nan": ([0.0, float("nan"), 1.0, 1.5], 4, 2.0),
            "nested": ([[0.0, 0.5], [1.0, 1.5]], 4, 2.0),
            "text": (["a", "b", "c", "d"], 4, 2.0),
            "bad TR": (good, 4, 0.0),
        }
        for name, (times, n, tr) in cases.items():
            with self.subTest(case=name), self.assertRaises(TimingError):
                validate_slice_timing(times, n, tr)

    def test_k_minus_reverses(self) -> None:
        times = [0.0, 0.4, 0.8, 1.2, 1.6]
        np.testing.assert_allclose(afni_slice_times(times, 5, 2.0, "k"), times)
        np.testing.assert_allclose(afni_slice_times(times, 5, 2.0, "k-"), times[::-1])
        for direction in ("i", "j-", "z"):
            with self.subTest(direction=direction), self.assertRaises(TimingError):
                afni_slice_times(times, 5, 2.0, direction)

    def test_file_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "times.txt"
            path.write_text("# seconds\n0.0\n1.0\n0.5\n1.5\n", encoding="utf-8")
            np.testing.assert_allclose(load_slice_times_file(f"file:{path}"), [0.0, 1.0, 0.5, 1.5])
            np.testing.assert_allclose(load_slice_times_file("file:times.txt", base_dir=tmp), [0.0, 1.0, 0.5, 1.5])
            with self.assertRaises(TimingError):
                load_slice_times_file("file:missing.txt", base_dir=tmp)
            with self.assertRaises(TimingError):
                load_slice_times_file("file:")
            (Path(tmp) / "bad.txt").write_text("0.0\nabc\n", encoding="utf-8")
            with self.assertRaises(TimingError):
                load_slice_times_file("file:bad.txt", base_dir=tmp)


class HeaderTR(unittest.TestCase):
    def _header(self, value: float, units: str | None) -> nib.Nifti1Header:
        header = nib.Nifti1Header()
        header.set_data_shape((4, 4, 4, 10))
        header["pixdim"][4] = value
        if units is not None:
            header.set_xyzt_units("mm", units)
        return header

    def test_units(self) -> None:
        self.assertAlmostEqual(header_tr(self._header(2.0, "sec")).seconds, 2.0)
        self.assertAlmostEqual(header_tr(self._header(2500.0, "msec")).seconds, 2.5)
        unknown = header_tr(self._header(3.0, None))
        self.assertAlmostEqual(unknown.seconds, 3.0)
        self.assertTrue(unknown.note)

    def test_unit_errors_are_reinterpreted(self) -> None:
        wrong = header_tr(self._header(2000.0, "sec"))
        self.assertAlmostEqual(wrong.seconds, 2.0)
        self.assertAlmostEqual(wrong.literal_s, 2000.0)
        self.assertIn("implausible", wrong.note)
        self.assertAlmostEqual(header_tr(self._header(2.0, "msec")).seconds, 2.0)

    def test_garbage(self) -> None:
        for value in (0.0, -2.0, float("nan"), 1e9):
            with self.subTest(value=value):
                self.assertIsNone(header_tr(self._header(value, "sec")).seconds)

    def test_unknown_units_with_millisecond_value(self) -> None:
        found = header_tr(self._header(2000.0, None))
        self.assertAlmostEqual(found.seconds, 2.0)
        self.assertAlmostEqual(found.literal_s, 2000.0)
        self.assertIn("msec", found.note)


class AcquisitionTable(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "acq.tsv"
        self.path.write_text(
            ACQ_HEADER
            + "SITE_A\t2\t6\tIA\tapply\t4\tj-\tA:protocol\t-\n"
            + "SITE_B\t2\t6\tunknown\tskip\t2\t-\tC:parity\tcustom sequence\n"
            + "SITE_C\t2\t6\tSA\tskip\t4\t-\tB:doc\t-\n"
            + "SITE_F\t2\t4\tfile:times.txt\tapply\t0\t-\tA:measured\t-\n"
            + "*\t-\t-\tunknown\tskip\t4\t-\tC:no_information\tdefault\n",
            encoding="utf-8",
        )
        (self.dir / "times.txt").write_text("0.0\n1.0\n0.5\n1.5\n", encoding="utf-8")
        self.table = load_acq_table(self.path)

    def test_parsing(self) -> None:
        row = self.table["SITE_A"]
        self.assertEqual((row.tr, row.n_slices, row.slice_order, row.stc, row.drop_volumes, row.pe_dir),
                         (2.0, 6, "IA", "apply", 4, "j-"))
        default = self.table["*"]
        self.assertIsNone(default.tr)
        self.assertIsNone(default.n_slices)
        self.assertIsNone(default.pe_dir)
        self.assertEqual(default.drop_volumes, 4)

    def test_lookup_exact_then_default(self) -> None:
        self.assertEqual(lookup_acq(self.table, "SITE_A").matched, "exact")
        fallback = lookup_acq(self.table, "SITE_Z")
        self.assertEqual((fallback.group, fallback.matched), ("*", "default"))
        self.assertEqual(lookup_acq(self.table, "-").matched, "default")
        self.assertIsNone(lookup_acq(None, "SITE_A"))
        no_default = {k: v for k, v in self.table.items() if k != "*"}
        self.assertIsNone(lookup_acq(no_default, "SITE_Z"))
        self.assertIsNone(lookup_acq(no_default, "site_a"))      # exact match only, no case folding

    def test_apply(self) -> None:
        decision = table_timing(lookup_acq(self.table, "SITE_A"), 6, 2.0, self.path)
        self.assertEqual(decision.stc, "apply")
        np.testing.assert_allclose(decision.slice_times, slice_times_from_order([1, 3, 5, 2, 4, 6], 2.0))
        self.assertEqual(decision.evidence, "A:protocol")
        self.assertIn("SITE_A", decision.source)
        self.assertIn("acq.tsv", decision.source)

    def test_n_slices_mismatch_skips(self) -> None:
        decision = table_timing(lookup_acq(self.table, "SITE_A"), 7, 2.0, self.path)
        self.assertEqual(decision.stc, "skip")
        self.assertIsNone(decision.slice_times)
        self.assertIn("n_slices", decision.reason)

    def test_tr_mismatch_skips(self) -> None:
        self.assertEqual(table_timing(lookup_acq(self.table, "SITE_A"), 6, 2.5, self.path).stc, "skip")
        self.assertEqual(table_timing(lookup_acq(self.table, "SITE_A"), 6, 2.0, self.path, tr_conflict=True).stc, "skip")
        self.assertEqual(table_timing(lookup_acq(self.table, "SITE_A"), 6, 2.0005, self.path).stc, "apply")

    def test_unknown_skip_and_default(self) -> None:
        for group in ("SITE_B", "SITE_C", "SITE_Z"):
            with self.subTest(group=group):
                decision = table_timing(lookup_acq(self.table, group), 6, 2.0, self.path)
                self.assertEqual(decision.stc, "skip")
                self.assertTrue(decision.reason)
        self.assertEqual(table_timing(None, 6, 2.0).stc, "skip")

    def test_file_order(self) -> None:
        decision = table_timing(lookup_acq(self.table, "SITE_F"), 4, 2.0, self.path)
        self.assertEqual(decision.stc, "apply")
        np.testing.assert_allclose(decision.slice_times, [0.0, 1.0, 0.5, 1.5])
        (self.dir / "times.txt").write_text("0.0\n1.0\n0.5\n2.5\n", encoding="utf-8")
        self.assertEqual(table_timing(lookup_acq(self.table, "SITE_F"), 4, 2.0, self.path).stc, "skip")

    def test_invalid_tables(self) -> None:
        bad_rows = [
            "SITE_A\t2\t6\tZIGZAG\tapply\t4\t-\tx\t-\n",
            "SITE_A\t2\t6\tIA\tmaybe\t4\t-\tx\t-\n",
            "SITE_A\ttwo\t6\tIA\tapply\t4\t-\tx\t-\n",
            "SITE_A\t2\t6.5\tIA\tapply\t4\t-\tx\t-\n",
            "SITE_A\t2\t6\tIA\tapply\t4\tAP\tx\t-\n",
            "SITE_A\t2\t6\tIA\tapply\t4\t-\tx\t-\nSITE_A\t2\t6\tIA\tapply\t4\t-\tx\t-\n",
        ]
        for text in bad_rows:
            with self.subTest(row=text):
                self.path.write_text(ACQ_HEADER + text, encoding="utf-8")
                with self.assertRaises(TimingError):
                    load_acq_table(self.path)

    def test_repository_table_loads(self) -> None:
        repo_table = Path(__file__).resolve().parents[1] / "config" / "datasets" / "abide_acquisition.tsv"
        if not repo_table.is_file():
            self.skipTest("abide_acquisition.tsv not present")
        table = load_acq_table(repo_table)
        self.assertIn("*", table)
        self.assertEqual(table["ABIDEII-NYU_2"].stc, "skip")
        decision = table_timing(lookup_acq(table, "ABIDEII-EMC_1"), 37, 2.0, repo_table)
        self.assertEqual(decision.stc, "skip")
        self.assertIsNone(decision.slice_times)
        # USM_1 remains skipped; the site convention is not confirmed.
        self.assertEqual(table_timing(lookup_acq(table, "ABIDEII-USM_1"), 41, 2.0, repo_table).stc, "skip")


class AfniTpatternCli(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.bold = self.dir / "sub-01_task-rest_bold.nii.gz"
        self.sidecar = self.dir / "sub-01_task-rest_bold.json"
        self.out_1d = self.dir / "work" / "slice_timing.1D"
        self.out_json = self.dir / "work" / "timing.json"
        write_bold(self.bold)
        self.times = slice_times_from_order(slice_order_vector("IA", 6), 2.0)

    def run_cli(self, stc: str) -> int:
        return quiet(timing.main, ["afni-tpattern", "--bold", str(self.bold), "--json", str(self.sidecar),
                                   "--stc", stc, "--out-1d", str(self.out_1d), "--out-json", str(self.out_json)])

    def write_sidecar(self, **meta) -> None:
        self.sidecar.write_text(json.dumps(meta), encoding="utf-8")

    def result(self) -> dict:
        return json.loads(self.out_json.read_text(encoding="utf-8"))

    def test_apply(self) -> None:
        self.write_sidecar(RepetitionTime=2.0, SliceTiming=self.times.tolist(), SliceEncodingDirection="k",
                           SliceTimingSource="acquisition_table:x.tsv[group=A]", SliceTimingEvidence="A:protocol")
        self.assertEqual(self.run_cli("auto"), 0)
        result = self.result()
        self.assertEqual(set(result), {"stc", "reason", "tzero", "n_slices", "tr", "source", "evidence"})
        self.assertEqual(result["stc"], "apply")
        self.assertEqual((result["n_slices"], result["tr"]), (6, 2.0))
        self.assertAlmostEqual(result["tzero"], (self.times.min() + self.times.max()) / 2)
        self.assertEqual(result["evidence"], "A:protocol")
        rows = self.out_1d.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(rows), 1)                        # AFNI @file: one row of per-slice times
        np.testing.assert_allclose([float(v) for v in rows[0].split()], self.times, atol=1e-9)

    def test_k_minus_is_reversed(self) -> None:
        self.write_sidecar(RepetitionTime=2.0, SliceTiming=self.times.tolist(), SliceEncodingDirection="k-")
        self.assertEqual(self.run_cli("require"), 0)
        np.testing.assert_allclose(np.loadtxt(self.out_1d), self.times[::-1], atol=1e-9)

    def test_missing_timing(self) -> None:
        self.write_sidecar(RepetitionTime=2.0, SliceTimingSkipReason="slice order unknown")
        self.assertEqual(self.run_cli("auto"), 0)
        result = self.result()
        self.assertEqual(result["stc"], "skip")
        self.assertIn("slice order unknown", result["reason"])
        self.assertFalse(self.out_1d.exists())
        self.assertEqual(self.run_cli("require"), 1)

    def test_missing_sidecar(self) -> None:
        self.assertEqual(self.run_cli("auto"), 0)
        self.assertEqual(self.result()["stc"], "skip")
        self.assertAlmostEqual(self.result()["tr"], 2.0)      # falls back to the header TR
        self.assertEqual(self.run_cli("require"), 1)

    def test_off(self) -> None:
        self.write_sidecar(RepetitionTime=2.0, SliceTiming=self.times.tolist())
        self.assertEqual(self.run_cli("auto"), 0)
        self.assertTrue(self.out_1d.exists())
        self.assertEqual(self.run_cli("off"), 0)
        self.assertEqual((self.result()["stc"], self.result()["reason"]), ("skip", "STC=off"))
        self.assertFalse(self.out_1d.exists())                # a stale @file must not survive

    def test_invalid_timing(self) -> None:
        cases = {
            "wrong length": dict(RepetitionTime=2.0, SliceTiming=self.times.tolist()[:-1]),
            "milliseconds": dict(RepetitionTime=2.0, SliceTiming=(self.times * 1000).tolist()),
            "slice axis j": dict(RepetitionTime=2.0, SliceTiming=self.times.tolist(), SliceEncodingDirection="j"),
            "TR conflict": dict(RepetitionTime=2.5, SliceTiming=self.times.tolist()),
        }
        for name, meta in cases.items():
            with self.subTest(case=name):
                self.write_sidecar(**meta)
                self.assertEqual(self.run_cli("auto"), 0)
                self.assertEqual(self.result()["stc"], "skip")
                self.assertTrue(self.result()["reason"])
                self.assertEqual(self.run_cli("require"), 1)

    def test_not_4d(self) -> None:
        write_bold(self.bold, shape=(4, 4, 6))
        self.write_sidecar(RepetitionTime=2.0)
        self.assertEqual(self.run_cli("auto"), 1)

    def test_off_without_sidecar_uses_header_tr(self) -> None:
        self.assertEqual(self.run_cli("off"), 0)
        self.assertEqual((self.result()["stc"], self.result()["reason"], self.result()["tr"]), ("skip", "STC=off", 2.0))


class OrderCli(unittest.TestCase):
    def test_order_subcommand(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(timing.main(["order", "--code", "ID", "--n-slices", "5", "--tr", "2"]), 0)
        self.assertIn("5 3 1 4 2", out.getvalue())
        self.assertEqual(quiet(timing.main, ["order", "--code", "ZIGZAG", "--n-slices", "5", "--tr", "2"]), 1)



class AbideEvidenceRegression(unittest.TestCase):
    def test_unconfirmed_conventions_skip_without_blanket_b_rule(self):
        from fmriproc.timing import load_acq_table, table_timing
        table = load_acq_table(Path(__file__).resolve().parents[1] / "config/datasets/abide_acquisition.tsv")
        for site in ("ABIDEII-EMC_1", "ABIDEII-UCD_1", "ABIDEII-USM_1"):
            row = table[site]
            self.assertEqual(table_timing(row, row.n_slices, 2).stc, "skip")
        self.assertEqual(table_timing(table["UM_1"], 40, 2).stc, "apply")
        self.assertEqual(table_timing(table["ABIDEII-GU_1"], 43, 2).stc, "apply")
        self.assertEqual(table_timing(table["ABIDEII-NYU_2"], 34, 2).stc, "skip")


if __name__ == "__main__":
    unittest.main()
