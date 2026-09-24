"""fmriproc.inclusion: run inclusion criteria, phenotype groups, motion by group."""
from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from fmriproc import inclusion
from fmriproc.inclusion import Criteria

STRATEGIES = ["wmcsf24", "36p"]


def run_row(subject: str, **values) -> dict:
    row = {"subject": subject, "session": "-", "run": f"{subject}_task-rest", "group": "SITE",
           "fd_mean": 0.15, "fd_max": 0.8, "fd_pct_gt_02": 12.0, "minutes_retained": 5.5, "pct_censored": 1.0,
           "overall_flag": "pass", "flag.fd_mean": "pass", "flag.coreg_dice": "pass",
           "wmcsf24.dof_remaining": 30.0, "36p.dof_remaining": 20.0, "flag.36p.dof_remaining": "pass"}
    row.update(values)
    return row


def decide(rows: list[dict], criteria: Criteria = Criteria()) -> pd.DataFrame:
    return inclusion.decide(pd.DataFrame(rows), criteria, STRATEGIES).set_index("subject")


class CriteriaTest(unittest.TestCase):
    def test_clean_run_is_included_everywhere(self) -> None:
        out = decide([run_row("sub-01")])
        self.assertEqual(out.at["sub-01", "included"], "yes")
        self.assertEqual(out.at["sub-01", "reasons"], "-")
        self.assertEqual((out.at["sub-01", "included_wmcsf24"], out.at["sub-01", "included_36p"]), ("yes", "yes"))
        self.assertEqual(out.at["sub-01", "dof_36p"], 20.0)

    def test_each_run_criterion(self) -> None:
        cases = {
            "sub-01": ({"fd_mean": 0.6}, "mean FD 0.6 mm > 0.5 mm"),
            "sub-02": ({"fd_max": 6.2}, "max FD 6.2 mm > 5 mm"),
            "sub-03": ({"minutes_retained": 3.2}, "retained 3.2 min < 4 min"),
            "sub-04": ({"flag.coreg_dice": "fail", "overall_flag": "fail"}, "QC coreg_dice fail"),
            "sub-05": ({"overall_flag": "incomplete"}, "QC incomplete"),
            "sub-06": ({"fd_mean": None}, "mean FD n/a"),
        }
        out = decide([run_row(s, **values) for s, (values, _) in cases.items()])
        for subject, (_, reason) in cases.items():
            self.assertEqual(out.at[subject, "included"], "no", subject)
            self.assertIn(reason, out.at[subject, "reasons"], subject)
            self.assertEqual(out.at[subject, "included_wmcsf24"], "no", subject)

    def test_boundaries_are_strict(self) -> None:
        out = decide([run_row("sub-01", fd_mean=0.5, fd_max=5.0, minutes_retained=4.0)])
        self.assertEqual(out.at["sub-01", "included"], "yes")

    def test_zero_switches_criteria_off(self) -> None:
        off = Criteria(fd_mean=0, fd_max=0, pct_fd_gt02=0, min_retained_min=0, min_dof=0, qc_fail=False)
        out = decide([run_row("sub-01", fd_mean=2.0, fd_max=9.0, minutes_retained=1.0, overall_flag="fail",
                              **{"flag.coreg_dice": "fail", "36p.dof_remaining": 3.0})], off)
        self.assertEqual((out.at["sub-01", "included"], out.at["sub-01", "included_36p"]), ("yes", "yes"))
        self.assertEqual(off.describe(), [])

    def test_stringent_percentage_criterion(self) -> None:
        out = decide([run_row("sub-01", fd_pct_gt_02=24.7), run_row("sub-02", fd_pct_gt_02=17.6)],
                     Criteria(pct_fd_gt02=20))
        self.assertEqual(out["included"].to_dict(), {"sub-01": "no", "sub-02": "yes"})
        self.assertIn("frames with FD > 0.2 mm 24.7% > 20%", out.at["sub-01", "reasons"])

    def test_mean_fd_flag_is_not_counted_twice(self) -> None:
        # QC_FD_MEAN_FAIL may be lower than EXCLUDE_FD_MEAN: the dedicated criterion decides
        out = decide([run_row("sub-01", fd_mean=0.45, overall_flag="fail", **{"flag.fd_mean": "fail"})])
        self.assertEqual(out.at["sub-01", "included"], "yes")

    def test_dof_is_judged_per_strategy(self) -> None:
        out = decide([run_row("sub-01", **{"36p.dof_remaining": 14.0, "flag.36p.dof_remaining": "warn"})])
        self.assertEqual(out.at["sub-01", "included"], "yes")
        self.assertEqual((out.at["sub-01", "included_wmcsf24"], out.at["sub-01", "included_36p"]), ("yes", "no"))
        self.assertEqual(out.at["sub-01", "reasons"], "36p: DOF 14 < 15")
        # a strategy that was not run (no DOF) is not judged on DOF
        row = run_row("sub-02")
        del row["36p.dof_remaining"]
        self.assertEqual(decide([row]).at["sub-02", "included_36p"], "yes")
        # a run excluded as a whole stays excluded for every strategy
        out = decide([run_row("sub-03", fd_mean=0.7)])
        self.assertEqual(out.at["sub-03", "included_36p"], "no")

    def test_included_runs(self) -> None:
        table = inclusion.decide(pd.DataFrame([run_row("sub-01"), run_row("sub-02", **{"36p.dof_remaining": 9.0})]),
                                 Criteria(), STRATEGIES)
        self.assertEqual(inclusion.included_runs(table), {"sub-01_task-rest", "sub-02_task-rest"})
        self.assertEqual(inclusion.included_runs(table, "36p"), {"sub-01_task-rest"})
        self.assertEqual(inclusion.included_runs(table, "unknown"), {"sub-01_task-rest", "sub-02_task-rest"})
        self.assertEqual(inclusion.included_runs(inclusion.decide(pd.DataFrame(), Criteria(), STRATEGIES)), set())

    def test_command_line_options(self) -> None:
        parser = argparse.ArgumentParser()
        inclusion.add_arguments(parser)
        args = parser.parse_args(["--exclude-fd-mean", "0.25", "--exclude-pct-fd-gt02", "20", "--exclude-qc-fail", "no"])
        criteria = inclusion.criteria_from_args(args)
        self.assertEqual((criteria.fd_mean, criteria.pct_fd_gt02, criteria.qc_fail, criteria.min_dof), (0.25, 20.0, False, 15.0))
        self.assertIn("mean FD > 0.25 mm", criteria.describe())
        for bad in (["--exclude-fd-mean", "-1"], ["--exclude-min-dof", "x"], ["--exclude-qc-fail", "maybe"]):
            with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                parser.parse_args(bad)


class PhenotypeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_ids_and_labels(self) -> None:
        self.assertEqual({inclusion.normalize_id(v) for v in ("sub-0028744", "0028744", 28744, " 28744 ")}, {"28744"})
        self.assertEqual(inclusion.normalize_id("sub-ctrl01"), "ctrl01")
        self.assertEqual(inclusion.parse_labels("1=ASD 2=TD"), {"1": "ASD", "2": "TD"})
        self.assertEqual(inclusion.parse_labels("1=ASD,2=TD"), {"1": "ASD", "2": "TD"})
        self.assertEqual(inclusion.parse_labels(""), {})
        with self.assertRaises(ValueError):
            inclusion.parse_labels("ASD")

    def test_csv_with_bom_and_tsv(self) -> None:
        csv = self.tmp / "pheno.csv"
        csv.write_bytes((chr(0xFEFF) + "SUB_ID,DX_GROUP,AGE\n28744,1,10\n29150,2,12\n").encode("utf-8"))
        self.assertEqual(inclusion.read_phenotype(csv, "SUB_ID", "DX_GROUP", {"1": "ASD", "2": "TD"}),
                         {"28744": "ASD", "29150": "TD"})
        tsv = self.tmp / "participants.tsv"
        tsv.write_text("participant_id\tgroup\nsub-01\tpatient\nsub-02\t\n", encoding="utf-8")
        self.assertEqual(inclusion.read_phenotype(tsv, "participant_id", "group"), {"1": "patient", "2": "n/a"})

    def test_errors(self) -> None:
        path = self.tmp / "pheno.tsv"
        path.write_text("participant_id\tdiagnosis\nsub-01\tASD\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "column\\(s\\) group not found"):
            inclusion.read_phenotype(path, "participant_id", "group")
        path.write_text("participant_id\tgroup\nsub-01\tASD\n01\tTD\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "two groups"):
            inclusion.read_phenotype(path, "participant_id", "group")

    def test_attach(self) -> None:
        table = pd.DataFrame({"subject": ["sub-0028744", "sub-0000001"]})
        out = inclusion.attach_phenotype(table, {"28744": "ASD"})
        self.assertEqual(out["pheno_group"].tolist(), ["ASD", "n/a"])


class MotionByGroupTest(unittest.TestCase):
    def test_two_groups_with_tests(self) -> None:
        rows = []
        rng = np.random.default_rng(0)
        for k in range(6):
            rows.append(run_row(f"sub-a{k}", pheno_group="ASD", fd_mean=float(0.25 + 0.05 * rng.random())))
            rows.append(run_row(f"sub-t{k}", pheno_group="TD", fd_mean=float(0.10 + 0.05 * rng.random())))
        rows[0]["fd_mean"] = 0.9                   # one ASD subject excluded
        rows.append(run_row("sub-x", pheno_group="n/a"))
        decision = inclusion.decide(pd.DataFrame(rows), Criteria(), STRATEGIES)
        table, tests = inclusion.motion_by_group(decision)
        table = table.set_index("pheno_group")
        self.assertEqual(table.at["ASD", "n_subjects"], 6)
        self.assertEqual(table.at["ASD", "n_subjects_excluded"], 1)
        self.assertAlmostEqual(table.at["ASD", "pct_subjects_excluded"], 100 / 6)
        self.assertEqual(table.at["TD", "n_subjects_excluded"], 0)
        self.assertGreater(table.at["ASD", "fd_mean_median_all"], table.at["ASD", "fd_mean_median_included"] - 1e-9)
        self.assertIn("n/a", table.index)                      # unmatched subjects are listed, not tested
        self.assertEqual(tests["groups"], "ASD vs TD")
        self.assertLess(tests["mannwhitney_p_fd_included"], 0.05)
        self.assertGreater(tests["fisher_p_subjects_excluded"], 0.05)

    def test_no_tests_for_small_or_many_groups(self) -> None:
        rows = [run_row("sub-1", pheno_group="ASD"), run_row("sub-2", pheno_group="TD"), run_row("sub-3", pheno_group="TD")]
        table, tests = inclusion.motion_by_group(inclusion.decide(pd.DataFrame(rows), Criteria(), STRATEGIES))
        self.assertEqual(len(table), 2)
        self.assertEqual(tests, {})
        table, tests = inclusion.motion_by_group(inclusion.decide(pd.DataFrame([run_row("sub-1")]), Criteria(), STRATEGIES))
        self.assertTrue(table.empty)                           # no phenotype at all


if __name__ == "__main__":
    unittest.main()
