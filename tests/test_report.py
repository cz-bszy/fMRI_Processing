"""Tests of fmriproc.plots, fmriproc.report and fmriproc.group_report (stages 08/09).

They run on the synthetic derivatives tree of tests/test_qc_metrics.py: metrics
-> figures -> per-subject HTML, and several runs -> group table / QC-FC / HTML.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fmriproc import group_report, plots, qc_metrics, report
from fmriproc.utils import read_tsv, write_json, write_tsv

try:
    from tests import test_qc_metrics as synth
except ImportError:  # started from inside tests/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_qc_metrics as synth  # type: ignore[no-redef]

SUB, RUN, TPL, RES, ATLAS, STRATEGIES = synth.SUB, synth.RUN, synth.TPL, synth.RES, synth.ATLAS, synth.STRATEGIES
RUN_FIGURES = ("boldref-mask", "epi-to-t1", "epi-to-template", "motion", "carpet", "tsnr", "confound-corr",
               f"atlas-{ATLAS}_fc", f"atlas-{ATLAS}_roiqc")


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="fmriproc_report_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.deriv = self.tmp / "derivatives"
        self.atlas_dir = synth.write_atlas_dir(self.tmp / "resources" / "atlases")

    def run_metrics(self, sub: str, run: str, cache: Path | None = None, **kwargs: object) -> None:
        self.assertEqual(qc_metrics.main(synth.qc_args(self.tmp, sub, run, cache, **kwargs)), 0)

    def render_report(self, sub: str, runs: str, manifest: Path | None = None) -> str:
        out = self.deriv / f"{sub}.html"
        args = ["--deriv-dir", str(self.deriv), "--subject", sub, "--runs", runs, "--template", TPL,
                "--strategies", " ".join(STRATEGIES), "--atlases", ATLAS, "--out", str(out)]
        if manifest is not None:
            args += ["--manifest", str(manifest)]
        self.assertEqual(report.main(args), 0)
        return out.read_text(encoding="utf-8")


class FiguresAndReportTest(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        synth.build_subject(self.tmp, surfqc=True)
        self.manifest = synth.write_manifest(self.tmp, [(SUB, RUN, "SITE_A")])
        self.fig_dir = self.deriv / SUB / "figures"
        self.cache = self.tmp / "work" / SUB / "func" / RUN / "qc"
        synth.write_tpl_tissue_masks(self.cache)
        self.run_metrics(SUB, RUN, self.cache, tpl_masks_dir=self.cache)

    def make_figures(self) -> None:
        rc = plots.main(["anat", "--anat-dir", str(self.deriv / SUB / "anat"), "--fig-dir", str(self.fig_dir),
                         "--subject", SUB, "--template", TPL, "--template-brain", "none", "--template-mask", "none"])
        self.assertEqual(rc, 0)
        rc = plots.main(["run", "--func-dir", str(self.deriv / SUB / "func"), "--anat-dir", str(self.deriv / SUB / "anat"),
                         "--fig-dir", str(self.fig_dir), "--run", RUN, "--template", TPL, "--mni-res", RES,
                         "--strategies", " ".join(STRATEGIES), "--atlases", ATLAS, "--cache-dir", str(self.cache),
                         "--template-brain", "none", "--template-mask", "none",
                         "--atlas-labels-dir", str(self.atlas_dir), "--resource-dir", str(self.tmp / "resources")])
        self.assertEqual(rc, 0)

    def test_figures_and_html(self) -> None:
        self.make_figures()
        expected = [f"{SUB}_anat-mosaic.png", f"{SUB}_anat-template.png"] + [f"{RUN}_{n}.png" for n in RUN_FIGURES]
        for name in expected:
            self.assertTrue((self.fig_dir / name).is_file(), name)
            self.assertGreater((self.fig_dir / name).stat().st_size, 2000, name)
        # no surfaces, no fsLR tSNR: those figures are skipped, not written empty
        self.assertFalse((self.fig_dir / f"{SUB}_anat-surfaces.png").exists())
        self.assertFalse((self.fig_dir / f"{RUN}_surface-tsnr.png").exists())

        html = self.render_report(SUB, RUN, self.manifest)
        for needle in (RUN, "data:image/png;base64,", "Time-series validation", "B: site protocol",
                       'class="badge pass"', "总览", "SITE_A", "surface - volume", "fc_similarity", "tSNR cortex",
                       "bbregister", "brain_x_warped_gm", "per-ROI tSNR of both streams"):
            self.assertIn(needle, html, needle)
        self.assertEqual(html.count("data:image/png;base64,"), len(expected))
        self.assertNotIn("figure not available", html)
        self.assertNotIn('src="http', html)                     # self-contained: no external resources
        self.assertNotIn("<script", html.lower())

    def test_report_marks_missing_figures(self) -> None:
        html = self.render_report(SUB, RUN)                    # metrics exist, figures were never drawn
        self.assertIn(RUN, html)
        self.assertIn("figure not available", html)
        self.assertIn(f"{RUN}_carpet.png", html)
        self.assertNotIn("data:image/png;base64,", html)
        self.assertIn('class="badge pass"', html)

    def test_report_survives_missing_metrics(self) -> None:
        other = "sub-02"
        run = f"{other}_task-rest"
        func = self.deriv / other / "func"
        func.mkdir(parents=True)
        write_json(func / f"{run}_desc-prep_info.json", {"run_label": run, "tr": 2.0, "stc_applied": False,
                                                          "stc_reason": "no verified slice timing"})
        html = self.render_report(other, "")                   # runs discovered from the prep_info file
        self.assertIn(run, html)
        self.assertIn("were not computed", html)
        self.assertIn("no verified slice timing", html)
        self.assertIn('class="badge na"', html)
        self.assertNotIn("Time-series validation", html)

    def test_missing_subject_is_an_error(self) -> None:
        self.assertEqual(report.main(["--deriv-dir", str(self.deriv), "--subject", "sub-99", "--template", TPL]), 2)


class ReportHelpersTest(TempDirCase):
    def test_streamcompare_pivot(self) -> None:
        func = self.deriv / SUB / "func"
        synth.build_run(func, RUN, images=False)
        tables = report.streamcompare_tables(func / f"{RUN}_desc-streamcompare.tsv")
        self.assertIsNotNone(tables)
        run_table, roi_table = tables["run"], tables["roi"]
        self.assertEqual(run_table["columns"][0], "metric")
        self.assertIn(f"wmcsf24 | {ATLAS} | volume", run_table["columns"])
        self.assertIn(f"wmcsf24 | {ATLAS} | surface - volume", run_table["columns"])
        metrics = [row[0] for row in run_table["rows"]]
        self.assertEqual(metrics, ["fc_similarity", "roi_tsnr_median"])
        self.assertEqual(len(roi_table["rows"]), len(STRATEGIES) * len(synth.ROI_NAMES))
        self.assertNotIn("scope", roi_table["columns"])
        self.assertIsNone(report.streamcompare_tables(func / "absent.tsv"))
        validation = report.validation_table(func / f"{RUN}_desc-validation.tsv")
        self.assertEqual(validation["columns"][0], "metric")
        self.assertIn(f"surface | wmcsf24 | {ATLAS}", validation["columns"])
        self.assertEqual(len(validation["rows"]), 3)

    def test_fmt(self) -> None:
        self.assertEqual(report.fmt(None), "n/a")
        self.assertEqual(report.fmt(float("nan")), "n/a")
        self.assertEqual(report.fmt(True), "yes")
        self.assertEqual(report.fmt(3.0), "3")
        self.assertEqual(report.fmt(0.123456), "0.1235")
        self.assertEqual(report.fmt([3.0, 3.0, 3.5]), "3 x 3 x 3.5")


class GroupReportTest(TempDirCase):
    ENTRIES = [("sub-01", "sub-01_task-rest_run-1", "SITE_A", 0.08), ("sub-01", "sub-01_task-rest_run-2", "SITE_A", 0.12),
               ("sub-02", "sub-02_task-rest_run-1", "SITE_B", 0.20), ("sub-03", "sub-03_task-rest_run-1", "SITE_B", 0.30)]

    def setUp(self) -> None:
        super().setUp()
        for k, (sub, run, _, fd) in enumerate(self.ENTRIES):
            if not (self.deriv / sub / "anat").is_dir():
                synth.build_anat(self.deriv / sub / "anat", sub)
            synth.build_run(self.deriv / sub / "func", run, seed=10 * k + 1, fd_level=fd, validation=False)
            self.run_metrics(sub, run)
        self.manifest = synth.write_manifest(self.tmp, [(s, r, g) for s, r, g, _ in self.ENTRIES])
        self.out_dir = self.deriv / "group"

    def run_group(self, min_subjects: int) -> str:
        rc = group_report.main(["--deriv-dir", str(self.deriv), "--manifest", str(self.manifest), "--out-dir", str(self.out_dir),
                                "--template", TPL, "--strategies", " ".join(STRATEGIES), "--atlases", ATLAS,
                                "--atlas-dir", str(self.atlas_dir), "--qcfc-min-subjects", str(min_subjects),
                                "--qc-fd-mean-warn", "0.15", "--qc-fd-mean-fail", "0.25"])
        self.assertEqual(rc, 0)
        return (self.out_dir / "group_report.html").read_text(encoding="utf-8")

    def test_group_table_qcfc_and_html(self) -> None:
        html = self.run_group(min_subjects=3)
        table = read_tsv(self.out_dir / "group_qc.tsv")
        self.assertEqual(len(table), 4)
        for column in ("subject", "run", "group", "overall_flag", "n_fail", "n_warn", "fd_mean", "tsnr_gm_median",
                       "flag.fd_mean", "z_fd_mean", "outlier_fd_mean", "z_tsnr_gm_median", "z_scope", "n_outliers",
                       f"{STRATEGIES[0]}.dof_remaining", f"{STRATEGIES[0]}.{ATLAS}.roi_tsnr_median"):
            self.assertIn(column, table.columns, column)
        self.assertEqual(sorted(table["group"].unique()), ["SITE_A", "SITE_B"])
        self.assertTrue((table["z_scope"] == "all").all())         # fewer than 5 runs per site
        self.assertTrue(table["z_fd_mean"].notna().all())
        self.assertEqual(table["overall_flag"].tolist(), sorted(table["overall_flag"], key=lambda f: -qc_metrics.FLAG_RANK[f]))

        for strategy in STRATEGIES:
            edges = read_tsv(self.out_dir / f"qcfc_{strategy}_{ATLAS}.tsv")
            self.assertEqual(list(edges.columns), ["roi_i", "roi_j", "qcfc_r", "p", "n", "q_bh", "distance_mm"])
            self.assertEqual(len(edges), 15)                     # 6 ROIs
            self.assertTrue(edges["distance_mm"].notna().all())  # centroids from the atlas volume under --atlas-dir
            self.assertEqual(edges["qcfc_r"].notna().sum(), 10)  # edges of the uncovered ROI are n/a
            self.assertTrue((edges.loc[edges["qcfc_r"].notna(), "n"] == 3).all())
        summary = read_tsv(self.out_dir / "qcfc_summary.tsv")
        self.assertEqual(len(summary), len(STRATEGIES))
        self.assertTrue((summary["n_runs"] == 4).all())
        self.assertTrue(summary["pct_sig_p05"].notna().all())
        self.assertTrue(summary["median_abs_r"].between(0, 1).all())
        self.assertTrue(summary["dist_dep_spearman"].notna().all())
        self.assertTrue((summary["n_subjects"] == 3).all())
        for needle in ("QC-FC", "SITE_A", "SITE_B", "data:image/png;base64,", "sub-01_task-rest_run-2", "flag thresholds",
                       "warn 0.15 / fail 0.25", 'class="badge'):
            self.assertIn(needle, html, needle)
        self.assertTrue((self.out_dir / "figures" / "group_distributions.png").is_file())
        self.assertTrue((self.out_dir / "figures" / f"qcfc_{ATLAS}.png").is_file())

    def test_qcfc_needs_enough_runs(self) -> None:
        self.run_group(min_subjects=3)
        html = self.run_group(min_subjects=10)
        self.assertFalse(list(self.out_dir.glob("qcfc_*.tsv")))      # stale tables of the first call are removed
        self.assertIn("required: 10", html)
        self.assertIn("subjects with FC: 3", html)
        self.assertTrue((self.out_dir / "group_qc.tsv").is_file())

    def test_group_without_metrics(self) -> None:
        empty = self.tmp / "empty_derivatives"
        empty.mkdir()
        rc = group_report.main(["--deriv-dir", str(empty), "--template", TPL])
        self.assertEqual(rc, 0)
        self.assertTrue((empty / "group" / "group_report.html").is_file())
        self.assertEqual(len(read_tsv(empty / "group" / "group_qc.tsv")), 0)

    def test_group_without_metrics_with_configured_names(self) -> None:
        # stage 09 always passes the configured strategies/atlases, also when every
        # subject failed upstream: QC-FC must then report nothing instead of crashing
        empty = self.tmp / "empty_derivatives2"
        empty.mkdir()
        rc = group_report.main(["--deriv-dir", str(empty), "--template", TPL, "--strategies", "wmcsf24 wmcsf24gsr",
                                "--atlases", "Schaefer2018_100Parcels_7Networks"])
        self.assertEqual(rc, 0)
        self.assertTrue((empty / "group" / "group_report.html").is_file())
        self.assertFalse(list((empty / "group").glob("qcfc_*.tsv")))


class GroupHelpersTest(unittest.TestCase):
    def test_robust_z_and_outliers(self) -> None:
        values = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 5.0])
        z = group_report.robust_z(values)
        self.assertGreater(abs(z[-1]), 3.0)
        self.assertTrue((np.abs(z[:-1]) < 3.0).all())
        self.assertTrue(np.isnan(group_report.robust_z(np.array([1.0, 2.0]))).all())
        table = pd.DataFrame({
            "subject": [f"sub-{k:02d}" for k in range(12)], "run": [f"sub-{k:02d}_task-rest" for k in range(12)],
            "group": ["A"] * 6 + ["B"] * 6, "overall_flag": ["pass"] * 12,
            "fd_mean": [0.1, 0.11, 0.09, 0.1, 0.12, 0.9, 0.3, 0.31, 0.29, 0.3, 0.32, 0.3],
            "tsnr_gm_median": [50.0] * 12, "coreg_dice": [0.9] * 12, "norm_dice": [0.95] * 12, "pct_censored": [5.0] * 12,
        })
        out = group_report.add_outliers(table, min_site_n=5)
        self.assertTrue((out["z_scope"] == "site").all())
        self.assertEqual(out["outlier_fd_mean"].tolist(), [False] * 5 + [True] + [False] * 6)
        self.assertEqual(out.loc[5, "outlier_metrics"], "fd_mean")
        self.assertTrue(np.isnan(out["z_tsnr_gm_median"]).all())    # no spread: no z-score, no outlier

    def test_edgewise_correlation(self) -> None:
        rng = np.random.default_rng(0)
        fd = rng.uniform(0.05, 0.5, 20)
        fc = np.column_stack([2.0 * fd + 0.01 * rng.normal(size=20), rng.normal(size=20), np.full(20, np.nan)])
        r, p, n = group_report.edgewise_correlation(fd, fc, min_n=5)
        self.assertGreater(r[0], 0.99)
        self.assertLess(p[0], 1e-6)
        self.assertTrue(np.isnan(r[2]))
        self.assertEqual(n.tolist(), [20, 20, 0])

    def test_find_atlas_volume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = synth.write_atlas_dir(Path(tmp))
            found = group_report.find_atlas_volume(root, ATLAS, "MNI152NLin6Asym")
            self.assertIsNotNone(found)
            self.assertTrue(found.name.endswith("_dseg.nii.gz"))
            self.assertIsNone(group_report.find_atlas_volume(root, "NoSuchAtlas", "MNI152NLin6Asym"))
            self.assertIsNone(group_report.find_atlas_volume(None, ATLAS, "MNI152NLin6Asym"))


if __name__ == "__main__":
    unittest.main()
