"""Tests of fmriproc.strategies (stage 05 regressor selection and DOF accounting)."""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fmriproc import confounds, strategies
from fmriproc.utils import read_tsv, write_tsv

REPO = Path(__file__).resolve().parents[1]
N_T = 150
TR = 2.0
MOTION6 = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]


def make_confounds(n_t: int = N_T, seed: int = 0, n_csf_voxels: int = 30) -> pd.DataFrame:
    """A real stage-04 table from random data, so that the two modules are tested
    against each other's column names."""
    rng = np.random.default_rng(seed)
    n_brain, n_wm = 120, 40
    series = (10000.0 + rng.normal(0.0, 30.0, size=(n_t, n_brain + n_wm + n_csf_voxels))).astype(np.float32)
    index = np.arange(series.shape[1])
    in_brain = index < n_brain
    in_wm = (index >= n_brain) & (index < n_brain + n_wm)
    in_csf = index >= n_brain + n_wm
    motion = np.cumsum(rng.normal(0.0, 0.01, size=(n_t, 6)), axis=0)
    motion[:, 3:] *= 0.02
    frame, _ = confounds.build_confounds(
        series, in_brain, in_wm, in_csf, motion,
        relrms=np.abs(rng.normal(0.05, 0.01, size=n_t - 1)), outliers=np.abs(rng.normal(0.01, 0.002, size=n_t)),
        tr=TR, acompcor_n=5, highpass_sec=128.0, censor_fd=0.5, censor_prev=False, censor_dvars=0.0,
    )
    return frame


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stderr(io.StringIO()):
        return function(*args, **kwargs)


class TestColumns(unittest.TestCase):
    def test_motion_sets(self) -> None:
        self.assertEqual(strategies.motion_columns(6), MOTION6)
        self.assertEqual(strategies.motion_columns(12), MOTION6 + [f"{m}_derivative1" for m in MOTION6])
        p24 = strategies.motion_columns(24)
        self.assertEqual(len(p24), 24)
        self.assertEqual(len(set(p24)), 24)
        for m in MOTION6:
            for suffix in ("", "_derivative1", "_power2", "_derivative1_power2"):
                self.assertIn(m + suffix, p24)

    def test_every_strategy(self) -> None:
        p24 = strategies.motion_columns(24)
        p12 = strategies.motion_columns(12)
        compcor = [f"w_comp_cor_{i:02d}" for i in range(5)] + [f"c_comp_cor_{i:02d}" for i in range(5)]
        expansions = ["", "_derivative1", "_power2", "_derivative1_power2"]
        expected = {
            "wmcsf24": p24 + ["white_matter", "csf"],
            "wmcsf24gsr": p24 + ["white_matter", "csf", "global_signal"],
            "36p": p24 + [s + e for s in ("white_matter", "csf", "global_signal") for e in expansions],
            "acompcor": p12 + compcor,
            "acompcorgsr": p12 + ["global_signal"] + compcor,
            "legacy8": MOTION6 + ["white_matter", "csf"],
            "legacy9gsr": MOTION6 + ["white_matter", "csf", "global_signal"],
        }
        self.assertEqual(set(strategies.STRATEGIES), set(expected))
        sizes = {"wmcsf24": 26, "wmcsf24gsr": 27, "36p": 36, "acompcor": 22, "acompcorgsr": 23, "legacy8": 8, "legacy9gsr": 9}
        for name, columns in expected.items():
            got = strategies.strategy_columns(name)
            self.assertEqual(sorted(got), sorted(columns), name)
            self.assertEqual(len(got), sizes[name], name)
            self.assertEqual(len(set(got)), len(got), name)
            # principle 10: the name says whether the global signal is regressed
            self.assertEqual(any(c.startswith("global_signal") for c in got), "gsr" in name or name == "36p", name)

    def test_all_strategies_find_their_columns_in_a_stage04_table(self) -> None:
        frame = make_confounds()
        for name in strategies.STRATEGIES:
            notes: list[str] = []
            columns = quiet(strategies.select_columns, name, frame, TR, "bandpass", 0.01, 5, notes)
            self.assertEqual(columns, strategies.strategy_columns(name) + ([c for c in frame if c.startswith("cosine_")] if strategies.STRATEGIES[name]["compcor"] else []), name)
            self.assertEqual(notes, [], name)

    def test_acompcor_n_selects_that_many_components(self) -> None:
        # stage 05 passes ACOMPCOR_N; the table of a run with ACOMPCOR_N=3 holds 3 per tissue
        frame = make_confounds()
        three = strategies.strategy_columns("acompcor", 3)
        self.assertEqual([c for c in three if "_comp_cor_" in c],
                         [f"w_comp_cor_{i:02d}" for i in range(3)] + [f"c_comp_cor_{i:02d}" for i in range(3)])
        notes: list[str] = []
        columns = quiet(strategies.select_columns, "acompcorgsr", frame, TR, "bandpass", 0.01, 3, notes)
        self.assertEqual(columns, strategies.strategy_columns("acompcorgsr", 3) + [c for c in frame if c.startswith("cosine_")])
        self.assertEqual(notes, [])
        self.assertNotIn("w_comp_cor_03", columns)

    def test_missing_compcor_component_is_tolerated(self) -> None:
        frame = make_confounds(n_csf_voxels=2)
        self.assertNotIn("c_comp_cor_04", frame.columns)
        notes: list[str] = []
        columns = quiet(strategies.select_columns, "acompcor", frame, TR, "bandpass", 0.01, 5, notes)
        self.assertIn("w_comp_cor_04", columns)
        self.assertNotIn("c_comp_cor_04", columns)
        self.assertTrue(any("c_comp_cor_04" in note for note in notes))

    def test_missing_other_column_is_an_error(self) -> None:
        frame = make_confounds().drop(columns=["rot_y_power2"])
        with self.assertRaises(ValueError) as caught:
            quiet(strategies.select_columns, "wmcsf24", frame, TR, "bandpass", 0.01)
        self.assertIn("rot_y_power2", str(caught.exception))
        quiet(strategies.select_columns, "legacy8", frame, TR, "bandpass", 0.01)

    def test_acompcor_keeps_full_prefilter_basis(self) -> None:
        frame = make_confounds()
        cosines = [c for c in frame if c.startswith("cosine_")]
        for mode in ("none", "highpass", "bandpass"):
            self.assertEqual(strategies.cosine_columns_needed(list(frame), N_T, TR, mode, .01), cosines)
            self.assertEqual(quiet(strategies.select_columns, "acompcor", frame, TR, mode, .01),
                             strategies.strategy_columns("acompcor") + cosines)
        self.assertEqual(quiet(strategies.select_columns, "36p", frame, TR, "none", .01), strategies.strategy_columns("36p"))

    def test_dct_is_not_contained_in_fourier_stopband(self) -> None:
        # Independently constructed low-frequency Fourier basis, including
        # polort2: odd DCT half-periods are not spanned by integer Fourier modes.
        t = np.arange(150)
        dct = np.cos(np.pi * (t + .5) / 150)
        basis = np.column_stack([np.ones(150), t, t*t] +
            [f(2*np.pi*j*t/150) for j in (1, 2, 3) for f in (np.sin, np.cos)])
        residual = dct - basis @ np.linalg.lstsq(basis, dct, rcond=None)[0]
        self.assertGreater(np.linalg.norm(residual) / np.linalg.norm(dct), 1e-4)


class TestConditioning(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(5)
        self.frame = pd.DataFrame({
            "a": 10000.0 + rng.normal(0.0, 30.0, size=50),
            "b": rng.normal(0.0, 1e-4, size=50),
            "c": np.linspace(-3.0, 8.0, 50) ** 2,
        })

    def test_centred_unit_sd_float64(self) -> None:
        matrix, kept, dropped = strategies.prepare_regressors(self.frame, ["a", "b", "c"])
        self.assertEqual(matrix.dtype, np.float64)
        self.assertEqual((kept, dropped), (["a", "b", "c"], {}))
        np.testing.assert_allclose(matrix.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(matrix.std(axis=0), 1.0, atol=1e-12)
        for j, name in enumerate(kept):
            self.assertAlmostEqual(np.corrcoef(matrix[:, j], self.frame[name])[0, 1], 1.0, places=9)

    def test_statistics_over_retained_frames(self) -> None:
        keep = np.ones(50, dtype=bool)
        keep[[3, 4, 20]] = False
        frame = self.frame.copy()
        frame.loc[[3, 4, 20], "a"] += 5000.0          # motion spikes live in censored frames
        matrix, _, _ = strategies.prepare_regressors(frame, ["a", "b", "c"], keep)
        self.assertEqual(matrix.shape, (50, 3))
        np.testing.assert_allclose(matrix[keep].mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(matrix[keep].std(axis=0), 1.0, atol=1e-12)
        self.assertGreater(matrix[3, 0], 50.0)

    def test_constant_and_duplicate_columns_are_dropped(self) -> None:
        frame = self.frame.copy()
        frame["flat"] = 3.0
        frame["a_again"] = frame["a"] * 2.0 + 7.0
        frame["a_negative"] = -frame["a"]
        matrix, kept, dropped = strategies.prepare_regressors(frame, ["a", "flat", "b", "a_again", "c", "a_negative"])
        self.assertEqual(kept, ["a", "b", "c"])
        self.assertEqual(matrix.shape, (50, 3))
        self.assertIn("constant", dropped["flat"])
        self.assertEqual(dropped["a_again"], "duplicate of a")
        self.assertEqual(dropped["a_negative"], "duplicate of a")

    def test_constant_over_retained_frames_only(self) -> None:
        frame = self.frame.copy()
        frame["spike"] = 0.0
        frame.loc[7, "spike"] = 1.0
        keep = np.ones(50, dtype=bool)
        keep[7] = False
        _, kept, dropped = strategies.prepare_regressors(frame, ["a", "spike"], keep)
        self.assertEqual(kept, ["a"])
        self.assertIn("spike", dropped)

    def test_bad_input(self) -> None:
        frame = self.frame.copy()
        frame.loc[2, "b"] = np.nan
        with self.assertRaises(ValueError):
            strategies.prepare_regressors(frame, ["a", "b"])
        with self.assertRaises(ValueError):
            strategies.prepare_regressors(self.frame, ["a"], np.ones(49, dtype=bool))
        flat = pd.DataFrame({"x": np.ones(10), "y": np.zeros(10)})
        with self.assertRaises(ValueError):
            strategies.prepare_regressors(flat, ["x", "y"])


class TestDegreesOfFreedom(unittest.TestCase):
    def test_bandpass_cost_hand_computed(self) -> None:
        # N = 150, TR = 2: df = 1/300 Hz, 75 positive frequencies.
        # low stop band 0..0.0099 Hz  -> j = 1..3   (0 Hz belongs to polort)
        # high stop band 0.1001..inf  -> j = 30..75 (46 frequencies, Nyquist has no sine)
        self.assertEqual(strategies.bandpass_cost(150, 2.0, "bandpass", 0.01, 0.1), 2 * (3 + 46) - 1)
        # N = 151: df = 1/302, no exact Nyquist term -> j = 1..3 and 30..75, two each
        self.assertEqual(strategies.bandpass_cost(151, 2.0, "bandpass", 0.01, 0.1), 2 * (3 + 46))

    def test_highpass_and_none(self) -> None:
        # "-passband 0.01 99999": j = 1..3 plus the clipped upper band = Nyquist (j = 75)
        self.assertEqual(strategies.bandpass_cost(150, 2.0, "highpass", 0.01, 0.1), 2 * 3 + 1)
        self.assertEqual(strategies.bandpass_cost(151, 2.0, "highpass", 0.01, 0.1), 2 * 3 + 2)
        self.assertEqual(strategies.bandpass_cost(150, 2.0, "none", 0.01, 0.1), 0)
        self.assertEqual(strategies.bandpass_cost(1, 2.0, "bandpass", 0.01, 0.1), 0)

    def test_close_to_the_textbook_approximation(self) -> None:
        for n_t, tr in ((120, 2.0), (176, 2.0), (240, 2.5), (137, 3.0), (300, 2.0)):
            exact = strategies.bandpass_cost(n_t, tr, "bandpass", 0.01, 0.1)
            approx = n_t * (1.0 - 2.0 * tr * (0.1 - 0.01))
            self.assertLessEqual(abs(exact - approx), 4.0, (n_t, tr))
            exact = strategies.bandpass_cost(n_t, tr, "highpass", 0.01, 0.1)
            self.assertLessEqual(abs(exact - 2.0 * tr * 0.01 * n_t), 3.0, (n_t, tr))

    def test_bandpass_cost_second_hand_computed_case(self) -> None:
        # N = 100, TR = 2: df = 0.005 Hz, 50 positive frequencies, 3dTproject rounds
        # rintf((f +- df/6) / df): low band 0..0.0099 -> j = 0..2 (0 goes to polort),
        # high band 0.1001..inf -> j = 20..50, j = 50 is Nyquist (cosine only)
        self.assertEqual(strategies.bandpass_cost(100, 2.0, "bandpass", 0.01, 0.1), 2 * 2 + 2 * 30 + 1)
        self.assertEqual(strategies.bandpass_cost(100, 2.0, "highpass", 0.01, 0.1), 2 * 2 + 1)
        # a lower edge exactly on a Fourier frequency (0.01 = 2 df) is removed as well
        self.assertEqual(strategies.bandpass_cost(100, 2.0, "highpass", 0.01, 0.1),
                         strategies.bandpass_cost(100, 2.0, "highpass", 0.0125, 0.1))

    def test_zero_lower_edge_costs_nothing_below(self) -> None:
        # "-passband 0 0.1": the low stop band collapses onto 0 Hz, which polort covers
        self.assertEqual(strategies.bandpass_cost(100, 2.0, "bandpass", 0.0, 0.1), 2 * 30 + 1)
        self.assertEqual(strategies.dof_accounting(100, 0, 5, -1, 2.0, "bandpass", 0.0, 0.1, 15)["dof_remaining"],
                         100 - 5 - 1 - 61)

    def test_band_above_nyquist_costs_only_the_low_side(self) -> None:
        # TR = 3 s: Nyquist 0.1667 Hz; an upper edge of 0.2 Hz removes nothing but Nyquist
        self.assertEqual(
            strategies.bandpass_cost(100, 3.0, "bandpass", 0.01, 0.2),
            strategies.bandpass_cost(100, 3.0, "highpass", 0.01, 0.2),
        )

    def test_dof_arithmetic(self) -> None:
        dof = strategies.dof_accounting(150, 10, 26, 2, 2.0, "bandpass", 0.01, 0.1, 15)
        self.assertEqual(dof["n_retained"], 140)
        self.assertEqual(dof["dof_bandpass_cost"], 97)
        self.assertEqual(dof["dof_remaining"], 140 - 26 - 3 - 97)
        self.assertIs(dof["low_dof"], True)
        self.assertIs(strategies.dof_accounting(150, 10, 26, 2, 2.0, "bandpass", 0.01, 0.1, 14)["low_dof"], False)
        dof = strategies.dof_accounting(150, 0, 9, 2, 2.0, "none", 0.01, 0.1, 15)
        self.assertEqual((dof["dof_bandpass_cost"], dof["dof_remaining"], dof["low_dof"]), (0, 150 - 9 - 3, False))
        dof = strategies.dof_accounting(150, 0, 8, 1, 2.0, "highpass", 0.01, 0.1, 15)
        self.assertEqual(dof["dof_remaining"], 150 - 8 - 2 - 7)

    def test_polort_minus_one(self) -> None:
        # with a stop band starting at 0 Hz 3dTproject still removes the mean
        self.assertEqual(strategies.dof_accounting(100, 0, 5, -1, 2.0, "none", 0.01, 0.1, 15)["dof_remaining"], 95)
        cost = strategies.bandpass_cost(100, 2.0, "highpass", 0.01, 0.1)
        self.assertEqual(strategies.dof_accounting(100, 0, 5, -1, 2.0, "highpass", 0.01, 0.1, 15)["dof_remaining"], 95 - 1 - cost)


class TestJointDesign(unittest.TestCase):
    def test_ntrp_preserves_columns_different_at_censored_frames(self):
        a = np.arange(50, dtype=float)
        b = a.copy(); b[5] += 10
        frame = pd.DataFrame({"a": a, "b": b})
        keep = np.ones(50, bool); keep[5] = False
        full, columns, _ = strategies.prepare_regressors(frame, ["a", "b"])
        self.assertEqual(columns, ["a", "b"])
        removed, columns, _ = strategies.prepare_regressors(frame, ["a", "b"], keep)
        self.assertEqual(columns, ["a"])
        info = strategies.design_accounting(full, keep, "NTRP", 0, 2, "none", .01, .1, 15)
        self.assertEqual((info["fit_rows"], info["design_rank"], info["retained_observations"]), (50, 3, 49))
        info = strategies.design_accounting(removed, keep, "ZERO", 0, 2, "none", .01, .1, 15)
        self.assertEqual((info["fit_rows"], info["design_rank"]), (49, 2))

    def test_overlap_with_legendre_is_counted_once(self):
        t = np.linspace(-1, 1, 50)
        matrix = np.column_stack([t, t*t])
        keep = np.ones(50, bool); keep[::9] = False
        for mode in ("NTRP", "ZERO", "KILL"):
            info = strategies.design_accounting(matrix, keep, mode, 2, 2, "none", .01, .1, 15)
            self.assertEqual(info["design_columns"], 5)
            self.assertEqual(info["design_rank"], 3)
            rows = 50 if mode == "NTRP" else int(keep.sum())
            self.assertEqual(info["algebraic_dof"], rows - 3)

    def test_no_intercept_retains_nonzero_offset_after_full_run_demeaning(self):
        frame = pd.DataFrame({"spike": np.r_[1., np.zeros(19)]})
        keep = np.ones(20, bool); keep[0] = False
        matrix, columns, _ = strategies.prepare_regressors(frame, ["spike"], keep, has_intercept=False)
        self.assertEqual(columns, ["spike"])
        design = strategies.joint_design(matrix, -1, 2, "none", .01, .1)
        self.assertTrue(np.all(np.abs(design[keep]) > 0))
        self.assertEqual(np.linalg.matrix_rank(design[keep]), 1)

    def test_afni_gate_uses_nominal_count_even_for_ntrp(self):
        t = np.linspace(-1, 1, 20)
        matrix = np.column_stack([t]*8)
        keep = np.zeros(20, bool); keep[:9] = True
        info = strategies.design_accounting(matrix, keep, "NTRP", 2, 2, "none", .01, .1, 15)
        self.assertEqual(info["design_rank"], 3)
        self.assertGreater(info["algebraic_dof"], 0)
        self.assertFalse(info["afni_model_feasible"])

    def test_fourier_basis_matches_hand_computed_indices(self):
        n = 100; t = np.arange(n)
        design = strategies.joint_design(np.empty((n, 0)), 0, 2, "bandpass", .01, .1)
        # Independently specified j=1,2,20..50; Nyquist has only cosine.
        expected = [np.ones(n)]
        for j in [1, 2, *range(20, 51)]:
            expected.append(np.cos(2*np.pi*j*t/n))
            if j != 50: expected.append(np.sin(2*np.pi*j*t/n))
        np.testing.assert_allclose(design, np.column_stack(expected), atol=3e-5, rtol=0)

    def test_cli_refuses_nominally_overfull_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            frame = make_confounds()
            write_tsv(tmp / "c.tsv", frame)
            np.savetxt(tmp / "keep.1D", np.r_[np.ones(120), np.zeros(30)], fmt="%d")
            code = quiet(strategies.main, ["--confounds", str(tmp/"c.tsv"), "--strategy", "wmcsf24",
                "--tr", "2", "--n-volumes-censored-from", str(tmp/"keep.1D"),
                "--out-1d", str(tmp/"reg.1D"), "--out-json", str(tmp/"out.json")])
            self.assertEqual(code, 1)
            self.assertFalse((tmp/"reg.1D").exists())


class TestCli(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = Path(self._tmp.name)
        self.frame = make_confounds()
        self.confounds = self.tmp / "confounds.tsv"
        write_tsv(self.confounds, self.frame, float_format="%.10g")
        self.keep = np.ones(N_T, dtype=int)
        self.keep[[10, 11, 70, 71, 72]] = 0
        self.censor = self.tmp / "censor.1D"
        self.censor.write_text("".join(f"{v}\n" for v in self.keep), encoding="ascii")
        self.out_1d = self.tmp / "out" / "regressors.1D"
        self.out_json = self.tmp / "out" / "denoise.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _argv(self, strategy: str, *extra: str) -> list[str]:
        return [
            "--confounds", str(self.confounds), "--strategy", strategy, "--tr", str(TR),
            "--n-volumes-censored-from", str(self.censor), "--polort", "2", "--filter-mode", "bandpass",
            "--band-low", "0.01", "--band-high", "0.1", "--min-dof", "15",
            "--out-1d", str(self.out_1d), "--out-json", str(self.out_json), *extra,
        ]

    def test_every_strategy_runs(self) -> None:
        for name in strategies.STRATEGIES:
            self.assertEqual(quiet(strategies.main, self._argv(name)), 0, name)
            matrix = np.loadtxt(self.out_1d, ndmin=2)
            info = json.loads(self.out_json.read_text(encoding="utf-8"))
            expected = strategies.strategy_columns(name) + ([c for c in self.frame if c.startswith("cosine_")] if strategies.STRATEGIES[name]["compcor"] else [])
            self.assertEqual(matrix.shape, (N_T, len(expected)), name)
            self.assertEqual(info["columns"], expected, name)
            self.assertEqual(info["n_regressors"], matrix.shape[1], name)
            retained = matrix  # default NTRP fits all rows
            np.testing.assert_allclose(retained.mean(axis=0), 0.0, atol=1e-8)
            np.testing.assert_allclose(retained.std(axis=0), 1.0, atol=1e-8)

    def test_json_contract_and_dof(self) -> None:
        argv = self._argv("wmcsf24gsr", "--censor-mode", "KILL", "--smooth-fwhm", "6",
                          "--input", "bold=/x/bold.nii.gz", "--input", "mask=/x/mask.nii.gz")
        self.assertEqual(quiet(strategies.main, argv), 0)
        info = json.loads(self.out_json.read_text(encoding="utf-8"))
        for key in ("strategy", "columns", "n_regressors", "polort", "filter_mode", "band", "n_volumes", "n_censored",
                    "censor_mode", "dof_bandpass_cost", "dof_remaining", "low_dof", "smooth_fwhm", "inputs"):
            self.assertIn(key, info)
        self.assertEqual(info["strategy"], "wmcsf24gsr")
        self.assertEqual(info["n_regressors"], 27)
        self.assertEqual(info["band"], [0.01, 0.1])
        self.assertEqual((info["n_volumes"], info["n_censored"], info["n_volumes_out"]), (N_T, 5, N_T - 5))
        self.assertEqual(info["dof_bandpass_cost"], 97)
        self.assertEqual(info["dof_remaining"], 145 - 27 - 3 - 97)
        self.assertIs(info["low_dof"], False)
        self.assertEqual(info["censor_mode"], "KILL")
        self.assertEqual(info["smooth_fwhm"], 6.0)
        self.assertEqual(info["inputs"]["bold"], "/x/bold.nii.gz")
        self.assertEqual(info["inputs"]["confounds"], str(self.confounds))

    def test_low_dof_is_flagged_not_fatal(self) -> None:
        self.assertEqual(quiet(strategies.main, self._argv("36p", "--min-dof", "20")), 0)
        info = json.loads(self.out_json.read_text(encoding="utf-8"))
        self.assertEqual(info["dof_remaining"], info["fit_rows"] - info["design_rank"])
        self.assertEqual(info["afni_nominal_dof"], 145 - 36 - 3 - 97)
        self.assertIs(info["low_dof"], True)
        self.assertEqual(info["n_volumes_out"], N_T)           # NTRP keeps the length
        self.assertTrue(any("degrees of freedom" in note for note in info["warnings"]))

    def test_highpass_and_none_bands(self) -> None:
        argv = self._argv("legacy8")
        argv[argv.index("--filter-mode") + 1] = "highpass"
        self.assertEqual(quiet(strategies.main, argv), 0)
        info = json.loads(self.out_json.read_text(encoding="utf-8"))
        self.assertEqual(info["band"], [0.01, None])
        self.assertEqual(info["dof_remaining"], 150 - 8 - 3 - 7)
        argv[argv.index("--filter-mode") + 1] = "none"
        self.assertEqual(quiet(strategies.main, argv), 0)
        info = json.loads(self.out_json.read_text(encoding="utf-8"))
        self.assertEqual(info["band"], [None, None])
        self.assertEqual(info["dof_remaining"], 150 - 8 - 3)

    def test_precision_of_the_1d_file(self) -> None:
        self.assertEqual(quiet(strategies.main, self._argv("legacy9gsr")), 0)
        columns = strategies.strategy_columns("legacy9gsr")
        expected, _, _ = strategies.prepare_regressors(read_tsv(self.confounds), columns)
        self.assertNotIn(b"\r", self.out_1d.read_bytes())
        self.assertNotIn("nan", self.out_1d.read_text(encoding="ascii").lower())
        written = np.loadtxt(self.out_1d, ndmin=2)
        np.testing.assert_allclose(written, expected, rtol=0, atol=1e-9)
        # the confounds TSV holds 10 significant digits: a tissue mean near 10000 with
        # an SD of a few units keeps about 6 digits after scaling
        unrounded, _, _ = strategies.prepare_regressors(self.frame, columns)
        np.testing.assert_allclose(written, unrounded, rtol=0, atol=1e-5)

    def test_duplicate_column_in_table(self) -> None:
        frame = self.frame.copy()
        frame["csf"] = frame["white_matter"] * 3.0 - 5.0
        write_tsv(self.confounds, frame, float_format="%.10g")
        self.assertEqual(quiet(strategies.main, self._argv("legacy8")), 0)
        info = json.loads(self.out_json.read_text(encoding="utf-8"))
        self.assertEqual(info["n_regressors"], 7)
        self.assertNotIn("csf", info["columns"])
        self.assertEqual(info["dropped_columns"], {"csf": "duplicate of white_matter"})
        self.assertEqual(np.loadtxt(self.out_1d, ndmin=2).shape, (N_T, 7))

    def test_unknown_strategy_exits_2(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = strategies.main(self._argv("NoGRS"))
        self.assertEqual(code, 2)
        for name in strategies.STRATEGIES:
            self.assertIn(name, stderr.getvalue())
        self.assertFalse(self.out_1d.exists())

    def test_module_entry_point(self) -> None:
        env = dict(os.environ, PYTHONPATH=str(REPO / "py"))
        done = subprocess.run([sys.executable, "-m", "fmriproc.strategies", *self._argv("bogus")],
                              env=env, capture_output=True, text=True)
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertIn("wmcsf24gsr", done.stderr)
        done = subprocess.run([sys.executable, "-m", "fmriproc.strategies", *self._argv("wmcsf24")],
                              env=env, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue(self.out_1d.is_file() and self.out_json.is_file())

    def test_acompcor_n_from_the_command_line(self) -> None:
        self.assertEqual(quiet(strategies.main, self._argv("acompcor", "--acompcor-n", "2")), 0)
        info = json.loads(self.out_json.read_text(encoding="utf-8"))
        self.assertEqual(info["n_regressors"], 12 + 2 + 2 + len([c for c in self.frame if c.startswith("cosine_")]))
        self.assertEqual([c for c in info["columns"] if "_comp_cor_" in c],
                         ["w_comp_cor_00", "w_comp_cor_01", "c_comp_cor_00", "c_comp_cor_01"])
        self.assertEqual(info["warnings"], [])

    def test_bad_censor_file(self) -> None:
        self.censor.write_text("1\n0\n1\n", encoding="ascii")
        self.assertEqual(quiet(strategies.main, self._argv("legacy8")), 1)
        self.assertFalse(self.out_1d.exists())

    def test_na_in_a_needed_column(self) -> None:
        frame = self.frame.copy()
        frame["csf"] = np.nan                                  # empty CSF mask in stage 04
        write_tsv(self.confounds, frame, float_format="%.10g")
        self.assertEqual(quiet(strategies.main, self._argv("legacy8")), 1)
        self.assertEqual(quiet(strategies.main, self._argv("acompcor")), 0)


if __name__ == "__main__":
    unittest.main()
