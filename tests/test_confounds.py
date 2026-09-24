"""Tests of fmriproc.confounds (stage 04) on tiny synthetic data."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from fmriproc import confounds
from fmriproc.utils import longest_true_run, read_tsv

AFFINE = np.diag([3.0, 3.0, 3.0, 1.0])
SHAPE = (6, 6, 5)
N_T = 60
TR = 2.0


def reference_std_dvars(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plain float64 transcription of nipype.algorithms.confounds.compute_dvars."""
    data = np.asarray(series, dtype=np.float64)
    q75 = np.percentile(data, 75, axis=0, method="lower")
    q25 = np.percentile(data, 25, axis=0, method="lower")
    sd = (q75 - q25) / 1.349
    good = sd > 1e-7
    data, sd = data[:, good], sd[good]
    centred = data - data.mean(axis=0)
    ar1 = (centred[1:] * centred[:-1]).sum(axis=0) / (centred ** 2).sum(axis=0)
    expected = np.mean(np.sqrt(2.0 * (1.0 - ar1)) * sd)
    dvars = np.sqrt(np.mean(np.diff(data, axis=0) ** 2, axis=1))
    return np.concatenate([[0.0], dvars]), np.concatenate([[0.0], dvars / expected])


class TestMotion(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_par_columns_are_reordered(self) -> None:
        # mcflirt writes rx ry rz tx ty tz separated by two blanks
        path = self.tmp / "motion.par"
        path.write_text("0.001  0.002  0.003  1  2  3  \n-0.001  0  0  4  5  6  \n", encoding="ascii")
        motion = confounds.read_motion_par(path)
        np.testing.assert_allclose(motion[0], [1, 2, 3, 0.001, 0.002, 0.003])
        np.testing.assert_allclose(motion[1], [4, 5, 6, -0.001, 0, 0])

    def test_par_with_wrong_column_count(self) -> None:
        path = self.tmp / "bad.par"
        path.write_text("0 0 0 0 0\n0 0 0 0 0\n", encoding="ascii")
        with self.assertRaises(ValueError):
            confounds.read_motion_par(path)

    def test_fd_known_answer(self) -> None:
        # columns: trans_x trans_y trans_z (mm) rot_x rot_y rot_z (rad)
        motion = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.001, 0.0, 0.0],      # 0.1 + 50 * 0.001           = 0.15
            [0.1, 0.3, -0.2, 0.001, -0.002, 0.0],  # 0.3 + 0.2 + 50 * 0.002     = 0.60
            [0.0, 0.3, -0.2, 0.001, -0.002, 0.004],  # 0.1 + 50 * 0.004         = 0.30
        ])
        fd = confounds.framewise_displacement(motion)
        np.testing.assert_allclose(fd, [0.0, 0.15, 0.60, 0.30], atol=1e-12)

    def test_fd_from_par_file(self) -> None:
        path = self.tmp / "motion.par"
        path.write_text("0  0  0  0  0  0\n0.002  0  0  0  0  0.25\n", encoding="ascii")
        fd = confounds.framewise_displacement(confounds.read_motion_par(path))
        np.testing.assert_allclose(fd, [0.0, 0.25 + 50 * 0.002])

    def test_relrms_padding(self) -> None:
        padded = confounds.pad_relrms(np.array([[0.1], [0.2], [0.3]]), 4)
        np.testing.assert_allclose(padded, [0.0, 0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            confounds.pad_relrms(np.array([0.1, 0.2]), 5)


class TestExpansions(unittest.TestCase):
    def test_derivative_and_powers(self) -> None:
        out = confounds.expand_signal("csf", np.array([1.0, 3.0, 6.0, 10.0]))
        self.assertEqual(list(out), ["csf", "csf_derivative1", "csf_power2", "csf_derivative1_power2"])
        np.testing.assert_allclose(out["csf"], [1, 3, 6, 10])
        np.testing.assert_allclose(out["csf_derivative1"], [0, 2, 3, 4])
        np.testing.assert_allclose(out["csf_power2"], [1, 9, 36, 100])
        np.testing.assert_allclose(out["csf_derivative1_power2"], [0, 4, 9, 16])

    def test_backward_difference_2d(self) -> None:
        values = np.array([[1.0, 10.0], [2.0, 8.0], [4.0, 9.0]])
        np.testing.assert_allclose(confounds.backward_difference(values), [[0, 0], [1, -2], [2, 1]])


class TestDctBasis(unittest.TestCase):
    def test_order_and_orthonormality(self) -> None:
        basis = confounds.dct_basis(100, 2.0, 128.0)
        self.assertEqual(basis.shape, (100, int(np.floor(2 * 100 * 2.0 / 128.0))))
        np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10)
        np.testing.assert_allclose(basis.sum(axis=0), 0.0, atol=1e-10)

    def test_short_run_has_no_cosines(self) -> None:
        self.assertEqual(confounds.dct_basis(20, 2.0, 128.0).shape, (20, 0))
        self.assertEqual(confounds.dct_basis(20, 2.0, 0.0).shape, (20, 0))

    def test_matches_nilearn(self) -> None:
        try:
            from nilearn.glm.first_level.design_matrix import _cosine_drift as nilearn_drift
        except ImportError:
            try:
                from nilearn.glm.first_level.design_matrix import create_cosine_drift as nilearn_drift
            except ImportError:
                self.skipTest("nilearn cosine drift function not importable")
        frame_times = np.arange(90) * 2.5
        expected = nilearn_drift(1.0 / 128.0, frame_times)[:, :-1]
        np.testing.assert_allclose(confounds.dct_basis(90, 2.5, 128.0), expected, atol=1e-10)


class TestACompCor(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(7)
        self.n_t = 120
        self.basis = confounds.dct_basis(self.n_t, TR, 128.0)
        times = np.arange(self.n_t) * TR
        self.planted = np.sin(2 * np.pi * 0.05 * times) + 0.5 * np.sin(2 * np.pi * 0.11 * times + 1.0)

    def _series(self, n_vox: int = 80) -> np.ndarray:
        loading = self.rng.uniform(2.0, 4.0, size=n_vox) * self.rng.choice([-1.0, 1.0], size=n_vox)
        noise = self.rng.normal(0.0, 1.0, size=(self.n_t, n_vox))
        return (10000.0 + np.outer(self.planted, loading) + noise).astype(np.float32)

    def test_recovers_planted_component(self) -> None:
        components, variance, n_used = confounds.acompcor(self._series(), self.basis, 5)
        self.assertEqual(components.shape, (self.n_t, 5))
        self.assertEqual(n_used, 80)
        r = np.corrcoef(components[:, 0], self.planted)[0, 1]
        self.assertGreater(abs(r), 0.99)
        self.assertGreater(variance[0], 0.5)
        self.assertTrue(np.all(np.diff(variance) <= 1e-12))
        self.assertAlmostEqual(float(variance.sum()), 1.0, places=10)

    def test_components_are_orthonormal_and_drift_free(self) -> None:
        components, _, _ = confounds.acompcor(self._series(), self.basis, 5)
        np.testing.assert_allclose(components.T @ components, np.eye(5), atol=1e-8)
        np.testing.assert_allclose(self.basis.T @ components, 0.0, atol=1e-8)
        np.testing.assert_allclose(components.sum(axis=0), 0.0, atol=1e-8)

    def test_slow_drift_does_not_win(self) -> None:
        # a drift inside the DCT space, ten times larger than the planted signal
        slow = 300.0 * self.basis[:, 0] / np.abs(self.basis[:, 0]).max()
        series = self._series() + slow[:, None].astype(np.float32)
        raw = series - series.mean(axis=0)
        self.assertGreater(abs(np.corrcoef(np.linalg.svd(raw, full_matrices=False)[0][:, 0], slow)[0, 1]), 0.9)
        components, _, _ = confounds.acompcor(series, self.basis, 3)
        self.assertGreater(abs(np.corrcoef(components[:, 0], self.planted)[0, 1]), 0.99)

    def test_sign_is_deterministic(self) -> None:
        series = self._series()
        a, _, _ = confounds.acompcor(series, self.basis, 3)
        b, _, _ = confounds.acompcor(series[:, ::-1].copy(), self.basis, 3)
        np.testing.assert_allclose(a[:, 0], b[:, 0], atol=1e-6)
        peak = np.abs(a).argmax(axis=0)
        self.assertTrue(np.all(a[peak, np.arange(3)] > 0))

    def test_fewer_voxels_than_components(self) -> None:
        components, variance, n_used = confounds.acompcor(self._series(3), self.basis, 5)
        self.assertEqual(n_used, 3)
        self.assertLessEqual(components.shape[1], 3)
        self.assertGreaterEqual(components.shape[1], 1)
        self.assertEqual(variance.size, 3)

    def test_constant_voxels_and_empty_mask(self) -> None:
        series = self._series(10)
        series[:, :4] = 10000.0
        _, _, n_used = confounds.acompcor(series, self.basis, 2)
        self.assertEqual(n_used, 6)
        components, variance, n_used = confounds.acompcor(np.zeros((self.n_t, 0), dtype=np.float32), self.basis, 5)
        self.assertEqual(components.shape, (self.n_t, 0))
        self.assertEqual((variance.size, n_used), (0, 0))


class TestDvars(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(3)
        self.series = (10000.0 + rng.normal(0.0, 50.0, size=(200, 400))).astype(np.float32)

    def test_first_zero_and_non_negative(self) -> None:
        dvars, std_dvars = confounds.compute_dvars(self.series)
        self.assertEqual(dvars.shape, (200,))
        self.assertEqual((dvars[0], std_dvars[0]), (0.0, 0.0))
        self.assertTrue(np.all(dvars >= 0) and np.all(std_dvars >= 0))
        self.assertTrue(np.all(dvars[1:] > 0))

    def test_white_noise_is_about_one(self) -> None:
        dvars, std_dvars = confounds.compute_dvars(self.series)
        self.assertAlmostEqual(float(np.mean(std_dvars[1:])), 1.0, delta=0.08)
        self.assertAlmostEqual(float(np.mean(dvars[1:])), 50.0 * np.sqrt(2.0), delta=3.0)

    def test_matches_reference_implementation(self) -> None:
        dvars, std_dvars = confounds.compute_dvars(self.series)
        ref_dvars, ref_std = reference_std_dvars(self.series)
        np.testing.assert_allclose(dvars, ref_dvars, rtol=1e-4)
        np.testing.assert_allclose(std_dvars, ref_std, rtol=1e-3)

    def test_spike_and_scale_invariance(self) -> None:
        spiked = self.series.copy()
        spiked[120] += 400.0
        _, std_dvars = confounds.compute_dvars(spiked)
        # the spike enters the difference series twice: going in (120) and coming back (121)
        self.assertIn(int(np.argmax(std_dvars)), (120, 121))
        self.assertGreater(std_dvars[120], 3.0)
        self.assertGreater(std_dvars[121], 3.0)
        _, scaled = confounds.compute_dvars(spiked * np.float32(0.01))
        np.testing.assert_allclose(scaled, std_dvars, rtol=1e-3)

    def test_constant_voxels_are_ignored(self) -> None:
        series = self.series.copy()
        series[:, :100] = 10000.0
        dvars, std_dvars = confounds.compute_dvars(series)
        ref_dvars, ref_std = reference_std_dvars(self.series[:, 100:])
        np.testing.assert_allclose(dvars, ref_dvars, rtol=1e-4)
        np.testing.assert_allclose(std_dvars, ref_std, rtol=1e-3)

    def test_degenerate_input(self) -> None:
        dvars, std_dvars = confounds.compute_dvars(np.full((10, 5), 100.0, dtype=np.float32))
        self.assertTrue(np.all(dvars == 0))
        self.assertEqual(std_dvars[0], 0.0)
        self.assertTrue(np.all(np.isnan(std_dvars[1:])))


class TestCensor(unittest.TestCase):
    FD = np.array([0.0, 0.1, 0.6, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1])
    DV = np.array([0.0, 1.0, 1.1, 1.0, 1.0, 1.0, 1.0, 2.5, 1.0])

    def test_fd_only(self) -> None:
        censor = confounds.censor_vector(self.FD, self.DV, 0.5, False, 0.0)
        np.testing.assert_array_equal(censor, [1, 1, 0, 1, 1, 0, 1, 1, 1])
        self.assertEqual(longest_true_run(censor == 1), 3)

    def test_previous_frame(self) -> None:
        censor = confounds.censor_vector(self.FD, self.DV, 0.5, True, 0.0)
        np.testing.assert_array_equal(censor, [1, 0, 0, 1, 0, 0, 1, 1, 1])
        self.assertEqual(longest_true_run(censor == 1), 3)

    def test_previous_frame_at_run_start(self) -> None:
        censor = confounds.censor_vector(np.array([0.0, 0.9, 0.1]), np.zeros(3), 0.5, True, 0.0)
        np.testing.assert_array_equal(censor, [0, 0, 1])

    def test_dvars_criterion(self) -> None:
        censor = confounds.censor_vector(self.FD, self.DV, 0.5, False, 2.0)
        np.testing.assert_array_equal(censor, [1, 1, 0, 1, 1, 0, 1, 0, 1])
        self.assertEqual(longest_true_run(censor == 1), 2)
        # the previous-frame rule belongs to FD only
        censor = confounds.censor_vector(self.FD, self.DV, 0.0, True, 2.0)
        np.testing.assert_array_equal(censor, [1, 1, 1, 1, 1, 1, 1, 0, 1])

    def test_thresholds_are_strict_and_zero_is_off(self) -> None:
        censor = confounds.censor_vector(np.array([0.0, 0.5, 0.5000001]), np.zeros(3), 0.5, False, 0.0)
        np.testing.assert_array_equal(censor, [1, 1, 0])
        censor = confounds.censor_vector(self.FD, self.DV, 0.0, True, 0.0)
        self.assertTrue(np.all(censor == 1))
        self.assertEqual(longest_true_run(censor == 1), self.FD.size)

    def test_previous_frame_with_consecutive_high_fd(self) -> None:
        # frames 2, 3 and 4 move; with prev the block 1..4 goes, nothing after it
        fd = np.array([0.0, 0.1, 0.9, 0.8, 0.7, 0.1, 0.1])
        censor = confounds.censor_vector(fd, np.zeros(7), 0.5, True, 0.0)
        np.testing.assert_array_equal(censor, [1, 0, 0, 0, 0, 1, 1])
        self.assertEqual(longest_true_run(censor == 1), 2)
        # the last frame moving censors itself and its predecessor only
        censor = confounds.censor_vector(np.array([0.0, 0.1, 0.1, 0.9]), np.zeros(4), 0.5, True, 0.0)
        np.testing.assert_array_equal(censor, [1, 1, 0, 0])

    def test_nan_dvars_never_censors(self) -> None:
        censor = confounds.censor_vector(np.zeros(4), np.array([0.0, np.nan, np.nan, np.nan]), 0.5, False, 1.5)
        self.assertTrue(np.all(censor == 1))

    def test_frames_after(self) -> None:
        censor = confounds.censor_vector(self.FD, self.DV, 0.5, False, 0.0, censor_next=2)
        np.testing.assert_array_equal(censor, [1, 1, 0, 0, 0, 0, 0, 0, 1])
        # with prev as well, Power et al. 2014 style (1 before, 2 after)
        censor = confounds.censor_vector(self.FD, self.DV, 0.5, True, 0.0, censor_next=2)
        np.testing.assert_array_equal(censor, [1, 0, 0, 0, 0, 0, 0, 0, 1])
        # frames after the end of the run are ignored; next extends FD only, not DVARS
        censor = confounds.censor_vector(np.array([0.0, 0.1, 0.9]), np.zeros(3), 0.5, False, 0.0, censor_next=5)
        np.testing.assert_array_equal(censor, [1, 1, 0])
        censor = confounds.censor_vector(self.FD, self.DV, 0.0, False, 2.0, censor_next=2)
        np.testing.assert_array_equal(censor, [1, 1, 1, 1, 1, 1, 1, 0, 1])

    def test_minimum_segment(self) -> None:
        # kept stretches: 0-1 (2 frames), 3-4 (2), 6-8 (3)
        censor = confounds.censor_vector(self.FD, self.DV, 0.5, False, 0.0, min_segment=3)
        np.testing.assert_array_equal(censor, [0, 0, 0, 0, 0, 0, 1, 1, 1])
        censor = confounds.censor_vector(self.FD, self.DV, 0.5, False, 0.0, min_segment=4)
        self.assertTrue(np.all(censor == 0))
        # a run without censored frames is never shortened
        censor = confounds.censor_vector(np.zeros(4), np.zeros(4), 0.5, False, 0.0, min_segment=10)
        self.assertTrue(np.all(censor == 1))
        # 0 and 1 switch the rule off
        for value in (0, 1):
            censor = confounds.censor_vector(self.FD, self.DV, 0.5, False, 0.0, min_segment=value)
            np.testing.assert_array_equal(censor, [1, 1, 0, 1, 1, 0, 1, 1, 1])

    def test_short_segments_helper(self) -> None:
        keep = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
        np.testing.assert_array_equal(confounds.short_segments(keep, 3),
                                      [1, 0, 1, 1, 0, 0, 0, 0, 0, 1])

    def test_negative_options_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            confounds.censor_vector(self.FD, self.DV, 0.5, False, 0.0, censor_next=-1)
        with self.assertRaises(ValueError):
            confounds.censor_vector(self.FD, self.DV, 0.5, False, 0.0, min_segment=-2)


class TestCli(unittest.TestCase):
    """The whole module on a 6x6x5x60 series."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = Path(self._tmp.name)
        rng = np.random.default_rng(11)
        self.brain = np.zeros(SHAPE, dtype=bool)
        self.brain[1:5, 1:5, 1:4] = True
        self.wm = np.zeros(SHAPE, dtype=bool)
        self.wm[2:4, 2:4, 1:4] = True           # 12 voxels
        self.csf = np.zeros(SHAPE, dtype=bool)
        self.csf[1, 1, 1:3] = True              # 2 voxels: fewer than ACOMPCOR_N
        data = rng.normal(10000.0, 40.0, size=SHAPE + (N_T,))
        times = np.arange(N_T) * TR
        self.wm_signal = 80.0 * np.sin(2 * np.pi * 0.06 * times)
        data[self.wm] += self.wm_signal * rng.uniform(0.8, 1.2, size=(int(self.wm.sum()), 1))
        self.data = data.astype(np.float32)

        self.par = rng.normal(0.0, 0.0005, size=(N_T, 6))
        self.par[:, 3:] = rng.normal(0.0, 0.02, size=(N_T, 3))
        self.par[30, 3] += 1.0                   # one big translation step
        self.paths = {
            "bold": self.tmp / "bold.nii.gz", "brain": self.tmp / "brain.nii.gz",
            "wm": self.tmp / "wm.nii.gz", "csf": self.tmp / "csf.nii.gz",
            "par": self.tmp / "motion.par", "relrms": self.tmp / "rel.rms", "outliers": self.tmp / "out.1D",
            "tsv": self.tmp / "out" / "confounds.tsv", "json": self.tmp / "out" / "confounds.json",
            "censor": self.tmp / "out" / "censor.1D",
        }
        nib.save(nib.Nifti1Image(self.data, AFFINE), str(self.paths["bold"]))
        for key, mask in (("brain", self.brain), ("wm", self.wm), ("csf", self.csf)):
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), AFFINE), str(self.paths[key]))
        np.savetxt(self.paths["par"], self.par, fmt="%.8g", delimiter="  ")
        np.savetxt(self.paths["relrms"], np.abs(rng.normal(0.05, 0.01, size=N_T - 1)), fmt="%.6f")
        np.savetxt(self.paths["outliers"], np.abs(rng.normal(0.01, 0.002, size=N_T)), fmt="%.6f")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _argv(self, **override: str) -> list[str]:
        values = {
            "--bold": self.paths["bold"], "--brain-mask": self.paths["brain"], "--wm-mask": self.paths["wm"],
            "--csf-mask": self.paths["csf"], "--motion-par": self.paths["par"], "--relrms": self.paths["relrms"],
            "--outliers": self.paths["outliers"], "--tr": TR, "--acompcor-n": 5, "--highpass-sec": 64,
            "--censor-fd": 0.5, "--censor-prev": "no", "--censor-dvars": 0,
            "--out-tsv": self.paths["tsv"], "--out-json": self.paths["json"], "--out-censor": self.paths["censor"],
        }
        values.update(override)
        return [str(item) for pair in values.items() for item in pair]

    def test_outputs(self) -> None:
        self.assertEqual(confounds.main(self._argv()), 0)
        frame = read_tsv(self.paths["tsv"])
        self.assertEqual(len(frame), N_T)

        expected = []
        for base in confounds.MOTION_COLUMNS + ["csf", "white_matter", "global_signal"]:
            expected += [base + suffix for suffix in confounds.EXPANSIONS]
        expected += [f"w_comp_cor_{i:02d}" for i in range(5)]
        expected += ["framewise_displacement", "fd_jenkinson", "dvars", "std_dvars", "outlier_fraction", "censor"]
        for column in expected:
            self.assertIn(column, frame.columns)
        compcor_csf = [c for c in frame.columns if c.startswith("c_comp_cor_")]
        self.assertLessEqual(len(compcor_csf), 2)
        self.assertGreaterEqual(len(compcor_csf), 1)
        n_cos = int(np.floor(2 * N_T * TR / 64))
        self.assertEqual([c for c in frame.columns if c.startswith("cosine_")], [f"cosine_{i:02d}" for i in range(n_cos)])

        # nothing that can become a regressor may be n/a, and derivatives start at 0
        regressors = [c for c in frame.columns if c not in ("std_dvars",)]
        self.assertFalse(frame[regressors].isna().any().any())
        derivative = [c for c in frame.columns if "_derivative1" in c]
        self.assertEqual(len(derivative), 18)
        self.assertTrue((frame.loc[0, derivative] == 0).all())
        self.assertEqual(frame.loc[0, "framewise_displacement"], 0)
        self.assertEqual(frame.loc[0, "fd_jenkinson"], 0)
        self.assertEqual(frame.loc[0, "dvars"], 0)
        self.assertEqual(frame.loc[0, "std_dvars"], 0)

        # values against direct computation
        np.testing.assert_allclose(frame["global_signal"], self.data[self.brain].mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(frame["white_matter"], self.data[self.wm].mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(frame["csf"], self.data[self.csf].mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(frame["trans_x"], self.par[:, 3], rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(frame["rot_z"], self.par[:, 2], rtol=1e-6, atol=1e-9)
        motion = np.column_stack([self.par[:, 3:], self.par[:, :3]])
        np.testing.assert_allclose(frame["framewise_displacement"], confounds.framewise_displacement(motion), rtol=1e-6)
        ref_dvars, ref_std = reference_std_dvars(self.data[self.brain].T)
        np.testing.assert_allclose(frame["dvars"], ref_dvars, rtol=1e-4)
        np.testing.assert_allclose(frame["std_dvars"], ref_std, rtol=1e-3)
        self.assertGreater(abs(np.corrcoef(frame["w_comp_cor_00"], self.wm_signal)[0, 1]), 0.95)

        # censoring: the planted step moves frame 30 (and back at 31)
        censor = frame["censor"].to_numpy()
        self.assertEqual(censor[30], 0)
        self.assertEqual(censor[31], 0)
        lines = self.paths["censor"].read_text(encoding="ascii").split("\n")
        self.assertEqual(lines[-1], "")
        self.assertEqual([int(v) for v in lines[:-1]], censor.tolist())
        self.assertNotIn(b"\r", self.paths["tsv"].read_bytes())

    def test_json_contract(self) -> None:
        self.assertEqual(confounds.main(self._argv(**{"--censor-prev": "yes"})), 0)
        meta = json.loads(self.paths["json"].read_text(encoding="utf-8"))
        frame = read_tsv(self.paths["tsv"])
        for column in frame.columns:
            self.assertIn(column, meta, f"column {column} is not described")
        censor = frame["censor"].to_numpy()
        info = meta["censor"]
        for key in ("fd_threshold", "prev", "dvars_threshold", "n_censored", "n_volumes", "minutes_retained", "longest_segment"):
            self.assertIn(key, info)
        self.assertEqual(info["fd_threshold"], 0.5)
        self.assertIs(info["prev"], True)
        self.assertEqual(info["n_volumes"], N_T)
        self.assertEqual(info["n_censored"], int((censor == 0).sum()))
        self.assertGreaterEqual(info["n_censored"], 3)     # frames 29, 30, 31
        self.assertEqual(censor[29], 0)
        self.assertAlmostEqual(info["minutes_retained"], (censor == 1).sum() * TR / 60.0)
        self.assertEqual(info["longest_segment"], longest_true_run(censor == 1))
        acc = meta["acompcor"]
        self.assertEqual(acc["n_wm_voxels"], 12)
        self.assertEqual(acc["n_csf_voxels"], 2)
        self.assertEqual(len(acc["variance_explained_wm"]), 5)
        self.assertEqual(len(acc["variance_explained_csf"]), sum(c.startswith("c_comp_cor_") for c in frame.columns))
        self.assertLess(len(acc["variance_explained_csf"]), 5)

    def test_block_reading_equals_full_read(self) -> None:
        img = nib.load(str(self.paths["bold"]))
        union = self.brain | self.wm | self.csf
        old = confounds.CHUNK_BYTES
        try:
            confounds.CHUNK_BYTES = 7 * int(np.prod(SHAPE)) * 8      # 7 volumes per block
            series = confounds.load_masked_series(img, union)
        finally:
            confounds.CHUNK_BYTES = old
        self.assertEqual(series.dtype, np.float32)
        np.testing.assert_array_equal(series, self.data[union].T)

    def test_no_censoring(self) -> None:
        self.assertEqual(confounds.main(self._argv(**{"--censor-fd": "0"})), 0)
        self.assertEqual(set(self.paths["censor"].read_text(encoding="ascii").split()), {"1"})

    def test_frames_after_and_minimum_segment_options(self) -> None:
        self.assertEqual(confounds.main(self._argv(**{"--censor-next": "2", "--censor-min-segment": "5"})), 0)
        meta = json.loads(self.paths["json"].read_text(encoding="utf-8"))
        censor = read_tsv(self.paths["tsv"])["censor"].to_numpy()
        self.assertEqual((meta["censor"]["next"], meta["censor"]["min_segment"]), (2, 5))
        self.assertTrue(np.all(censor[30:34] == 0))           # the step at 30 and back at 31, plus 2 after
        self.assertEqual(meta["censor"]["n_censored"], int((censor == 0).sum()))
        # the defaults record the rules as off
        self.assertEqual(confounds.main(self._argv()), 0)
        meta = json.loads(self.paths["json"].read_text(encoding="utf-8"))
        self.assertEqual((meta["censor"]["next"], meta["censor"]["min_segment"]), (0, 0))

    def test_bad_option_values_exit_with_usage_error(self) -> None:
        for option, value in (("--censor-next", "-1"), ("--censor-min-segment", "2.5")):
            with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(io.StringIO()):
                confounds.main(self._argv(**{option: value}))
            self.assertEqual(caught.exception.code, 2)

    def test_empty_tissue_mask_gives_na(self) -> None:
        nib.save(nib.Nifti1Image(np.zeros(SHAPE, dtype=np.uint8), AFFINE), str(self.paths["csf"]))
        self.assertEqual(confounds.main(self._argv()), 0)
        frame = read_tsv(self.paths["tsv"])
        self.assertTrue(frame["csf"].isna().all())
        self.assertFalse(any(c.startswith("c_comp_cor_") for c in frame.columns))
        self.assertFalse(frame["white_matter"].isna().any())

    def test_bad_inputs_fail(self) -> None:
        shifted = AFFINE.copy()
        shifted[0, 3] = 9.0
        nib.save(nib.Nifti1Image(self.wm.astype(np.uint8), shifted), str(self.paths["wm"]))
        self.assertEqual(confounds.main(self._argv()), 1)
        self.assertFalse(self.paths["tsv"].exists())

    def test_row_mismatch_fails(self) -> None:
        np.savetxt(self.paths["par"], self.par[:-2], fmt="%.8g")
        self.assertEqual(confounds.main(self._argv()), 1)
        self.assertFalse(self.paths["tsv"].exists())

    def test_relrms_length(self) -> None:
        # N rows (some mcflirt builds) are tolerated, anything else is an error
        np.savetxt(self.paths["relrms"], np.full(N_T, 0.05), fmt="%.6f")
        self.assertEqual(confounds.main(self._argv()), 0)
        frame = read_tsv(self.paths["tsv"])
        np.testing.assert_allclose(frame["fd_jenkinson"], 0.05)
        np.savetxt(self.paths["relrms"], np.full(N_T - 3, 0.05), fmt="%.6f")
        self.assertEqual(confounds.main(self._argv()), 1)

    def test_mask_stored_as_4d_single_volume(self) -> None:
        nib.save(nib.Nifti1Image(self.wm.astype(np.uint8)[..., None], AFFINE), str(self.paths["wm"]))
        self.assertEqual(confounds.main(self._argv()), 0)
        frame = read_tsv(self.paths["tsv"])
        np.testing.assert_allclose(frame["white_matter"], self.data[self.wm].mean(axis=0), rtol=1e-6)
        meta = json.loads(self.paths["json"].read_text(encoding="utf-8"))
        self.assertEqual(meta["acompcor"]["n_wm_voxels"], 12)

    def test_outliers_file_with_afni_comment_header(self) -> None:
        text = "# 3dToutcount output\n" + "".join(f"{v:.6f}\n" for v in np.linspace(0.0, 0.05, N_T))
        self.paths["outliers"].write_text(text, encoding="ascii")
        self.assertEqual(confounds.main(self._argv()), 0)
        frame = read_tsv(self.paths["tsv"])
        np.testing.assert_allclose(frame["outlier_fraction"], np.linspace(0.0, 0.05, N_T), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
