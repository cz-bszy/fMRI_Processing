"""Tests for fmriproc.anat_qc (synthetic volumes only)."""
from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from fmriproc import anat_qc

# keys of sub-X_desc-anatqc.json, docs/DESIGN.md section 10
CONTRACT_KEYS = [
    "anat_mode", "euler_lh", "euler_rh", "holes_total", "brain_volume_mm3", "norm_quality",
    "norm_dice", "template_corr", "jacobian_p01", "jacobian_p50", "jacobian_p99",
    "jacobian_nonpos_frac", "wm_voxels", "csf_voxels", "gm_voxels",
]


def _cube(shape: tuple[int, int, int], lo: int, hi: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[lo:hi, lo:hi, lo:hi] = 1
    return mask


def _save(path: Path, data: np.ndarray, affine: np.ndarray) -> Path:
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


class DiceTest(unittest.TestCase):
    def test_identical_masks(self) -> None:
        mask = _cube((10, 10, 10), 2, 8)
        self.assertEqual(anat_qc.dice(mask, mask), 1.0)

    def test_disjoint_masks(self) -> None:
        a = np.zeros((10, 10, 10), dtype=np.uint8)
        b = np.zeros((10, 10, 10), dtype=np.uint8)
        a[:5] = 1
        b[5:] = 1
        self.assertEqual(anat_qc.dice(a, b), 0.0)

    def test_partial_overlap(self) -> None:
        a = np.zeros((4, 4, 4), dtype=np.uint8)
        b = np.zeros((4, 4, 4), dtype=np.uint8)
        a[:2] = 1          # 32 voxels
        b[1:3] = 1         # 32 voxels, 16 shared
        self.assertAlmostEqual(anat_qc.dice(a, b), 0.5)

    def test_non_binary_values_count_as_inside(self) -> None:
        a = _cube((8, 8, 8), 1, 5).astype(np.float32) * 110.0
        b = _cube((8, 8, 8), 1, 5)
        self.assertEqual(anat_qc.dice(a, b), 1.0)

    def test_both_empty_is_nan(self) -> None:
        empty = np.zeros((4, 4, 4), dtype=np.uint8)
        self.assertTrue(math.isnan(anat_qc.dice(empty, empty)))

    def test_one_empty_mask_is_zero(self) -> None:
        a = _cube((6, 6, 6), 1, 4)
        self.assertEqual(anat_qc.dice(a, np.zeros_like(a)), 0.0)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            anat_qc.dice(np.ones((4, 4, 4)), np.ones((4, 4, 5)))


class MaskedCorrelationTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.x = rng.normal(100.0, 10.0, size=(8, 8, 8)).astype(np.float32)
        self.mask = _cube((8, 8, 8), 1, 7)

    def test_linear_relation_is_one(self) -> None:
        self.assertAlmostEqual(anat_qc.masked_correlation(self.x, 2.0 * self.x + 5.0, self.mask), 1.0, places=5)

    def test_inverted_is_minus_one(self) -> None:
        self.assertAlmostEqual(anat_qc.masked_correlation(self.x, -self.x, self.mask), -1.0, places=5)

    def test_outside_of_mask_is_ignored(self) -> None:
        y = self.x.copy()
        y[self.mask == 0] = 1e6 * np.arange(int((self.mask == 0).sum()))
        self.assertAlmostEqual(anat_qc.masked_correlation(self.x, y, self.mask), 1.0, places=5)

    def test_matches_numpy(self) -> None:
        rng = np.random.default_rng(1)
        y = self.x + rng.normal(0.0, 10.0, size=self.x.shape).astype(np.float32)
        inside = self.mask > 0
        expected = np.corrcoef(self.x[inside].astype(float), y[inside].astype(float))[0, 1]
        self.assertAlmostEqual(anat_qc.masked_correlation(self.x, y, self.mask), expected, places=6)

    def test_nan_voxels_are_dropped(self) -> None:
        y = 3.0 * self.x
        y[3, 3, 3] = np.nan
        self.assertAlmostEqual(anat_qc.masked_correlation(self.x, y, self.mask), 1.0, places=5)

    def test_constant_image_is_nan(self) -> None:
        self.assertTrue(math.isnan(anat_qc.masked_correlation(self.x, np.ones_like(self.x), self.mask)))

    def test_fewer_than_three_voxels_is_nan(self) -> None:
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[0, 0, :2] = 1
        self.assertTrue(math.isnan(anat_qc.masked_correlation(self.x, 2.0 * self.x, mask)))

    def test_grid_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            anat_qc.masked_correlation(self.x, self.x[:-1], self.mask)


class JacobianStatsTest(unittest.TestCase):
    def test_identity_warp(self) -> None:
        stats = anat_qc.jacobian_stats(np.ones((6, 6, 6), dtype=np.float32), _cube((6, 6, 6), 1, 5))
        self.assertEqual(stats["jacobian_p01"], 1.0)
        self.assertEqual(stats["jacobian_p50"], 1.0)
        self.assertEqual(stats["jacobian_p99"], 1.0)
        self.assertEqual(stats["jacobian_nonpos_frac"], 0.0)

    def test_percentiles_use_only_the_mask(self) -> None:
        jac = np.full((10, 10, 10), 50.0, dtype=np.float32)
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[0] = 1                                    # 100 voxels
        jac[0] = np.linspace(0.5, 1.5, 100).reshape(10, 10)
        stats = anat_qc.jacobian_stats(jac, mask)
        self.assertAlmostEqual(stats["jacobian_p01"], 0.51, places=5)
        self.assertAlmostEqual(stats["jacobian_p50"], 1.0, places=5)
        self.assertAlmostEqual(stats["jacobian_p99"], 1.49, places=5)
        self.assertEqual(stats["jacobian_nonpos_frac"], 0.0)

    def test_folded_fraction(self) -> None:
        jac = np.ones((10, 10, 10), dtype=np.float32)
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[:2] = 1                                   # 200 voxels
        jac[0, 0, :5] = -0.3                           # folded
        jac[0, 1, :4] = 0.0                            # collapsed
        jac[0, 2, 0] = np.nan                          # counted as bad, excluded from percentiles
        jac[5] = -1.0                                  # outside the mask: ignored
        stats = anat_qc.jacobian_stats(jac, mask)
        self.assertAlmostEqual(stats["jacobian_nonpos_frac"], 10 / 200)
        self.assertTrue(math.isfinite(stats["jacobian_p50"]))
        self.assertEqual(stats["jacobian_p50"], 1.0)

    def test_empty_mask_is_nan(self) -> None:
        stats = anat_qc.jacobian_stats(np.ones((4, 4, 4)), np.zeros((4, 4, 4)))
        self.assertEqual(sorted(stats), ["jacobian_nonpos_frac", "jacobian_p01", "jacobian_p50", "jacobian_p99"])
        self.assertTrue(all(math.isnan(v) for v in stats.values()))

    def test_all_nonfinite_inside_mask(self) -> None:
        jac = np.full((4, 4, 4), np.nan, dtype=np.float32)
        stats = anat_qc.jacobian_stats(jac, np.ones((4, 4, 4)))
        self.assertEqual(stats["jacobian_nonpos_frac"], 1.0)
        self.assertTrue(math.isnan(stats["jacobian_p01"]))
        self.assertTrue(math.isnan(stats["jacobian_p50"]))
        self.assertTrue(math.isnan(stats["jacobian_p99"]))

    def test_grid_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            anat_qc.jacobian_stats(np.ones((4, 4, 4)), np.ones((4, 4, 3)))


class SmallHelpersTest(unittest.TestCase):
    def test_parse_optional_int(self) -> None:
        self.assertEqual(anat_qc.parse_optional_int("-116"), -116)
        self.assertEqual(anat_qc.parse_optional_int(" 2 "), 2)
        for text in ("n/a", "", "NaN", "None", "-", None):
            self.assertIsNone(anat_qc.parse_optional_int(text))
        with self.assertRaises(ValueError):
            anat_qc.parse_optional_int("two")

    def test_holes_from_euler(self) -> None:
        self.assertEqual(anat_qc.holes_from_euler(2), 0)
        self.assertEqual(anat_qc.holes_from_euler(-116), 59)   # mris_euler_number: -116 --> 59 holes
        self.assertEqual(anat_qc.holes_from_euler(0), 1)
        self.assertIsNone(anat_qc.holes_from_euler(None))

    def test_voxel_volume_and_count(self) -> None:
        affine = np.diag([1.0, 1.2, 2.0, 1.0])
        affine[0, 0] = -1.0                                    # radiological storage: determinant < 0
        img = nib.Nifti1Image(_cube((8, 8, 8), 2, 6), affine)
        self.assertAlmostEqual(anat_qc.voxel_volume_mm3(img), 2.4)
        self.assertEqual(anat_qc.mask_voxels(img), 64)


class CommandLineTest(unittest.TestCase):
    """End-to-end run on a tiny synthetic 'template' grid."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        shape = (12, 12, 12)
        self.tpl_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        self.t1_affine = np.diag([-1.0, 1.0, 1.0, 1.0])
        rng = np.random.default_rng(3)
        tpl_mask = _cube(shape, 2, 10)
        tpl_brain = (rng.uniform(50, 150, size=shape) * tpl_mask).astype(np.float32)
        warped_mask = _cube(shape, 3, 10)                      # 343 of the 512 template voxels
        jac = np.ones(shape, dtype=np.float32)
        jac[2, 2, 2:6] = -0.5                                  # 4 folded voxels inside the template mask
        self.files = {
            "template_mask": _save(self.dir / "tpl_mask.nii.gz", tpl_mask, self.tpl_affine),
            "template_brain": _save(self.dir / "tpl_brain.nii.gz", tpl_brain, self.tpl_affine),
            "warped_brain": _save(self.dir / "warped.nii.gz", (0.5 * tpl_brain + 3).astype(np.float32), self.tpl_affine),
            "warped_mask": _save(self.dir / "warped_mask.nii.gz", warped_mask, self.tpl_affine),
            "jacobian": _save(self.dir / "jac.nii.gz", jac, self.tpl_affine),
            "brain_mask": _save(self.dir / "brain_mask.nii.gz", _cube((16, 16, 16), 3, 13), self.t1_affine),
            "wm": _save(self.dir / "wm.nii.gz", _cube((16, 16, 16), 6, 10), self.t1_affine),
            "csf": _save(self.dir / "csf.nii.gz", _cube((16, 16, 16), 7, 9), self.t1_affine),
            "gm": _save(self.dir / "gm.nii.gz", _cube((16, 16, 16), 4, 12), self.t1_affine),
        }
        self.out = self.dir / "out" / "sub-01_desc-anatqc.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _argv(self, **override: str) -> list[str]:
        f = self.files
        argv = [
            "--anat-mode", "freesurfer", "--norm-quality", "precise",
            "--brain-mask", str(f["brain_mask"]),
            "--wm-mask", str(f["wm"]), "--csf-mask", str(f["csf"]), "--gm-mask", str(f["gm"]),
            "--warped-brain", str(f["warped_brain"]), "--warped-mask", str(f["warped_mask"]),
            "--template-brain", str(f["template_brain"]), "--template-mask", str(f["template_mask"]),
            "--jacobian", str(f["jacobian"]),
            "--euler-lh=-116", "--euler-rh=-20", "--holes-lh=59", "--holes-rh=11",
            "--template-name", "MNI152NLin6Asym", "--wm-erode", "2",
            "--csf-erode", "0", "--csf-erode-requested", "1",
            "--out", str(self.out),
        ]
        for key, value in override.items():
            flag = "--" + key.replace("_", "-")
            argv = [a for a in argv if not a.startswith(flag + "=")]
            if flag in argv:
                at = argv.index(flag)
                del argv[at: at + 2]
            argv.append(f"{flag}={value}")
        return argv

    def _run(self, argv: list[str]) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return anat_qc.main(argv)

    def test_writes_every_contract_key(self) -> None:
        self.assertEqual(self._run(self._argv()), 0)
        with open(self.out, encoding="utf-8") as fh:
            text = fh.read()
        data = json.loads(text)
        for key in CONTRACT_KEYS:
            self.assertIn(key, data)
        self.assertNotIn("\r", text)
        self.assertEqual(data["anat_mode"], "freesurfer")
        self.assertEqual(data["norm_quality"], "precise")
        self.assertEqual((data["euler_lh"], data["euler_rh"], data["holes_total"]), (-116, -20, 70))
        self.assertAlmostEqual(data["brain_volume_mm3"], 1000.0)
        self.assertAlmostEqual(data["norm_dice"], 2 * 343 / (343 + 512))
        self.assertAlmostEqual(data["template_corr"], 1.0, places=5)
        self.assertAlmostEqual(data["jacobian_nonpos_frac"], 4 / 512)
        self.assertEqual(data["jacobian_p50"], 1.0)
        self.assertEqual((data["wm_voxels"], data["csf_voxels"], data["gm_voxels"]), (64, 8, 512))
        self.assertTrue(data["csf_erosion_relaxed"])
        # flat JSON: scalars only, so json_get of lib/common.sh can read every key
        self.assertFalse([k for k, v in data.items() if isinstance(v, (dict, list))])

    def test_synth_mode_has_null_surface_metrics(self) -> None:
        argv = self._argv(anat_mode="synth", euler_lh="n/a", euler_rh="n/a", holes_lh="n/a", holes_rh="n/a")
        self.assertEqual(self._run(argv), 0)
        data = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(data["anat_mode"], "synth")
        self.assertIsNone(data["euler_lh"])
        self.assertIsNone(data["euler_rh"])
        self.assertIsNone(data["holes_total"])

    def test_holes_are_derived_from_euler_when_missing(self) -> None:
        self.assertEqual(self._run(self._argv(holes_lh="n/a", holes_rh="n/a")), 0)
        data = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(data["holes_total"], 59 + 11)

    def test_without_jacobian(self) -> None:
        argv = self._argv()
        at = argv.index("--jacobian")
        del argv[at: at + 2]
        self.assertEqual(self._run(argv), 0)
        data = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertIsNone(data["jacobian_p01"])
        self.assertIsNone(data["jacobian_nonpos_frac"])
        self.assertIsNotNone(data["norm_dice"])

    def test_image_on_another_grid_fails(self) -> None:
        wrong = _save(self.dir / "wrong_grid.nii.gz", _cube((12, 12, 12), 3, 10), np.diag([1.0, 1.0, 1.0, 1.0]))
        self.assertEqual(self._run(self._argv(warped_mask=str(wrong))), 1)
        self.assertFalse(self.out.exists())

    def test_missing_input_fails(self) -> None:
        self.assertEqual(self._run(self._argv(jacobian=str(self.dir / "absent.nii.gz"))), 1)
        self.assertFalse(self.out.exists())

    def test_empty_template_mask_fails(self) -> None:
        empty = _save(self.dir / "empty_mask.nii.gz", np.zeros((12, 12, 12), dtype=np.uint8), self.tpl_affine)
        self.assertEqual(self._run(self._argv(template_mask=str(empty))), 1)
        self.assertFalse(self.out.exists())

    def test_grid_tolerance_accepts_float_rounding(self) -> None:
        # ITK (antsApplyTransforms) and FSL write the same 1 mm grid with sub-voxel
        # differences in the sform; that must not count as a grid mismatch
        affine = self.tpl_affine.copy()
        affine[:3, 3] += 1e-3
        rounded = _save(self.dir / "warped_mask_rounded.nii.gz", _cube((12, 12, 12), 3, 10), affine)
        self.assertEqual(self._run(self._argv(warped_mask=str(rounded))), 0)
        data = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertAlmostEqual(data["norm_dice"], 2 * 343 / (343 + 512))

    def test_single_frame_4d_input_is_accepted(self) -> None:
        jac4d = np.ones((12, 12, 12, 1), dtype=np.float32)
        path = _save(self.dir / "jac4d.nii.gz", jac4d, self.tpl_affine)
        self.assertEqual(self._run(self._argv(jacobian=str(path))), 0)
        data = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(data["jacobian_nonpos_frac"], 0.0)
        self.assertEqual(data["jacobian_p50"], 1.0)

    def test_negative_euler_as_separate_argument(self) -> None:
        # the stage passes --euler-lh=-N; make sure the space-separated form works too
        argv = [a for a in self._argv() if not a.startswith("--euler-lh=")] + ["--euler-lh", "-40"]
        self.assertEqual(self._run(argv), 0)
        data = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(data["euler_lh"], -40)
        self.assertEqual(data["holes_total"], 59 + 11)     # explicit --holes-lh=59 still wins


if __name__ == "__main__":
    unittest.main()
