"""Tests of fmriproc.prep_utils on synthetic data (unittest; no pytest in the image)."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from fmriproc import prep_utils as pu


def _affine(voxel: tuple[float, float, float] = (3.0, 3.0, 4.0), origin: tuple[float, float, float] = (-30.0, -40.0, -20.0)) -> np.ndarray:
    affine = np.diag([*voxel, 1.0])
    affine[:3, 3] = origin
    return affine


def _rotation_x(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def _save(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> Path:
    nib.save(nib.Nifti1Image(data, _affine() if affine is None else affine), str(path))
    return path


def _head_series(n_t: int = 60, seed: int = 0) -> np.ndarray:
    """(12, 12, 8, T) float32: bright 'head' block in a dark background + noise."""
    rng = np.random.default_rng(seed)
    data = rng.normal(5.0, 1.0, size=(12, 12, 8, n_t)).astype(np.float32)
    data[3:9, 3:9, 2:6, :] += 1000.0
    return data


def _run_cli(argv: list[str]) -> tuple[int, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
        code = pu.main(argv)
    return code, buffer.getvalue().strip()


class DetectNssTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(1)
        self.steady = 1000.0 + rng.normal(0.0, 1.0, size=50)

    def test_steady_series_has_none(self) -> None:
        self.assertEqual(pu.detect_nss(self.steady), 0)

    def test_saturation_recovery_is_counted(self) -> None:
        signal = self.steady.copy()
        signal[:3] += [300.0, 120.0, 40.0]
        self.assertEqual(pu.detect_nss(signal), 3)

    def test_only_contiguous_from_start(self) -> None:
        signal = self.steady.copy()
        signal[0] += 200.0
        signal[5] += 200.0          # a later spike is not a dummy scan
        self.assertEqual(pu.detect_nss(signal), 1)

    def test_low_outlier_is_ignored(self) -> None:
        signal = self.steady.copy()
        signal[0] -= 200.0
        self.assertEqual(pu.detect_nss(signal), 0)

    def test_capped(self) -> None:
        signal = self.steady.copy()
        signal[:20] += 500.0
        self.assertEqual(pu.detect_nss(signal), pu.NSS_CAP)

    def test_flat_and_short_series(self) -> None:
        self.assertEqual(pu.detect_nss(np.full(50, 100.0)), 0)
        self.assertEqual(pu.detect_nss([5.0, 1.0]), 0)
        short = [150.0, 100.0, 100.2, 99.8, 100.1, 99.9, 100.0, 100.3]
        self.assertEqual(pu.detect_nss(short), 1)

    def test_global_signal_from_image(self) -> None:
        data = _head_series(n_t=60)
        data[3:9, 3:9, 2:6, 0] *= 1.4
        data[3:9, 3:9, 2:6, 1] *= 1.1
        img = nib.Nifti1Image(data, _affine())
        signal = pu.global_signal(img, 50)
        self.assertEqual(signal.shape, (50,))
        self.assertGreater(signal[0], signal[10] * 1.3)
        self.assertEqual(pu.detect_nss(signal), 2)


class OneDTest(unittest.TestCase):
    def test_min_outlier_index(self) -> None:
        self.assertEqual(pu.min_outlier_index([0.3, 0.1, 0.05, 0.2]), 2)

    def test_tie_prefers_the_middle_of_the_run(self) -> None:
        self.assertEqual(pu.min_outlier_index([0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0]), 3)

    def test_nan_is_skipped(self) -> None:
        self.assertEqual(pu.min_outlier_index([np.nan, 0.4, 0.2]), 2)
        with self.assertRaises(ValueError):
            pu.min_outlier_index([np.nan, np.nan])

    def test_max_index(self) -> None:
        self.assertEqual(pu.max_index([0.0, 0.5, 2.5, 1.0]), (2, 2.5))

    def test_cli_reads_column_and_row_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            column = Path(tmp) / "outliers.1D"
            column.write_text("# 3dToutcount\n0.30\n0.01\n0.20\n", encoding="utf-8")
            row = Path(tmp) / "row.1D"
            row.write_text("0.30 0.20 0.01\n", encoding="utf-8")
            self.assertEqual(_run_cli(["min-outlier", "--file", str(column)]), (0, "1"))
            self.assertEqual(_run_cli(["min-outlier", "--file", str(row)]), (0, "2"))
            self.assertEqual(_run_cli(["max-index", "--file", str(column)]), (0, "0 0.3"))
            code, text = _run_cli(["mean-1d", "--file", str(column)])
            self.assertEqual(code, 0)
            self.assertAlmostEqual(float(text), 0.17, places=6)

    def test_cli_missing_file_is_an_error(self) -> None:
        code, text = _run_cli(["min-outlier", "--file", "/nonexistent/x.1D"])
        self.assertEqual((code, text), (1, ""))


class DespikeFractionTest(unittest.TestCase):
    def test_identical_series(self) -> None:
        data = _head_series()
        img = nib.Nifti1Image(data, _affine())
        self.assertEqual(pu.despike_fraction(img, img), 0.0)

    def test_known_fraction_inside_the_head(self) -> None:
        data = _head_series(n_t=30)
        clean = data.copy()
        clean[3:9, 3:9, 2:6, 10] -= 50.0      # one of 30 volumes changed everywhere in the head
        clean[0, 0, 0, :] += 99.0             # background changes do not count
        before = nib.Nifti1Image(data, _affine())
        after = nib.Nifti1Image(clean, _affine())
        self.assertAlmostEqual(pu.despike_fraction(before, after, max_vols=40), 1.0 / 30.0, places=6)

    def test_subsampling_uses_evenly_spaced_volumes(self) -> None:
        data = _head_series(n_t=100)
        clean = data.copy()
        clean[3:9, 3:9, 2:6, 0::3] -= 50.0    # step = ceil(100 / 40) = 3 -> every sampled volume differs
        fraction = pu.despike_fraction(nib.Nifti1Image(data, _affine()), nib.Nifti1Image(clean, _affine()), max_vols=40)
        self.assertAlmostEqual(fraction, 1.0, places=6)

    def test_shape_mismatch(self) -> None:
        a = nib.Nifti1Image(_head_series(n_t=10), _affine())
        b = nib.Nifti1Image(_head_series(n_t=12), _affine())
        with self.assertRaises(ValueError):
            pu.despike_fraction(a, b)


class ScaleDiceCorrTest(unittest.TestCase):
    def test_scale_factor(self) -> None:
        mean = np.full((4, 4, 4), 50.0, dtype=np.float32)
        mask = np.zeros((4, 4, 4), dtype=bool)
        mask[1:3, 1:3, 1:3] = True
        mean[mask] = 500.0
        mean[1, 1, 1] = np.nan
        self.assertAlmostEqual(pu.scale_factor(mean, mask, 10000.0), 20.0)

    def test_scale_factor_refuses_empty_or_dark_masks(self) -> None:
        mean = np.zeros((3, 3, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            pu.scale_factor(mean, np.zeros((3, 3, 3), dtype=bool), 10000.0)
        with self.assertRaises(ValueError):
            pu.scale_factor(mean, np.ones((3, 3, 3), dtype=bool), 10000.0)

    def test_dice(self) -> None:
        a = np.zeros((10, 10, 10), dtype=bool)
        b = np.zeros((10, 10, 10), dtype=bool)
        a[2:6] = True          # 400 voxels
        b[4:8] = True          # 400 voxels, 200 shared
        self.assertAlmostEqual(pu.dice(a, b), 0.5)
        self.assertAlmostEqual(pu.dice(a, a), 1.0)
        self.assertTrue(np.isnan(pu.dice(np.zeros_like(a), np.zeros_like(a))))

    def test_dice_within_field_of_view(self) -> None:
        a = np.zeros((10, 10, 10), dtype=bool)
        b = np.zeros((10, 10, 10), dtype=bool)
        a[2:6] = True
        b[2:9] = True          # the anatomical mask continues outside the EPI field of view
        fov = np.zeros((10, 10, 10), dtype=bool)
        fov[:6] = True
        self.assertLess(pu.dice(a, b), 0.8)
        self.assertAlmostEqual(pu.dice(a, b, within=fov), 1.0)

    def test_masked_corr(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.normal(size=(8, 8, 8)).astype(np.float32)
        mask = np.zeros((8, 8, 8), dtype=bool)
        mask[2:6, 2:6, 2:6] = True
        b = 3.0 * a + 7.0
        b[~mask] = rng.normal(size=int((~mask).sum()))     # disagreement outside the mask is irrelevant
        self.assertAlmostEqual(pu.masked_corr(a, b, mask), 1.0, places=5)
        self.assertAlmostEqual(pu.masked_corr(a, -b, mask), -1.0, places=5)
        self.assertTrue(np.isnan(pu.masked_corr(a, np.ones_like(a), mask)))

    def test_cli_on_files_and_grid_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mask = np.zeros((6, 6, 6), dtype=np.uint8)
            mask[1:5, 1:5, 1:5] = 1
            mean = np.where(mask > 0, 250.0, 3.0).astype(np.float32)
            mask_file = _save(tmp_path / "mask.nii.gz", mask)
            mean_file = _save(tmp_path / "mean.nii.gz", mean)
            code, text = _run_cli(["scale-factor", "--mean", str(mean_file), "--mask", str(mask_file), "--target", "10000"])
            self.assertEqual(code, 0)
            self.assertAlmostEqual(float(text), 40.0)
            code, text = _run_cli(["dice", "--a", str(mask_file), "--b", str(mask_file)])
            self.assertEqual((code, float(text)), (0, 1.0))
            code, text = _run_cli(["masked-corr", "--a", str(mean_file), "--b", str(mean_file), "--mask", str(mask_file)])
            self.assertEqual((code, text), (0, "nan"))      # constant inside the mask
            shifted = _save(tmp_path / "shifted.nii.gz", mask, _affine(origin=(0.0, 0.0, 0.0)))
            code, _ = _run_cli(["dice", "--a", str(mask_file), "--b", str(shifted)])
            self.assertEqual(code, 1)


class GeometryTest(unittest.TestCase):
    def test_obliquity(self) -> None:
        self.assertAlmostEqual(pu.obliquity_deg(_affine()), 0.0, places=5)
        self.assertAlmostEqual(pu.obliquity_deg(_rotation_x(23.0) @ _affine()), 23.0, places=4)
        las = _affine()
        las[0, 0] *= -1.0
        self.assertAlmostEqual(pu.obliquity_deg(las), 0.0, places=5)

    def test_voxel_sizes(self) -> None:
        self.assertEqual(pu.voxel_sizes(_rotation_x(10.0) @ _affine((3.0, 3.4375, 4.0))), [3.0, 3.4375, 4.0])

    def _check_grid_encloses(self, mask_img: nib.Nifti1Image, grid: nib.Nifti1Image, pad: float) -> None:
        index = np.argwhere(np.asarray(mask_img.dataobj) > 0)
        world = nib.affines.apply_affine(mask_img.affine, index)
        edges_low = nib.affines.apply_affine(grid.affine, np.array([-0.5, -0.5, -0.5]))
        edges_high = nib.affines.apply_affine(grid.affine, np.array(grid.shape) - 0.5)
        box_low = np.minimum(edges_low, edges_high)
        box_high = np.maximum(edges_low, edges_high)
        self.assertTrue(np.all(world.min(axis=0) - pad >= box_low - 1e-4))
        self.assertTrue(np.all(world.max(axis=0) + pad <= box_high + 1e-4))

    def test_make_grid_conformed_lia(self) -> None:
        # FreeSurfer conformed geometry: LIA storage, 1 mm
        affine = np.array([[-1.0, 0, 0, 128.0], [0, 0, 1.0, -110.0], [0, -1.0, 0, 128.0], [0, 0, 0, 1.0]])
        data = np.zeros((64, 64, 64), dtype=np.uint8)
        data[20:40, 10:50, 25:45] = 1
        mask_img = nib.Nifti1Image(data, affine)
        grid = pu.make_grid(mask_img, 3.0, pad_mm=10.0)
        self.assertEqual(nib.aff2axcodes(grid.affine), ("L", "A", "S"))
        np.testing.assert_allclose(nib.affines.voxel_sizes(grid.affine), [3.0, 3.0, 3.0])
        self.assertAlmostEqual(pu.obliquity_deg(grid.affine), 0.0, places=5)
        self._check_grid_encloses(mask_img, grid, pad=10.0)
        # 20 x 20 x 40 mm box (world x, y, z) + 2 * 10 mm, never more than one voxel too large
        self.assertEqual(grid.shape, (14, 14, 20))
        self.assertEqual(int(grid.header["qform_code"]), 1)
        self.assertEqual(int(grid.header["sform_code"]), 1)
        np.testing.assert_allclose(grid.get_qform(), grid.get_sform(), atol=1e-4)

    def test_make_grid_oblique_mask(self) -> None:
        affine = _rotation_x(15.0) @ _affine((1.0, 1.0, 1.2))
        data = np.zeros((40, 40, 30), dtype=np.uint8)
        data[5:35, 8:30, 4:26] = 1
        mask_img = nib.Nifti1Image(data, affine)
        grid = pu.make_grid(mask_img, 2.5, pad_mm=5.0)
        self.assertAlmostEqual(pu.obliquity_deg(grid.affine), 0.0, places=5)
        self._check_grid_encloses(mask_img, grid, pad=5.0)

    def test_make_grid_rejects_empty_mask(self) -> None:
        with self.assertRaises(ValueError):
            pu.make_grid(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.uint8), _affine()), 3.0)

    def test_cli_make_grid_and_obliquity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data = np.zeros((20, 20, 20), dtype=np.uint8)
            data[5:15, 5:15, 5:15] = 1
            mask_file = _save(tmp_path / "mask.nii.gz", data, _rotation_x(23.0) @ _affine((1.0, 1.0, 1.0)))
            out = tmp_path / "grid.nii.gz"
            code, text = _run_cli(["make-grid", "--mask", str(mask_file), "--res", "3", "--out", str(out)])
            self.assertEqual(code, 0)
            grid = nib.load(str(out))
            self.assertEqual(text, " ".join(str(n) for n in grid.shape))
            self.assertEqual(grid.get_data_dtype(), np.uint8)
            self.assertEqual(_run_cli(["obliquity", "--image", str(mask_file)]), (0, "23.000"))
            self.assertEqual(_run_cli(["obliquity", "--image", str(out)]), (0, "0.000"))


class WriteInfoTest(unittest.TestCase):
    def test_infer_value(self) -> None:
        self.assertIs(pu.infer_value("true"), True)
        self.assertIs(pu.infer_value("yes"), True)
        self.assertIs(pu.infer_value("False"), False)
        self.assertIs(pu.infer_value("no"), False)
        self.assertEqual(pu.infer_value("4"), 4)
        self.assertIsInstance(pu.infer_value("4"), int)
        self.assertEqual(pu.infer_value("2.0"), 2.0)
        self.assertEqual(pu.infer_value("1e-3"), 0.001)
        self.assertEqual(pu.infer_value("[3.0, 3.0, 4.0]"), [3.0, 3.0, 4.0])
        self.assertEqual(pu.infer_value("bbregister"), "bbregister")
        self.assertEqual(pu.infer_value("sub-0050952_task-rest"), "sub-0050952_task-rest")
        for empty in ("", "nan", "NaN", "n/a", "null", "None"):
            self.assertIsNone(pu.infer_value(empty), empty)

    def test_parse_pairs(self) -> None:
        pairs = pu.parse_pairs(["tr=2", "stc_reason=STC=off", "despike=yes"])
        self.assertEqual(pairs, {"tr": 2, "stc_reason": "STC=off", "despike": True})
        self.assertEqual(pu.parse_pairs(["run_label=007", "note=no"], as_string=True), {"run_label": "007", "note": "no"})
        with self.assertRaises(ValueError):
            pu.parse_pairs(["novalue"])

    def test_build_info_reports_missing_contract_keys(self) -> None:
        info, missing = pu.build_info({"tr": 2.0, "extra_key": 1}, versions={"fsl": "6.0.7"})
        self.assertEqual(list(info)[: len(pu.PREP_INFO_KEYS)], list(pu.PREP_INFO_KEYS))
        self.assertEqual(list(info)[-1], "extra_key")
        self.assertIn("coreg_method", missing)
        self.assertNotIn("tr", missing)
        self.assertNotIn("tool_versions", missing)
        self.assertIsNone(info["coreg_method"])
        self.assertEqual(info["tool_versions"], {"fsl": "6.0.7"})

    def test_cli_writes_every_contract_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bold = _save(tmp_path / "bold.nii.gz", _head_series(n_t=4), _rotation_x(12.5) @ _affine((3.0, 3.0, 4.0)))
            versions = tmp_path / "versions.json"
            versions.write_text(json.dumps({"pipeline": "2.0.0", "fsl": "6.0.7.18"}), encoding="utf-8")
            out = tmp_path / "sub-1_task-rest_desc-prep_info.json"
            argv = [
                "write-info", "--out", str(out), "--versions-json", str(versions), "--geometry-from", str(bold),
                "--set-str", "source=/out/rawdata/sub-1/func/sub-1_task-rest_bold.nii.gz",
                "--set-str", "run_label=sub-1_task-rest",
                "--set-str", "stc_reason=verified slice timing", "--set-str", "stc_interp=quintic",
                "--set-str", "slice_timing_source=acquisition table", "--set-str", "slice_timing_evidence=A:direct",
                "--set-str", "hmc_reference=median of mcflirt pass 1 (initial reference: volume 17)",
                "--set-str", "coreg_method=bbregister", "--set-str", "epi_mask_method=synthstrip",
                "--set-str", "template=MNI152NLin6Asym",
                "--set", "tr=2.0", "--set", "n_volumes_raw=180", "--set", "n_dropped=4", "--set", "n_volumes=176",
                "--set", "nss_detected=1", "--set", "despike=true", "--set", "despike_fraction=0.0123",
                "--set", "stc_applied=true", "--set", "tzero=0.969697", "--set", "bbr_cost=0.612",
                "--set", "bbr_vs_init_mm=1.84", "--set", "bbr_rejected=false", "--set", "scale_factor=12.75",
                "--set", "func_t1w_res=3", "--set", "mni_res=2", "--set", "wm_mask_voxels=2100",
                "--set", "csf_mask_voxels=55", "--set", "gm_mask_voxels=15000",
                "--set", "tissue_erosion_relaxed=false", "--set", "coreg_dice=0.94",
                "--set", "hmc_consistency_r=nan",
            ]
            code, _ = _run_cli(argv)
            self.assertEqual(code, 0)
            text = out.read_text(encoding="utf-8")
            self.assertNotIn("\r", text)
            info = json.loads(text)
            for key in pu.PREP_INFO_KEYS:
                self.assertIn(key, info)
            self.assertEqual(info["voxel_size"], [3.0, 3.0, 4.0])
            self.assertAlmostEqual(info["obliquity_deg"], 12.5, places=2)
            self.assertIs(info["despike"], True)
            self.assertIs(info["bbr_rejected"], False)
            self.assertEqual(info["n_volumes"], 176)
            self.assertEqual(info["run_label"], "sub-1_task-rest")
            self.assertIsNone(info["hmc_consistency_r"])
            self.assertEqual(info["tool_versions"]["fsl"], "6.0.7.18")

    def test_cli_rejects_malformed_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, _ = _run_cli(["write-info", "--out", str(Path(tmp) / "x.json"), "--set", "broken"])
            self.assertEqual(code, 1)
            self.assertFalse((Path(tmp) / "x.json").exists())


if __name__ == "__main__":
    unittest.main()
