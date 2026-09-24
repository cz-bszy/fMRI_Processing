"""Tests of fmriproc.timeseries (stage 07) on tiny synthetic images."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from fmriproc import timeseries
from fmriproc.utils import read_tsv

REPO = Path(__file__).resolve().parents[1]
AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
SHAPE = (6, 6, 4)
N_T = 40


def save_nifti(path: Path, data: np.ndarray, affine: np.ndarray = AFFINE) -> Path:
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def make_atlas() -> np.ndarray:
    """Three ROIs of 8 voxels each: labels 1, 2 and 5."""
    atlas = np.zeros(SHAPE, dtype=np.int16)
    atlas[0:2, 0:2, 0:2] = 1
    atlas[2:4, 0:2, 0:2] = 2
    atlas[4:6, 0:2, 0:2] = 5
    return atlas


def make_bold(atlas: np.ndarray, seed: int = 0) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """BOLD whose ROI voxels are an ROI signal plus a voxel-specific offset."""
    rng = np.random.default_rng(seed)
    data = rng.normal(100.0, 1.0, size=SHAPE + (N_T,)).astype(np.float32)
    signals = {}
    for label in np.unique(atlas[atlas > 0]):
        signal = rng.normal(0.0, 5.0, size=N_T)
        signals[int(label)] = signal
        idx = np.argwhere(atlas == label)
        for k, (x, y, z) in enumerate(idx):
            data[x, y, z, :] = (1000.0 + k + signal).astype(np.float32)
    return data, signals


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        # open gzip handles (keep_file_open) may outlive a test on Windows
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write_inputs(self, mask: np.ndarray | None = None, seed: int = 0) -> dict[str, Path]:
        atlas = make_atlas()
        bold, self.signals = make_bold(atlas, seed)
        self.atlas, self.bold = atlas, bold
        mask = np.ones(SHAPE, dtype=np.uint8) if mask is None else mask
        return {
            "bold": save_nifti(self.tmp / "bold.nii.gz", bold),
            "atlas": save_nifti(self.tmp / "atlas.nii.gz", atlas),
            "mask": save_nifti(self.tmp / "mask.nii.gz", mask.astype(np.uint8)),
        }

    def run_volume(self, files: dict[str, Path], prefix: Path, *extra: str) -> int:
        argv = ["volume", "--bold", str(files["bold"]), "--atlas", str(files["atlas"]),
                "--mask", str(files["mask"]), "--out-prefix", str(prefix), *extra]
        return timeseries.main(argv)


class CoverageTest(TempDirCase):
    def test_low_coverage_roi_is_nan_and_written_as_na(self) -> None:
        mask = np.ones(SHAPE, dtype=np.uint8)
        mask[2:4, 0:2, 0:2] = 0
        mask[2, 0, 0:2] = 1                      # ROI 2: 2 of 8 voxels = 0.25
        mask[4, 0, 0, ] = 0                      # ROI 5: 7 of 8 voxels
        files = self.write_inputs(mask)
        prefix = self.tmp / "sub-1_task-rest_space-MNI_atlas-Test_Atlas_desc-wmcsf24"
        self.assertEqual(self.run_volume(files, prefix, "--min-coverage", "0.5"), 0)

        text = Path(f"{prefix}_timeseries.tsv").read_text(encoding="utf-8")
        lines = text.splitlines()
        self.assertEqual(lines[0].split("\t"), ["roi_1", "roi_2", "roi_5"])
        self.assertEqual(len(lines), N_T + 1)
        self.assertTrue(all(line.split("\t")[1] == "n/a" for line in lines[1:]))
        self.assertNotIn("\r", text)

        series = read_tsv(f"{prefix}_timeseries.tsv")
        self.assertTrue(series["roi_2"].isna().all())
        expected = self.bold[self.atlas == 1].astype(np.float64).mean(axis=0)
        np.testing.assert_allclose(series["roi_1"].to_numpy(), expected, rtol=1e-6)

        coverage = read_tsv(f"{prefix}_coverage.tsv")
        self.assertEqual(coverage["roi"].tolist(), [1, 2, 5])
        self.assertEqual(coverage["atlas_voxels"].tolist(), [8, 8, 8])
        self.assertEqual(coverage["valid_voxels"].tolist(), [8, 2, 7])
        np.testing.assert_allclose(coverage["coverage_fraction"], [1.0, 0.25, 0.875])
        self.assertEqual([bool(v) for v in coverage["included"]], [True, False, True])

        info = json.loads(Path(f"{prefix}_timeseries.json").read_text(encoding="utf-8"))
        self.assertEqual(info["missing_rois"], ["roi_2"])
        self.assertEqual(info["atlas"], "Test_Atlas")
        self.assertEqual(info["strategy"], "wmcsf24")
        self.assertEqual(info["n_volumes"], N_T)
        self.assertEqual(info["n_retained"], N_T)
        self.assertIn("definition", info)

        fc = read_tsv(f"{prefix}_connectivity.tsv")
        self.assertEqual(fc.shape, (3, 3))
        self.assertEqual(list(fc.columns), ["roi_1", "roi_2", "roi_5"])
        self.assertTrue(np.isnan(fc.to_numpy()[1]).all())
        self.assertTrue(np.isnan(fc.to_numpy()[:, 1]).all())
        self.assertAlmostEqual(fc.to_numpy()[0, 0], 1.0)

    def test_lower_threshold_keeps_the_roi(self) -> None:
        mask = np.ones(SHAPE, dtype=np.uint8)
        mask[2:4, 0:2, 0:2] = 0
        mask[2, 0, 0:2] = 1
        files = self.write_inputs(mask)
        prefix = self.tmp / "out"
        self.assertEqual(self.run_volume(files, prefix, "--min-coverage", "0.2"), 0)
        series = read_tsv(f"{prefix}_timeseries.tsv")
        expected = self.bold[2, 0, 0:2].astype(np.float64).mean(axis=0)
        np.testing.assert_allclose(series["roi_2"].to_numpy(), expected, rtol=1e-6)

    def test_constant_and_nonfinite_voxels_do_not_count(self) -> None:
        files = self.write_inputs()
        bold = self.bold.copy()
        bold[0, 0, 0, :] = 0.0                   # constant voxel in ROI 1
        bold[0, 1, 0, 3] = np.nan                # non-finite voxel in ROI 1
        save_nifti(files["bold"], bold)
        prefix = self.tmp / "out"
        self.assertEqual(self.run_volume(files, prefix), 0)
        coverage = read_tsv(f"{prefix}_coverage.tsv")
        self.assertEqual(int(coverage.loc[coverage["roi"] == 1, "valid_voxels"].iloc[0]), 6)

    def test_grid_mismatch_is_an_error(self) -> None:
        files = self.write_inputs()
        save_nifti(files["atlas"], make_atlas(), np.diag([3.0, 3.0, 3.0, 1.0]))
        self.assertEqual(self.run_volume(files, self.tmp / "out"), 1)
        self.assertFalse(Path(f"{self.tmp / 'out'}_timeseries.tsv").exists())

    def test_chunked_reading_equals_full_read(self) -> None:
        files = self.write_inputs()
        select = self.atlas > 0
        img = nib.load(str(files["bold"]), keep_file_open=True)
        chunked = timeseries.load_roi_voxels(img, select, max_chunk_bytes=1)   # one volume per block
        del img
        np.testing.assert_array_equal(chunked, self.bold[select])
        self.assertEqual(chunked.dtype, np.float32)


class LabelTest(TempDirCase):
    def write_labels(self, rows: list[tuple[object, str, str]], name: str = "labels.tsv") -> Path:
        path = self.tmp / name
        lines = ["index\tname\tnetwork"] + [f"{i}\t{n}\t{net}" for i, n, net in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_header_uses_label_names_and_keeps_every_table_row(self) -> None:
        files = self.write_inputs()
        # label 5 is not in the table; label 3 of the table is not in the atlas;
        # "NA" is a legitimate region name (nucleus accumbens), not a missing value
        labels = self.write_labels([(0, "Background", "n/a"), (1, "LH_Vis_1", "Vis"), (2, "NA", "Limbic"), (3, "RH_Lost", "Default")])
        prefix = self.tmp / "out"
        self.assertEqual(self.run_volume(files, prefix, "--labels", str(labels)), 0)
        header = Path(f"{prefix}_timeseries.tsv").read_text(encoding="utf-8").splitlines()[0].split("\t")
        self.assertEqual(header, ["LH_Vis_1", "NA", "RH_Lost", "roi_5"])

        series = pd.read_csv(f"{prefix}_timeseries.tsv", sep="\t", na_values=["n/a"], keep_default_na=False)
        self.assertTrue(series["RH_Lost"].isna().all())
        self.assertFalse(series["NA"].isna().any())

        coverage = pd.read_csv(f"{prefix}_coverage.tsv", sep="\t", na_values=["n/a"], keep_default_na=False)
        self.assertEqual(coverage["roi"].tolist(), [1, 2, 3, 5])
        self.assertEqual(coverage["name"].tolist(), header)
        self.assertEqual(coverage["network"].tolist()[:3], ["Vis", "Limbic", "Default"])
        self.assertEqual(coverage["atlas_voxels"].tolist(), [8, 8, 0, 8])
        info = json.loads(Path(f"{prefix}_timeseries.json").read_text(encoding="utf-8"))
        self.assertEqual(info["missing_rois"], ["RH_Lost"])

    def test_duplicate_and_empty_names(self) -> None:
        table = pd.DataFrame({"index": [1, 2, 5], "name": ["A", "A", ""], "network": ["n/a"] * 3})
        columns = timeseries.build_columns(np.array([1, 2, 5]), table)
        self.assertEqual(columns["name"].tolist(), ["A_1", "A_2", "roi_5"])

    def test_bad_label_table_is_an_error(self) -> None:
        files = self.write_inputs()
        bad = self.tmp / "bad.tsv"
        bad.write_text("id\tlabel\n1\tx\n", encoding="utf-8")
        self.assertEqual(self.run_volume(files, self.tmp / "out", "--labels", str(bad)), 1)

    def test_custom_atlas_without_labels(self) -> None:
        files = self.write_inputs()
        # float-typed .nii atlas as shipped in parcellations/, no label table
        custom = save_nifti(self.tmp / "custom.nii", make_atlas().astype(np.float32))
        files["atlas"] = custom
        prefix = self.tmp / "out"
        self.assertEqual(self.run_volume(files, prefix, "--labels", "none"), 0)
        header = Path(f"{prefix}_timeseries.tsv").read_text(encoding="utf-8").splitlines()[0].split("\t")
        self.assertEqual(header, ["roi_1", "roi_2", "roi_5"])
        coverage = read_tsv(f"{prefix}_coverage.tsv")
        self.assertTrue(coverage["network"].isna().all())


class CensorTest(TempDirCase):
    def write_censor(self, keep: np.ndarray) -> Path:
        path = self.tmp / "censor.1D"
        np.savetxt(path, keep.astype(int), fmt="%d")
        return path

    def test_retained_frames_rules(self) -> None:
        censor = np.ones(10)
        censor[[2, 3]] = 0
        keep, how = timeseries.retained_frames(10, censor, "NTRP")
        self.assertEqual((int(keep.sum()), how), (8, "applied"))
        keep, how = timeseries.retained_frames(8, censor, "KILL")
        self.assertEqual((int(keep.sum()), keep.size, how), (8, 8, "already_removed"))
        keep, how = timeseries.retained_frames(7, None, "NTRP")
        self.assertEqual((int(keep.sum()), how), (7, "none"))
        with self.assertRaises(ValueError):
            timeseries.retained_frames(9, censor, "KILL")

    def test_fc_uses_retained_frames_only(self) -> None:
        files = self.write_inputs(seed=3)
        keep = np.ones(N_T, dtype=bool)
        keep[[5, 6, 20]] = False
        bold = self.bold.copy()
        # a shared artefact in the censored frames would push every r towards 1
        bold[..., ~keep] += 5000.0
        save_nifti(files["bold"], bold)
        prefix = self.tmp / "out"
        rc = self.run_volume(files, prefix, "--censor", str(self.write_censor(keep)), "--censor-mode", "NTRP")
        self.assertEqual(rc, 0)

        series = read_tsv(f"{prefix}_timeseries.tsv").to_numpy()
        self.assertEqual(series.shape[0], N_T)             # censored rows stay in the table
        fc = read_tsv(f"{prefix}_connectivity.tsv").to_numpy()
        np.testing.assert_allclose(fc, np.corrcoef(series[keep], rowvar=False), atol=1e-5)
        self.assertFalse(np.allclose(fc, np.corrcoef(series, rowvar=False), atol=1e-2))
        truth = np.corrcoef(np.vstack([self.signals[k][keep] for k in (1, 2, 5)]))
        np.testing.assert_allclose(fc, truth, atol=1e-4)

        info = json.loads(Path(f"{prefix}_timeseries.json").read_text(encoding="utf-8"))
        self.assertEqual((info["n_volumes"], info["n_retained"]), (N_T, N_T - 3))
        self.assertEqual(info["censor_handling"], "applied")

    def test_kill_mode_series_is_already_shortened(self) -> None:
        files = self.write_inputs(seed=4)
        keep = np.ones(N_T, dtype=bool)
        keep[[0, 1, 17, 30]] = False
        save_nifti(files["bold"], self.bold[..., keep])
        prefix = self.tmp / "out"
        rc = self.run_volume(files, prefix, "--censor", str(self.write_censor(keep)), "--censor-mode", "KILL")
        self.assertEqual(rc, 0)
        series = read_tsv(f"{prefix}_timeseries.tsv").to_numpy()
        self.assertEqual(series.shape[0], N_T - 4)
        fc = read_tsv(f"{prefix}_connectivity.tsv").to_numpy()
        np.testing.assert_allclose(fc, np.corrcoef(series, rowvar=False), atol=1e-5)
        info = json.loads(Path(f"{prefix}_timeseries.json").read_text(encoding="utf-8"))
        self.assertEqual((info["n_volumes"], info["n_retained"]), (N_T - 4, N_T - 4))
        self.assertEqual(info["censor_handling"], "already_removed")

    def test_length_mismatch_is_an_error(self) -> None:
        files = self.write_inputs()
        keep = np.ones(N_T + 5, dtype=bool)
        keep[:2] = False
        rc = self.run_volume(files, self.tmp / "out", "--censor", str(self.write_censor(keep)))
        self.assertEqual(rc, 1)

    def test_no_connectivity_flag(self) -> None:
        files = self.write_inputs()
        prefix = self.tmp / "sub-1_task-rest_space-MNI_atlas-X_desc-preproc"
        self.assertEqual(self.run_volume(files, prefix, "--no-connectivity", "--strategy", "preproc"), 0)
        self.assertTrue(Path(f"{prefix}_timeseries.tsv").exists())
        self.assertTrue(Path(f"{prefix}_coverage.tsv").exists())
        self.assertFalse(Path(f"{prefix}_connectivity.tsv").exists())
        series = read_tsv(f"{prefix}_timeseries.tsv")
        self.assertGreater(float(series["roi_1"].mean()), 900.0)   # the mean level survives (needed for ROI tSNR)


def make_ptseries(path: Path, data: np.ndarray, names: list[str], parcels_first: bool = False) -> Path:
    axes = nib.cifti2.cifti2_axes
    n_vertices = 4 * len(names)
    parcels = []
    for k, name in enumerate(names):
        member = np.zeros(n_vertices, dtype=bool)
        member[4 * k: 4 * k + 4] = True
        parcels.append((name, axes.BrainModelAxis.from_mask(member, name="CIFTI_STRUCTURE_CORTEX_LEFT")))
    parcel_axis = axes.ParcelsAxis.from_brain_models(parcels)
    series_axis = axes.SeriesAxis(start=0.0, step=2.0, size=data.shape[0])
    if parcels_first:
        img = nib.Cifti2Image(data.T.astype(np.float32), header=(parcel_axis, series_axis))
    else:
        img = nib.Cifti2Image(data.astype(np.float32), header=(series_axis, parcel_axis))
    img.nifti_header.set_intent("ConnParcelSries")
    nib.save(img, str(path))
    return path


N_VERTICES = 20
LEFT = "CIFTI_STRUCTURE_CORTEX_LEFT"


def dense_axis(vertices: list[int], with_voxels: bool = False) -> "nib.cifti2.BrainModelAxis":
    axes = nib.cifti2.cifti2_axes
    member = np.zeros(N_VERTICES, dtype=bool)
    member[vertices] = True
    axis = axes.BrainModelAxis.from_mask(member, name=LEFT)
    if with_voxels:
        axis = axis + axes.BrainModelAxis.from_mask(np.ones((2, 1, 1), dtype=bool), affine=np.eye(4),
                                                    name="CIFTI_STRUCTURE_THALAMUS_LEFT")
    return axis


def make_dtseries(path: Path, data: np.ndarray, vertices: list[int], with_voxels: bool = False) -> Path:
    """data (T, G): one column per listed vertex (ascending), then two thalamus voxels."""
    series_axis = nib.cifti2.cifti2_axes.SeriesAxis(start=0.0, step=2.0, size=data.shape[0])
    img = nib.Cifti2Image(data.astype(np.float32), header=(series_axis, dense_axis(vertices, with_voxels)))
    img.nifti_header.set_intent("ConnDenseSeries")
    nib.save(img, str(path))
    return path


def make_dscalar(path: Path, values: list[float], vertices: list[int], with_voxels: bool = False) -> Path:
    """One map on the grayordinates of dense_axis(vertices, with_voxels)."""
    scalar_axis = nib.cifti2.cifti2_axes.ScalarAxis(["sampled"])
    img = nib.Cifti2Image(np.array([values], dtype=np.float32), header=(scalar_axis, dense_axis(vertices, with_voxels)))
    img.nifti_header.set_intent("ConnDenseScalar")
    nib.save(img, str(path))
    return path


def make_dlabel(path: Path, keys: dict[int, int], table: dict[int, str], with_voxels: bool = False,
                voxel_keys: tuple[int, int] = (0, 0)) -> Path:
    """keys: vertex -> label key, defined on every vertex (0 = unlabeled '???')."""
    values = [keys.get(v, 0) for v in range(N_VERTICES)] + (list(voxel_keys) if with_voxels else [])
    labels = {0: ("???", (1.0, 1.0, 1.0, 0.0))}
    labels.update({key: (name, (1.0, 0.0, 0.0, 1.0)) for key, name in table.items()})
    label_axis = nib.cifti2.cifti2_axes.LabelAxis(["parcels"], [labels])
    img = nib.Cifti2Image(np.array([values], dtype=np.float32),
                          header=(label_axis, dense_axis(list(range(N_VERTICES)), with_voxels)))
    img.nifti_header.set_intent("ConnDenseLabel")
    nib.save(img, str(path))
    return path


class CiftiCoverageTest(TempDirCase):
    """dlabel: P_a = vertices 0-3 (key 3), P_b = 4-7 (key 1), P_c = 8-11 + 2 voxels (key 2),
    P_lost = 16-19 (key 7, not in the dense series), P_unused (key 9, no grayordinate)."""

    VERTICES = list(range(12))
    TABLE = {3: "P_a", 1: "P_b", 2: "P_c", 7: "P_lost", 9: "P_unused"}
    ORDER = ["P_b", "P_c", "P_a", "P_lost", "P_unused"]          # ascending key

    def setUp(self) -> None:
        super().setUp()
        rng = np.random.default_rng(11)
        self.n_t = 25
        self.dense = rng.normal(1000.0, 10.0, size=(self.n_t, 14))
        self.dense[:, 4:7] = 0.0                   # P_b: 3 of 4 vertices outside the field of view
        self.dense[:, 8] = 0.0                     # P_c: 1 of 6 grayordinates without signal
        keys = {v: 3 for v in range(0, 4)} | {v: 1 for v in range(4, 8)} | {v: 2 for v in range(8, 12)} \
            | {v: 7 for v in range(16, 20)}
        self.dlabel = make_dlabel(self.tmp / "a.dlabel.nii", keys, self.TABLE, with_voxels=True, voxel_keys=(2, 2))
        self.dtseries = make_dtseries(self.tmp / "b.dtseries.nii", self.dense, self.VERTICES, with_voxels=True)
        # what wb_command -cifti-parcellate -legacy-mode returns: overlap means, key order, empty parcels dropped
        wb = np.column_stack([self.dense[:, 4:8].mean(axis=1), self.dense[:, 8:14].mean(axis=1), self.dense[:, 0:4].mean(axis=1)])
        self.ptseries = make_ptseries(self.tmp / "c.ptseries.nii", wb, ["P_b", "P_c", "P_a"])

    def run_cifti(self, prefix: Path, *extra: str) -> int:
        return timeseries.main(["cifti", "--ptseries", str(self.ptseries), "--dlabel", str(self.dlabel),
                                "--out-prefix", str(prefix), *extra])

    def test_dlabel_parcels_in_key_order(self) -> None:
        parcels, keys, axis = timeseries.load_dlabel(self.dlabel)
        self.assertEqual(parcels["roi"].tolist(), [1, 2, 3, 7, 9])
        self.assertEqual(parcels["name"].tolist(), self.ORDER)
        self.assertEqual(keys.size, N_VERTICES + 2)
        self.assertEqual(len(axis), N_VERTICES + 2)

    def test_dropped_parcels_are_restored_from_the_dlabel(self) -> None:
        prefix = self.tmp / "out"
        self.assertEqual(self.run_cifti(prefix), 0)
        series = read_tsv(f"{prefix}_timeseries.tsv")
        self.assertEqual(list(series.columns), self.ORDER)
        self.assertTrue(series["P_lost"].isna().all() and series["P_unused"].isna().all())
        self.assertFalse(series["P_b"].isna().any())           # no --dtseries: no coverage rule
        self.assertFalse(Path(f"{prefix}_coverage.tsv").exists())
        info = json.loads(Path(f"{prefix}_timeseries.json").read_text(encoding="utf-8"))
        self.assertEqual(info["roi_names"], self.ORDER)
        self.assertFalse(info["coverage_checked"])
        self.assertEqual(info["column_order"], "dlabel label keys, ascending")

    def test_coverage_rule_with_dense_series(self) -> None:
        labels = self.tmp / "labels.tsv"
        labels.write_text("index\tname\tnetwork\n1\t7Networks_b\tVis\n2\t7Networks_c\tDefault\n3\t7Networks_a\tVis\n", encoding="utf-8")
        prefix = self.tmp / "out"
        rc = self.run_cifti(prefix, "--dtseries", str(self.dtseries), "--labels", str(labels), "--min-coverage", "0.5")
        self.assertEqual(rc, 0)
        coverage = read_tsv(f"{prefix}_coverage.tsv")
        self.assertEqual(coverage["name"].tolist(), self.ORDER)
        self.assertEqual(coverage["roi"].tolist(), [1, 2, 3, 7, 9])
        self.assertEqual(coverage["network"].tolist()[:3], ["Vis", "Default", "Vis"])
        self.assertEqual(coverage["atlas_voxels"].tolist(), [4, 6, 4, 4, 0])
        self.assertEqual(coverage["data_voxels"].tolist(), [4, 6, 4, 0, 0])
        self.assertEqual(coverage["valid_voxels"].tolist(), [1, 5, 4, 0, 0])
        self.assertEqual([bool(v) for v in coverage["included"]], [False, True, True, False, False])

        series = read_tsv(f"{prefix}_timeseries.tsv")
        self.assertTrue(series["P_b"].isna().all())             # 1 of 4 valid < 0.5
        np.testing.assert_allclose(series["P_a"].to_numpy(), self.dense[:, 0:4].mean(axis=1), rtol=1e-5)
        # partly covered parcel: mean over the valid grayordinates, not diluted by the empty vertex
        np.testing.assert_allclose(series["P_c"].to_numpy(), self.dense[:, 9:14].mean(axis=1), rtol=1e-5)
        info = json.loads(Path(f"{prefix}_timeseries.json").read_text(encoding="utf-8"))
        self.assertTrue(info["coverage_checked"])
        self.assertEqual(info["recomputed_rois"], ["P_c"])
        self.assertLess(info["parcellate_max_abs_diff"], 1e-2)
        self.assertEqual(info["missing_rois"], ["P_b", "P_lost", "P_unused"])
        self.assertEqual(info["label_table_names"], ["7Networks_b", "7Networks_c", "7Networks_a"])
        fc = read_tsv(f"{prefix}_connectivity.tsv")
        self.assertEqual(list(fc.columns), self.ORDER)
        self.assertTrue(np.isfinite(fc.to_numpy()[1, 2]))

    def test_unsampled_grayordinates_do_not_count(self) -> None:
        # vertices 0 and 1 of P_a only hold a neighbour's copy (surface dilation):
        # they must neither count as covered nor enter the parcel mean
        mask = make_dscalar(self.tmp / "m.dscalar.nii", [0.0, 0.0] + [1.0] * 12, self.VERTICES, with_voxels=True)
        prefix = self.tmp / "out"
        rc = self.run_cifti(prefix, "--dtseries", str(self.dtseries), "--sampled-mask", str(mask), "--min-coverage", "0.5")
        self.assertEqual(rc, 0)
        coverage = read_tsv(f"{prefix}_coverage.tsv").set_index("name")
        self.assertEqual(int(coverage.loc["P_a", "valid_voxels"]), 2)
        self.assertAlmostEqual(float(coverage.loc["P_a", "coverage_fraction"]), 0.5)
        series = read_tsv(f"{prefix}_timeseries.tsv")
        np.testing.assert_allclose(series["P_a"].to_numpy(), self.dense[:, 2:4].mean(axis=1), rtol=1e-5)
        info = json.loads(Path(f"{prefix}_timeseries.json").read_text(encoding="utf-8"))
        self.assertIn("P_a", info["recomputed_rois"])
        self.assertEqual(info["unsampled_grayordinates"], 2)
        # a stricter coverage threshold turns the half-sampled parcel into n/a
        prefix = self.tmp / "out2"
        rc = self.run_cifti(prefix, "--dtseries", str(self.dtseries), "--sampled-mask", str(mask), "--min-coverage", "0.6")
        self.assertEqual(rc, 0)
        self.assertTrue(read_tsv(f"{prefix}_timeseries.tsv")["P_a"].isna().all())

    def test_sampled_mask_on_other_grayordinates_is_an_error(self) -> None:
        mask = make_dscalar(self.tmp / "m.dscalar.nii", [1.0] * 10, list(range(10)))   # no voxels, fewer vertices
        rc = self.run_cifti(self.tmp / "out", "--dtseries", str(self.dtseries), "--sampled-mask", str(mask))
        self.assertEqual(rc, 1)

    def test_no_connectivity_for_the_preproc_series(self) -> None:
        prefix = self.tmp / "sub-1_task-rest_space-fsLR_atlas-X_desc-preproc"
        self.assertEqual(self.run_cifti(prefix, "--dtseries", str(self.dtseries), "--no-connectivity"), 0)
        self.assertTrue(Path(f"{prefix}_timeseries.tsv").exists())
        self.assertFalse(Path(f"{prefix}_connectivity.tsv").exists())
        series = read_tsv(f"{prefix}_timeseries.tsv")
        self.assertGreater(float(series["P_a"].mean()), 900.0)

    def test_foreign_parcel_names_are_kept_as_they_are(self) -> None:
        self.ptseries = make_ptseries(self.tmp / "d.ptseries.nii", self.dense[:, :2], ["X_one", "X_two"])
        prefix = self.tmp / "out"
        self.assertEqual(self.run_cifti(prefix, "--dtseries", str(self.dtseries)), 0)
        self.assertEqual(list(read_tsv(f"{prefix}_timeseries.tsv").columns), ["X_one", "X_two"])
        info = json.loads(Path(f"{prefix}_timeseries.json").read_text(encoding="utf-8"))
        self.assertFalse(info["coverage_checked"])

    def test_frame_count_mismatch_and_missing_dlabel_are_errors(self) -> None:
        short = make_dtseries(self.tmp / "s.dtseries.nii", self.dense[:-3], self.VERTICES, with_voxels=True)
        self.assertEqual(self.run_cifti(self.tmp / "o1", "--dtseries", str(short)), 1)
        rc = timeseries.main(["cifti", "--ptseries", str(self.ptseries), "--dtseries", str(self.dtseries),
                              "--out-prefix", str(self.tmp / "o2")])
        self.assertEqual(rc, 1)


class CiftiCanonicalNamesTest(TempDirCase):
    """The volumetric label table and the dlabel of one Schaefer release can name
    the same parcel differently (CBIG renamed e.g. Default_PCC_1 -> Default_pCunPCC_1
    without changing keys or boundaries). Surface columns must then carry the
    label-table names, matched by atlas label key, so that the volume and surface
    streams share ROI identities; a key whose hemisphere or network differs is
    never renamed."""

    DLABEL = {1: "7Networks_LH_Vis_1", 2: "7Networks_LH_Default_pCunPCC_1", 3: "7Networks_RH_Default_pCunPCC_1"}
    LABELS = {1: ("7Networks_LH_Vis_1", "Vis"), 2: ("7Networks_LH_Default_PCC_1", "Default"),
              3: ("7Networks_RH_Default_PCC_1", "Default")}

    def setUp(self) -> None:
        super().setUp()
        rng = np.random.default_rng(5)
        self.dense = rng.normal(1000.0, 10.0, size=(25, 12))
        keys = {v: 1 for v in range(0, 4)} | {v: 2 for v in range(4, 8)} | {v: 3 for v in range(8, 12)}
        self.dlabel = make_dlabel(self.tmp / "a.dlabel.nii", keys, self.DLABEL)
        self.dtseries = make_dtseries(self.tmp / "b.dtseries.nii", self.dense, list(range(12)))
        means = np.column_stack([self.dense[:, 4 * k:4 * k + 4].mean(axis=1) for k in range(3)])
        self.ptseries = make_ptseries(self.tmp / "c.ptseries.nii", means, [self.DLABEL[k] for k in (1, 2, 3)])

    def write_labels(self, labels: dict[int, tuple[str, str]]) -> Path:
        path = self.tmp / "labels.tsv"
        rows = "".join(f"{k}\t{name}\t{net}\n" for k, (name, net) in labels.items())
        path.write_text("index\tname\tnetwork\n" + rows, encoding="utf-8")
        return path

    def run_cifti(self, labels: Path, prefix: Path) -> int:
        return timeseries.main(["cifti", "--ptseries", str(self.ptseries), "--dlabel", str(self.dlabel),
                                "--dtseries", str(self.dtseries), "--labels", str(labels), "--out-prefix", str(prefix)])

    def test_label_table_names_by_key(self) -> None:
        prefix = self.tmp / "out"
        self.assertEqual(self.run_cifti(self.write_labels(self.LABELS), prefix), 0)
        expected = [self.LABELS[k][0] for k in (1, 2, 3)]
        self.assertEqual(list(read_tsv(f"{prefix}_timeseries.tsv").columns), expected)
        self.assertEqual(list(read_tsv(f"{prefix}_connectivity.tsv").columns), expected)
        self.assertEqual(read_tsv(f"{prefix}_coverage.tsv")["name"].tolist(), expected)
        info = json.loads(Path(f"{prefix}_timeseries.json").read_text(encoding="utf-8"))
        self.assertEqual(info["roi_names"], expected)
        self.assertEqual(info["roi_names_source"], "label table, matched by atlas label key")
        self.assertEqual(info["renamed_from_dlabel"], {"7Networks_LH_Default_PCC_1": "7Networks_LH_Default_pCunPCC_1",
                                                       "7Networks_RH_Default_PCC_1": "7Networks_RH_Default_pCunPCC_1"})
        np.testing.assert_allclose(read_tsv(f"{prefix}_timeseries.tsv")[expected[1]].to_numpy(),
                                   self.dense[:, 4:8].mean(axis=1), rtol=1e-5)

    def test_hemisphere_or_network_conflict_keeps_dlabel_names(self) -> None:
        for bad in ({**self.LABELS, 3: ("7Networks_LH_Default_PCC_2", "Default")},       # hemisphere differs
                    {**self.LABELS, 2: ("7Networks_LH_Cont_PCC_1", "Cont")}):           # network differs
            prefix = self.tmp / f"out{len(list(self.tmp.glob('out*')))}"
            self.assertEqual(self.run_cifti(self.write_labels(bad), prefix), 0)
            self.assertEqual(list(read_tsv(f"{prefix}_timeseries.tsv").columns), [self.DLABEL[k] for k in (1, 2, 3)])
            info = json.loads(Path(f"{prefix}_timeseries.json").read_text(encoding="utf-8"))
            self.assertEqual(info["roi_names_source"], "dlabel")
            self.assertEqual(info["renamed_from_dlabel"], {})

    def test_different_key_sets_keep_dlabel_names(self) -> None:
        prefix = self.tmp / "out"
        labels = {k: v for k, v in self.LABELS.items() if k != 3}
        self.assertEqual(self.run_cifti(self.write_labels(labels), prefix), 0)
        self.assertEqual(list(read_tsv(f"{prefix}_timeseries.tsv").columns), [self.DLABEL[k] for k in (1, 2, 3)])


class CiftiTest(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        rng = np.random.default_rng(7)
        self.names = ["7Networks_LH_Vis_1", "7Networks_LH_Default_1", "7Networks_RH_Vis_1"]
        self.data = rng.normal(size=(30, 3))
        self.data[:, 2] = 0.0                     # empty parcel as filled by wb_command

    def test_ptseries_to_tsv_and_fc(self) -> None:
        pt = make_ptseries(self.tmp / "x.ptseries.nii", self.data, self.names)
        keep = np.ones(30, dtype=int)
        keep[[3, 4]] = 0
        censor = self.tmp / "censor.1D"
        np.savetxt(censor, keep, fmt="%d")
        prefix = self.tmp / "sub-1_task-rest_space-fsLR_atlas-Schaefer2018_100Parcels_7Networks_desc-acompcor"
        rc = timeseries.main(["cifti", "--ptseries", str(pt), "--censor", str(censor), "--out-prefix", str(prefix)])
        self.assertEqual(rc, 0)
        series = read_tsv(f"{prefix}_timeseries.tsv")
        self.assertEqual(list(series.columns), self.names)
        np.testing.assert_allclose(series.iloc[:, :2].to_numpy(), self.data[:, :2], rtol=1e-5, atol=1e-6)
        self.assertTrue(series[self.names[2]].isna().all())
        fc = read_tsv(f"{prefix}_connectivity.tsv").to_numpy()
        expected = np.corrcoef(self.data[keep > 0, :2], rowvar=False)[0, 1]
        self.assertAlmostEqual(fc[0, 1], expected, places=4)
        self.assertTrue(np.isnan(fc[2]).all())
        info = json.loads(Path(f"{prefix}_timeseries.json").read_text(encoding="utf-8"))
        self.assertEqual(info["atlas"], "Schaefer2018_100Parcels_7Networks")
        self.assertEqual(info["strategy"], "acompcor")
        self.assertEqual(info["n_retained"], 28)
        self.assertEqual(info["missing_rois"], [self.names[2]])

    def test_transposed_ptseries(self) -> None:
        pt = make_ptseries(self.tmp / "t.ptseries.nii", self.data, self.names, parcels_first=True)
        data, names = timeseries.load_ptseries(pt)
        self.assertEqual(data.shape, (30, 3))
        self.assertEqual(names, self.names)

    def test_label_table_never_renames_or_reorders_the_parcels(self) -> None:
        # the templateflow TSV may spell parcels differently: the dlabel names and order win
        pt = make_ptseries(self.tmp / "y.ptseries.nii", self.data[:, :2], [self.names[1], self.names[0]])
        labels = self.tmp / "labels.tsv"
        labels.write_text("index\tname\tnetwork\n" + "".join(f"{i + 1}\t{n}\tX\n" for i, n in enumerate(self.names)), encoding="utf-8")
        prefix = self.tmp / "out"
        rc = timeseries.main(["cifti", "--ptseries", str(pt), "--labels", str(labels), "--out-prefix", str(prefix)])
        self.assertEqual(rc, 0)
        series = read_tsv(f"{prefix}_timeseries.tsv")
        self.assertEqual(list(series.columns), [self.names[1], self.names[0]])
        np.testing.assert_allclose(series[self.names[0]].to_numpy(), self.data[:, 1], rtol=1e-5, atol=1e-6)
        info = json.loads(Path(f"{prefix}_timeseries.json").read_text(encoding="utf-8"))
        self.assertEqual(info["roi_names"], [self.names[1], self.names[0]])
        self.assertEqual(info["label_table_names"], self.names)
        self.assertEqual(info["column_order"], "parcels axis of the ptseries")

    def test_not_a_cifti_file(self) -> None:
        path = save_nifti(self.tmp / "plain.nii.gz", np.zeros(SHAPE, dtype=np.float32))
        self.assertEqual(timeseries.main(["cifti", "--ptseries", str(path), "--out-prefix", str(self.tmp / "o")]), 1)


class GridcheckTest(TempDirCase):
    def test_gridcheck_cli(self) -> None:
        a = save_nifti(self.tmp / "a.nii.gz", np.zeros(SHAPE, dtype=np.uint8))
        b = save_nifti(self.tmp / "b.nii.gz", np.zeros(SHAPE + (3,), dtype=np.float32))
        c = save_nifti(self.tmp / "c.nii.gz", np.zeros((4, 4, 3), dtype=np.uint8), np.diag([3.0, 3.0, 3.0, 1.0]))
        env = dict(os.environ, PYTHONPATH=str(REPO / "py"))
        for other, verdict in ((b, "same"), (c, "different")):
            out = subprocess.run(
                [sys.executable, "-m", "fmriproc.timeseries", "gridcheck", "--image", str(a), "--reference", str(other)],
                capture_output=True, text=True, env=env, check=True,
            )
            self.assertEqual(out.stdout.strip(), verdict)


# ----------------------------------------------------------------------------
# stage script with stub tools (POSIX hosts only: needs a native bash)
# ----------------------------------------------------------------------------

ANTS_STUB = """\
#!/bin/bash
# stub: nearest-neighbour label resampling; records its command line
echo "antsApplyTransforms $*" >> "$STUB_LOG"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i) in="$2"; shift 2 ;;
        -r) ref="$2"; shift 2 ;;
        -o) out="$2"; shift 2 ;;
        *) shift ;;
    esac
done
exec "$PYTHON_BIN" - "$in" "$ref" "$out" <<'PY'
import sys
import nibabel as nib
import numpy as np
src, ref = nib.load(sys.argv[1]), nib.load(sys.argv[2])
ijk = np.indices(ref.shape[:3]).reshape(3, -1)
xyz = nib.affines.apply_affine(ref.affine, ijk.T)
vox = np.rint(nib.affines.apply_affine(np.linalg.inv(src.affine), xyz)).astype(int)
inside = ((vox >= 0) & (vox < np.array(src.shape[:3]))).all(axis=1)
data = np.zeros(ijk.shape[1], dtype=np.int32)
source = np.asanyarray(src.dataobj)
data[inside] = source[tuple(vox[inside].T)]
nib.save(nib.Nifti1Image(data.reshape(ref.shape[:3]), ref.affine), sys.argv[3])
PY
"""

WB_STUB = """\
#!/bin/bash
# stub of -cifti-parcellate: fails without -legacy-mode, then writes the means of
# dense columns 0-3 (P_one) and 4-7 (P_two); the third dlabel parcel is dropped
echo "wb_command $*" >> "$STUB_LOG"
[[ " $* " == *" -legacy-mode "* ]] || { echo "ERROR: label file has vertices that are missing from the data" >&2; exit 1; }
exec "$PYTHON_BIN" - "$5" "$2" <<'PY'
import sys
import nibabel as nib
import numpy as np
axes = nib.cifti2.cifti2_axes
dense = np.asarray(nib.load(sys.argv[2]).dataobj, dtype=np.float64)
parcels = []
for k, name in enumerate(["P_one", "P_two"]):
    member = np.zeros(20, dtype=bool)
    member[4 * k: 4 * k + 4] = True
    parcels.append((name, axes.BrainModelAxis.from_mask(member, name="CIFTI_STRUCTURE_CORTEX_LEFT")))
data = np.column_stack([dense[:, 0:4].mean(axis=1), dense[:, 4:8].mean(axis=1)]).astype(np.float32)
img = nib.Cifti2Image(data, header=(axes.SeriesAxis(0.0, 2.0, data.shape[0]), axes.ParcelsAxis.from_brain_models(parcels)))
nib.save(img, sys.argv[1])
PY
"""


def native_bash() -> str | None:
    if os.name == "nt" and not os.environ.get("FMRIPROC_TEST_BASH"):
        return None
    return os.environ.get("FMRIPROC_TEST_BASH") or shutil.which("bash")


@unittest.skipUnless(native_bash(), "needs a native bash (set FMRIPROC_TEST_BASH=<path to bash> on Windows)")
class StageScriptTest(TempDirCase):
    SUB = "sub-0001"
    RUN = "sub-0001_task-rest"
    TPL = "MNI152NLin6Asym"
    ATLAS = "Schaefer2018_100Parcels_7Networks"

    def build_dataset(self) -> dict[str, str]:
        out = self.tmp / "out"
        func = out / "derivatives" / self.SUB / "func"
        func.mkdir(parents=True)
        (out / "rawdata").mkdir()
        (out / "rawdata" / "manifest.tsv").write_text(
            "subject\tsession\ttask\trun\tgroup\tbold\tt1w\trun_label\n"
            f"{self.SUB}\t-\trest\t-\tNYU\t/x/bold.nii.gz\t/x/t1.nii.gz\t{self.RUN}\n", encoding="utf-8")
        atlas = make_atlas()
        bold, _ = make_bold(atlas)
        stem = f"{self.RUN}_space-{self.TPL}_res-2"
        save_nifti(func / f"{stem}_boldref.nii.gz", bold.mean(axis=3))
        save_nifti(func / f"{stem}_desc-brain_mask.nii.gz", np.ones(SHAPE, dtype=np.uint8))
        save_nifti(func / f"{stem}_desc-preproc_bold.nii.gz", bold)
        keep = np.ones(N_T, dtype=int)
        keep[[4, 5]] = 0
        np.savetxt(func / f"{self.RUN}_desc-censor.1D", keep, fmt="%d")
        dense = np.random.default_rng(5).normal(1000.0, 10.0, size=(N_T, 12))
        for strategy in ("preproc", "wmcsf24", "wmcsf24gsr"):
            if strategy != "preproc":
                save_nifti(func / f"{stem}_desc-{strategy}_bold.nii.gz", bold - bold.mean(axis=3, keepdims=True))
            make_dtseries(func / f"{self.RUN}_space-fsLR_den-91k_desc-{strategy}_bold.dtseries.nii",
                          dense if strategy == "preproc" else dense - dense.mean(axis=0), list(range(12)))

        atlas_dir = out / "resources" / "atlases" / self.ATLAS
        atlas_dir.mkdir(parents=True)
        save_nifti(atlas_dir / f"{self.ATLAS}_space-{self.TPL}_res-02_dseg.nii.gz", atlas)
        (atlas_dir / "labels.tsv").write_text("index\tname\tnetwork\n1\tA\tVis\n2\tB\tVis\n5\tC\tDefault\n", encoding="utf-8")
        # third parcel on vertices that the dense series lacks: needs -legacy-mode
        keys = {v: 1 for v in range(0, 4)} | {v: 2 for v in range(4, 8)} | {v: 5 for v in range(16, 20)}
        make_dlabel(atlas_dir / f"{self.ATLAS}.dlabel.nii", keys, {1: "P_one", 2: "P_two", 5: "P_three"})

        # custom atlas on a finer grid (1 mm), no label table
        fine = np.kron(atlas, np.ones((2, 2, 2), dtype=np.int16))
        fine_affine = np.diag([1.0, 1.0, 1.0, 1.0])
        fine_affine[:3, 3] = -0.5
        custom = save_nifti(self.tmp / "custom_atlas.nii", fine.astype(np.float32), fine_affine)

        bindir = self.tmp / "bin"
        bindir.mkdir()
        for name, text in (("antsApplyTransforms", ANTS_STUB), ("wb_command", WB_STUB)):
            path = bindir / name
            path.write_text(textwrap.dedent(text), encoding="utf-8", newline="\n")
            path.chmod(0o755)
        return {
            "OUT_DIR": str(out),
            "PYTHON_BIN": sys.executable,
            "ATLASES": self.ATLAS,
            "CUSTOM_ATLASES": f"Mini={custom}",
            "DENOISE_STRATEGIES": "wmcsf24 wmcsf24gsr",
            "SURFACE": "yes",
            "MNI_RES": "2",
            "STUB_LOG": str(self.tmp / "stub.log"),
            "STUB_BIN": str(bindir),
        }

    def run_stage(self, settings: dict[str, str]) -> subprocess.CompletedProcess:
        env = dict(os.environ, **settings)
        env.pop("FMRIPROC_CONFIG", None)
        # lib/common.sh must provide the package path on its own (and a mixed
        # POSIX/Windows list would not be translated by MSYS bash)
        env.pop("PYTHONPATH", None)
        script = (REPO / "stages" / "07_timeseries.sh").as_posix()
        command = f'export PATH="$(cd "$STUB_BIN" && pwd):$PATH"; exec bash "{script}" {self.SUB}'
        return subprocess.run([native_bash(), "-c", command], env=env, capture_output=True, text=True)

    def test_stage_end_to_end_with_stubs(self) -> None:
        settings = self.build_dataset()
        result = self.run_stage(settings)
        self.assertEqual(result.returncode, 0, msg=result.stderr[-3000:])
        func = Path(settings["OUT_DIR"]) / "derivatives" / self.SUB / "func"
        for atlas in (self.ATLAS, "Mini"):
            for desc in ("preproc", "wmcsf24", "wmcsf24gsr"):
                prefix = func / f"{self.RUN}_space-{self.TPL}_atlas-{atlas}_desc-{desc}"
                self.assertTrue(Path(f"{prefix}_timeseries.tsv").exists(), msg=str(prefix))
                self.assertTrue(Path(f"{prefix}_coverage.tsv").exists())
                self.assertTrue(Path(f"{prefix}_timeseries.json").exists())
                self.assertEqual(Path(f"{prefix}_connectivity.tsv").exists(), desc != "preproc")
        named = read_tsv(func / f"{self.RUN}_space-{self.TPL}_atlas-{self.ATLAS}_desc-wmcsf24_timeseries.tsv")
        mini = read_tsv(func / f"{self.RUN}_space-{self.TPL}_atlas-Mini_desc-wmcsf24_timeseries.tsv")
        self.assertEqual(list(named.columns), ["A", "B", "C"])
        self.assertEqual(list(mini.columns), ["roi_1", "roi_2", "roi_5"])
        np.testing.assert_allclose(named.to_numpy(), mini.to_numpy(), rtol=1e-6, atol=1e-6)

        # CIFTI only for the atlas with a dlabel, after the -legacy-mode retry
        for desc in ("preproc", "wmcsf24", "wmcsf24gsr"):
            prefix = func / f"{self.RUN}_space-fsLR_atlas-{self.ATLAS}_desc-{desc}"
            table = read_tsv(f"{prefix}_timeseries.tsv")
            self.assertEqual(list(table.columns), ["P_one", "P_two", "P_three"])      # dlabel order, dropped parcel = n/a
            self.assertTrue(table["P_three"].isna().all())
            self.assertFalse(table["P_one"].isna().any())
            self.assertTrue(Path(f"{prefix}_coverage.tsv").exists())
            self.assertEqual(Path(f"{prefix}_connectivity.tsv").exists(), desc != "preproc")
        pre = read_tsv(func / f"{self.RUN}_space-fsLR_atlas-{self.ATLAS}_desc-preproc_timeseries.tsv")
        self.assertGreater(float(pre["P_one"].mean()), 900.0)
        self.assertFalse((func / f"{self.RUN}_space-fsLR_atlas-Mini_desc-wmcsf24_timeseries.tsv").exists())
        calls = Path(settings["STUB_LOG"]).read_text(encoding="utf-8").splitlines()
        self.assertEqual(sum(c.startswith("antsApplyTransforms") for c in calls), 1)   # only the off-grid atlas
        self.assertEqual(sum("-legacy-mode" in c for c in calls), 3)
        work = Path(settings["OUT_DIR"]) / "work" / self.SUB / "func" / self.RUN
        self.assertTrue((work / "atlas_Mini.nii.gz").exists())
        self.assertFalse((work / f"atlas_{self.ATLAS}.nii.gz").exists())
        marker = Path(settings["OUT_DIR"]) / "work" / self.SUB / ".done" / f"07_timeseries__{self.RUN}.hash"
        self.assertTrue(marker.exists())

        # unchanged parameters: second call skips the run
        again = self.run_stage(settings)
        self.assertEqual(again.returncode, 0, msg=again.stderr[-3000:])
        self.assertIn("up to date", again.stderr)

    def test_dry_run_prints_commands_and_writes_nothing(self) -> None:
        settings = self.build_dataset()
        settings["DRY_RUN"] = "yes"
        result = self.run_stage(settings)
        self.assertEqual(result.returncode, 0, msg=result.stderr[-3000:])
        self.assertIn("-cifti-parcellate", result.stderr)
        self.assertIn("-n GenericLabel", result.stderr)
        self.assertIn("desc-preproc_bold.dtseries.nii", result.stderr)
        func = Path(settings["OUT_DIR"]) / "derivatives" / self.SUB / "func"
        self.assertEqual(list(func.glob("*_timeseries.tsv")), [])
        self.assertFalse(Path(settings["STUB_LOG"]).exists())
        marker = Path(settings["OUT_DIR"]) / "work" / self.SUB / ".done" / f"07_timeseries__{self.RUN}.hash"
        self.assertFalse(marker.exists())

    def test_missing_atlas_points_to_fetch_resources(self) -> None:
        settings = self.build_dataset()
        settings["ATLASES"] = "Schaefer2018_400Parcels_7Networks"
        result = self.run_stage(settings)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fetch_resources.sh", result.stderr)

    def test_failing_run_fails_the_stage(self) -> None:
        settings = self.build_dataset()
        func = Path(settings["OUT_DIR"]) / "derivatives" / self.SUB / "func"
        (func / f"{self.RUN}_space-{self.TPL}_res-2_desc-wmcsf24gsr_bold.nii.gz").unlink()
        result = self.run_stage(settings)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("1 of 1 run(s) failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
