"""Tests of fmriproc.validate and fmriproc.compare_streams (stage 10) on synthetic ROI tables."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import pandas as pd

from fmriproc import compare_streams, validate
from fmriproc.utils import fc_matrix, fisher_z, read_tsv, write_tsv

REPO = Path(__file__).resolve().parents[1]
TPL = "MNI152NLin6Asym"
ATLAS = "Schaefer2018_100Parcels_7Networks"
NETWORKS = ("Vis", "SomMot", "Default", "Cont")
N_T = 160
TR = 2.0


# ----------------------------------------------------------------------------
# synthetic data
# ----------------------------------------------------------------------------

def roi_layout() -> pd.DataFrame:
    """16 parcels: 4 networks x 2 parcels x 2 hemispheres; RH parcel k mirrors LH parcel k.

    The Default parcels are one PFC and one pCunPCC parcel per hemisphere.
    """
    rows = []
    index = 1
    for hemi, sign in (("LH", -1.0), ("RH", 1.0)):
        for n, network in enumerate(NETWORKS):
            for k in range(2):
                part = {("Default", 0): "PFC_1", ("Default", 1): "pCunPCC_1"}.get((network, k), str(k + 1))
                rows.append({
                    "index": index, "name": f"7Networks_{hemi}_{network}_{part}", "network": network, "pair": f"{network}_{k}",
                    "hemi": hemi[0], "x": sign * (12.0 + 8.0 * k), "y": -60.0 + 30.0 * n, "z": 10.0 + 6.0 * k,
                })
                index += 1
    return pd.DataFrame(rows)


def planted_series(layout: pd.DataFrame, n_t: int = N_T, seed: int = 0, noise: float = 1.0) -> np.ndarray:
    """Network signal (within > between) + a signal shared by each homotopic pair + noise."""
    rng = np.random.default_rng(seed)
    network_signal = {net: rng.normal(size=n_t) for net in NETWORKS}
    pair_signal: dict[str, np.ndarray] = {}
    series = np.empty((n_t, len(layout)))
    for col, row in enumerate(layout.itertuples(index=False)):
        if row.pair not in pair_signal:                     # same network and parcel part in both hemispheres
            pair_signal[row.pair] = rng.normal(size=n_t)
        series[:, col] = 1.2 * network_signal[row.network] + 1.0 * pair_signal[row.pair] + noise * rng.normal(size=n_t)
    return series


def to_pre(post: np.ndarray, level: float = 10000.0, seed: int = 1) -> np.ndarray:
    """Pre-denoise series: signal level + the final series + removed nuisance variance."""
    rng = np.random.default_rng(seed)
    nuisance = rng.normal(size=(post.shape[0], 1)) * 30.0
    return level + 10.0 * post + nuisance


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="fmriproc_validate_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


def write_atlas_dir(root: Path, layout: pd.DataFrame, atlas: str = ATLAS) -> Path:
    """labels.tsv + a 2 mm dseg volume whose parcel centroids are the layout coordinates."""
    base = root / atlas
    base.mkdir(parents=True)
    layout[["index", "name", "network"]].to_csv(base / "labels.tsv", sep="\t", index=False, lineterminator="\n")
    affine = np.array([[2.0, 0, 0, -40.0], [0, 2.0, 0, -80.0], [0, 0, 2.0, 0.0], [0, 0, 0, 1.0]])
    data = np.zeros((41, 61, 21), dtype=np.int16)
    for row in layout.itertuples(index=False):
        i, j, k = np.rint(nib.affines.apply_affine(np.linalg.inv(affine), [row.x, row.y, row.z])).astype(int)
        data[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2] = row.index
    nib.save(nib.Nifti1Image(data, affine), str(base / f"{atlas}_space-{TPL}_res-02_dseg.nii.gz"))
    return root


def write_run(
    func: Path, run: str, layout: pd.DataFrame, strategies: tuple[str, ...] = ("wmcsf24",), surface: bool = True,
    keep: np.ndarray | None = None, kill: bool = False, seed: int = 0, nan_volume: tuple[int, ...] = (),
    nan_surface: tuple[int, ...] = (), surface_pre: bool = True, atlas: str = ATLAS,
) -> None:
    """Stage-07/04/05 outputs of one run, as far as stage 10 reads them."""
    func.mkdir(parents=True, exist_ok=True)
    keep = np.ones(N_T, dtype=bool) if keep is None else keep
    np.savetxt(func / f"{run}_desc-censor.1D", keep.astype(int), fmt="%d")
    rng = np.random.default_rng(seed + 100)
    fd = np.abs(rng.normal(0.15, 0.05, size=N_T))
    fd[~keep] = 0.9
    fd_column = pd.DataFrame({"trans_x": rng.normal(size=N_T), "framewise_displacement": fd})
    fd_column.loc[0, "framewise_displacement"] = np.nan
    write_tsv(func / f"{run}_desc-confounds_timeseries.tsv", fd_column)
    names = layout["name"].tolist()
    for s, strategy in enumerate(strategies):
        (func / f"{run}_desc-{strategy}_denoise.json").write_text(
            json.dumps({"strategy": strategy, "polort": 2, "dof_remaining": 40 - 5 * s}), encoding="utf-8")
        post = planted_series(layout, seed=seed)
        post = post - post[keep].mean(axis=0)
        # the synthetic surface stream is the less noisy one
        streams = [("volume", TPL, nan_volume, 0.9)]
        if surface:
            streams.append(("surface", "fsLR", nan_surface, 0.3))
        for stream, space, nan_cols, noise_sd in streams:
            extra = np.random.default_rng(1000 * seed + 7 + len(stream)).normal(size=post.shape) * noise_sd
            final = post + extra
            pre = to_pre(final)
            final[:, list(nan_cols)] = np.nan
            pre[:, list(nan_cols)] = np.nan
            stem = func / f"{run}_space-{space}_atlas-{atlas}"
            rows = final[keep] if kill else final
            write_tsv(f"{stem}_desc-{strategy}_timeseries.tsv", pd.DataFrame(rows, columns=names), float_format="%.8g")
            write_tsv(f"{stem}_desc-{strategy}_connectivity.tsv", pd.DataFrame(fc_matrix(final, keep), columns=names))
            if stream == "volume" or surface_pre:
                write_tsv(f"{stem}_desc-preproc_timeseries.tsv", pd.DataFrame(pre, columns=names), float_format="%.8g")
            if stream == "volume":
                coverage = pd.DataFrame({"roi": layout["index"], "name": names, "network": layout["network"],
                                         "coverage_fraction": 1.0, "included": True})
                write_tsv(f"{stem}_desc-{strategy}_coverage.tsv", coverage)


def cli(func: Path, run: str, atlas_dir: Path | str, strategies: str = "wmcsf24", atlases: str = ATLAS,
        censor_mode: str = "NTRP") -> int:
    return validate.main([
        "--func-dir", str(func), "--run", run, "--template", TPL, "--strategies", strategies, "--atlases", atlases,
        "--atlas-dir", str(atlas_dir), "--tr", str(TR), "--censor-mode", censor_mode,
        "--out-tsv", str(func / f"{run}_desc-validation.tsv"), "--out-json", str(func / f"{run}_desc-validation.json"),
        "--out-compare-tsv", str(func / f"{run}_desc-streamcompare.tsv"),
    ])


def metric_value(table: pd.DataFrame, stream: str, metric: str, strategy: str = "wmcsf24") -> float:
    rows = table[(table["stream"] == stream) & (table["metric"] == metric) & (table["strategy"] == strategy)]
    assert len(rows) == 1, (stream, metric, len(rows))
    return float(rows["value"].iloc[0])


# ----------------------------------------------------------------------------
# pure metric functions
# ----------------------------------------------------------------------------

class SignalLevelTest(unittest.TestCase):
    def test_roi_tsnr_is_pre_mean_over_post_sd(self) -> None:
        rng = np.random.default_rng(0)
        post = rng.normal(0.0, 50.0, size=(4000, 3))
        pre = 10000.0 + rng.normal(0.0, 300.0, size=(4000, 3))
        keep = np.ones(4000, dtype=bool)
        np.testing.assert_allclose(validate.roi_tsnr(pre, post, keep, keep), 200.0, rtol=0.05)

    def test_roi_tsnr_ignores_censored_frames(self) -> None:
        rng = np.random.default_rng(1)
        post = rng.normal(0.0, 10.0, size=(300, 2))
        pre = 1000.0 + post
        keep = np.ones(300, dtype=bool)
        keep[10:20] = False
        spiky = post.copy()
        spiky[10:20] += 500.0
        np.testing.assert_allclose(validate.roi_tsnr(pre, spiky, keep, keep), validate.roi_tsnr(pre, post, keep, keep))

    def test_variance_removed(self) -> None:
        rng = np.random.default_rng(2)
        n_t = 400
        signal = rng.normal(size=(n_t, 4))
        trend = np.linspace(-1, 1, n_t)[:, None] ** 2 * 50.0
        keep = np.ones(n_t, dtype=bool)
        detrended = validate.detrend(signal, keep, 2)
        removed = validate.variance_removed(1000.0 + signal + trend, 0.5 * detrended, keep, keep, polort=2)
        np.testing.assert_allclose(removed, 0.75, atol=1e-6)

    def test_gs_residual_sd_between_independent_and_global(self) -> None:
        rng = np.random.default_rng(3)
        independent = rng.normal(size=(2000, 100))
        keep = np.ones(2000, dtype=bool)
        self.assertAlmostEqual(validate.gs_residual_sd(independent, keep), 0.1, delta=0.01)
        shared = independent + 2.0 * rng.normal(size=(2000, 1))
        self.assertGreater(validate.gs_residual_sd(shared, keep), 0.8)

    def test_gs_residual_sd_ignores_constant_columns(self) -> None:
        rng = np.random.default_rng(7)
        series = rng.normal(size=(500, 20))
        keep = np.ones(500, dtype=bool)
        with_constant = np.column_stack([series, np.full(500, 3.0)])
        self.assertAlmostEqual(validate.gs_residual_sd(with_constant, keep), validate.gs_residual_sd(series, keep), places=12)
        self.assertTrue(np.isnan(validate.gs_residual_sd(np.full((500, 3), 1.0), keep)))


class SpectrumTest(unittest.TestCase):
    def test_sinusoids_inside_and_outside_the_band(self) -> None:
        t = np.arange(200) * TR
        slow = np.sin(2 * np.pi * 0.05 * t)[:, None] + 1000.0
        fast = np.sin(2 * np.pi * 0.2 * t)[:, None] + 1000.0
        self.assertGreater(validate.lowfreq_power_fraction(slow, TR), 0.95)
        self.assertLess(validate.lowfreq_power_fraction(fast, TR), 0.05)

    def test_white_noise_baseline_with_and_without_censoring(self) -> None:
        rng = np.random.default_rng(4)
        noise = rng.normal(size=(300, 60))
        baseline = (0.1 - 0.01) / (0.25 - 0.01)
        self.assertAlmostEqual(validate.lowfreq_power_fraction(noise, TR), baseline, delta=0.04)
        keep = np.ones(300, dtype=bool)
        keep[rng.choice(300, size=60, replace=False)] = False
        self.assertAlmostEqual(validate.lowfreq_power_fraction(noise, TR, keep), baseline, delta=0.05)

    def test_censored_spikes_do_not_leak_into_the_spectrum(self) -> None:
        t = np.arange(200) * TR
        slow = np.sin(2 * np.pi * 0.05 * t)[:, None] + 1000.0
        keep = np.ones(200, dtype=bool)
        keep[50:56] = False
        spiky = slow.copy()
        spiky[50:56] += np.array([40.0, -60.0, 80.0, -70.0, 50.0, -30.0])[:, None]
        self.assertGreater(validate.lowfreq_power_fraction(spiky, TR, keep), 0.95)
        self.assertLess(validate.lowfreq_power_fraction(spiky, TR), 0.9)

    def test_undefined_cases(self) -> None:
        noise = np.random.default_rng(5).normal(size=(100, 3))
        self.assertTrue(np.isnan(validate.lowfreq_power_fraction(noise, 0.0)))
        self.assertTrue(np.isnan(validate.lowfreq_power_fraction(noise, 6.0)))     # Nyquist below 0.1 Hz


class NetworkStructureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = roi_layout()
        self.series = planted_series(self.layout, n_t=600)
        self.fc_z = fisher_z(fc_matrix(self.series))

    def test_network_contrast_detects_planted_structure(self) -> None:
        networks = self.layout["network"].tolist()
        planted = validate.network_contrast(self.fc_z, networks)
        self.assertGreater(planted, 2.0)
        shuffled = list(np.random.default_rng(0).permutation(networks))
        self.assertLess(abs(validate.network_contrast(self.fc_z, shuffled)), planted / 3)

    def test_network_contrast_needs_labels(self) -> None:
        self.assertTrue(np.isnan(validate.network_contrast(self.fc_z, [""] * len(self.layout))))
        self.assertTrue(np.isnan(validate.network_contrast(self.fc_z, ["Vis"] * len(self.layout))))

    def test_network_contrast_ignores_nan_rois(self) -> None:
        fc_z = self.fc_z.copy()
        fc_z[3, :] = fc_z[:, 3] = np.nan
        self.assertGreater(validate.network_contrast(fc_z, self.layout["network"].tolist()), 2.0)

    def test_dmn_contrast(self) -> None:
        names, networks = self.layout["name"].tolist(), self.layout["network"].tolist()
        self.assertGreater(validate.dmn_contrast(self.fc_z, names, networks), 0.3)
        # a Cont parcel called PFC must not count: only Default parcels are DMN nodes
        renamed = [n.replace("Default", "Cont") for n in names]
        self.assertTrue(np.isnan(validate.dmn_contrast(self.fc_z, renamed, ["Cont" if n == "Default" else n for n in networks])))
        self.assertTrue(np.isnan(validate.dmn_contrast(self.fc_z, [f"roi_{k}" for k in range(len(names))], [""] * len(names))))

    def test_17_network_names(self) -> None:
        names = ["17Networks_LH_DefaultA_PFCd_1", "17Networks_RH_DefaultA_pCunPCC_1", "17Networks_LH_SomMotA_1", "17Networks_RH_SomMotB_S2_1"]
        networks = [validate.network_of(n) for n in names]
        self.assertEqual(networks, ["DefaultA", "DefaultA", "SomMotA", "SomMotB"])
        fc_z = np.array([[np.inf, 0.8, 0.1, 0.0], [0.8, np.inf, 0.2, 0.1], [0.1, 0.2, np.inf, 0.5], [0.0, 0.1, 0.5, np.inf]])
        self.assertAlmostEqual(validate.dmn_contrast(fc_z, names, networks), 0.8 - 0.1, places=6)


class NameParsingTest(unittest.TestCase):
    def test_schaefer_7_and_17_network_names(self) -> None:
        cases = {
            "7Networks_LH_Vis_1": ("L", "Vis"),
            "7Networks_RH_DorsAttn_Post_3": ("R", "DorsAttn"),
            "7Networks_LH_Default_pCunPCC_2": ("L", "Default"),
            "17Networks_LH_VisCent_ExStr_1": ("L", "VisCent"),
            "17Networks_RH_SomMotB_S2_1": ("R", "SomMotB"),
        }
        for name, (hemi, network) in cases.items():
            self.assertEqual(validate.hemisphere_of(name), hemi, name)
            self.assertEqual(validate.network_of(name), network, name)
        for name in ("roi_12", "Left-Thalamus", "LHVis", "7Networks_Vis_1", ""):
            self.assertEqual(validate.hemisphere_of(name), "", name)
            self.assertEqual(validate.network_of(name), "", name)


class HomotopicTest(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = roi_layout()
        self.centroids = self.layout[["x", "y", "z"]].to_numpy()
        self.hemis = self.layout["hemi"].tolist()
        self.networks = self.layout["network"].tolist()

    def test_pairs_are_the_mirrored_parcels(self) -> None:
        pairs = validate.find_homotopic_pairs(self.hemis, self.networks, self.centroids)
        self.assertEqual(pairs, [(k, k + 8) for k in range(8)])

    def test_distance_limit_network_rule_and_unknown_centroids(self) -> None:
        centroids = self.centroids.copy()
        centroids[8] += np.array([0.0, 0.0, 45.0])         # mirrored partner of parcel 0 moved far away
        centroids[9] = np.nan                               # partner of parcel 1 unknown
        networks = list(self.networks)
        networks[10] = "Limbic"                             # partner of parcel 2 now in another network
        pairs = dict(validate.find_homotopic_pairs(self.hemis, networks, centroids, max_mm=20.0))
        self.assertNotIn(0, pairs)                          # nearest same-network RH parcel is > 20 mm away
        self.assertNotIn(1, pairs)
        self.assertNotIn(8, pairs.values())
        self.assertNotIn(9, pairs.values())
        self.assertEqual(pairs[2], 11)                      # next-nearest parcel of the same network, 10 mm off
        self.assertEqual(pairs[4], 12)

    def test_homotopic_contrast_detects_planted_pairs(self) -> None:
        fc_z = fisher_z(fc_matrix(planted_series(self.layout, n_t=600)))
        pairs = validate.find_homotopic_pairs(self.hemis, self.networks, self.centroids)
        self.assertGreater(validate.homotopic_contrast(fc_z, self.hemis, pairs), 0.3)
        wrong = [(left, 8 + (left + 4) % 8) for left in range(8)]
        self.assertLess(validate.homotopic_contrast(fc_z, self.hemis, wrong), 0.0)

    def test_without_hemispheres_is_nan(self) -> None:
        fc_z = fisher_z(fc_matrix(planted_series(self.layout)))
        self.assertTrue(np.isnan(validate.homotopic_contrast(fc_z, [""] * 16, [])))


class CentroidTest(TempDirCase):
    def test_centroids_in_world_coordinates_and_cache(self) -> None:
        layout = roi_layout()
        root = write_atlas_dir(self.tmp / "atlases", layout)
        table = validate.atlas_centroids(root, ATLAS)
        merged = table.merge(layout, on="index", suffixes=("", "_true"))
        self.assertEqual(len(merged), 16)
        np.testing.assert_allclose(merged[["x", "y", "z"]].to_numpy(), merged[["x_true", "y_true", "z_true"]].to_numpy(), atol=1.01)
        cache = root / ATLAS / "centroids.tsv"
        self.assertTrue(cache.is_file())
        with mock.patch.object(validate, "compute_centroids", side_effect=AssertionError("cache not used")):
            again = validate.atlas_centroids(root, ATLAS)
        np.testing.assert_allclose(again[["x", "y", "z"]].to_numpy(), table[["x", "y", "z"]].to_numpy(), atol=1e-3)

    def test_read_only_resource_dir_is_tolerated(self) -> None:
        root = write_atlas_dir(self.tmp / "atlases", roi_layout())
        with mock.patch.object(validate, "write_tsv", side_effect=PermissionError("read-only file system")):
            table = validate.atlas_centroids(root, ATLAS)
        self.assertEqual(len(table), 16)
        self.assertFalse((root / ATLAS / "centroids.tsv").exists())

    def test_missing_atlas_volume(self) -> None:
        self.assertIsNone(validate.atlas_centroids(self.tmp, "Yeo100"))

    def test_flipped_affine_gives_world_coordinates(self) -> None:
        # radiological storage (voxel i grows towards the left): world x must come from the affine
        affine = np.array([[-2.0, 0, 0, 40.0], [0, 2.0, 0, -80.0], [0, 0, 2.0, 0.0], [0, 0, 0, 1.0]])
        data = np.zeros((41, 61, 21), dtype=np.int16)
        data[5:8, 10:13, 3:6] = 1       # voxel centre (6, 11, 4)
        data[30:33, 10:13, 3:6] = 2     # voxel centre (31, 11, 4)
        table = validate.compute_centroids(nib.Nifti1Image(data, affine)).set_index("index")
        np.testing.assert_allclose(table.loc[1, ["x", "y", "z"]].to_numpy(dtype=float), [28.0, -58.0, 8.0], atol=1e-9)
        np.testing.assert_allclose(table.loc[2, ["x", "y", "z"]].to_numpy(dtype=float), [-22.0, -58.0, 8.0], atol=1e-9)

    def test_unreadable_atlas_volume_is_tolerated(self) -> None:
        root = write_atlas_dir(self.tmp / "atlases", roi_layout())
        (root / ATLAS / f"{ATLAS}_space-{TPL}_res-02_dseg.nii.gz").write_bytes(b"not a nifti")
        self.assertIsNone(validate.atlas_centroids(root, ATLAS))
        self.assertFalse((root / ATLAS / "centroids.tsv").exists())


class RoiTableTest(unittest.TestCase):
    def test_labels_by_name_by_order_and_from_column_names(self) -> None:
        layout = roi_layout()
        labels = layout[["index", "name", "network"]].copy()
        by_name = validate.roi_table(layout["name"].tolist()[::-1], labels)
        self.assertEqual(by_name["index"].tolist(), layout["index"].tolist()[::-1])
        self.assertEqual(by_name["hemi"].tolist(), layout["hemi"].tolist()[::-1])
        # Different names cannot be mapped by position.
        by_order = validate.roi_table([f"parcel {k}" for k in range(16)], labels)
        self.assertEqual(by_order["index"].tolist(), [-1] * 16)
        self.assertEqual(by_order["network"].tolist(), [""] * 16)
        self.assertEqual(by_order["hemi"].tolist(), [""] * 16)
        no_labels = validate.roi_table(layout["name"].tolist(), None)
        self.assertEqual(no_labels["network"].tolist(), layout["network"].tolist())
        self.assertTrue((no_labels["index"] == -1).all())
        custom = validate.roi_table(["roi_1", "roi_2"], None)
        self.assertEqual(custom["network"].tolist(), ["", ""])
        self.assertEqual(custom["hemi"].tolist(), ["", ""])


class MotionCouplingTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(6)
        self.n_t = 240
        self.clean = planted_series(roi_layout(), n_t=self.n_t, seed=3)
        self.fd = 0.04 + rng.exponential(0.08, size=self.n_t)        # right-skewed, like real FD
        self.fd[0] = np.nan
        self.keep = np.ones(self.n_t, dtype=bool)
        # motion-coupled artefact: every ROI jumps in proportion to FD; the direction of
        # the jump changes from frame to frame, its size across ROIs does not
        gain = rng.choice([-1.0, 1.0], size=self.clean.shape[1]) * rng.uniform(15.0, 25.0, size=self.clean.shape[1])
        direction = rng.choice([-1.0, 1.0], size=(self.n_t, 1))
        self.dirty = self.clean + np.nan_to_num(self.fd)[:, None] * gain[None, :] * direction

    def test_artefact_raises_the_coupling(self) -> None:
        clean = validate.fd_fc_coupling(self.clean, self.keep, self.fd)
        dirty = validate.fd_fc_coupling(self.dirty, self.keep, self.fd)
        self.assertLess(clean, 0.2)
        self.assertGreater(dirty, 0.5)

    def test_cofluctuation_rss_equals_the_explicit_edge_time_series(self) -> None:
        z = validate.zscore_columns(self.clean[:30, :6])
        rows, cols = np.triu_indices(6, k=1)
        explicit = np.sqrt(((z[:, rows] * z[:, cols]) ** 2).sum(axis=1))
        np.testing.assert_allclose(validate.cofluctuation_rss(z), explicit, rtol=1e-10)

    def test_kill_mode_series_gives_the_same_value(self) -> None:
        keep = self.keep.copy()
        keep[40:52] = False
        full = validate.fd_fc_coupling(self.dirty, keep, self.fd)
        keep_rows, frames = validate.resolve_frames(int(keep.sum()), keep)
        self.assertTrue(keep_rows.all())
        np.testing.assert_array_equal(frames, np.flatnonzero(keep))
        killed = validate.fd_fc_coupling(self.dirty[keep], keep_rows, self.fd[frames])
        self.assertAlmostEqual(full, killed, places=12)

    def test_fd_length_mismatch_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            validate.fd_fc_coupling(self.clean, self.keep, self.fd[:-5])

    def test_no_fd_is_nan(self) -> None:
        self.assertTrue(np.isnan(validate.fd_fc_coupling(self.clean, self.keep, None)))

    def test_constant_fd_is_nan(self) -> None:
        self.assertTrue(np.isnan(validate.fd_fc_coupling(self.dirty, self.keep, np.full(self.n_t, 0.2))))
        self.assertTrue(np.isnan(validate.fd_fc_coupling(np.full(self.dirty.shape, 1.0), self.keep, self.fd)))

    def test_constant_roi_column_does_not_poison_the_value(self) -> None:
        with_constant = self.dirty.copy()
        with_constant[:, 2] = 7.0
        mask = np.ones(self.dirty.shape[1], dtype=bool)
        mask[2] = False
        value = validate.fd_fc_coupling(with_constant, self.keep, self.fd)
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(value, validate.fd_fc_coupling(self.dirty[:, mask], self.keep, self.fd), places=12)


class ResolveFramesTest(unittest.TestCase):
    def test_rules(self) -> None:
        keep = np.array([1, 1, 0, 1, 0, 1], dtype=bool)
        rows, frames = validate.resolve_frames(6, keep)
        np.testing.assert_array_equal(rows, keep)
        np.testing.assert_array_equal(frames, np.arange(6))
        rows, frames = validate.resolve_frames(4, keep)
        np.testing.assert_array_equal(rows, np.ones(4, dtype=bool))
        np.testing.assert_array_equal(frames, [0, 1, 3, 5])
        rows, frames = validate.resolve_frames(5, None)
        self.assertTrue(rows.all())
        with self.assertRaises(ValueError):
            validate.resolve_frames(5, keep)


class StreamComparisonTest(unittest.TestCase):
    def make(self, post: np.ndarray, layout: pd.DataFrame) -> validate.StreamData:
        keep = np.ones(post.shape[0], dtype=bool)
        table = validate.roi_table(layout["name"].tolist(), layout[["index", "name", "network"]])
        return validate.StreamData(post, to_pre(np.nan_to_num(post)) + 0 * post, keep, keep, None, table,
                                   layout[["x", "y", "z"]].to_numpy())

    def test_common_rois_only(self) -> None:
        layout = roi_layout()
        base = planted_series(layout, n_t=300)
        volume, surface = base.copy(), base + np.random.default_rng(9).normal(size=base.shape) * 0.2
        volume[:, 3] = np.nan
        surface[:, 7] = np.nan
        vol, surf = self.make(volume, layout), self.make(surface, layout)
        own = {"volume": validate.stream_metrics(vol, TR), "surface": validate.stream_metrics(surf, TR)}
        self.assertEqual(own["volume"]["n_roi_nan"], 1)
        result = validate.compare_streams(vol, surf, TR, per_stream=own)
        self.assertEqual(result["n_roi_common"], 14)
        self.assertGreater(result["fc_similarity"], 0.9)
        self.assertEqual(result["metrics"]["volume"]["n_roi_nan"], 1)
        self.assertEqual(result["metrics"]["surface"]["n_roi_nan"], 1)
        # the paired values come from the same 14 ROIs in both streams
        mask = np.ones(16, dtype=bool)
        mask[[3, 7]] = False
        expected = validate.stream_metrics(self.make(base[:, mask], layout[mask].reset_index(drop=True)), TR)
        self.assertAlmostEqual(result["metrics"]["volume"]["network_contrast"], expected["network_contrast"], places=10)
        self.assertAlmostEqual(result["metrics"]["volume"]["homotopic_contrast"], expected["homotopic_contrast"], places=10)
        roi = result["roi_tsnr"]
        self.assertEqual(len(roi), 16)
        self.assertTrue(np.isnan(roi.loc[3, "volume"]) and np.isfinite(roi.loc[3, "surface"]))

    def test_different_column_counts_are_not_compared(self) -> None:
        layout = roi_layout()
        base = planted_series(layout)
        short = layout.iloc[:12].reset_index(drop=True)
        with self.assertRaises(ValueError):
            validate.compare_streams(self.make(base, layout), self.make(base[:, :12], short), TR)

    def test_constant_and_nan_rois_are_dropped_before_fc(self) -> None:
        layout = roi_layout()
        post = planted_series(layout, n_t=300)
        post[:, 0] = 5.0
        post[10, 1] = np.nan
        metrics = validate.stream_metrics(self.make(post, layout), TR)
        self.assertEqual(metrics["n_roi_nan"], 2)
        self.assertTrue(np.isfinite(metrics["network_contrast"]))
        self.assertTrue(np.isfinite(metrics["split_half_r"]))


# ----------------------------------------------------------------------------
# per-run CLI
# ----------------------------------------------------------------------------

class RunCliTest(TempDirCase):
    RUN = "sub-0001_task-rest"

    def setUp(self) -> None:
        super().setUp()
        self.layout = roi_layout()
        self.atlas_dir = write_atlas_dir(self.tmp / "atlases", self.layout)
        self.func = self.tmp / "derivatives" / "sub-0001" / "func"

    def test_both_streams(self) -> None:
        keep = np.ones(N_T, dtype=bool)
        keep[[20, 21, 90]] = False
        write_run(self.func, self.RUN, self.layout, strategies=("wmcsf24", "wmcsf24gsr"), keep=keep, nan_volume=(5,))
        self.assertEqual(cli(self.func, self.RUN, self.atlas_dir, strategies="wmcsf24 wmcsf24gsr"), 0)

        table = read_tsv(self.func / f"{self.RUN}_desc-validation.tsv")
        self.assertEqual(list(table.columns), ["stream", "strategy", "atlas", "metric", "value"])
        self.assertEqual(len(table), 2 * 2 * len(validate.METRICS))
        self.assertEqual(metric_value(table, "volume", "n_retained"), N_T - 3)
        self.assertEqual(metric_value(table, "volume", "n_roi_nan"), 1)
        self.assertEqual(metric_value(table, "surface", "n_roi_nan"), 0)
        self.assertEqual(metric_value(table, "volume", "dof_remaining"), 40)
        self.assertEqual(metric_value(table, "volume", "dof_remaining", "wmcsf24gsr"), 35)
        for metric in ("roi_tsnr_median", "roi_tsnr_p10", "variance_removed_median", "split_half_r", "network_contrast",
                       "homotopic_contrast", "dmn_contrast", "lowfreq_power_fraction", "fd_fc_coupling", "gs_residual_sd"):
            for stream in ("volume", "surface"):
                self.assertTrue(np.isfinite(metric_value(table, stream, metric)), msg=f"{stream} {metric}")
        self.assertGreater(metric_value(table, "volume", "network_contrast"), 1.5)
        self.assertGreater(metric_value(table, "volume", "homotopic_contrast"), 0.2)
        self.assertGreater(metric_value(table, "volume", "roi_tsnr_median"), 100)
        # less added noise on the synthetic surface stream
        self.assertGreater(metric_value(table, "surface", "network_contrast"), metric_value(table, "volume", "network_contrast"))

        info = json.loads((self.func / f"{self.RUN}_desc-validation.json").read_text(encoding="utf-8"))
        nested = info["streams"]["surface"]["wmcsf24gsr"][ATLAS]
        self.assertAlmostEqual(nested["split_half_r"], metric_value(table, "surface", "split_half_r", "wmcsf24gsr"), places=5)
        self.assertEqual(info["n_censored"], 3)
        self.assertAlmostEqual(info["lowfreq_white_noise_baseline"], 0.375, places=6)
        self.assertIn("fc_similarity", info["stream_comparison"]["wmcsf24"][ATLAS])

        compare = read_tsv(self.func / f"{self.RUN}_desc-streamcompare.tsv")
        self.assertEqual(list(compare.columns), ["strategy", "atlas", "scope", "roi", "metric", "volume", "surface", "value"])
        run_rows = compare[(compare["scope"] == "run") & (compare["strategy"] == "wmcsf24")].set_index("metric")
        self.assertEqual(run_rows.at["n_roi_common", "value"], 15)
        self.assertGreater(run_rows.at["fc_similarity", "value"], 0.8)
        self.assertAlmostEqual(run_rows.at["network_contrast", "value"],
                               run_rows.at["network_contrast", "surface"] - run_rows.at["network_contrast", "volume"], places=5)
        self.assertEqual(run_rows.at["n_roi_nan", "volume"], 1)
        roi_rows = compare[(compare["scope"] == "roi") & (compare["strategy"] == "wmcsf24")]
        self.assertEqual(len(roi_rows), 16)
        self.assertEqual(int(roi_rows["volume"].isna().sum()), 1)
        self.assertTrue((self.atlas_dir / ATLAS / "centroids.tsv").is_file())

    def test_kill_mode_tables(self) -> None:
        keep = np.ones(N_T, dtype=bool)
        keep[30:45] = False
        write_run(self.func, self.RUN, self.layout, keep=keep, kill=False)
        self.assertEqual(cli(self.func, self.RUN, self.atlas_dir), 0)
        full = read_tsv(self.func / f"{self.RUN}_desc-validation.tsv")
        write_run(self.func, self.RUN, self.layout, keep=keep, kill=True)
        self.assertEqual(cli(self.func, self.RUN, self.atlas_dir, censor_mode="KILL"), 0)
        killed = read_tsv(self.func / f"{self.RUN}_desc-validation.tsv")
        self.assertEqual(metric_value(killed, "volume", "n_retained"), N_T - 15)
        # a shortened series and a full-length series with the same retained frames agree
        np.testing.assert_allclose(killed["value"].to_numpy(), full["value"].to_numpy(), rtol=1e-5, equal_nan=True)
        self.assertTrue(np.isfinite(metric_value(killed, "volume", "fd_fc_coupling")))

    def test_missing_or_invalid_censor_fails_with_existing_roi_tables(self):
        write_run(self.func, self.RUN, self.layout, surface=False)
        censor = self.func / f"{self.RUN}_desc-censor.1D"
        censor.unlink()
        for content in (None, "1\nNaN\n0\n"):
            if content is not None:
                censor.write_text(content, encoding="utf-8")
            self.assertEqual(cli(self.func, self.RUN, self.atlas_dir), 1)
            self.assertFalse((self.func / f"{self.RUN}_desc-validation.tsv").exists())

    def test_length_mismatch_skips_the_table(self) -> None:
        write_run(self.func, self.RUN, self.layout, surface=False)
        np.savetxt(self.func / f"{self.RUN}_desc-censor.1D", np.ones(N_T + 7, dtype=int), fmt="%d")
        self.assertEqual(cli(self.func, self.RUN, self.atlas_dir), 1)

    def test_volume_only_removes_a_stale_comparison(self) -> None:
        write_run(self.func, self.RUN, self.layout, surface=False)
        stale = self.func / f"{self.RUN}_desc-streamcompare.tsv"
        stale.write_text("old\n", encoding="utf-8")
        self.assertEqual(cli(self.func, self.RUN, self.atlas_dir), 0)
        self.assertFalse(stale.exists())
        table = read_tsv(self.func / f"{self.RUN}_desc-validation.tsv")
        self.assertEqual(set(table["stream"]), {"volume"})

    def test_surface_without_its_own_preproc_table(self) -> None:
        write_run(self.func, self.RUN, self.layout, surface_pre=False)
        self.assertEqual(cli(self.func, self.RUN, self.atlas_dir), 0)
        table = read_tsv(self.func / f"{self.RUN}_desc-validation.tsv")
        self.assertTrue(np.isfinite(metric_value(table, "surface", "roi_tsnr_median")))
        self.assertTrue(np.isnan(metric_value(table, "surface", "variance_removed_median")))
        self.assertTrue(np.isnan(metric_value(table, "surface", "lowfreq_power_fraction")))
        info = json.loads((self.func / f"{self.RUN}_desc-validation.json").read_text(encoding="utf-8"))
        self.assertTrue(any("volume pre-denoise" in note for note in info["notes"]))

    def test_custom_atlas_without_labels(self) -> None:
        layout = self.layout.copy()
        layout["name"] = [f"roi_{k}" for k in layout["index"]]
        write_run(self.func, self.RUN, layout, surface=False, atlas="Yeo100")
        self.assertEqual(cli(self.func, self.RUN, self.atlas_dir, atlases="Yeo100=/opt/x/ThomasYeo_100.nii"), 0)
        table = read_tsv(self.func / f"{self.RUN}_desc-validation.tsv")
        self.assertEqual(set(table["atlas"]), {"Yeo100"})
        for metric in ("network_contrast", "homotopic_contrast", "dmn_contrast"):
            self.assertTrue(np.isnan(metric_value(table, "volume", metric)), msg=metric)
        self.assertTrue(np.isfinite(metric_value(table, "volume", "split_half_r")))
        text = (self.func / f"{self.RUN}_desc-validation.tsv").read_text(encoding="utf-8")
        self.assertIn("\tn/a\n", text)

    def test_nothing_found_is_an_error(self) -> None:
        self.func.mkdir(parents=True)
        self.assertEqual(cli(self.func, self.RUN, self.atlas_dir), 1)
        self.assertFalse((self.func / f"{self.RUN}_desc-validation.tsv").exists())

    def test_fd_length_mismatch_costs_only_the_coupling(self) -> None:
        write_run(self.func, self.RUN, self.layout, surface=False)
        confounds = self.func / f"{self.RUN}_desc-confounds_timeseries.tsv"
        write_tsv(confounds, read_tsv(confounds).iloc[:-3])
        self.assertEqual(cli(self.func, self.RUN, self.atlas_dir), 0)
        table = read_tsv(self.func / f"{self.RUN}_desc-validation.tsv")
        self.assertTrue(np.isnan(metric_value(table, "volume", "fd_fc_coupling")))
        self.assertTrue(np.isfinite(metric_value(table, "volume", "split_half_r")))
        info = json.loads((self.func / f"{self.RUN}_desc-validation.json").read_text(encoding="utf-8"))
        self.assertTrue(any("FD has" in note for note in info["notes"]))

    def test_corrupt_atlas_volume_costs_only_the_homotopic_contrast(self) -> None:
        write_run(self.func, self.RUN, self.layout, surface=False)
        (self.atlas_dir / ATLAS / f"{ATLAS}_space-{TPL}_res-02_dseg.nii.gz").write_bytes(b"garbage")
        self.assertEqual(cli(self.func, self.RUN, self.atlas_dir), 0)
        table = read_tsv(self.func / f"{self.RUN}_desc-validation.tsv")
        self.assertTrue(np.isnan(metric_value(table, "volume", "homotopic_contrast")))
        self.assertTrue(np.isfinite(metric_value(table, "volume", "network_contrast")))
        self.assertTrue(np.isfinite(metric_value(table, "volume", "dmn_contrast")))


# ----------------------------------------------------------------------------
# group level
# ----------------------------------------------------------------------------

class GroupFunctionsTest(unittest.TestCase):
    def test_parse_connectivity_name(self) -> None:
        parsed = compare_streams.parse_connectivity_name(
            "sub-01_ses-2_task-rest_run-1_space-MNI152NLin6Asym_atlas-Schaefer2018_100Parcels_7Networks_desc-wmcsf24gsr_connectivity.tsv")
        self.assertEqual(parsed, ("sub-01_ses-2_task-rest_run-1", "volume", "Schaefer2018_100Parcels_7Networks", "wmcsf24gsr"))
        parsed = compare_streams.parse_connectivity_name("sub-01_task-rest_space-fsLR_atlas-A_desc-36p_connectivity.tsv")
        self.assertEqual(parsed, ("sub-01_task-rest", "surface", "A", "36p"))
        self.assertIsNone(compare_streams.parse_connectivity_name("sub-01_task-rest_desc-validation.tsv"))

    def test_loo_typicality_flags_the_odd_run(self) -> None:
        rng = np.random.default_rng(0)
        pattern = rng.normal(size=300)
        vectors = pattern[None, :] + rng.normal(size=(8, 300)) * 0.5
        vectors[5] = rng.normal(size=300)
        vectors[2, :40] = np.nan
        values = compare_streams.loo_typicality(vectors)
        self.assertEqual(int(np.argmin(values)), 5)
        self.assertLess(values[5], 0.3)
        self.assertGreater(np.delete(values, 5).min(), 0.7)
        others = np.nanmean(np.delete(vectors, 0, axis=0), axis=0)
        self.assertAlmostEqual(values[0], np.corrcoef(vectors[0], others)[0, 1], places=10)

    def test_wilcoxon_rules(self) -> None:
        self.assertTrue(np.isnan(compare_streams.wilcoxon_p(np.array([0.1, 0.2, 0.3, 0.2, 0.1]))))     # n < 6
        self.assertTrue(np.isnan(compare_streams.wilcoxon_p(np.zeros(10))))
        self.assertLess(compare_streams.wilcoxon_p(np.linspace(0.05, 0.4, 10)), 0.01)
        # zeros among the differences are dropped by scipy, NaN pairs never count
        mixed = np.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.15, 0.25, np.nan])
        self.assertTrue(np.isfinite(compare_streams.wilcoxon_p(mixed)))
        self.assertTrue(np.isnan(compare_streams.wilcoxon_p(np.array([0.1, 0.2, np.nan, np.nan, np.nan, np.nan, np.nan]))))

    def test_direction_table(self) -> None:
        self.assertEqual(compare_streams.better_stream("split_half_r", 0.1), "surface")
        self.assertEqual(compare_streams.better_stream("split_half_r", -0.1), "volume")
        self.assertEqual(compare_streams.better_stream("fd_fc_coupling", -0.1), "surface")
        self.assertEqual(compare_streams.better_stream("n_roi_nan", 2.0), "volume")
        self.assertEqual(compare_streams.better_stream("n_roi_nan", 0.0), "equal")
        for metric in ("variance_removed_median", "lowfreq_power_fraction", "gs_residual_sd", "fc_similarity"):
            self.assertEqual(compare_streams.better_stream(metric, 0.5), "descriptive")
        for metric in validate.METRICS:
            self.assertIn(metric, compare_streams.DIRECTION)


class GroupCliTest(TempDirCase):
    SITES = ("NYU", "NYU", "NYU", "NYU", "UM_1", "UM_1", "UM_1", "UM_1")

    def build_tree(self, surface: bool = True) -> tuple[Path, Path]:
        layout = roi_layout()
        atlas_dir = write_atlas_dir(self.tmp / "atlases", layout)
        deriv = self.tmp / "out" / "derivatives"
        rows = ["subject\tsession\ttask\trun\tgroup\tbold\tt1w\trun_label"]
        for k, site in enumerate(self.SITES):
            sub = f"sub-{k + 1:04d}"
            run = f"{sub}_task-rest"
            rows.append(f"{sub}\t-\trest\t-\t{site}\t/x/b.nii.gz\t/x/t.nii.gz\t{run}")
            func = deriv / sub / "func"
            write_run(func, run, layout, strategies=("wmcsf24", "wmcsf24gsr"), surface=surface, seed=k)
            self.assertEqual(cli(func, run, atlas_dir, strategies="wmcsf24 wmcsf24gsr"), 0)
        manifest = self.tmp / "out" / "rawdata" / "manifest.tsv"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return deriv, manifest

    def test_group_outputs(self) -> None:
        deriv, manifest = self.build_tree()
        out = deriv / "group"
        self.assertEqual(compare_streams.main(["--deriv-dir", str(deriv), "--manifest", str(manifest), "--out-dir", str(out)]), 0)

        long = read_tsv(out / "validation_long.tsv")
        self.assertEqual(list(long.columns), compare_streams.LONG_COLUMNS)
        self.assertEqual(long["run_label"].nunique(), 8)
        self.assertEqual(set(long["group"]), {"NYU", "UM_1"})
        self.assertIn("fc_typicality", set(long["metric"]))

        comparison = read_tsv(out / "stream_comparison.tsv")
        row = comparison[(comparison["strategy"] == "wmcsf24") & (comparison["metric"] == "network_contrast")].iloc[0]
        self.assertEqual(row["n"], 8)
        self.assertGreater(row["median_diff"], 0)           # the synthetic surface stream is the less noisy one
        self.assertLess(row["wilcoxon_p"], 0.05)
        self.assertEqual(row["better"], "surface")
        self.assertEqual(row["significant"], "yes")
        self.assertAlmostEqual(row["median_diff"], self.paired_median(long, "network_contrast"), delta=0.05)
        similarity = comparison[(comparison["strategy"] == "wmcsf24") & (comparison["metric"] == "fc_similarity")].iloc[0]
        self.assertEqual(similarity["better"], "descriptive")
        self.assertGreater(similarity["median_value"], 0.8)
        descriptive = comparison[comparison["metric"] == "variance_removed_median"]
        self.assertEqual(set(descriptive["better"]), {"descriptive"})
        self.assertIn("fc_typicality", set(comparison["metric"]))

        typicality = read_tsv(out / "fc_typicality.tsv")
        self.assertEqual(len(typicality), 8 * 2 * 2)
        self.assertTrue((typicality["n_runs"] == 8).all())
        self.assertGreater(typicality["fc_typicality"].min(), 0.5)

        strategies = read_tsv(out / "strategy_comparison.tsv")
        self.assertEqual(set(strategies["stream"]), {"volume", "surface"})
        dof = strategies[(strategies["metric"] == "dof_remaining") & (strategies["stream"] == "volume")].set_index("strategy")
        self.assertEqual(dof.at["wmcsf24", "median"], 40)
        self.assertEqual(dof.at["wmcsf24gsr", "median"], 35)

        png = out / "stream_comparison.png"
        self.assertGreater(png.stat().st_size, 5000)
        report = (out / "validation_report.html").read_text(encoding="utf-8")
        self.assertIn("data:image/png;base64,", report)
        self.assertIn("fd_fc_coupling", report)
        self.assertIn("split_half_r", report)
        self.assertNotIn("<script", report)

    @staticmethod
    def paired_median(long: pd.DataFrame, metric: str) -> float:
        part = long[(long["metric"] == metric) & (long["strategy"] == "wmcsf24")]
        wide = part.pivot(index="run_label", columns="stream", values="value")
        return float((wide["surface"] - wide["volume"]).median())

    def test_too_few_runs_for_wilcoxon_and_typicality(self) -> None:
        deriv, manifest = self.build_tree()
        for k in range(3, 8):
            shutil.rmtree(deriv / f"sub-{k + 1:04d}")
        out = deriv / "group"
        self.assertEqual(compare_streams.main(["--deriv-dir", str(deriv), "--manifest", str(manifest), "--out-dir", str(out)]), 0)
        comparison = read_tsv(out / "stream_comparison.tsv")
        self.assertTrue((comparison.loc[comparison["metric"] == "split_half_r", "n"] == 3).all())
        self.assertTrue(comparison["wilcoxon_p"].isna().all())
        self.assertEqual(len(read_tsv(out / "fc_typicality.tsv")), 0)

    def test_volume_only_dataset(self) -> None:
        deriv, manifest = self.build_tree(surface=False)
        out = deriv / "group"
        self.assertEqual(compare_streams.main(["--deriv-dir", str(deriv), "--manifest", str(manifest), "--out-dir", str(out)]), 0)
        self.assertEqual(len(read_tsv(out / "stream_comparison.tsv")), 0)
        self.assertFalse((out / "stream_comparison.png").exists())
        strategies = read_tsv(out / "strategy_comparison.tsv")
        self.assertEqual(set(strategies["stream"]), {"volume"})
        self.assertTrue((out / "validation_report.html").is_file())

    def test_pairs_without_streamcompare_files_fall_back_to_the_long_table(self) -> None:
        deriv, manifest = self.build_tree()
        for path in deriv.glob("sub-*/func/*_desc-streamcompare.tsv"):
            path.unlink()
        out = deriv / "group"
        self.assertEqual(compare_streams.main(["--deriv-dir", str(deriv), "--manifest", str(manifest), "--out-dir", str(out)]), 0)
        comparison = read_tsv(out / "stream_comparison.tsv")
        self.assertNotIn("split_half_r", set(comparison["metric"]))  # no verified common-ROI comparison
        self.assertNotIn("fc_similarity", set(comparison["metric"]))

    def test_empty_tree_is_an_error(self) -> None:
        (self.tmp / "empty").mkdir()
        self.assertEqual(compare_streams.main(["--deriv-dir", str(self.tmp / "empty"), "--out-dir", str(self.tmp / "g")]), 1)

    def test_preproc_connectivity_is_not_a_strategy(self) -> None:
        deriv, manifest = self.build_tree(surface=False)
        for path in deriv.glob("sub-*/func/*_desc-wmcsf24_connectivity.tsv"):
            shutil.copy(path, path.with_name(path.name.replace("_desc-wmcsf24_", "_desc-preproc_")))
        out = deriv / "group"
        self.assertEqual(compare_streams.main(["--deriv-dir", str(deriv), "--manifest", str(manifest), "--out-dir", str(out)]), 0)
        typicality = read_tsv(out / "fc_typicality.tsv")
        self.assertEqual(set(typicality["strategy"]), {"wmcsf24", "wmcsf24gsr"})
        self.assertNotIn("preproc", set(read_tsv(out / "validation_long.tsv")["strategy"]))


# ----------------------------------------------------------------------------
# stage script (POSIX hosts only: needs a native bash)
# ----------------------------------------------------------------------------

def native_bash() -> str | None:
    if os.name == "nt" and not os.environ.get("FMRIPROC_TEST_BASH"):
        return None
    return os.environ.get("FMRIPROC_TEST_BASH") or shutil.which("bash")


@unittest.skipUnless(native_bash(), "needs a native bash (set FMRIPROC_TEST_BASH=<path to bash> on Windows)")
class StageScriptTest(TempDirCase):
    SUB = "sub-0001"
    RUN = "sub-0001_task-rest"

    def build_dataset(self) -> dict[str, str]:
        out = self.tmp / "out"
        layout = roi_layout()
        write_atlas_dir(out / "resources" / "atlases", layout)
        func = out / "derivatives" / self.SUB / "func"
        write_run(func, self.RUN, layout, strategies=("wmcsf24", "wmcsf24gsr"))
        (func / f"{self.RUN}_desc-prep_info.json").write_text(json.dumps({"tr": 2.0, "n_volumes": N_T}), encoding="utf-8")
        (out / "rawdata").mkdir()
        (out / "rawdata" / "manifest.tsv").write_text(
            "subject\tsession\ttask\trun\tgroup\tbold\tt1w\trun_label\n"
            f"{self.SUB}\t-\trest\t-\tNYU\t/x/bold.nii.gz\t/x/t1.nii.gz\t{self.RUN}\n", encoding="utf-8")
        return {
            "OUT_DIR": str(out), "PYTHON_BIN": sys.executable, "ATLASES": ATLAS, "CUSTOM_ATLASES": "",
            "DENOISE_STRATEGIES": "wmcsf24 wmcsf24gsr", "SURFACE": "yes", "MNI_RES": "2", "CENSOR_MODE": "NTRP",
        }

    def run_stage(self, settings: dict[str, str], target: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, **settings)
        env.pop("FMRIPROC_CONFIG", None)
        env.pop("PYTHONPATH", None)
        script = (REPO / "stages" / "10_validate.sh").as_posix()
        return subprocess.run([native_bash(), "-c", f'exec bash "{script}" {target}'], env=env, capture_output=True, text=True)

    def test_subject_then_group(self) -> None:
        settings = self.build_dataset()
        result = self.run_stage(settings, self.SUB)
        self.assertEqual(result.returncode, 0, msg=result.stderr[-3000:])
        func = Path(settings["OUT_DIR"]) / "derivatives" / self.SUB / "func"
        for suffix in ("validation.tsv", "validation.json", "streamcompare.tsv"):
            self.assertTrue((func / f"{self.RUN}_desc-{suffix}").is_file(), msg=suffix)
        marker = Path(settings["OUT_DIR"]) / "work" / self.SUB / ".done" / f"10_validate__{self.RUN}.hash"
        self.assertTrue(marker.is_file())
        again = self.run_stage(settings, self.SUB)
        self.assertEqual(again.returncode, 0, msg=again.stderr[-3000:])
        self.assertIn("up to date", again.stderr)

        group = self.run_stage(settings, "--group")
        self.assertEqual(group.returncode, 0, msg=group.stderr[-3000:])
        out = Path(settings["OUT_DIR"]) / "derivatives" / "group"
        for name in ("validation_long.tsv", "stream_comparison.tsv", "strategy_comparison.tsv", "validation_report.html"):
            self.assertTrue((out / name).is_file(), msg=name)
        self.assertTrue((Path(settings["OUT_DIR"]) / "logs" / "dataset" / "10_validate_group.log").is_file())

    def test_dry_run_writes_nothing(self) -> None:
        settings = self.build_dataset()
        settings["DRY_RUN"] = "yes"
        result = self.run_stage(settings, self.SUB)
        self.assertEqual(result.returncode, 0, msg=result.stderr[-3000:])
        self.assertIn("fmriproc.validate", result.stderr)
        func = Path(settings["OUT_DIR"]) / "derivatives" / self.SUB / "func"
        self.assertFalse((func / f"{self.RUN}_desc-validation.tsv").exists())

    def test_missing_tables_fail_the_stage(self) -> None:
        settings = self.build_dataset()
        func = Path(settings["OUT_DIR"]) / "derivatives" / self.SUB / "func"
        for path in func.glob("*_timeseries.tsv"):
            path.unlink()
        result = self.run_stage(settings, self.SUB)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("1 of 1 run(s) failed", result.stderr)

    def test_unknown_option(self) -> None:
        result = self.run_stage(self.build_dataset(), "--grp")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown option", result.stderr)


class UpgradeRegressionTest(unittest.TestCase):
    def test_named_roi_permutation(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(100, 4)); x[:, 1] += 2*x[:, 0]
        names = ["A", "B", "C", "D"]; perm = [2, 0, 3, 1]
        keep = np.ones(100, bool)
        def make(values, labels):
            return validate.StreamData(values, None, keep, None, None,
                                       validate.roi_table(labels, None), np.full((4, 3), np.nan))
        result = validate.compare_streams(make(x, names), make(x[:, perm], [names[i] for i in perm]), 2)
        self.assertAlmostEqual(result["fc_similarity"], 1.)
        with self.assertRaises(ValueError):
            validate.compare_streams(make(x, names), make(x, ["A", "B", "C", "unknown"]), 2)

    def test_repeated_runs_do_not_inflate_n(self):
        pairs = pd.DataFrame([("sub-1", f"run-{i}", "g", "s", "a", "network_contrast", 1., 2.)
                              for i in range(10)], columns=compare_streams.PAIR_COLUMNS)
        result = compare_streams.stream_comparison(pairs, pd.DataFrame(columns=["strategy", "atlas", "metric", "value"]))
        self.assertEqual(result.iloc[0]["n"], 1)
        self.assertTrue(np.isnan(result.iloc[0]["wilcoxon_p"]))

    def test_bh_known_answer(self):
        np.testing.assert_allclose(compare_streams.bh_adjust([.01, .04, .03]), [.03, .04, .04])


if __name__ == "__main__":
    unittest.main()
