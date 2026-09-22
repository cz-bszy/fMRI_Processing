"""Tests of fmriproc.fetch_resources (names, layout, --check, download logic) and
of fmriproc.surface_utils. Synthetic files only, no network access."""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
import urllib.error
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.cifti2 import cifti2_axes

from fmriproc import fetch_resources as fr
from fmriproc import surface_utils as su
from fmriproc.utils import read_json, read_tsv

ATLAS = "Schaefer2018_100Parcels_7Networks"
GZIP = b"\x1f\x8b" + b"\x00" * 64
GIFTI = b'<?xml version="1.0"?><GIFTI></GIFTI>'
CIFTI = b"\x00" * 600
LABEL_TSV = (
    "index\tname\tcolor\n"
    "1\t7Networks_LH_Vis_1\t#781286\n"
    "2\t7Networks_LH_SomMot_1\t#4682b4\n"
    "3\t7Networks_RH_Default_PFCdPFCm_2\t#cd3e4e\n"
)


def touch(path: Path, payload: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def quiet_main(argv: list[str]) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = fr.main(argv)
    return code, out.getvalue()


def fill_resources(root: Path, specs: list[fr.SchaeferSpec], surface: bool) -> None:
    for _, path in fr.expected_files(root, specs, surface):
        touch(path)


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeWeb:
    """Stands in for the downloader: serves payloads by URL suffix, records the calls."""

    def __init__(self, payloads: dict[str, bytes], broken: tuple[str, ...] = ()) -> None:
        self.payloads = payloads
        self.broken = broken
        self.urls: list[str] = []

    def __call__(self, url: str, dest: Path) -> None:
        self.urls.append(url)
        if any(url.endswith(suffix) for suffix in self.broken):
            raise fr.DownloadError(f"{url}: offline")
        for suffix, payload in self.payloads.items():
            if url.endswith(suffix):
                touch(Path(dest), payload)
                return
        raise fr.DownloadError(f"{url}: 404")


def write_label_gii(path: Path, keys: np.ndarray) -> None:
    darray = nib.gifti.GiftiDataArray(keys.astype(np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.gifti.GiftiImage(darrays=[darray]), str(path))


# ----------------------------------------------------------------------------
# names
# ----------------------------------------------------------------------------

class AtlasNameTests(unittest.TestCase):
    def test_parse_schaefer(self) -> None:
        spec = fr.parse_schaefer(ATLAS)
        assert spec is not None
        self.assertEqual((spec.n_parcels, spec.n_networks), (100, 7))
        self.assertEqual(spec.tf_desc, "100Parcels7Networks")
        self.assertEqual(
            spec.dseg_filename,
            "tpl-MNI152NLin6Asym_res-02_atlas-Schaefer2018_desc-100Parcels7Networks_dseg.nii.gz",
        )
        self.assertEqual(spec.tsv_filename, "tpl-MNI152NLin6Asym_atlas-Schaefer2018_desc-100Parcels7Networks_dseg.tsv")

    def test_parse_other_names(self) -> None:
        self.assertIsNone(fr.parse_schaefer("Yeo100"))
        self.assertIsNone(fr.parse_schaefer("Schaefer2018_100Parcels"))
        self.assertIsNone(fr.parse_schaefer("schaefer2018_100Parcels_7Networks"))

    def test_impossible_schaefer_is_an_error(self) -> None:
        for name in ("Schaefer2018_150Parcels_7Networks", "Schaefer2018_100Parcels_9Networks",
                     "Schaefer2018_1100Parcels_17Networks"):
            with self.assertRaises(ValueError):
                fr.parse_schaefer(name)

    def test_dlabel_url(self) -> None:
        spec = fr.parse_schaefer("Schaefer2018_400Parcels_17Networks")
        assert spec is not None
        self.assertEqual(
            spec.dlabel_url(),
            "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/"
            "Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/"
            "Schaefer2018_400Parcels_17Networks_order.dlabel.nii",
        )
        self.assertTrue(spec.dlabel_url("https://mirror.example/raw/").startswith("https://mirror.example/raw/Thomas"))

    def test_network_from_label(self) -> None:
        self.assertEqual(fr.network_from_label("7Networks_LH_Vis_1"), "Vis")
        self.assertEqual(fr.network_from_label("7Networks_RH_Default_PFCdPFCm_2"), "Default")
        self.assertEqual(fr.network_from_label("17Networks_RH_DefaultA_PFCd_2"), "DefaultA")
        self.assertEqual(fr.network_from_label("17Networks_LH_SalVentAttnB_Ins_1"), "SalVentAttnB")
        self.assertEqual(fr.network_from_label("Background"), "n/a")
        self.assertEqual(fr.network_from_label("7Networks_XX_Vis_1"), "n/a")
        self.assertEqual(fr.network_from_label(""), "n/a")

    def test_split_atlas_names(self) -> None:
        specs, other = fr.split_atlas_names([f"{ATLAS} Schaefer2018_200Parcels_7Networks", "Yeo100", ATLAS])
        self.assertEqual([s.name for s in specs], [ATLAS, "Schaefer2018_200Parcels_7Networks"])
        self.assertEqual(other, ["Yeo100"])
        self.assertEqual(fr.split_atlas_names([]), ([], []))

    def test_hcp_urls(self) -> None:
        urls = {asset.relpath: asset.url() for asset in fr.hcp_assets()}
        base = "https://raw.githubusercontent.com/Washington-University/HCPpipelines/master/global/templates"
        self.assertEqual(urls["hcp/Atlas_ROIs.2.nii.gz"], f"{base}/91282_Greyordinates/Atlas_ROIs.2.nii.gz")
        self.assertEqual(urls["hcp/L.atlasroi.32k_fs_LR.shape.gii"],
                         f"{base}/standard_mesh_atlases/L.atlasroi.32k_fs_LR.shape.gii")
        self.assertIn("hcp/R.atlasroi.32k_fs_LR.shape.gii", urls)

    def test_surface_templateflow_assets(self) -> None:
        assets = {asset.key: asset for asset in fr.surface_tf_assets()}
        self.assertEqual(len(assets), 10)
        sphere = assets["fslr_fsaverage_sphere_R"]
        self.assertEqual(sphere.filename, "tpl-fsLR_space-fsaverage_hemi-R_den-32k_sphere.surf.gii")
        self.assertEqual(sphere.query["space"], "fsaverage")
        self.assertEqual(sphere.url(), "https://templateflow.s3.amazonaws.com/tpl-fsLR/" + sphere.filename)
        self.assertIsNone(assets["fslr_sphere_L"].query["space"])
        self.assertEqual(assets["fslr_nomedialwall_L"].query["desc"], "nomedialwall")


# ----------------------------------------------------------------------------
# layout and --check
# ----------------------------------------------------------------------------

class LayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        spec = fr.parse_schaefer(ATLAS)
        assert spec is not None
        self.spec = spec

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_atlas_layout_is_the_stage07_contract(self) -> None:
        paths = fr.atlas_paths(self.root, ATLAS)
        base = self.root / "atlases" / ATLAS
        self.assertEqual(paths["dseg"], base / f"{ATLAS}_space-MNI152NLin6Asym_res-02_dseg.nii.gz")
        self.assertEqual(paths["labels"], base / "labels.tsv")
        self.assertEqual(paths["dlabel"], base / f"{ATLAS}.dlabel.nii")

    def test_expected_files_depend_on_surface(self) -> None:
        with_surface = fr.expected_files(self.root, [self.spec], True)
        without = fr.expected_files(self.root, [self.spec], False)
        self.assertEqual(len(with_surface), 10 + 3 + 3)
        self.assertEqual([p.name for _, p in without], [f"{ATLAS}_space-MNI152NLin6Asym_res-02_dseg.nii.gz", "labels.tsv"])

    def test_missing_files_ignores_empty_placeholders(self) -> None:
        self.assertEqual(len(fr.missing_files(self.root, [self.spec], False)), 2)
        paths = fr.atlas_paths(self.root, ATLAS)
        touch(paths["dseg"], b"")            # TemplateFlow-style zero-byte placeholder
        touch(paths["labels"])
        missing = fr.missing_files(self.root, [self.spec], False)
        self.assertEqual([p for _, p in missing], [paths["dseg"]])
        touch(paths["dseg"])
        self.assertEqual(fr.missing_files(self.root, [self.spec], False), [])

    def test_templateflow_entity_order_is_free(self) -> None:
        asset = {a.key: a for a in fr.surface_tf_assets()}["fslr_midthickness_L"]
        tf_home = self.root / "templateflow"
        self.assertIsNone(asset.find(tf_home))
        other_order = touch(tf_home / "tpl-fsLR" / "tpl-fsLR_hemi-L_den-32k_midthickness.surf.gii")
        touch(tf_home / "tpl-fsLR" / "tpl-fsLR_hemi-R_den-32k_midthickness.surf.gii")
        self.assertEqual(asset.find(tf_home), other_order)
        self.assertEqual(fr.locate(self.root, ["fslr_midthickness_L"])["fslr_midthickness_L"], other_order)

    def test_check_mode_exit_codes(self) -> None:
        argv = ["--resource-dir", str(self.root), "--atlases", ATLAS, "Yeo100", "--surface", "yes", "--check"]
        code, out = quiet_main(argv)
        self.assertEqual(code, 1)
        self.assertEqual(out.count("MISSING\t"), 16)
        self.assertIn(f"{ATLAS}.dlabel.nii", out)
        self.assertFalse(any(self.root.rglob("*.nii*")), "--check must not create files")

        fill_resources(self.root, [self.spec], surface=False)
        code, out = quiet_main(argv)
        self.assertEqual(code, 1)                        # surface assets still missing
        self.assertNotIn("labels.tsv", out)
        code, _ = quiet_main(["--resource-dir", str(self.root), "--atlases", ATLAS, "--surface", "no", "--check"])
        self.assertEqual(code, 0)

        fill_resources(self.root, [self.spec], surface=True)
        code, out = quiet_main(argv)
        self.assertEqual((code, out), (0, ""))

    def test_check_without_atlases(self) -> None:
        code, _ = quiet_main(["--resource-dir", str(self.root), "--atlases", "--surface", "no", "--check"])
        self.assertEqual(code, 0)

    def test_check_rejects_impossible_atlas(self) -> None:
        code, _ = quiet_main(["--resource-dir", str(self.root), "--atlases", "Schaefer2018_150Parcels_7Networks", "--check"])
        self.assertEqual(code, 2)

    def test_locate_cli(self) -> None:
        keys = ["fslr_fsaverage_sphere_L", "hcp_atlas_rois"]
        code, _ = quiet_main(["--resource-dir", str(self.root), "--locate", *keys])
        self.assertEqual(code, 1)
        sphere = touch(self.root / "templateflow/tpl-fsLR/tpl-fsLR_space-fsaverage_hemi-L_den-32k_sphere.surf.gii")
        rois = touch(self.root / "hcp/Atlas_ROIs.2.nii.gz")
        code, out = quiet_main(["--resource-dir", str(self.root), "--locate", *keys])
        self.assertEqual(code, 0)
        self.assertEqual([Path(line) for line in out.splitlines()], [sphere.resolve(), rois.resolve()])
        code, _ = quiet_main(["--resource-dir", str(self.root), "--locate", "nonsense"])
        self.assertEqual(code, 2)


# ----------------------------------------------------------------------------
# download logic (no network: fake opener / fake downloader)
# ----------------------------------------------------------------------------

class DownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_validate_payload(self) -> None:
        self.assertIsNone(fr.validate_payload(GZIP, len(GZIP), "a.nii.gz"))
        self.assertIsNone(fr.validate_payload(GIFTI, len(GIFTI), "a.surf.gii"))
        self.assertIsNone(fr.validate_payload(CIFTI, len(CIFTI), "a.dlabel.nii"))
        self.assertIsNone(fr.validate_payload(b"index\tname\n", 11, "a.tsv"))
        self.assertIsNotNone(fr.validate_payload(b"", 0, "a.tsv"))
        pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n"
        self.assertIn("LFS", fr.validate_payload(pointer, len(pointer), "a.dlabel.nii") or "")
        self.assertIn("HTML", fr.validate_payload(b"<!DOCTYPE html><html>", 21, "a.nii.gz") or "")
        self.assertIsNotNone(fr.validate_payload(b"plain text", 10, "a.nii.gz"))
        self.assertIsNotNone(fr.validate_payload(b"plain text", 10, "a.shape.gii"))
        self.assertIsNotNone(fr.validate_payload(b"\x00" * 100, 100, "a.dlabel.nii"))

    def test_download_retries_then_succeeds(self) -> None:
        calls: list[float] = []
        naps: list[float] = []

        def opener(request: object, timeout: float) -> FakeResponse:
            calls.append(timeout)
            if len(calls) < 3:
                raise urllib.error.URLError("connection reset")
            return FakeResponse(GZIP)

        dest = self.root / "hcp" / "Atlas_ROIs.2.nii.gz"
        with contextlib.redirect_stderr(io.StringIO()):
            fr.download("https://example.invalid/x.nii.gz", dest, retries=3, timeout=7.0, opener=opener, sleep=naps.append)
        self.assertEqual(dest.read_bytes(), GZIP)
        self.assertEqual(calls, [7.0, 7.0, 7.0])
        self.assertEqual(len(naps), 2)
        self.assertEqual([p.name for p in dest.parent.iterdir()], [dest.name], "temporary files must be removed")

    def test_download_gives_up_and_leaves_nothing(self) -> None:
        def opener(request: object, timeout: float) -> FakeResponse:
            return FakeResponse(b"version https://git-lfs.github.com/spec/v1\n")

        dest = self.root / "atlases" / "x.dlabel.nii"
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(fr.DownloadError):
            fr.download("https://example.invalid/x.dlabel.nii", dest, retries=2, opener=opener, sleep=lambda s: None)
        self.assertEqual(list(dest.parent.iterdir()), [])

    def test_fetch_atlas_builds_the_layout(self) -> None:
        spec = fr.parse_schaefer(ATLAS)
        assert spec is not None
        web = FakeWeb({spec.dseg_filename: GZIP, spec.tsv_filename: LABEL_TSV.encode(), "_order.dlabel.nii": CIFTI})
        with contextlib.redirect_stderr(io.StringIO()):
            errors = fr.fetch_atlas(spec, self.root, True, web, fr.TEMPLATEFLOW_S3, fr.GITHUB_RAW, use_api=False)
        self.assertEqual(errors, [])
        paths = fr.atlas_paths(self.root, ATLAS)
        self.assertEqual(paths["dseg"].read_bytes(), GZIP)
        self.assertEqual(paths["dlabel"].read_bytes(), CIFTI)
        labels = read_tsv(paths["labels"])
        self.assertEqual(list(labels.columns), ["index", "name", "network"])
        self.assertEqual(labels["index"].tolist(), [1, 2, 3])
        self.assertEqual(labels["network"].tolist(), ["Vis", "SomMot", "Default"])
        self.assertTrue((self.root / "templateflow" / "tpl-MNI152NLin6Asym" / spec.dseg_filename).is_file())
        self.assertEqual(fr.missing_files(self.root, [spec], True)[0][0], "fslr_sphere_L")   # only surface assets left

        # second call: everything present, no request is made
        n_requests = len(web.urls)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(fr.fetch_atlas(spec, self.root, True, web, fr.TEMPLATEFLOW_S3, fr.GITHUB_RAW, use_api=False), [])
        self.assertEqual(len(web.urls), n_requests)

    def test_fetch_atlas_without_surface_skips_dlabel(self) -> None:
        spec = fr.parse_schaefer(ATLAS)
        assert spec is not None
        web = FakeWeb({spec.dseg_filename: GZIP, spec.tsv_filename: LABEL_TSV.encode()})
        with contextlib.redirect_stderr(io.StringIO()):
            errors = fr.fetch_atlas(spec, self.root, False, web, fr.TEMPLATEFLOW_S3, fr.GITHUB_RAW, use_api=False)
        self.assertEqual(errors, [])
        self.assertFalse(any("dlabel" in url for url in web.urls))
        self.assertEqual(fr.missing_files(self.root, [spec], False), [])

    def test_fetch_atlas_reports_failures(self) -> None:
        spec = fr.parse_schaefer(ATLAS)
        assert spec is not None
        web = FakeWeb({spec.dseg_filename: GZIP, spec.tsv_filename: LABEL_TSV.encode()}, broken=("_order.dlabel.nii",))
        with contextlib.redirect_stderr(io.StringIO()):
            errors = fr.fetch_atlas(spec, self.root, True, web, fr.TEMPLATEFLOW_S3, fr.GITHUB_RAW, use_api=False)
        self.assertEqual(len(errors), 1)
        self.assertIn("dlabel", errors[0])

    def test_atlasroi_falls_back_to_templateflow_label(self) -> None:
        keys = np.array([0, 1, 1, 0, 1], dtype=np.int32)

        class Web(FakeWeb):
            def __call__(self, url: str, dest: Path) -> None:
                if "nomedialwall" in url:
                    self.urls.append(url)
                    write_label_gii(Path(dest), keys)
                    return
                super().__call__(url, dest)

        web = Web({".surf.gii": GIFTI, "Atlas_ROIs.2.nii.gz": GZIP}, broken=(".atlasroi.32k_fs_LR.shape.gii",))
        with contextlib.redirect_stderr(io.StringIO()):
            errors = fr.fetch_surface_assets(self.root, web, fr.TEMPLATEFLOW_S3, fr.GITHUB_RAW, use_api=False)
        self.assertEqual(errors, [])
        self.assertEqual(fr.missing_files(self.root, [], True), [])
        roi = su.load_metric(self.root / "hcp" / "R.atlasroi.32k_fs_LR.shape.gii")
        np.testing.assert_array_equal(roi, [0, 1, 1, 0, 1])


# ----------------------------------------------------------------------------
# surface_utils
# ----------------------------------------------------------------------------

def save_volume(path: Path, data: np.ndarray, affine: np.ndarray) -> Path:
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def save_shape(path: Path, values: list[float]) -> Path:
    su.save_metric(np.asarray(values, dtype=np.float32), path)
    return path


class SurfaceUtilsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_grid_relation(self) -> None:
        # FSL MNI152 2 mm style grid: x stored right-to-left (LAS)
        las = np.array([[-2.0, 0, 0, 90], [0, 2, 0, -126], [0, 0, 2, -72], [0, 0, 0, 1]])
        data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
        ref = save_volume(self.root / "ref.nii.gz", data, las)
        same = save_volume(self.root / "same.nii.gz", np.zeros((4, 5, 6, 3), dtype=np.float32), las)
        canonical = nib.as_closest_canonical(nib.load(str(ref)))
        flipped = self.root / "ras.nii.gz"
        nib.save(canonical, str(flipped))
        shifted = las.copy()
        shifted[0, 3] += 2.0
        moved = save_volume(self.root / "moved.nii.gz", data, shifted)
        coarse = save_volume(self.root / "coarse.nii.gz", data, np.diag([3.0, 3.0, 3.0, 1.0]))

        self.assertEqual(su.grid_relation(same, ref)[0], "same")
        self.assertEqual(su.grid_relation(flipped, ref)[0], "reordered")
        self.assertEqual(su.grid_relation(moved, ref)[0], "different")
        self.assertEqual(su.grid_relation(coarse, ref)[0], "different")
        self.assertTrue(su.compare_grids(same, ref)[0])
        self.assertFalse(su.compare_grids(flipped, ref)[0])

        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = su.main(["check-grid", str(flipped), str(ref)])
        self.assertEqual((code, out.getvalue().strip()), (1, "reordered"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = su.main(["check-grid", str(same), str(ref)])
        self.assertEqual((code, out.getvalue().strip()), (0, "same"))

    def test_voxel_subdiv(self) -> None:
        self.assertEqual(su.choose_voxel_subdiv([3.0, 3.0, 3.0]), 5)
        self.assertEqual(su.choose_voxel_subdiv([3.0, 3.0, 3.5]), 7)
        self.assertEqual(su.choose_voxel_subdiv([3.4375, 3.4375, 4.0, 2.0]), 7)   # a 4th entry (TR) is ignored
        self.assertEqual(su.choose_voxel_subdiv([3.0, 3.0, 3.0, 3.6]), 5)
        with self.assertRaises(ValueError):
            su.choose_voxel_subdiv([])

        info = self.root / "prep_info.json"
        info.write_text('{"voxel_size": [3.0, 3.0, 4.0]}', encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(su.main(["voxel-subdiv", "--prep-info", str(info)]), 0)
        self.assertEqual(out.getvalue().strip(), "7")
        info.write_text("{}", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(su.main(["voxel-subdiv", "--prep-info", str(info)]), 1)

    def test_label_to_roi(self) -> None:
        label = self.root / "label.label.gii"
        write_label_gii(label, np.array([0, 3, 3, 0, 1, 1]))
        out = self.root / "roi.shape.gii"
        self.assertEqual(su.label_gii_to_roi(label, out, "CortexLeft"), 4)
        np.testing.assert_array_equal(su.load_metric(out), [0, 1, 1, 0, 1, 1])
        self.assertEqual(nib.load(str(out)).meta["AnatomicalStructurePrimary"], "CortexLeft")
        self.assertEqual([p.name for p in self.root.iterdir() if p.name.startswith(".tmp")], [])

    def test_bad_vertices_and_goodvoxels(self) -> None:
        bad = [np.array([1, 0, 0, 1.0]), np.array([0, 0, 1.0])]
        roi = [np.array([1, 1, 1, 0.0]), np.array([1, 1, 1.0])]
        self.assertAlmostEqual(su.pct_bad_vertices(bad, roi), 100.0 * 2 / 6)
        with self.assertRaises(ValueError):
            su.pct_bad_vertices([np.zeros(3)], [np.zeros(4)])

        ribbon = np.zeros((2, 2, 2))
        ribbon[0] = 1                                   # 4 ribbon voxels
        mask = np.ones((2, 2, 2))
        mask[0, 0, 0] = 0                               # one of them has no data
        good = mask.copy()
        good[0, 1, 1] = 0                               # one is excluded by the CoV rule
        good[1, 1, 1] = 0                               # outside the ribbon: irrelevant
        stats = su.goodvoxel_stats(ribbon, good, mask)
        self.assertEqual(stats["n_ribbon_voxels"], 4)
        self.assertAlmostEqual(stats["pct_ribbon_outside_mask"], 25.0)
        self.assertAlmostEqual(stats["pct_goodvoxels_excluded"], 100.0 / 3)

    def _write_tsnr(self, path: Path) -> None:
        left = cifti2_axes.BrainModelAxis.from_surface(np.arange(4), 6, name="CortexLeft")
        right = cifti2_axes.BrainModelAxis.from_surface(np.arange(3), 6, name="CortexRight")
        volume_mask = np.zeros((3, 3, 3), dtype=bool)
        volume_mask[1, 1, :2] = True
        thalamus = cifti2_axes.BrainModelAxis.from_mask(volume_mask, affine=np.eye(4), name="thalamus_left")
        values = np.array([[10, 20, np.nan, 0, 30, 40, 50, 5, 7]], dtype=np.float32)
        header = (cifti2_axes.ScalarAxis(["tsnr"]), left + right + thalamus)
        nib.save(nib.Cifti2Image(values, header=header), str(path))

    def test_surfqc_cli(self) -> None:
        tsnr = self.root / "tsnr.dscalar.nii"
        self._write_tsnr(tsnr)
        affine = np.diag([3.0, 3.0, 3.0, 1.0])
        ribbon = np.zeros((2, 2, 2), dtype=np.float32)
        ribbon[0] = 1
        mask = np.ones((2, 2, 2), dtype=np.float32)
        good = mask.copy()
        good[0, 0, 0] = 0
        save_volume(self.root / "ribbon.nii.gz", ribbon, affine)
        save_volume(self.root / "mask.nii.gz", mask, affine)
        save_volume(self.root / "good.nii.gz", good, affine)
        bad = [save_shape(self.root / f"bad{h}.shape.gii", v) for h, v in (("L", [1, 0, 0, 0]), ("R", [0, 0, 0]))]
        roi = [save_shape(self.root / f"roi{h}.shape.gii", v) for h, v in (("L", [1, 1, 1, 1]), ("R", [1, 1, 0]))]
        valid = [save_shape(self.root / f"valid{h}.shape.gii", v) for h, v in (("L", [1, 1, 0, 1]), ("R", [1, 1, 1]))]
        atlas = [save_shape(self.root / f"atlas{h}.shape.gii", v) for h, v in (("L", [1, 1, 1, 0]), ("R", [1, 1, 1]))]
        out = self.root / "sub-1_task-rest_desc-surfqc.json"

        code = su.main([
            "surfqc", "--badvert", str(bad[0]), str(bad[1]), "--roi", str(roi[0]), str(roi[1]),
            "--ribbon", str(self.root / "ribbon.nii.gz"), "--goodvoxels", str(self.root / "good.nii.gz"),
            "--mask", str(self.root / "mask.nii.gz"), "--tsnr", str(tsnr),
            "--valid", str(valid[0]), str(valid[1]), "--atlasroi", str(atlas[0]), str(atlas[1]),
            "--voxel-subdiv", "7", "--strategies", "wmcsf24", "acompcor", "--smooth-fwhm", "5", "--out", str(out),
        ])
        self.assertEqual(code, 0)
        qc = read_json(out)
        for key in ("pct_badvertices", "pct_goodvoxels_excluded", "tsnr_cortex_median", "n_vertices_valid"):
            self.assertIn(key, qc)                       # DESIGN.md / stage 08 contract
        self.assertAlmostEqual(qc["pct_badvertices"], 100.0 / 6)
        self.assertAlmostEqual(qc["pct_goodvoxels_excluded"], 25.0)
        self.assertAlmostEqual(qc["tsnr_cortex_median"], 30.0)   # NaN and 0 are not counted
        self.assertEqual(qc["n_vertices_valid"], 5)
        self.assertEqual(qc["n_vertices_total"], 7)
        self.assertAlmostEqual(qc["tsnr_subcortex_median"], 6.0)
        self.assertAlmostEqual(qc["pct_fslr_vertices_nodata"], 100.0 / 6)
        self.assertEqual(qc["voxel_subdiv"], 7)
        self.assertEqual(qc["strategies"], "wmcsf24 acompcor")

        # without the optional resampling ROIs the key is simply absent
        qc2 = su.compute_surfqc(bad, roi, self.root / "ribbon.nii.gz", self.root / "good.nii.gz",
                                self.root / "mask.nii.gz", tsnr)
        self.assertNotIn("pct_fslr_vertices_nodata", qc2)
        self.assertAlmostEqual(qc2["tsnr_cortex_median"], 30.0)

    def test_cifti_structure_values(self) -> None:
        tsnr = self.root / "tsnr.dscalar.nii"
        self._write_tsnr(tsnr)
        cortex = su.cifti_structure_values(tsnr, su.CORTEX_STRUCTURES)
        self.assertEqual(cortex.shape, (7,))
        np.testing.assert_array_equal(np.isfinite(cortex), [True, True, False, True, True, True, True])
        left = su.cifti_structure_values(tsnr, ["CIFTI_STRUCTURE_CORTEX_LEFT"])
        np.testing.assert_allclose(np.nan_to_num(left, nan=-1), [10, 20, -1, 0])
        rest = su.cifti_structure_values(tsnr, None, exclude=su.CORTEX_STRUCTURES)
        np.testing.assert_allclose(rest, [5, 7])
        self.assertEqual(su.cifti_structure_values(tsnr, ["CIFTI_STRUCTURE_CEREBELLUM"]).size, 0)
        not_cifti = save_shape(self.root / "not_cifti.shape.gii", [1.0, 0.0])
        with self.assertRaises(ValueError):
            su.cifti_structure_values(not_cifti)

    def test_grid_relation_permuted_axes(self) -> None:
        # the same lattice stored with the axes permuted (x<->z) and one flipped
        las = np.array([[-2.0, 0, 0, 90], [0, 2, 0, -126], [0, 0, 2, -72], [0, 0, 0, 1]])
        data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
        ref = save_volume(self.root / "ref.nii.gz", data, las)
        permuted_affine = np.array([[0, 0, -2.0, 90], [0, 2, 0, -126], [2.0, 0, 0, -72], [0, 0, 0, 1]])
        permuted = save_volume(self.root / "perm.nii.gz", np.transpose(data, (2, 1, 0)).copy(), permuted_affine)
        self.assertEqual(su.grid_relation(permuted, ref)[0], "reordered")
        # a permutation that changes the extent is a different grid
        wrong_shape = save_volume(self.root / "wrong.nii.gz", data, permuted_affine)
        self.assertEqual(su.grid_relation(wrong_shape, ref)[0], "different")
        # header-only comparison must not care about the number of volumes
        self.assertEqual(su.grid_of(ref)[0], (4, 5, 6))


if __name__ == "__main__":
    unittest.main()
