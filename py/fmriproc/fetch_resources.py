"""One-time download of the external resources (templates, atlases, fsLR meshes).

Layout written under ``--resource-dir`` (= ``$RESOURCE_DIR``)::

    templateflow/                      TEMPLATEFLOW_HOME (tpl-fsLR/..., tpl-MNI152NLin6Asym/...)
    hcp/Atlas_ROIs.2.nii.gz            HCP 91282 subcortical label volume (FSL MNI 2 mm grid)
    hcp/{L,R}.atlasroi.32k_fs_LR.shape.gii   fsLR-32k medial-wall ROI (1 = cortex)
    atlases/<A>/<A>_space-MNI152NLin6Asym_res-02_dseg.nii.gz
    atlases/<A>/labels.tsv             columns: index, name, network
    atlases/<A>/<A>.dlabel.nii         fsLR-32k CIFTI label file (CBIG), surface branch only

``<A>`` = ``Schaefer2018_<N>Parcels_<M>Networks``. Other names are ignored with a
warning (custom atlases are given to the pipeline as files, not fetched).

The module never needs the network for ``--check`` and ``--locate``.
"""
from __future__ import annotations

import argparse
import http.client
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

GITHUB_RAW = "https://raw.githubusercontent.com"
TEMPLATEFLOW_S3 = "https://templateflow.s3.amazonaws.com"
HCP_TEMPLATES = "Washington-University/HCPpipelines/master/global/templates"
CBIG_FSLR = (
    "ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/"
    "Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti"
)
ATLAS_TEMPLATE = "MNI152NLin6Asym"
ATLAS_RES_LABEL = "res-02"
USER_AGENT = "fmriproc-fetch/2.0 (python-urllib)"
HEMIS = ("L", "R")
STRUCTURES = {"L": "CortexLeft", "R": "CortexRight"}

SCHAEFER_RE = re.compile(r"^Schaefer2018_(\d+)Parcels_(\d+)Networks$")
SCHAEFER_PARCELS = tuple(range(100, 1001, 100))
SCHAEFER_NETWORKS = (7, 17)

Downloader = Callable[[str, Path], None]


def _log(message: str) -> None:
    print(f"[fetch_resources] {message}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------
# atlas names
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SchaeferSpec:
    name: str
    n_parcels: int
    n_networks: int

    @property
    def tf_desc(self) -> str:
        """TemplateFlow ``desc`` entity, e.g. ``100Parcels7Networks``."""
        return f"{self.n_parcels}Parcels{self.n_networks}Networks"

    @property
    def dseg_filename(self) -> str:
        return f"tpl-{ATLAS_TEMPLATE}_{ATLAS_RES_LABEL}_atlas-Schaefer2018_desc-{self.tf_desc}_dseg.nii.gz"

    @property
    def tsv_filename(self) -> str:
        return f"tpl-{ATLAS_TEMPLATE}_atlas-Schaefer2018_desc-{self.tf_desc}_dseg.tsv"

    def dlabel_url(self, github_raw: str = GITHUB_RAW) -> str:
        return f"{github_raw.rstrip('/')}/{CBIG_FSLR}/{self.name}_order.dlabel.nii"


def parse_schaefer(name: str) -> SchaeferSpec | None:
    """``Schaefer2018_100Parcels_7Networks`` -> spec; any other name -> None.

    A name that has the Schaefer pattern but an impossible parcel/network count
    raises ValueError (a typo must not be skipped silently).
    """
    match = SCHAEFER_RE.match(name.strip())
    if match is None:
        return None
    n_parcels, n_networks = int(match.group(1)), int(match.group(2))
    if n_parcels not in SCHAEFER_PARCELS or n_networks not in SCHAEFER_NETWORKS:
        raise ValueError(
            f"{name}: Schaefer2018 exists with {SCHAEFER_PARCELS[0]}..{SCHAEFER_PARCELS[-1]} parcels "
            f"(steps of 100) and {SCHAEFER_NETWORKS} networks"
        )
    return SchaeferSpec(name.strip(), n_parcels, n_networks)


def network_from_label(label: str) -> str:
    """``7Networks_LH_Vis_1`` -> ``Vis``; ``17Networks_RH_DefaultA_PFCd_2`` -> ``DefaultA``.

    Labels that do not follow ``<M>Networks_<LH|RH>_<network>_...`` give ``n/a``.
    """
    tokens = str(label).strip().split("_")
    if len(tokens) >= 3 and tokens[0].endswith("Networks") and tokens[1] in ("LH", "RH") and tokens[2]:
        return tokens[2]
    return "n/a"


def split_atlas_names(names: Iterable[str]) -> tuple[list[SchaeferSpec], list[str]]:
    """Schaefer specs (unique, input order) and the names that are not Schaefer."""
    specs: list[SchaeferSpec] = []
    other: list[str] = []
    seen: set[str] = set()
    for raw in names:
        for name in str(raw).split():
            if name in seen:
                continue
            seen.add(name)
            spec = parse_schaefer(name)
            if spec is None:
                other.append(name)
            else:
                specs.append(spec)
    return specs, other


# ----------------------------------------------------------------------------
# resource plan
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class TFAsset:
    """A TemplateFlow file: query for ``templateflow.api.get`` + canonical name."""
    key: str
    template: str
    filename: str
    query: dict = field(default_factory=dict, hash=False, compare=False)

    def directory(self, tf_home: Path) -> Path:
        return Path(tf_home) / f"tpl-{self.template}"

    def url(self, s3_base: str = TEMPLATEFLOW_S3) -> str:
        return f"{s3_base.rstrip('/')}/tpl-{self.template}/{self.filename}"

    def find(self, tf_home: Path) -> Path | None:
        """Non-empty local copy. TemplateFlow unpacks a skeleton of ZERO-BYTE
        placeholders on first import, so existence alone proves nothing. The
        entity order of upstream names is not uniform (``den-32k_hemi-L`` and
        ``hemi-L_den-32k`` both occur), hence the order-free comparison."""
        directory = self.directory(tf_home)
        canonical = directory / self.filename
        if _nonempty(canonical):
            return canonical
        if not directory.is_dir():
            return None
        wanted = _entity_key(self.filename)
        for candidate in sorted(directory.iterdir()):
            if _entity_key(candidate.name) == wanted and _nonempty(candidate):
                return candidate
        return None


@dataclass(frozen=True)
class URLAsset:
    key: str
    relpath: str            # below the resource dir
    url_path: str           # below the GitHub raw base

    def path(self, resource_dir: Path) -> Path:
        return Path(resource_dir) / self.relpath

    def url(self, github_raw: str = GITHUB_RAW) -> str:
        return f"{github_raw.rstrip('/')}/{self.url_path}"


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _entity_key(filename: str) -> tuple[frozenset[str], str]:
    """``tpl-fsLR_den-32k_hemi-L_sphere.surf.gii`` -> ({den-32k, hemi-L, tpl-fsLR}, 'sphere.surf.gii')."""
    parts = filename.split("_")
    return frozenset(parts[:-1]), parts[-1]


def surface_tf_assets() -> list[TFAsset]:
    assets: list[TFAsset] = []
    for hemi in HEMIS:
        assets += [
            TFAsset(f"fslr_sphere_{hemi}", "fsLR", f"tpl-fsLR_hemi-{hemi}_den-32k_sphere.surf.gii",
                    dict(hemi=hemi, density="32k", suffix="sphere", space=None, extension=".surf.gii")),
            TFAsset(f"fslr_fsaverage_sphere_{hemi}", "fsLR",
                    f"tpl-fsLR_space-fsaverage_hemi-{hemi}_den-32k_sphere.surf.gii",
                    dict(hemi=hemi, density="32k", suffix="sphere", space="fsaverage", extension=".surf.gii")),
            TFAsset(f"fslr_nomedialwall_{hemi}", "fsLR",
                    f"tpl-fsLR_hemi-{hemi}_den-32k_desc-nomedialwall_dparc.label.gii",
                    dict(hemi=hemi, density="32k", desc="nomedialwall", suffix="dparc", extension=".label.gii")),
            TFAsset(f"fslr_midthickness_{hemi}", "fsLR", f"tpl-fsLR_den-32k_hemi-{hemi}_midthickness.surf.gii",
                    dict(hemi=hemi, density="32k", suffix="midthickness", extension=".surf.gii")),
            TFAsset(f"fslr_inflated_{hemi}", "fsLR", f"tpl-fsLR_den-32k_hemi-{hemi}_inflated.surf.gii",
                    dict(hemi=hemi, density="32k", suffix="inflated", extension=".surf.gii")),
        ]
    return assets


def hcp_assets() -> list[URLAsset]:
    assets = [URLAsset("hcp_atlas_rois", "hcp/Atlas_ROIs.2.nii.gz",
                       f"{HCP_TEMPLATES}/91282_Greyordinates/Atlas_ROIs.2.nii.gz")]
    for hemi in HEMIS:
        name = f"{hemi}.atlasroi.32k_fs_LR.shape.gii"
        assets.append(URLAsset(f"hcp_atlasroi_{hemi}", f"hcp/{name}",
                               f"{HCP_TEMPLATES}/standard_mesh_atlases/{name}"))
    return assets


def atlas_tf_assets(spec: SchaeferSpec) -> tuple[TFAsset, TFAsset]:
    dseg = TFAsset(f"{spec.name}_dseg", ATLAS_TEMPLATE, spec.dseg_filename,
                   dict(resolution=2, atlas="Schaefer2018", desc=spec.tf_desc, suffix="dseg", extension=".nii.gz"))
    tsv = TFAsset(f"{spec.name}_tsv", ATLAS_TEMPLATE, spec.tsv_filename,
                  dict(resolution=None, atlas="Schaefer2018", desc=spec.tf_desc, suffix="dseg", extension=".tsv"))
    return dseg, tsv


def atlas_paths(resource_dir: Path, name: str) -> dict[str, Path]:
    """The atlases/<A>/ layout other stages rely on."""
    base = Path(resource_dir) / "atlases" / name
    return {
        "dseg": base / f"{name}_space-{ATLAS_TEMPLATE}_{ATLAS_RES_LABEL}_dseg.nii.gz",
        "labels": base / "labels.tsv",
        "dlabel": base / f"{name}.dlabel.nii",
    }


def expected_files(resource_dir: Path, specs: Sequence[SchaeferSpec], surface: bool) -> list[tuple[str, Path]]:
    """(description, path) of every final file the pipeline reads.

    TemplateFlow source files of the atlases are intermediate (the copy under
    atlases/ is what counts), fsLR meshes are reported through their canonical
    path even when an equivalent file with another entity order is present.
    """
    resource_dir = Path(resource_dir)
    tf_home = resource_dir / "templateflow"
    items: list[tuple[str, Path]] = []
    if surface:
        for asset in surface_tf_assets():
            found = asset.find(tf_home)
            items.append((asset.key, found if found is not None else asset.directory(tf_home) / asset.filename))
        for url_asset in hcp_assets():
            items.append((url_asset.key, url_asset.path(resource_dir)))
    for spec in specs:
        paths = atlas_paths(resource_dir, spec.name)
        items.append((f"{spec.name} volume", paths["dseg"]))
        items.append((f"{spec.name} labels", paths["labels"]))
        if surface:
            items.append((f"{spec.name} dlabel", paths["dlabel"]))
    return items


def missing_files(resource_dir: Path, specs: Sequence[SchaeferSpec], surface: bool) -> list[tuple[str, Path]]:
    return [(desc, path) for desc, path in expected_files(resource_dir, specs, surface) if not _nonempty(path)]


def locate(resource_dir: Path, keys: Sequence[str]) -> dict[str, Path | None]:
    """Key -> existing non-empty path (None when absent). Keys: see ``known_keys``."""
    resource_dir = Path(resource_dir)
    tf_home = resource_dir / "templateflow"
    table: dict[str, Path | None] = {}
    tf_by_key = {asset.key: asset for asset in surface_tf_assets()}
    url_by_key = {asset.key: asset for asset in hcp_assets()}
    for key in keys:
        if key in tf_by_key:
            table[key] = tf_by_key[key].find(tf_home)
        elif key in url_by_key:
            path = url_by_key[key].path(resource_dir)
            table[key] = path if _nonempty(path) else None
        else:
            raise KeyError(key)
    return table


def known_keys() -> list[str]:
    return [a.key for a in surface_tf_assets()] + [a.key for a in hcp_assets()]


# ----------------------------------------------------------------------------
# download
# ----------------------------------------------------------------------------

class DownloadError(RuntimeError):
    pass


def validate_payload(head: bytes, size: int, filename: str) -> str | None:
    """Reason why a downloaded file cannot be the real thing, or None.

    GitHub raw answers with a ~130 byte text pointer for Git-LFS files and
    proxies/captive portals answer with HTML; both would otherwise be stored as
    a plausible-looking atlas.
    """
    if size == 0:
        return "empty file"
    text = head[:200].lstrip().lower()
    if text.startswith(b"version https://git-lfs"):
        return "Git-LFS pointer instead of the file"
    if text.startswith(b"<!doctype html") or text.startswith(b"<html"):
        return "HTML page instead of the file"
    if filename.endswith(".gz") and head[:2] != b"\x1f\x8b":
        return "not a gzip file"
    if filename.endswith(".gii") and b"<" not in head[:200]:
        return "not a GIFTI (XML) file"
    if filename.endswith(".nii") and size < 540:
        return "too small for a NIfTI/CIFTI file"
    return None


def download(url: str, dest: Path, retries: int = 3, timeout: float = 60.0,
             wait: float = 3.0, opener: Callable[..., object] | None = None,
             sleep: Callable[[float], None] = time.sleep) -> None:
    """GET ``url`` into ``dest`` (temporary file + rename), ``retries`` attempts."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    opener = opener or urllib.request.urlopen
    last_error = "no attempt made"
    for attempt in range(1, max(1, retries) + 1):
        fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", dir=str(dest.parent))
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            size = 0
            head = b""
            with opener(request, timeout=timeout) as response, os.fdopen(fd, "wb") as fh:
                fd = -1
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    if len(head) < 512:
                        head += chunk[: 512 - len(head)]
                    fh.write(chunk)
                    size += len(chunk)
            problem = validate_payload(head, size, dest.name)
            if problem is not None:
                raise DownloadError(problem)
            os.replace(tmp, dest)
            _log(f"downloaded {dest.name} ({size} bytes)")
            return
        except (urllib.error.URLError, http.client.HTTPException, OSError, DownloadError, ValueError) as err:
            last_error = f"{type(err).__name__}: {err}"
            _log(f"attempt {attempt}/{retries} failed for {url}: {last_error}")
        finally:
            if fd >= 0:
                os.close(fd)
            if os.path.exists(tmp):
                os.unlink(tmp)
        if attempt < retries:
            sleep(wait * attempt)
    raise DownloadError(f"{url}: {last_error}")


def _templateflow_get(asset: TFAsset, tf_home: Path) -> Path | None:
    """``templateflow.api.get``; None when the package or the file is unavailable."""
    os.environ["TEMPLATEFLOW_HOME"] = str(tf_home)   # must be set before the import
    try:
        from templateflow import api  # type: ignore[import-not-found]
    except Exception as err:  # noqa: BLE001 - optional dependency, any failure means "use the URL"
        _log(f"templateflow not usable ({type(err).__name__}: {err}); using direct URLs")
        return None
    try:
        result = api.get(asset.template, **asset.query)
    except Exception as err:  # noqa: BLE001 - network/S3 errors surface as many exception types
        _log(f"templateflow.api.get failed for {asset.filename}: {type(err).__name__}: {err}")
        return None
    candidates = [Path(p) for p in (result if isinstance(result, (list, tuple)) else [result])]
    wanted = _entity_key(asset.filename)
    candidates.sort(key=lambda p: _entity_key(p.name) != wanted)
    for candidate in candidates:
        if _nonempty(candidate):
            return candidate
    return None


def fetch_tf_asset(asset: TFAsset, tf_home: Path, fetch: Downloader, s3_base: str = TEMPLATEFLOW_S3,
                   use_api: bool = True) -> Path:
    found = asset.find(tf_home)
    if found is not None:
        _log(f"present: {found.name}")
        return found
    if use_api:
        got = _templateflow_get(asset, tf_home)
        if got is not None:
            _log(f"templateflow: {got.name}")
            return got
    dest = asset.directory(tf_home) / asset.filename
    fetch(asset.url(s3_base), dest)
    return dest


def fetch_url_asset(asset: URLAsset, resource_dir: Path, fetch: Downloader, github_raw: str = GITHUB_RAW) -> Path:
    dest = asset.path(resource_dir)
    if _nonempty(dest):
        _log(f"present: {asset.relpath}")
        return dest
    fetch(asset.url(github_raw), dest)
    return dest


def _atomic_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", dir=str(dest.parent))
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_labels_tsv(source_tsv: Path, dest: Path) -> int:
    """TemplateFlow ``dseg.tsv`` -> ``labels.tsv`` (index, name, network). Returns the row count."""
    import pandas as pd

    from fmriproc.utils import read_tsv, write_tsv

    frame = read_tsv(source_tsv)
    columns = {str(c).lower(): c for c in frame.columns}
    index_col = columns.get("index", frame.columns[0])
    name_col = columns.get("name", columns.get("label", frame.columns[min(1, len(frame.columns) - 1)]))
    out = pd.DataFrame({
        "index": frame[index_col].astype(int),
        "name": frame[name_col].astype(str),
    })
    out = out[out["index"] > 0].sort_values("index").reset_index(drop=True)
    out["network"] = [network_from_label(name) for name in out["name"]]
    write_tsv(dest, out)
    return int(out.shape[0])


def fetch_atlas(spec: SchaeferSpec, resource_dir: Path, surface: bool, fetch: Downloader,
                s3_base: str, github_raw: str, use_api: bool = True) -> list[str]:
    """Fetch one Schaefer atlas; returns error messages (empty = complete)."""
    errors: list[str] = []
    tf_home = Path(resource_dir) / "templateflow"
    paths = atlas_paths(resource_dir, spec.name)
    dseg_asset, tsv_asset = atlas_tf_assets(spec)

    if _nonempty(paths["dseg"]):
        _log(f"present: {paths['dseg'].name}")
    else:
        try:
            _atomic_copy(fetch_tf_asset(dseg_asset, tf_home, fetch, s3_base, use_api), paths["dseg"])
        except (DownloadError, OSError) as err:
            errors.append(f"{spec.name} volume: {err}")

    if _nonempty(paths["labels"]):
        _log(f"present: atlases/{spec.name}/labels.tsv")
    else:
        try:
            n_rows = write_labels_tsv(fetch_tf_asset(tsv_asset, tf_home, fetch, s3_base, use_api), paths["labels"])
            if n_rows != spec.n_parcels:
                _log(f"WARNING {spec.name}: label table has {n_rows} rows, expected {spec.n_parcels}")
        except (DownloadError, OSError, ValueError, KeyError) as err:
            errors.append(f"{spec.name} labels: {err}")

    if surface:
        if _nonempty(paths["dlabel"]):
            _log(f"present: {paths['dlabel'].name}")
        else:
            try:
                fetch(spec.dlabel_url(github_raw), paths["dlabel"])
            except DownloadError as err:
                errors.append(f"{spec.name} dlabel: {err}")
    return errors


def fetch_surface_assets(resource_dir: Path, fetch: Downloader, s3_base: str, github_raw: str,
                         use_api: bool = True) -> list[str]:
    errors: list[str] = []
    resource_dir = Path(resource_dir)
    tf_home = resource_dir / "templateflow"
    for asset in surface_tf_assets():
        try:
            fetch_tf_asset(asset, tf_home, fetch, s3_base, use_api)
        except (DownloadError, OSError) as err:
            errors.append(f"{asset.key}: {err}")
    for url_asset in hcp_assets():
        try:
            fetch_url_asset(url_asset, resource_dir, fetch, github_raw)
        except DownloadError as err:
            if url_asset.key.startswith("hcp_atlasroi_") and _atlasroi_from_templateflow(url_asset, resource_dir):
                continue
            errors.append(f"{url_asset.key}: {err}")
    return errors


def _atlasroi_from_templateflow(asset: URLAsset, resource_dir: Path) -> bool:
    """Fallback for the HCP medial-wall ROI: TemplateFlow ``desc-nomedialwall`` label -> metric."""
    hemi = asset.key[-1]
    label = locate(resource_dir, [f"fslr_nomedialwall_{hemi}"])[f"fslr_nomedialwall_{hemi}"]
    if label is None:
        return False
    from fmriproc.surface_utils import label_gii_to_roi

    n_cortex = label_gii_to_roi(label, asset.path(resource_dir), STRUCTURES[hemi])
    _log(f"{asset.relpath}: derived from {label.name} ({n_cortex} cortical vertices)")
    return True


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _yes(value: str) -> bool:
    return str(value).strip().lower() in ("yes", "true", "1", "on")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fmriproc.fetch_resources",
        description="Download templates/atlases once into the resource directory (needs network).",
    )
    parser.add_argument("--resource-dir", required=True, type=Path, help="$RESOURCE_DIR")
    parser.add_argument("--atlases", nargs="*", default=[], metavar="NAME",
                        help="atlas names (ATLASES); only Schaefer2018_<N>Parcels_<M>Networks are fetched")
    parser.add_argument("--surface", default="yes", choices=["yes", "no"],
                        help="no = skip fsLR meshes, HCP files and CIFTI dlabels")
    parser.add_argument("--check", action="store_true", help="only report missing files (exit 1 if any)")
    parser.add_argument("--locate", nargs="+", metavar="KEY",
                        help=f"print the path of resources, one per line (exit 1 if one is absent); keys: {' '.join(known_keys())}")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per connection")
    parser.add_argument("--no-templateflow-api", action="store_true",
                        help="do not call templateflow.api.get, use the S3 URLs directly")
    parser.add_argument("--github-raw", default=os.environ.get("FMRIPROC_GITHUB_RAW", GITHUB_RAW),
                        help="base URL replacing https://raw.githubusercontent.com (mirror)")
    parser.add_argument("--templateflow-s3", default=os.environ.get("FMRIPROC_TEMPLATEFLOW_S3", TEMPLATEFLOW_S3),
                        help="base URL replacing https://templateflow.s3.amazonaws.com (mirror)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resource_dir: Path = args.resource_dir.resolve()
    surface = _yes(args.surface)

    if args.locate:
        try:
            table = locate(resource_dir, args.locate)
        except KeyError as err:
            _log(f"unknown key {err}; known: {' '.join(known_keys())}")
            return 2
        absent = [key for key, path in table.items() if path is None]
        if absent:
            _log(f"not found under {resource_dir}: {' '.join(absent)} (run stages/fetch_resources.sh)")
            return 1
        for key in args.locate:
            print(table[key])
        return 0

    try:
        specs, other = split_atlas_names(args.atlases)
    except ValueError as err:
        _log(f"ERROR {err}")
        return 2
    for name in other:
        _log(f"WARNING atlas '{name}' is not a Schaefer2018 name: nothing to fetch (use CUSTOM_ATLASES for files)")

    if args.check:
        missing = missing_files(resource_dir, specs, surface)
        total = len(expected_files(resource_dir, specs, surface))
        for desc, path in missing:
            print(f"MISSING\t{desc}\t{path}")
        _log(f"{total - len(missing)}/{total} resources present under {resource_dir}")
        return 1 if missing else 0

    def fetch(url: str, dest: Path) -> None:
        download(url, dest, retries=args.retries, timeout=args.timeout)

    use_api = not args.no_templateflow_api
    errors: list[str] = []
    if surface:
        errors += fetch_surface_assets(resource_dir, fetch, args.templateflow_s3, args.github_raw, use_api)
    for spec in specs:
        errors += fetch_atlas(spec, resource_dir, surface, fetch, args.templateflow_s3, args.github_raw, use_api)

    for message in errors:
        _log(f"ERROR {message}")
    missing = missing_files(resource_dir, specs, surface)
    for desc, path in missing:
        _log(f"still missing: {desc} -> {path}")
    if errors or missing:
        _log("incomplete. Check the network/proxy (https_proxy), or download the files listed above by hand "
             "to exactly these paths; then rerun with --check.")
        return 1
    _log(f"all resources present under {resource_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
