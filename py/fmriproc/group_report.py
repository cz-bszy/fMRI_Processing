"""Dataset-level QC (docs/DESIGN.md section 9, "Group").

    python -m fmriproc.group_report --deriv-dir $OUT_DIR/derivatives --manifest rawdata/manifest.tsv \
        --template MNI152NLin6Asym --atlas-volume A=/path/A_dseg.nii.gz --qcfc-min-subjects 10

Collects every ``<RUN>_desc-qc_metrics.json`` and writes into ``derivatives/group``:

* ``group_qc.tsv``       one row per run (subject, run, group/site from the manifest, every
                         flat metric, the flags, robust z-scores and outlier flags), worst runs first
* ``qcfc_<S>_<A>.tsv``   edge-wise QC-FC (Pearson r between mean FD and Fisher-z FC across subject means)
* ``qcfc_summary.tsv``   % edges with p < 0.05 (uncorrected), median |r|, distance dependence
* ``figures/*.png``      site-wise distributions, QC-FC figures
* ``group_report.html``  self-contained report

Robust z = 0.6745 (x - median) / MAD, within the site when it has at least
``--min-site-n`` runs, else across all runs; |z| > 3 is flagged.
QC-FC averages Fisher-z FC and FD within subject before correlation. All available
runs are included without automatic QC exclusion; site confounding remains possible.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import logging
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import BaseLoader, Environment, select_autoescape  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import ndimage, stats  # noqa: E402

from fmriproc import utils  # noqa: E402
from fmriproc.qc_metrics import FLAG_RANK, THRESHOLD_OPTIONS, THRESHOLDS, read_json_safe  # noqa: E402
from fmriproc.report import fmt  # noqa: E402

LOG = logging.getLogger("fmriproc.group_report")

DPI = 110
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS = {"pass": "#0ca30c", "warn": "#fab219", "fail": "#d03b3b", "n/a": "#898781"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

Z_METRICS = ("fd_mean", "tsnr_gm_median", "coreg_dice", "norm_dice", "pct_censored")
Z_LIMIT = 3.0
DIST_METRICS = [
    ("fd_mean", "FD mean (mm)"), ("pct_censored", "censored frames (%)"), ("minutes_retained", "retained (min)"),
    ("tsnr_gm_median", "tSNR GM, pre-denoise"), ("dvars_std_mean", "standardised DVARS"), ("gcor", "GCOR"),
    ("coreg_dice", "coregistration Dice"), ("norm_dice", "normalisation Dice"), ("dropout_fraction", "dropout fraction"),
]
ID_COLUMNS = ["subject", "session", "run", "group", "overall_flag", "n_fail", "n_warn", "n_outliers", "outlier_metrics",
              "z_scope"]

NOTES = {
    "table": "每个 run 一行，按问题严重程度排序（overall fail 在前，其次 warn 数量和 outlier 数量）。z_* 为稳健 z 分数 "
             "(0.6745 × (x − median) / MAD)：同一 site 的 run 数 ≥ 5 时在 site 内计算，否则在全部 run 中计算 (z_scope)；|z| > 3 记为 outlier。"
             "outlier 只说明该 run 与同组数据差异大，是否剔除仍需打开对应的 sub-X.html 看图确认。完整表格见 group_qc.tsv。",
    "dist": "各 site 的指标分布（箱线图 + 每个 run 一个点，点的颜色为该 run 的 overall flag；黄 / 红虚线为 QC_* 的 warn / fail 阈值）。"
            "site 之间 tSNR、FD 的系统差异在多中心数据中很常见，后续统计分析需要把 site 作为协变量或做 harmonisation。",
    "qcfc": "QC-FC 使用独立被试：同人各 run 的 Fisher-z FC 与 FD 先分别取均值，再作相关。"
            "表中同时保留原始 p 与边 family 内 BH q。距离相关仅描述，边之间不独立。"
            "不按 QC flag 自动剔除；站点与样本构成可能混杂结果，不能仅按指标高低判定策略优劣。",
    "qcfc_missing": "QC-FC 需要足够多的独立被试（QCFC_MIN_SUBJECTS）；当前不足，已跳过。",
}

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Group QC</title>
<style>
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; --page:#f9f9f7; --card:#ffffff;
        --pass:#0ca30c; --warn:#fab219; --fail:#d03b3b; --na:#898781; --accent:#2a78d6; }
* { box-sizing:border-box; }
body { margin:0; background:var(--page); color:var(--ink); font:14px/1.55 -apple-system,"Segoe UI","Microsoft YaHei","PingFang SC","Noto Sans CJK SC",Helvetica,Arial,sans-serif; }
header { background:var(--card); border-bottom:1px solid var(--line); padding:18px 28px; }
header h1 { margin:0 0 4px; font-size:22px; } header p { margin:0; color:var(--ink2); font-size:13px; }
main { max-width:1500px; margin:0 auto; padding:18px 28px 60px; }
section { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px 20px; margin:0 0 18px; }
h2 { font-size:18px; margin:2px 0 10px; } h3 { font-size:15px; margin:18px 0 8px; }
.note { background:#f3f6fb; border-left:3px solid var(--accent); padding:8px 12px; margin:8px 0 12px; color:var(--ink2); font-size:13px; }
.scroll { overflow-x:auto; max-height:640px; overflow-y:auto; }
table { border-collapse:collapse; font-size:13px; margin:6px 0 10px; }
th, td { padding:4px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th { color:var(--ink2); font-weight:600; background:#fafaf8; position:sticky; top:0; }
td:first-child, th:first-child { text-align:left; }
td.out { background:#fdf1f0; font-weight:600; }
.badge { display:inline-block; min-width:44px; padding:1px 8px; border-radius:10px; color:#fff; font-size:12px; font-weight:600; text-align:center; }
.badge.pass { background:var(--pass); } .badge.warn { background:var(--warn); color:#3a2a00; } .badge.incomplete { background:#8855aa; } .badge.fail { background:var(--fail); } .badge.na { background:var(--na); }
.tiles { display:flex; flex-wrap:wrap; gap:12px; margin:6px 0 4px; }
.tile { border:1px solid var(--line); border-radius:8px; padding:10px 16px; min-width:130px; }
.tile b { display:block; font-size:22px; } .tile span { color:var(--ink2); font-size:12px; }
figure { margin:10px 0 16px; } figure img { max-width:100%; height:auto; border:1px solid var(--line); border-radius:4px; }
.missing { color:var(--muted); font-style:italic; font-size:13px; }
a { color:var(--accent); }
footer { color:var(--muted); font-size:12px; text-align:center; padding:12px; }
</style></head><body>
{% macro badge(flag) %}<span class="badge {{ 'na' if flag == 'n/a' else flag }}">{{ flag }}</span>{% endmacro %}
<header><h1>Group QC <span style="font-weight:400;color:var(--ink2)">resting-state fMRI</span></h1>
<p>fMRI_Processing v2 &middot; {{ n_runs }} runs, {{ n_subjects }} subjects, {{ n_sites }} site(s) &middot; generated {{ generated }}</p></header>
<main>
<section><h2>Overview 总览</h2>
<div class="tiles">
  <div class="tile"><b>{{ n_runs }}</b><span>runs</span></div>
  {% for flag in ('pass', 'warn', 'fail', 'n/a') %}<div class="tile"><b>{{ counts.get(flag, 0) }}</b><span>{{ badge(flag) }} overall</span></div>{% endfor %}
  <div class="tile"><b>{{ n_outlier_runs }}</b><span>runs with a robust-z outlier</span></div>
</div>
{% if threshold_rows %}<p class="missing">flag thresholds (QC_*): {% for name, t in threshold_rows %}{{ name }} warn {{ t.warn | fmt }} / fail {{ t.fail | fmt }}{{ '; ' if not loop.last }}{% endfor %}</p>{% endif %}
{% if validation_link %}<p>Stage-10 validation report: <a href="{{ validation_link }}">{{ validation_link }}</a></p>{% endif %}
</section>

<section><h2>Runs sorted by severity 按严重程度排序</h2><div class="note">{{ notes.table }}</div>
<div class="scroll"><table><tr>{% for c in table.columns %}<th>{{ c }}</th>{% endfor %}</tr>
{% for row in table.rows %}<tr>{% for c in table.columns %}{% set v = row[c] %}
<td{% if c in row._out %} class="out"{% endif %}>{% if c == 'overall_flag' %}{{ badge(v or 'n/a') }}{% elif c == 'subject' %}<a href="../{{ v }}.html">{{ v }}</a>{% else %}{{ v | fmt }}{% endif %}</td>{% endfor %}</tr>
{% endfor %}</table></div></section>

<section><h2>Distributions per site 各 site 分布</h2><div class="note">{{ notes.dist }}</div>
{% if dist_figure %}<figure><img alt="distributions" src="data:image/png;base64,{{ dist_figure }}"></figure>{% else %}<p class="missing">no figure</p>{% endif %}
</section>

<section><h2>QC-FC</h2><div class="note">{{ notes.qcfc }}</div>
{% if qcfc_rows %}
<div class="scroll"><table><tr>{% for c in qcfc_columns %}<th>{{ c }}</th>{% endfor %}</tr>
{% for row in qcfc_rows %}<tr>{% for c in qcfc_columns %}<td>{{ row[c] | fmt }}</td>{% endfor %}</tr>{% endfor %}</table></div>
{% for item in qcfc_figures %}<h3>atlas {{ item.atlas }}</h3><figure><img alt="QC-FC {{ item.atlas }}" src="data:image/png;base64,{{ item.data }}"></figure>{% endfor %}
{% else %}<div class="note">{{ notes.qcfc_missing }} (subjects with FC: {{ qcfc_available }}, required: {{ qcfc_min }})</div>{% endif %}
</section>
</main><footer>fMRI_Processing v2 group QC &middot; {{ generated }}</footer></body></html>
"""


# ----------------------------------------------------------------------------
# table
# ----------------------------------------------------------------------------

def read_manifest(path: Path | None) -> pd.DataFrame:
    columns = ["subject", "session", "task", "run", "group", "bold", "t1w", "run_label"]
    if path is None or not Path(path).is_file():
        return pd.DataFrame(columns=columns)
    try:
        table = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig").fillna("-")
    except Exception as err:  # noqa: BLE001
        LOG.warning("cannot read manifest %s: %s", path, err)
        return pd.DataFrame(columns=columns)
    return table if "run_label" in table.columns else pd.DataFrame(columns=columns)


def _entity(run: str, key: str) -> str:
    for part in run.split("_"):
        if part.startswith(f"{key}-"):
            return part[len(key) + 1:]
    return "-"


def collect_metrics(deriv_dir: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per run: scalar metrics, ``flag.<name>`` columns, site from the manifest."""
    site = dict(zip(manifest.get("run_label", []), manifest.get("group", [])))
    rows = []
    for path in sorted(Path(deriv_dir).glob("sub-*/func/*_desc-qc_metrics.json")):
        metrics = read_json_safe(path)
        if not metrics:
            continue
        run = str(metrics.get("run") or path.name[: -len("_desc-qc_metrics.json")])
        row: dict[str, Any] = {
            "subject": metrics.get("subject") or run.split("_")[0], "session": _entity(run, "ses"), "run": run,
            "group": site.get(run, "-") or "-",
        }
        for key, value in metrics.items():
            if key in row or isinstance(value, (dict, list)):
                continue
            row[key] = value
        flags = metrics.get("flags") if isinstance(metrics.get("flags"), dict) else {}
        for name, flag in flags.items():
            row[f"flag.{name}"] = flag
        row["n_fail"] = sum(1 for f in flags.values() if f == "fail")
        row["n_warn"] = sum(1 for f in flags.values() if f == "warn")
        row.setdefault("overall_flag", "n/a")
        rows.append(row)
    return pd.DataFrame(rows)


def robust_z(values: np.ndarray) -> np.ndarray:
    """0.6745 (x - median) / MAD; mean absolute deviation when MAD is 0; NaN when there is no spread."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan)
    ok = np.isfinite(values)
    if ok.sum() < 3:
        return out
    centre = np.median(values[ok])
    mad = np.median(np.abs(values[ok] - centre))
    if mad > 0:
        out[ok] = 0.6745 * (values[ok] - centre) / mad
        return out
    mean_ad = np.mean(np.abs(values[ok] - centre))
    if mean_ad > 0:
        out[ok] = (values[ok] - centre) / (1.253314 * mean_ad)
    return out


def add_outliers(table: pd.DataFrame, min_site_n: int) -> pd.DataFrame:
    table = table.copy()
    if table.empty:
        return table
    sizes = table.groupby("group")["run"].transform("size")
    within = (sizes >= min_site_n).to_numpy() & (table["group"] != "-").to_numpy()
    table["z_scope"] = np.where(within, "site", "all")
    flagged: list[list[str]] = [[] for _ in range(len(table))]
    for metric in Z_METRICS:
        values = pd.to_numeric(table[metric], errors="coerce").to_numpy(float) if metric in table else np.full(len(table), np.nan)
        z = robust_z(values)                              # across all runs
        for site in table.loc[within, "group"].unique():
            rows = np.flatnonzero((table["group"] == site).to_numpy() & within)
            z[rows] = robust_z(values[rows])
        table[f"z_{metric}"] = z
        is_out = np.abs(z) > Z_LIMIT
        table[f"outlier_{metric}"] = is_out
        for i in np.flatnonzero(is_out):
            flagged[i].append(metric)
    table["n_outliers"] = [len(f) for f in flagged]
    table["outlier_metrics"] = [" ".join(f) if f else "-" for f in flagged]
    return table


def sort_by_severity(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    table = table.copy()
    table["_rank"] = table["overall_flag"].map(lambda f: FLAG_RANK.get(f, -1))
    table["_fd"] = pd.to_numeric(table.get("fd_mean"), errors="coerce") if "fd_mean" in table else np.nan
    table = table.sort_values(["_rank", "n_fail", "n_warn", "n_outliers", "_fd"], ascending=False, kind="stable")
    table = table.drop(columns=["_rank", "_fd"]).reset_index(drop=True)
    lead = [c for c in ID_COLUMNS if c in table.columns]
    return table[lead + [c for c in table.columns if c not in lead]]


# ----------------------------------------------------------------------------
# QC-FC
# ----------------------------------------------------------------------------

def roi_centroids(atlas_path: Path, roi_ids: np.ndarray) -> np.ndarray:
    """World-space centroids (mm) of the given labels; NaN rows for labels absent from the volume."""
    img = nib.load(str(atlas_path))
    labels = np.rint(np.asanyarray(img.dataobj)).astype(np.int64)
    while labels.ndim > 3:
        labels = labels[..., 0]
    out = np.full((len(roi_ids), 3), np.nan)
    present = np.isin(roi_ids, np.unique(labels[labels > 0]))
    if present.any():
        centres = ndimage.center_of_mass(labels > 0, labels, np.asarray(roi_ids)[present])
        out[present] = nib.affines.apply_affine(img.affine, np.asarray(centres, dtype=np.float64))
    return out


def edgewise_correlation(fd: np.ndarray, fc_z: np.ndarray, min_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pearson r (and two-sided p, n) between ``fd`` (runs) and every column of ``fc_z`` (runs x edges)."""
    fd = np.asarray(fd, dtype=np.float64)[:, None]
    valid = np.isfinite(fc_z) & np.isfinite(fd)
    n = valid.sum(axis=0)
    safe_n = np.maximum(n, 1)
    x = np.where(valid, fd, 0.0)
    y = np.where(valid, fc_z, 0.0)
    xc = np.where(valid, x - x.sum(axis=0) / safe_n, 0.0)
    yc = np.where(valid, y - y.sum(axis=0) / safe_n, 0.0)
    denom = np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom > 0, (xc * yc).sum(axis=0) / denom, np.nan)
        r = np.clip(r, -1.0, 1.0)
        t = r * np.sqrt((n - 2) / np.maximum(1.0 - r ** 2, 1e-12))
    p = 2.0 * stats.t.sf(np.abs(t), np.maximum(n - 2, 1))
    bad = (n < max(min_n, 3)) | ~np.isfinite(r)
    r[bad], p[bad] = np.nan, np.nan
    return r, p, n


def load_fc(path: Path) -> tuple[np.ndarray, list[str]] | None:
    if not path.is_file():
        return None
    try:
        frame = utils.read_tsv(path)
    except Exception as err:  # noqa: BLE001
        LOG.warning("cannot read %s: %s", path, err)
        return None
    frame = frame.drop(columns=[c for c in frame.columns if str(c).lower() in {"roi", "name", "index"}], errors="ignore")
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] < 2:
        return None
    return values, [str(c) for c in frame.columns]


def roi_ids_for(deriv_dir: Path, runs: pd.DataFrame, template: str, atlas: str, strategy: str, n_roi: int, names=None) -> np.ndarray | None:
    """Label value of every FC column, from the coverage table stage 07 writes next to the FC."""
    for subject, run in zip(runs["subject"], runs["run"]):
        path = Path(deriv_dir) / subject / "func" / f"{run}_space-{template}_atlas-{atlas}_desc-{strategy}_coverage.tsv"
        if path.is_file():
            try:
                coverage = utils.read_tsv(path)
            except Exception:  # noqa: BLE001
                continue
            if "roi" in coverage.columns and len(coverage) == n_roi:
                from fmriproc.validate import aligned_coverage
                try:
                    coverage = aligned_coverage(names, coverage)
                except ValueError:
                    continue
                return pd.to_numeric(coverage["roi"], errors="coerce").to_numpy(dtype=np.int64)
    return None


def qcfc(deriv_dir: Path, table: pd.DataFrame, template: str, strategy: str, atlas: str, atlas_volume: Path | None,
         min_runs: int) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    summary: dict[str, Any] = {"strategy": strategy, "atlas": atlas, "n_runs": 0, "n_edges": None, "pct_sig_p05": None,
                               "median_abs_r": None, "dist_dep_spearman": None, "dist_dep_p": None}
    matrices, fds, subjects, names = [], [], [], None
    for subject, run, fd in zip(table["subject"], table["run"], pd.to_numeric(table.get("fd_mean"), errors="coerce")):
        loaded = load_fc(Path(deriv_dir) / subject / "func" / f"{run}_space-{template}_atlas-{atlas}_desc-{strategy}_connectivity.tsv")
        if loaded is None or not np.isfinite(fd):
            continue
        if names is None:
            names = loaded[1]
        if loaded[0].shape[0] != len(names):
            LOG.warning("%s: FC of %s/%s has %d ROIs, expected %d - run left out", run, strategy, atlas,
                        loaded[0].shape[0], len(names))
            continue
        from fmriproc.validate import column_order
        try:
            order = column_order(names, loaded[1])
        except ValueError as err:
            LOG.warning("%s: FC identity mismatch: %s", run, err)
            continue
        matrices.append(utils.upper_triangle(utils.fisher_z(loaded[0][np.ix_(order, order)])))
        subjects.append(subject)
        fds.append(float(fd))
    summary["n_runs"] = len(matrices)
    summary["n_subjects"] = len(set(subjects))
    summary["aggregation"] = "equal subject weight; mean run Fisher-z FC and mean run FD"
    if names is None or len(set(subjects)) < max(min_runs, 3):
        return None, summary
    rows, cols = np.triu_indices(len(names), k=1)
    subject_fc = pd.DataFrame(np.vstack(matrices)).groupby(subjects, sort=True).mean()
    subject_fd = pd.Series(fds).groupby(subjects, sort=True).mean()
    r, p, n = edgewise_correlation(subject_fd.to_numpy(), subject_fc.to_numpy(), min_runs)
    edges = pd.DataFrame({"roi_i": [names[i] for i in rows], "roi_j": [names[j] for j in cols], "qcfc_r": r, "p": p, "n": n})
    from fmriproc.compare_streams import bh_adjust
    edges["q_bh"] = bh_adjust(p)
    edges["distance_mm"] = np.nan
    if atlas_volume is not None and Path(atlas_volume).is_file():
        roi_ids = roi_ids_for(deriv_dir, table, template, atlas, strategy, len(names), names)
        if roi_ids is not None:
            centres = roi_centroids(Path(atlas_volume), roi_ids)
            edges["distance_mm"] = np.linalg.norm(centres[rows] - centres[cols], axis=1)
    ok = np.isfinite(r)
    summary["n_edges"] = int(ok.sum())
    if ok.any():
        summary["pct_sig_p05"] = float(100.0 * np.mean(p[ok] < 0.05))
        summary["pct_sig_q05"] = float(100.0 * np.mean(edges.loc[ok, "q_bh"] < 0.05))
        summary["median_abs_r"] = float(np.median(np.abs(r[ok])))
    both = ok & np.isfinite(edges["distance_mm"].to_numpy())
    if both.sum() >= 10:
        rho, p_rho = stats.spearmanr(edges["distance_mm"].to_numpy()[both], r[both])
        summary["dist_dep_spearman"], summary["dist_dep_p"] = float(rho), None  # Edges are dependent; no naive edge-wise significance test.
    return edges, summary


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------

def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".tmp{os.getpid()}_{out.name}")
    fig.savefig(tmp, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor(), format="png")
    plt.close(fig)
    os.replace(tmp, out)
    return out


def plot_distributions(table: pd.DataFrame, out: Path, extra: list[tuple[str, str]],
                       thresholds: dict[str, tuple[str, float, float]] | None = None) -> Path | None:
    """Box + strip plot per site for every metric; dashed lines = warn / fail thresholds."""
    metrics = [(k, label) for k, label in [*DIST_METRICS, *extra]
               if k in table.columns and pd.to_numeric(table[k], errors="coerce").notna().any()]
    if table.empty or not metrics:
        return None
    sites = sorted(table["group"].astype(str).unique())
    n_cols = 3
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 2.9 * n_rows), facecolor=SURFACE, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (key, label) in zip(axes.ravel(), metrics):
        _style(ax)
        values = pd.to_numeric(table[key], errors="coerce")
        data = [values[(table["group"].astype(str) == s) & values.notna()].to_numpy() for s in sites]
        shown = [(k, d) for k, d in enumerate(data) if d.size]
        if shown:
            ax.boxplot([d for _, d in shown], positions=[k for k, _ in shown], widths=0.55, showfliers=False,
                       medianprops={"color": INK, "linewidth": 1.4}, boxprops={"color": MUTED}, whiskerprops={"color": MUTED},
                       capprops={"color": MUTED})
        for k, site in enumerate(sites):
            rows = table[(table["group"].astype(str) == site) & values.notna()]
            jitter = rng.uniform(-0.16, 0.16, len(rows))
            colours = [STATUS.get(f, STATUS["n/a"]) for f in rows["overall_flag"]]
            ax.scatter(k + jitter, values[rows.index], s=20, c=colours, edgecolors="white", linewidths=0.5, zorder=3)
        spec = (thresholds or {}).get(key)
        if spec is not None:
            for level, flag in ((spec[1], "warn"), (spec[2], "fail")):
                if level is not None and np.isfinite(level):
                    ax.axhline(level, color=STATUS[flag], linewidth=0.9, linestyle="--", zorder=2)
        # two-call form: the container's matplotlib may predate set_xticks(ticks, labels)
        ax.set_xticks(range(len(sites)))
        ax.set_xticklabels(sites, rotation=35 if len(sites) > 3 else 0, ha="right" if len(sites) > 3 else "center",
                           fontsize=8)
        ax.set_xlim(-0.6, len(sites) - 0.4)
        ax.set_title(label, fontsize=9, loc="left", color=INK)
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=6, markerfacecolor=c, markeredgecolor="white",
                      label=f"overall {name}") for name, c in STATUS.items()]
    fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("QC metrics per site (one dot per run)", fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out)


def plot_qcfc(atlas: str, results: dict[str, pd.DataFrame], out: Path) -> Path | None:
    results = {s: e for s, e in results.items() if e is not None and e["qcfc_r"].notna().any()}
    if not results:
        return None
    n = len(results)
    fig, axes = plt.subplots(1, n + 1, figsize=(4.2 * (n + 1), 3.8), facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    _style(axes[0])
    bins = np.linspace(-1, 1, 81)
    for k, (strategy, edges) in enumerate(results.items()):
        r = edges["qcfc_r"].dropna().to_numpy()
        axes[0].hist(r, bins=bins, histtype="step", density=True, linewidth=1.6, color=SERIES[k % len(SERIES)],
                     label=f"{strategy}: median |r| {np.median(np.abs(r)):.3f}")
    axes[0].axvline(0, color=MUTED, linewidth=0.8)
    axes[0].set_xlabel("QC-FC r (mean FD vs. edge z)", fontsize=8)
    axes[0].set_ylabel("density", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    axes[0].set_title("distribution over edges", fontsize=9, loc="left")
    for k, (ax, (strategy, edges)) in enumerate(zip(axes[1:], results.items())):
        _style(ax)
        ok = edges["qcfc_r"].notna() & edges["distance_mm"].notna()
        if ok.sum() >= 10:
            d, r = edges.loc[ok, "distance_mm"].to_numpy(), edges.loc[ok, "qcfc_r"].to_numpy()
            ax.scatter(d, r, s=3, color=SERIES[k % len(SERIES)], alpha=0.25, linewidths=0)
            slope, intercept = np.polyfit(d, r, 1)
            xs = np.array([d.min(), d.max()])
            ax.plot(xs, slope * xs + intercept, color=INK, linewidth=1.2)
            rho = stats.spearmanr(d, r)[0]
            ax.set_title(f"{strategy}: distance dependence rho = {rho:.3f}", fontsize=9, loc="left")
        else:
            ax.set_title(f"{strategy}: no ROI distances", fontsize=9, loc="left")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xlabel("distance between ROI centroids (mm)", fontsize=8)
        ax.set_ylabel("QC-FC r", fontsize=8)
    fig.suptitle(f"QC-FC, atlas {atlas}", fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out)


def _b64(path: Path | None) -> str | None:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii") if path and Path(path).is_file() else None


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def parse_pairs(items: list[str] | None) -> dict[str, Path]:
    """``name=path`` items (one per --atlas-volume option) -> {name: path}."""
    pairs = {}
    for item in items or []:
        if "=" in item:
            name, path = item.split("=", 1)
            pairs[name.strip()] = Path(path)
        else:
            LOG.warning("--atlas-volume expects name=path, got %r", item)
    return pairs


def find_atlas_volume(atlas_dir: Path | None, atlas: str, space: str) -> Path | None:
    """Label volume of an atlas under $RESOURCE_DIR/atlases/<A>/ (the fetch_resources layout)."""
    if atlas_dir is None or not Path(atlas_dir).is_dir():
        return None
    base = Path(atlas_dir) / atlas
    candidates = [base / f"{atlas}_space-{space}_res-02_dseg.nii.gz"]
    candidates += sorted(base.glob(f"{atlas}_space-*_dseg.nii*")) + sorted(base.glob("*_dseg.nii*"))
    for path in candidates:
        if path.is_file():
            return path
    return None


def threshold_specs(args: argparse.Namespace) -> dict[str, tuple[str, float, float]]:
    """metric -> (worse_when, warn, fail) from the --qc-* options (same names as qc_metrics)."""
    out = {}
    for name, spec in THRESHOLDS.items():
        stem = THRESHOLD_OPTIONS[name].replace("-", "_")
        out[name] = (spec[0], float(getattr(args, f"qc_{stem}_warn")), float(getattr(args, f"qc_{stem}_fail")))
    return out


def names_from(table: pd.DataFrame, column: str, given: str) -> list[str]:
    if given.split():
        return given.split()
    found: list[str] = []
    for value in table.get(column, pd.Series(dtype=str)).dropna():
        for name in str(value).split():
            if name not in found:
                found.append(name)
    return found


def html_table(table: pd.DataFrame, strategies: list[str], atlases: list[str]) -> dict[str, Any]:
    columns = ["subject", "run", "group", "overall_flag", "n_fail", "n_warn", "outlier_metrics", "fd_mean", "pct_censored",
               "minutes_retained", "tsnr_gm_median", "coreg_dice", "norm_dice"]
    columns += [f"{s}.dof_remaining" for s in strategies]
    if atlases:
        columns += [f"{s}.{atlases[0]}.roi_tsnr_median" for s in strategies]
    columns += [f"z_{m}" for m in Z_METRICS] + ["z_scope"]
    columns = [c for c in columns if c in table.columns]
    rows = []
    for record in table[columns].astype(object).where(table[columns].notna(), None).to_dict("records"):
        record = {k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()}
        record["_out"] = [f"z_{m}" for m in Z_METRICS
                          if isinstance(record.get(f"z_{m}"), float) and abs(record[f"z_{m}"]) > Z_LIMIT]
        rows.append(record)
    return {"columns": columns, "rows": rows}


def run(args: argparse.Namespace) -> int:
    deriv_dir = Path(args.deriv_dir)
    out_dir = Path(args.out_dir) if args.out_dir else deriv_dir / "group"
    fig_dir = out_dir / "figures"
    manifest = read_manifest(Path(args.manifest) if args.manifest else None)
    table = collect_metrics(deriv_dir, manifest)
    if table.empty:
        LOG.warning("no *_desc-qc_metrics.json under %s: nothing to summarise", deriv_dir)
        table = pd.DataFrame(columns=["subject", "session", "run", "group", "overall_flag", "n_fail", "n_warn"])
    table = sort_by_severity(add_outliers(table, args.min_site_n))
    out_dir.mkdir(parents=True, exist_ok=True)
    utils.write_tsv(out_dir / "group_qc.tsv", table)

    strategies = names_from(table, "strategies", args.strategies)
    atlases = names_from(table, "atlases", args.atlases)
    template = args.template or (str(table["template"].dropna().iloc[0]) if "template" in table and table["template"].notna().any() else "")
    volumes = parse_pairs(args.atlas_volume)
    for atlas in atlases:
        if atlas not in volumes:
            found = find_atlas_volume(Path(args.atlas_dir) if args.atlas_dir else None, atlas, args.atlas_space)
            if found is not None:
                volumes[atlas] = found
            else:
                LOG.info("atlas %s: no label volume found (QC-FC distance dependence will be n/a)", atlas)
    thresholds = threshold_specs(args)

    extra = [(f"{s}.{atlases[0]}.roi_tsnr_median", f"ROI tSNR (final), {s}") for s in strategies] if atlases else []
    extra += [(f"{s}.tsnr_gm_median_post", f"tSNR post, {s}") for s in strategies]
    dist_png = (plot_distributions(table, fig_dir / "group_distributions.png", extra, thresholds)
                if not table.empty else None)

    summaries, figures, available = [], [], 0
    for stale in out_dir.glob("qcfc_*.tsv"):
        stale.unlink()                                    # a strategy that was dropped must not linger
    for atlas in atlases:
        per_strategy: dict[str, pd.DataFrame] = {}
        for strategy in strategies:
            edges, summary = qcfc(deriv_dir, table, template, strategy, atlas, volumes.get(atlas), args.qcfc_min_subjects)
            available = max(available, int(summary["n_subjects"]))
            if edges is None:
                continue
            utils.write_tsv(out_dir / f"qcfc_{strategy}_{atlas}.tsv", edges)
            summaries.append(summary)
            per_strategy[strategy] = edges
        png = plot_qcfc(atlas, per_strategy, fig_dir / f"qcfc_{atlas}.png")
        if png is not None:
            figures.append({"atlas": atlas, "data": _b64(png)})
    if summaries:
        utils.write_tsv(out_dir / "qcfc_summary.tsv", pd.DataFrame(summaries))

    env = Environment(loader=BaseLoader(), autoescape=select_autoescape(default_for_string=True, default=True))
    env.filters["fmt"] = fmt
    counts = table["overall_flag"].value_counts().to_dict() if "overall_flag" in table else {}
    html = env.from_string(TEMPLATE).render(
        generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), n_runs=len(table),
        n_subjects=int(table["subject"].nunique()) if len(table) else 0,
        n_sites=int(table["group"].nunique()) if len(table) else 0, counts=counts,
        n_outlier_runs=int((table["n_outliers"] > 0).sum()) if "n_outliers" in table else 0,
        table=html_table(table, strategies, atlases) if len(table) else {"columns": [], "rows": []},
        dist_figure=_b64(dist_png), notes=NOTES, qcfc_rows=summaries,
        qcfc_columns=list(summaries[0].keys()) if summaries else [], qcfc_figures=figures,
        qcfc_available=available, qcfc_min=args.qcfc_min_subjects,
        threshold_rows=[(name, {"warn": spec[1], "fail": spec[2]}) for name, spec in thresholds.items()],
        validation_link="validation_report.html" if (out_dir / "validation_report.html").is_file() else None,
    )
    utils._atomic_write(out_dir / "group_report.html", html)
    LOG.info("group QC: %d runs -> %s", len(table), out_dir / "group_report.html")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deriv-dir", required=True, help="$OUT_DIR/derivatives")
    parser.add_argument("--manifest", default=None, help="rawdata/manifest.tsv (site of every run)")
    parser.add_argument("--out-dir", default=None, help="default: <deriv-dir>/group")
    parser.add_argument("--template", default="", help="template name in the FC file names; default: from the metrics")
    parser.add_argument("--strategies", default="", help="default: union of the strategies in the metrics files")
    parser.add_argument("--atlases", default="", help="default: union of the atlases in the metrics files")
    parser.add_argument("--atlas-dir", default=None,
                        help="$RESOURCE_DIR/atlases: <A>/<A>_space-<SPACE>_res-02_dseg.nii.gz gives the ROI centroids")
    parser.add_argument("--atlas-space", default="MNI152NLin6Asym", help="space label in the fetched atlas file names")
    parser.add_argument("--atlas-volume", action="append", default=[], metavar="NAME=PATH",
                        help="atlas label volume in template space (custom atlases; wins over --atlas-dir); repeatable")
    parser.add_argument("--qcfc-min-subjects", type=int, default=10)
    parser.add_argument("--min-site-n", type=int, default=5, help="runs per site needed for within-site z-scores")
    for name, (_, warn, fail) in THRESHOLDS.items():
        stem = THRESHOLD_OPTIONS[name]
        parser.add_argument(f"--qc-{stem}-warn", type=float, default=warn, help=f"{name}: warn threshold (lines in the plots)")
        parser.add_argument(f"--qc-{stem}-fail", type=float, default=fail, help=f"{name}: fail threshold")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    if not Path(args.deriv_dir).is_dir():
        LOG.error("derivatives directory not found: %s", args.deriv_dir)
        return 2
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
