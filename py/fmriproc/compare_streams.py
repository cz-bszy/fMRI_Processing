"""Group-level validation: volume-vs-surface and strategy comparison (stage 10 --group).

    python -m fmriproc.compare_streams --deriv-dir OUT/derivatives \
        --manifest OUT/rawdata/manifest.tsv --out-dir OUT/derivatives/group

Reads every ``*_desc-validation.tsv`` / ``*_desc-streamcompare.tsv`` (stage 10,
per run) and every ``*_connectivity.tsv`` (stage 07) below ``--deriv-dir`` and writes

    validation_long.tsv      all runs, long format (+ subject, run_label, group, included)
    fc_typicality.tsv        r(run FC, leave-one-out group-mean FC) per stream x strategy x atlas
    stream_comparison.tsv    per strategy x atlas x metric: paired volume-vs-surface statistics
    strategy_comparison.tsv  per stream x atlas x metric: median per denoising strategy
    stream_comparison.png    paired dot plots, colour = site
    validation_report.html   self-contained report (Chinese, technical terms in English)

The comparisons use the runs that the inclusion criteria (fmriproc.inclusion,
--exclude-* options) keep for each strategy; validation_long.tsv keeps every run.
"""
from __future__ import annotations

import argparse
import base64
import html
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from fmriproc.utils import NA, fisher_z, read_tsv, upper_triangle, write_tsv  # noqa: E402

HIGHER, LOWER, DESCRIPTIVE = "higher", "lower", "descriptive"
DIRECTION = {
    "roi_tsnr_median": HIGHER, "roi_tsnr_p10": HIGHER, "split_half_r": HIGHER, "network_contrast": HIGHER,
    "homotopic_contrast": HIGHER, "dmn_contrast": HIGHER, "fc_typicality": HIGHER,
    "fd_fc_coupling": LOWER, "n_roi_nan": LOWER,
    "variance_removed_median": DESCRIPTIVE, "lowfreq_power_fraction": DESCRIPTIVE,
    "gs_residual_sd": DESCRIPTIVE, "fc_similarity": DESCRIPTIVE, "n_roi_common": DESCRIPTIVE,
    "n_retained": DESCRIPTIVE, "dof_remaining": DESCRIPTIVE,
}
METRIC_ORDER = list(DIRECTION)
UNPAIRED = ("fc_similarity", "n_roi_common")
FIGURE_METRICS = (
    "roi_tsnr_median", "split_half_r", "network_contrast", "homotopic_contrast",
    "dmn_contrast", "fc_typicality", "fd_fc_coupling", "variance_removed_median",
)
MIN_WILCOXON = 6
MIN_TYPICALITY_RUNS = 4
MAX_FIGURE_ROWS = 6
ALPHA = 0.05

# categorical slots in fixed order (validated light-surface palette); further sites fold into "other"
SITE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
OTHER_COLOR = "#8a8a84"
INK, INK_MUTED, GRID = "#1a1a19", "#5f5e58", "#e4e3de"

LONG_COLUMNS = ["subject", "run_label", "group", "stream", "strategy", "atlas", "metric", "value"]
OUTPUT_LONG_COLUMNS = LONG_COLUMNS + ["included"]
PAIR_COLUMNS = ["subject", "run_label", "group", "strategy", "atlas", "metric", "volume", "surface"]


def log(message: str) -> None:
    print(f"[compare_streams] {message}", file=sys.stderr)


# ----------------------------------------------------------------------------
# gathering
# ----------------------------------------------------------------------------

def load_manifest(path: str | Path | None) -> dict[str, tuple[str, str]]:
    """run_label -> (subject, group); subjects are also keyed on their own."""
    out: dict[str, tuple[str, str]] = {}
    if not path or not Path(path).is_file():
        log("no manifest: the group/site column is n/a")
        return out
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for row in frame.to_dict("records"):
        subject = str(row.get("subject", "")).strip()
        group = str(row.get("group", "")).strip()
        group = NA if group in ("", "-") else group
        label = str(row.get("run_label", "")).strip()
        if label:
            out[label] = (subject, group)
        out.setdefault(subject, (subject, group))
    return out


def identify(run_label: str, path: Path, deriv_dir: Path, manifest: dict[str, tuple[str, str]]) -> tuple[str, str]:
    """(subject, group) of a run: manifest first, then the sub-X directory below `deriv_dir`."""
    try:
        subject = path.relative_to(deriv_dir).parts[0]
    except ValueError:
        subject = run_label.split("_", 1)[0]
    if run_label in manifest:
        return manifest[run_label][0] or subject, manifest[run_label][1]
    if subject in manifest:
        return subject, manifest[subject][1]
    return subject, NA


def run_files(deriv_dir: Path, suffix: str) -> list[Path]:
    return sorted(p for p in Path(deriv_dir).glob(f"sub-*/**/*{suffix}") if p.is_file())


def gather_validation(deriv_dir: Path, manifest: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """All ``*_desc-validation.tsv`` stacked, with subject / run_label / group in front."""
    frames = []
    for path in run_files(deriv_dir, "_desc-validation.tsv"):
        run_label = path.name[: -len("_desc-validation.tsv")]
        try:
            frame = read_tsv(path, dtype={"stream": str, "strategy": str, "atlas": str, "metric": str})
            frame = frame[["stream", "strategy", "atlas", "metric", "value"]].copy()
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as err:
            log(f"WARNING: cannot read {path}: {err}")
            continue
        subject, group = identify(run_label, path, Path(deriv_dir), manifest)
        frame.insert(0, "group", group)
        frame.insert(0, "run_label", run_label)
        frame.insert(0, "subject", subject)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=LONG_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out[LONG_COLUMNS]


def gather_streamcompare(deriv_dir: Path, manifest: dict[str, tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run-scope rows of every ``*_desc-streamcompare.tsv``.

    Returns (pairs with volume and surface values over the common ROIs,
    unpaired comparison metrics such as fc_similarity as long rows).
    """
    pairs, single = [], []
    for path in run_files(deriv_dir, "_desc-streamcompare.tsv"):
        run_label = path.name[: -len("_desc-streamcompare.tsv")]
        try:
            frame = read_tsv(path, dtype={"strategy": str, "atlas": str, "scope": str, "metric": str, "roi": str})
            frame = frame[frame["scope"] == "run"]
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as err:
            log(f"WARNING: cannot read {path}: {err}")
            continue
        subject, group = identify(run_label, path, Path(deriv_dir), manifest)
        for row in frame.to_dict("records"):
            head = (subject, run_label, group, row["strategy"], row["atlas"], row["metric"])
            if row["metric"] in UNPAIRED:
                single.append(head + (pd.to_numeric(row["value"], errors="coerce"),))
            else:
                pairs.append(head + (pd.to_numeric(row["volume"], errors="coerce"),
                                     pd.to_numeric(row["surface"], errors="coerce")))
    pair_frame = pd.DataFrame(pairs, columns=PAIR_COLUMNS)
    single_frame = pd.DataFrame(single, columns=PAIR_COLUMNS[:6] + ["value"])
    return pair_frame, single_frame


# ----------------------------------------------------------------------------
# FC typicality
# ----------------------------------------------------------------------------

def parse_connectivity_name(name: str) -> tuple[str, str, str, str] | None:
    """``<RUN>_space-<SPACE>_atlas-<A>_desc-<S>_connectivity.tsv`` -> (run, stream, atlas, strategy).

    Atlas names may contain '_', so the atlas is everything up to the last '_desc-'.
    """
    suffix = "_connectivity.tsv"
    if not name.endswith(suffix) or "_space-" not in name or "_atlas-" not in name:
        return None
    run_label, rest = name[: -len(suffix)].split("_space-", 1)
    if "_atlas-" not in rest:
        return None
    space, rest = rest.split("_atlas-", 1)
    if "_desc-" not in rest:
        return None
    atlas, strategy = rest.rsplit("_desc-", 1)
    if not (run_label and space and atlas and strategy):
        return None
    return run_label, ("surface" if space.startswith("fsLR") else "volume"), atlas, strategy


def loo_typicality(vectors: np.ndarray, min_others: int = MIN_TYPICALITY_RUNS - 1) -> np.ndarray:
    """r between every row of `vectors` (runs x edges, NaN allowed) and the mean of the
    OTHER rows; an edge enters only where at least `min_others` other runs have it."""
    vectors = np.asarray(vectors, dtype=np.float64)
    finite = np.isfinite(vectors)
    total = np.where(finite, vectors, 0.0).sum(axis=0)
    count = finite.sum(axis=0)
    out = np.full(vectors.shape[0], np.nan)
    for k in range(vectors.shape[0]):
        others = count - finite[k]
        ok = finite[k] & (others >= min_others)
        if ok.sum() < 3:
            continue
        mean_others = (total[ok] - vectors[k, ok]) / others[ok]
        mine = vectors[k, ok]
        if mine.std() > 0 and mean_others.std() > 0:
            out[k] = np.corrcoef(mine, mean_others)[0, 1]
    return out


def read_fc_vector(path: Path, names=None) -> np.ndarray | None:
    """Fisher-z upper triangle (float32) of a square connectivity TSV; None when unusable."""
    try:
        frame = read_tsv(path)
        from fmriproc.validate import column_order
        order = column_order(frame.columns if names is None else names, frame.columns)
        matrix = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        matrix = matrix[np.ix_(order, order)]
    except (OSError, ValueError, pd.errors.ParserError) as err:
        log(f"WARNING: cannot read {path}: {err}")
        return None
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 3:
        log(f"WARNING: {path.name}: not a square matrix ({matrix.shape}), ignored")
        return None
    return upper_triangle(fisher_z(matrix)).astype(np.float32)


def fc_typicality(deriv_dir: Path, manifest: dict[str, tuple[str, str]], min_runs: int = MIN_TYPICALITY_RUNS) -> pd.DataFrame:
    """One row per run x stream x strategy x atlas; combos with fewer than `min_runs` runs are left out."""
    combos: dict[tuple[str, str, str], list[tuple[str, Path]]] = {}
    for path in run_files(deriv_dir, "_connectivity.tsv"):
        parsed = parse_connectivity_name(path.name)
        if parsed is None:
            continue
        run_label, stream, atlas, strategy = parsed
        if strategy == "preproc":      # pre-denoise FC is not a strategy (stage 07 normally does not write it)
            continue
        combos.setdefault((stream, strategy, atlas), []).append((run_label, path))
    rows = []
    for (stream, strategy, atlas), items in sorted(combos.items()):
        names = read_tsv(items[0][1]).columns
        loaded = [(run_label, path, read_fc_vector(path, names)) for run_label, path in items]
        loaded = [item for item in loaded if item[2] is not None]
        if not loaded:
            continue
        sizes = pd.Series([item[2].size for item in loaded])
        size = int(sizes.mode().iloc[0])
        if (sizes != size).any():
            log(f"WARNING: {stream}/{strategy}/{atlas}: {int((sizes != size).sum())} run(s) with another ROI count ignored")
        loaded = [item for item in loaded if item[2].size == size]
        if len(loaded) < min_runs:
            log(f"{stream}/{strategy}/{atlas}: {len(loaded)} run(s), fc_typicality needs >= {min_runs}")
            continue
        subjects = [identify(run_label, path, Path(deriv_dir), manifest)[0] for run_label, path, _ in loaded]
        vectors = np.stack([item[2] for item in loaded])
        subject_means = pd.DataFrame(vectors).groupby(subjects).mean()
        values = []
        for subject, vector in zip(subjects, vectors):
            others = subject_means.drop(index=subject).to_numpy()
            values.append(loo_typicality(np.vstack([vector, others]), min_others=min_runs - 1)[0])
        for (run_label, path, _), value in zip(loaded, values):
            subject, group = identify(run_label, path, Path(deriv_dir), manifest)
            rows.append((subject, run_label, group, stream, strategy, atlas, len(loaded), value))
    return pd.DataFrame(rows, columns=["subject", "run_label", "group", "stream", "strategy", "atlas", "n_runs", "fc_typicality"])


# ----------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------

def pairs_from_long(long: pd.DataFrame) -> pd.DataFrame:
    """Volume/surface pairs from the long table (runs that have both streams)."""
    if long.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    keys = PAIR_COLUMNS[:6]
    sides = {}
    for stream in ("volume", "surface"):
        part = long.loc[long["stream"] == stream, keys + ["value"]].drop_duplicates(keys)
        sides[stream] = part.rename(columns={"value": stream})
    return sides["volume"].merge(sides["surface"], on=keys, how="inner")[PAIR_COLUMNS]


def build_pairs(long: pd.DataFrame, compare_pairs: pd.DataFrame) -> pd.DataFrame:
    """Paired values: the common-ROI values of the streamcompare files win; whatever
    they do not cover (fc_typicality, runs without such a file) comes from the long table."""
    fallback = pairs_from_long(long[long["metric"].eq("fc_typicality")])
    keys = ["run_label", "strategy", "atlas", "metric"]
    if compare_pairs.empty:
        merged = fallback
    else:
        covered = set(map(tuple, compare_pairs[keys].to_numpy().tolist()))
        extra = fallback[[tuple(k) not in covered for k in fallback[keys].to_numpy().tolist()]]
        merged = pd.concat([compare_pairs, extra], ignore_index=True)
    merged = merged[np.isfinite(merged["volume"].astype(float)) & np.isfinite(merged["surface"].astype(float))]
    return merged.reset_index(drop=True)


def bh_adjust(pvalues):
    """Benjamini-Hochberg q values for the finite tests in one declared family."""
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full(pvalues.shape, np.nan)
    ids = np.flatnonzero(np.isfinite(pvalues))
    order = ids[np.argsort(pvalues[ids])]
    if len(order):
        out[order] = np.minimum(1, np.minimum.accumulate((pvalues[order] * len(order) / np.arange(1, len(order)+1))[::-1])[::-1])
    return out


def wilcoxon_p(differences: np.ndarray, min_n: int = MIN_WILCOXON) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if differences.size < min_n or not np.any(differences != 0):
        return float("nan")
    try:
        return float(stats.wilcoxon(differences).pvalue)
    except ValueError:
        return float("nan")


def better_stream(metric: str, median_diff: float) -> str:
    """Which stream wins given the direction of the metric (difference = surface - volume)."""
    direction = DIRECTION.get(metric, DESCRIPTIVE)
    if direction == DESCRIPTIVE:
        return DESCRIPTIVE
    if not np.isfinite(median_diff):
        return NA
    if median_diff == 0:
        return "equal"
    surface_wins = (median_diff > 0) == (direction == HIGHER)
    return "surface" if surface_wins else "volume"


def _metric_rank(metric: str) -> int:
    return METRIC_ORDER.index(metric) if metric in METRIC_ORDER else len(METRIC_ORDER)


def stream_comparison(pairs: pd.DataFrame, single: pd.DataFrame) -> pd.DataFrame:
    columns = ["strategy", "atlas", "metric", "direction", "n", "median_volume", "median_surface", "median_diff",
               "n_surface_higher", "n_volume_higher", "wilcoxon_p", "significant", "better", "median_value"]
    rows = []
    # Equal subject weight: average paired run values within each subject first.
    pairs = pairs.groupby(["subject", "strategy", "atlas", "metric"], as_index=False)[["volume", "surface"]].mean()
    for (strategy, atlas, metric), block in pairs.groupby(["strategy", "atlas", "metric"], sort=False):
        volume, surface = block["volume"].to_numpy(dtype=float), block["surface"].to_numpy(dtype=float)
        diff = surface - volume
        p = wilcoxon_p(diff)
        median_diff = float(np.median(diff))
        rows.append((strategy, atlas, metric, DIRECTION.get(metric, DESCRIPTIVE), diff.size,
                     float(np.median(volume)), float(np.median(surface)), median_diff,
                     int((diff > 0).sum()), int((diff < 0).sum()), p,
                     NA if not np.isfinite(p) else ("yes" if p < ALPHA else "no"),
                     better_stream(metric, median_diff), np.nan))
    if not single.empty:
        single = single.groupby(["subject", "strategy", "atlas", "metric"], as_index=False)["value"].mean()
    for (strategy, atlas, metric), block in single.groupby(["strategy", "atlas", "metric"], sort=False):
        values = block["value"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            rows.append((strategy, atlas, metric, DESCRIPTIVE, values.size, np.nan, np.nan, np.nan,
                         NA, NA, np.nan, NA, DESCRIPTIVE, float(np.median(values))))
    table = pd.DataFrame(rows, columns=columns)
    if table.empty:
        return table
    table["wilcoxon_q"] = np.nan
    for _, indexes in table.groupby(["strategy", "atlas"]).groups.items():
        table.loc[indexes, "wilcoxon_q"] = bh_adjust(table.loc[indexes, "wilcoxon_p"].to_numpy(float))
    table["significant"] = [NA if not np.isfinite(q) else ("yes" if q < ALPHA else "no") for q in table["wilcoxon_q"]]
    table["n_subjects"] = table["n"]
    table["inference_unit"] = "subject; mean of paired runs"
    table["_rank"] = table["metric"].map(_metric_rank)
    return table.sort_values(["strategy", "atlas", "_rank"], kind="stable").drop(columns="_rank").reset_index(drop=True)


def strategy_comparison(long: pd.DataFrame) -> pd.DataFrame:
    """Descriptive distribution of subject means; legacy `best` remains unavailable."""
    columns = ["stream", "atlas", "metric", "direction", "strategy", "n", "median", "q25", "q75", "best"]
    rows = []
    data = long[np.isfinite(long["value"].astype(float))] if not long.empty else long
    if not data.empty:
        data = data.groupby(["subject", "stream", "atlas", "metric", "strategy"], as_index=False)["value"].mean()
    for (stream, atlas, metric), block in data.groupby(["stream", "atlas", "metric"], sort=False):
        direction = DIRECTION.get(metric, DESCRIPTIVE)
        best = None  # Availability differs across strategies; no automatic ranking.
        for strategy, values in block.groupby("strategy", sort=False)["value"]:
            values = values.to_numpy(dtype=float)
            flag = NA if direction == DESCRIPTIVE or best is None else ("yes" if strategy == best else "no")
            rows.append((stream, atlas, metric, direction, strategy, values.size, float(np.median(values)),
                         float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75)), flag))
    table = pd.DataFrame(rows, columns=columns)
    if table.empty:
        return table
    table["_rank"] = table["metric"].map(_metric_rank)
    return table.sort_values(["stream", "atlas", "_rank", "strategy"], kind="stable").drop(columns="_rank").reset_index(drop=True)


def verdict_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Per strategy x atlas: how many directional metrics favour each stream (BH q < ALPHA) or stay undecided."""
    rows = []
    directional = comparison[comparison["direction"] != DESCRIPTIVE] if not comparison.empty else comparison
    for (strategy, atlas), block in directional.groupby(["strategy", "atlas"], sort=False):
        significant = block["significant"] == "yes"
        rows.append((strategy, atlas, int(block["n"].max()),
                     ", ".join(block.loc[significant & (block["better"] == "surface"), "metric"]) or "-",
                     ", ".join(block.loc[significant & (block["better"] == "volume"), "metric"]) or "-",
                     ", ".join(block.loc[~significant, "metric"]) or "-"))
    return pd.DataFrame(rows, columns=["strategy", "atlas", "n_subjects", "surface_better", "volume_better", "undecided"])


# ----------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------

def site_colors(groups: pd.Series) -> dict[str, str]:
    """Fixed slot per site (alphabetical among the most frequent eight); the rest is 'other'."""
    counts = groups.astype(str).value_counts()
    chosen = sorted(counts.index[: len(SITE_COLORS)])
    return {site: SITE_COLORS[k] for k, site in enumerate(chosen)}


def direction_text(metric: str) -> str:
    return {HIGHER: "higher = better", LOWER: "lower = better"}.get(DIRECTION.get(metric, DESCRIPTIVE), "descriptive")


def plot_stream_comparison(pairs: pd.DataFrame, comparison: pd.DataFrame, out_png: Path) -> bool:
    """Paired dot plots: one row per strategy x atlas, one panel per metric. False when nothing to draw."""
    metrics = [m for m in FIGURE_METRICS if (pairs["metric"] == m).any()] if not pairs.empty else []
    if not metrics:
        return False
    combos = list(dict.fromkeys(map(tuple, pairs[["strategy", "atlas"]].to_numpy().tolist())))
    if len(combos) > MAX_FIGURE_ROWS:
        log(f"figure shows the first {MAX_FIGURE_ROWS} of {len(combos)} strategy x atlas combinations")
        combos = combos[:MAX_FIGURE_ROWS]
    colors = site_colors(pairs["group"])
    fig, axes = plt.subplots(len(combos), len(metrics), figsize=(2.3 * len(metrics), 2.5 * len(combos) + 0.8), squeeze=False)
    for r, (strategy, atlas) in enumerate(combos):
        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            block = pairs[(pairs["strategy"] == strategy) & (pairs["atlas"] == atlas) & (pairs["metric"] == metric)]
            jitter = (np.arange(len(block)) % 7 - 3) * 0.012
            for k, row in enumerate(block.itertuples(index=False)):
                color = colors.get(str(row.group), OTHER_COLOR)
                ax.plot([0 + jitter[k], 1 + jitter[k]], [row.volume, row.surface], color=color, linewidth=0.8,
                        alpha=0.55, marker="o", markersize=3.5, markeredgecolor="white", markeredgewidth=0.4, zorder=2)
            if len(block):
                ax.plot([0, 1], [block["volume"].median(), block["surface"].median()], color=INK, linewidth=2.0,
                        marker="D", markersize=5, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            stat = comparison[(comparison["strategy"] == strategy) & (comparison["atlas"] == atlas) & (comparison["metric"] == metric)]
            p_text = ""
            if len(stat) and np.isfinite(stat["wilcoxon_p"].iloc[0]):
                p_text = f", p={stat['wilcoxon_p'].iloc[0]:.3g}"
            ax.set_title(f"{metric}\n{direction_text(metric)} | n={len(block)}{p_text}", fontsize=7, color=INK, loc="left")
            ax.set_xlim(-0.35, 1.35)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["volume", "surface"], fontsize=7, color=INK_MUTED)
            ax.tick_params(axis="y", labelsize=6.5, colors=INK_MUTED, length=2)
            ax.tick_params(axis="x", length=0)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            if c == 0:
                ax.set_ylabel(f"{strategy}\n{atlas}", fontsize=7, color=INK)
    handles = [plt.Line2D([], [], color=color, marker="o", linewidth=0.8, markersize=4, label=site) for site, color in colors.items()]
    if (~pairs["group"].astype(str).isin(colors)).any():
        handles.append(plt.Line2D([], [], color=OTHER_COLOR, marker="o", linewidth=0.8, markersize=4, label="other"))
    handles.append(plt.Line2D([], [], color=INK, marker="D", linewidth=2.0, markersize=5, label="median"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 9), fontsize=7, frameon=False, title="site (group)", title_fontsize=7)
    fig.tight_layout(rect=(0, 0.8 / (2.5 * len(combos) + 0.8), 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_png.with_name(f".tmp_{out_png.name}")
    fig.savefig(tmp, dpi=130, facecolor="white", format="png")
    plt.close(fig)
    tmp.replace(out_png)
    return True


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------

# metric -> (what it measures, direction, pitfalls); Chinese with English terms
METRIC_DOC: dict[str, tuple[str, str, str]] = {
    "roi_tsnr_median / roi_tsnr_p10": (
        "ROI tSNR = 去噪前 (pre-denoise, scaled) ROI 均值序列的时间平均 ÷ 去噪后 (denoised) ROI 序列在保留帧 "
        "(retained frames) 上的 SD。median 是全部 ROI 的中位数, p10 是最差 10% 分位的 ROI。",
        "越高越好 (higher is better)。",
        "tSNR 随过度去噪 (over-aggressive denoising) 和 low-pass 单调上升: 回归量越多、通带越窄, 残差 SD 越小, "
        "但神经信号也被一起移除; surface 流的 ribbon-constrained mapping 相当于额外的空间平均, 也会抬高 tSNR。"
        "必须与 split_half_r、network_contrast 和 dof_remaining 一起读。",
    ),
    "variance_removed_median": (
        "1 − var(denoised) / var(pre-denoise, polort-detrended), ROI 中位数: 去噪 (含 bandpass) 移除的方差比例。",
        "描述性 (descriptive)。",
        "bandpass 自身就会移除大量方差, 0.5–0.9 常见; > 0.95 提示该 run 以伪影为主; 接近 0 说明去噪几乎没有作用。",
    ),
    "split_half_r": (
        "保留帧按时间对半 (contiguous halves), 两半各自的 Fisher-z FC 上三角之间的 Pearson r: FC 的 run 内可重复性 (reliability)。",
        "越高越好。",
        "每半少于 20 帧为 n/a; 扫描越短越低 (150 帧的站点天然低于 300 帧); 两半共有的伪影 (全局信号、持续的运动) "
        "同样会抬高它, 所以 reliability 高不等于 validity 高, 要对照 fd_fc_coupling 和 QC-FC。",
    ),
    "network_contrast": (
        "(网络内平均 z − 网络间平均 z) / SD(网络间), 使用 atlas 的 network 标签 (Schaefer 7/17 Networks)。",
        "越高越好。",
        "不做 GSR 时全脑正相关会抬高 between-network FC, 数值偏低; GSR 后普遍升高。跨策略比较时只在同为 GSR 或同为 "
        "non-GSR 的策略之间比。",
    ),
    "homotopic_contrast": (
        "同伦配对 (homotopic pairs) 的平均 Fisher-z FC − 其余跨半球配对的平均 z。配对: 每个 LH parcel 取同一 network 中质心 "
        "离其镜像 (x → −x) 最近的 RH parcel, 距离 ≤ 20 mm。",
        "越高越好。",
        "中线附近的 parcel 在 volume 流里左右共享体素邻域 (partial volume、插值), 会虚高; surface 流没有这种跨半球混叠。"
        "volume 略高不一定是优势, 看差值是否集中在 medial parcels。",
    ),
    "dmn_contrast": (
        "Default 网络中含 PCC/pCunPCC 的 parcel 与含 PFC 的 parcel 之间的平均 z − 这些 parcel 与 SomMot parcel 的平均 z: "
        "经典 DMN 前后节点耦合相对于感觉运动网络的分离度。",
        "越高越好。",
        "GSR 会引入负相关使其升高; 只能在同类策略内比较。自定义 atlas 没有这些名称时为 n/a。",
    ),
    "lowfreq_power_fraction": (
        "pre-denoise、polort 去趋势后的 ROI 序列中 0.01–0.1 Hz 功率占 0.01 Hz–Nyquist 功率的比例 (ROI 中位数); "
        "有 censoring 时用最小二乘谱 (Lomb-Scargle), 不做插值。",
        "描述性。",
        "依赖 TR: 白噪声基线 = 0.09 / (Nyquist − 0.01), TR=2 s 为 0.375, TR=3 s 为 0.57, 不能跨 TR 比较。高于基线说明 "
        "BOLD 样低频信号占优, 但低频漂移、呼吸和运动也落在这个频段。",
    ),
    "fd_fc_coupling": (
        "保留帧上 FD 与逐帧共波动幅度 (co-fluctuation amplitude: z-scored ROI 两两乘积即 edge time series 的 RSS) 的 "
        "|Spearman ρ|: 最终序列中残余的运动耦合。",
        "越低越好 (lower is better)。",
        "零假设下的期望约 0.8/√n_retained (150 帧 ≈ 0.065); < 0.1 可视为没有残余耦合, > 0.2 说明运动仍在驱动瞬时 FC。"
        "censoring 截断了 FD 的范围; 帧数不同的 run 之间不可直接比。",
    ),
    "gs_residual_sd": (
        "z-scored ROI 的跨 ROI 平均序列的 SD: ROI 相互独立时为 1/√R, 完全同步时为 1, 是平均 FC 的单调函数。",
        "描述性。",
        "GSR 策略应接近 1/√R; non-GSR 下数值大说明残余全局信号 (呼吸、运动、vigilance) 多, 但真实的全局神经信号也在其中。",
    ),
    "fc_similarity": (
        "同一 run 的 volume 与 surface 两个流的 Fisher-z FC 上三角之间的 Pearson r, 只用两个流都有效的 ROI。",
        "描述性。",
        "> 0.8: 两个流给出同一个 connectome, 选哪一个影响不大; < 0.6: 先查该被试的 bbregister / surface QC 和 ROI 覆盖。",
    ),
    "fc_typicality": (
        "该 run 的 FC 与 leave-one-out 组平均 FC 的 Pearson r (同一 stream × strategy × atlas, 至少 4 个 run)。",
        "越高越好。",
        "组平均混合了患者、对照和多个站点; 全组共有的伪影 (全局信号、平滑) 也会提高 typicality。",
    ),
    "n_roi_nan / n_retained / dof_remaining": (
        "覆盖率不足而为 NaN 的 ROI 数; 保留帧数; 去噪后剩余自由度 (来自 _denoise.json)。",
        "n_roi_nan 越低越好; 其余为描述性。",
        "任何指标的比较都要在可接受的 DOF (MIN_DOF) 和保留时长下进行: DOF 很低时 FC 估计本身不稳定, tSNR 却会很高。",
    ),
}

INTERPRETATION = (
    "没有任何单一指标可以做决定 (no single metric decides)。",
    "当 reliability (split_half_r)、network_contrast / homotopic_contrast / dmn_contrast 和 fc_typicality 上升, "
    "同时 fd_fc_coupling 和 QC-FC (stage 09 的 qcfc_*.tsv) 下降, 并且 dof_remaining 可接受时, 才优先选择该 stream / strategy。",
    "tSNR 单独升高不算证据: 它随过度去噪和平滑单调上升。",
    "stream_comparison 中的 volume / surface 取值在两个流都有效的 ROI 上重新计算 (common ROIs), 所以是成对可比的; "
    "Wilcoxon signed-rank 检验需要 n ≥ 6 名被试；同人配对 runs 先取均值。每个 strategy × atlas 内对指标检验作 BH 校正，显著标记依据 q < 0.05。",
    "GSR 与 non-GSR 策略的 network_contrast / dmn_contrast 不可直接互比; 建议两类各选一个并行报告。",
)

CSS = """
body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;margin:24px auto;max-width:1180px;padding:0 16px;
color:#1a1a19;background:#fff;line-height:1.55}
h1{font-size:22px}h2{font-size:17px;margin-top:32px;border-bottom:1px solid #e4e3de;padding-bottom:4px}
h3{font-size:14px;margin-top:20px}
table{border-collapse:collapse;font-size:12px;margin:8px 0 16px}
th,td{border-bottom:1px solid #e4e3de;padding:3px 8px;text-align:right;white-space:nowrap}
th{background:#f6f5f2;text-align:center}td.l,th.l{text-align:left}td.wrap{white-space:normal;text-align:left}
.surface{background:#e3eefb}.volume{background:#fde9df}.muted{color:#5f5e58}
.scroll{overflow-x:auto}img{max-width:100%}
.note{font-size:12px;color:#5f5e58}
"""


def fmt(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return NA if not np.isfinite(value) else f"{value:.4g}"
    if value is None:
        return NA
    return str(value)


def html_table(frame: pd.DataFrame, left: tuple[str, ...] = (), row_class: str | None = None, wrap: tuple[str, ...] = ()) -> str:
    if frame.empty:
        return '<p class="note">(no data)</p>'
    head = "".join(f'<th class="{"l" if c in left else ""}">{html.escape(str(c))}</th>' for c in frame.columns)
    body = []
    for row in frame.to_dict("records"):
        cls = row.get(row_class) if row_class else None
        cls = cls if cls in ("surface", "volume") and row.get("significant") == "yes" else ""
        cells = "".join(
            f'<td class="{"wrap" if c in wrap else ("l" if c in left else "")}">{html.escape(fmt(row[c]))}</td>'
            for c in frame.columns
        )
        body.append(f'<tr class="{cls}">{cells}</tr>')
    return f'<div class="scroll"><table><tr>{head}</tr>{"".join(body)}</table></div>'


def strategy_wide(table: pd.DataFrame, stream: str, atlas: str) -> pd.DataFrame:
    """metric x strategy medians of one stream x atlas; the best strategy is starred."""
    block = table[(table["stream"] == stream) & (table["atlas"] == atlas)]
    if block.empty:
        return pd.DataFrame()
    strategies = list(dict.fromkeys(block["strategy"]))
    rows = []
    for metric in dict.fromkeys(block["metric"]):
        part = block[block["metric"] == metric].set_index("strategy")
        row: dict[str, object] = {"metric": metric, "direction": part["direction"].iloc[0], "n": int(part["n"].max())}
        for strategy in strategies:
            if strategy in part.index:
                star = " *" if part.at[strategy, "best"] == "yes" else ""
                row[strategy] = f"{fmt(float(part.at[strategy, 'median']))}{star}"
            else:
                row[strategy] = NA
        rows.append(row)
    return pd.DataFrame(rows)


def build_report(
    long: pd.DataFrame, comparison: pd.DataFrame, strategies: pd.DataFrame, typicality: pd.DataFrame,
    figure_png: Path | None, qcfc_files: list[str], excluded: list[tuple[str, str]] | None = None,
) -> str:
    esc = html.escape
    parts = ['<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">',
             "<title>Validation report</title>", f"<style>{CSS}</style></head><body>",
             "<h1>时间序列验证与 volume-vs-surface 比较 (stage 10)</h1>"]
    n_runs = long["run_label"].nunique() if not long.empty else 0
    streams = sorted(set(long["stream"])) if not long.empty else []
    sites = sorted(set(long["group"].astype(str))) if not long.empty else []
    parts.append(
        "<p>Exploratory: paired tests use subject means over paired runs; BH q controls the metric family within each strategy × atlas. Raw p is retained; significant uses q &lt; 0.05. Strategy summaries are descriptive, without automatic ranking. FC typicality excludes every run of the same subject.</p>"
        f"<p>{n_runs} 个 run, {long['subject'].nunique() if not long.empty else 0} 名被试; streams: {esc(', '.join(streams) or NA)}; "
        f"strategies: {esc(', '.join(dict.fromkeys(long['strategy'])) if not long.empty else NA)}; "
        f"atlases: {esc(', '.join(dict.fromkeys(long['atlas'])) if not long.empty else NA)}; sites: {esc(', '.join(sites) or NA)}</p>"
    )
    if excluded:
        listed = ", ".join(f"{run} ({strategy})" for run, strategy in excluded[:20])
        more = f" … 共 {len(excluded)} 个" if len(excluded) > 20 else ""
        parts.append(f'<p class="note">按纳入标准 (EXCLUDE_*) 排除的 run × strategy 不进入第 2、3 节的统计: '
                     f"{esc(listed)}{esc(more)}。原因见 inclusion.tsv 与 group_report.html。</p>")
    else:
        parts.append('<p class="note">没有 run 被纳入标准 (EXCLUDE_*) 排除。</p>')
    parts.append("<h2>1. 如何解读 (interpretation rules)</h2><ul>")
    parts.extend(f"<li>{esc(rule)}</li>" for rule in INTERPRETATION)
    parts.append("</ul>")
    if qcfc_files:
        parts.append(f'<p class="note">同目录下的 QC-FC 表 (stage 09): {esc(", ".join(qcfc_files))}</p>')
    else:
        parts.append('<p class="note">未找到 stage 09 的 qcfc_*.tsv; 运行 09_group_qc 后请把 QC-FC 与本报告一起读。</p>')

    parts.append("<h2>2. Volume vs surface (paired, common ROIs)</h2>")
    if comparison.empty or "surface" not in streams:
        parts.append('<p class="note">没有同时具备 volume 与 surface 两个流的 run (SURFACE=no 或 stage 06/07 未运行), 无法比较。</p>')
    else:
        parts.append("<h3>小结: 每个 strategy × atlas 下, 哪些有方向的指标显著 (p &lt; 0.05) 支持某个流</h3>")
        parts.append(html_table(verdict_table(comparison), left=("strategy", "atlas"), wrap=("surface_better", "volume_better", "undecided")))
        parts.append('<p class="note">蓝色行 = surface 显著更好, 橙色行 = volume 显著更好; median_diff = surface − volume; '
                     "descriptive 指标不参与判断; median_value 只用于 fc_similarity / n_roi_common。</p>")
        parts.append(html_table(comparison, left=("strategy", "atlas", "metric"), row_class="better"))
        if figure_png is not None and figure_png.is_file():
            data = base64.b64encode(figure_png.read_bytes()).decode("ascii")
            parts.append(f'<img alt="paired volume-vs-surface dot plots" src="data:image/png;base64,{data}">')
            parts.append('<p class="note">每条细线是一个 run (颜色 = site), 黑色菱形 = 中位数; 每行一个 strategy × atlas, 每列一个指标。</p>')

    parts.append("<h2>3. 去噪策略比较 (within stream; median over subject means; descriptive)</h2>")
    if strategies.empty:
        parts.append('<p class="note">(no data)</p>')
    for stream in streams:
        for atlas in dict.fromkeys(strategies.loc[strategies["stream"] == stream, "atlas"]) if not strategies.empty else []:
            parts.append(f"<h3>{esc(stream)} | {esc(str(atlas))}</h3>")
            parts.append(html_table(strategy_wide(strategies, stream, atlas), left=("metric", "direction")))

    parts.append("<h2>4. FC typicality</h2>")
    if typicality.empty:
        parts.append(f'<p class="note">需要每个 stream × strategy × atlas 至少 {MIN_TYPICALITY_RUNS} 个 run 的 _connectivity.tsv。</p>')
    else:
        summary = (typicality.groupby(["stream", "strategy", "atlas"], sort=False)["fc_typicality"]
                   .agg(n="count", median="median", min="min").reset_index())
        parts.append(html_table(summary, left=("stream", "strategy", "atlas")))
        lowest = typicality.sort_values("fc_typicality").head(10)[["run_label", "group", "stream", "strategy", "atlas", "fc_typicality"]]
        parts.append("<h3>typicality 最低的 10 个 run (优先人工检查)</h3>")
        parts.append(html_table(lowest, left=("run_label", "group", "stream", "strategy", "atlas")))

    parts.append("<h2>5. 指标说明 (metric glossary)</h2>")
    glossary = pd.DataFrame(
        [(name, what, direction, pitfalls) for name, (what, direction, pitfalls) in METRIC_DOC.items()],
        columns=["metric", "含义 (what it measures)", "方向 (direction)", "常见陷阱 (pitfalls)"],
    )
    parts.append(html_table(glossary, left=("metric",), wrap=tuple(glossary.columns[1:])))
    parts.append('<p class="note">详细说明与未自动化的进一步验证思路见 docs/VALIDATION.md。</p></body></html>')
    return "\n".join(parts)


def inclusion_lookup(deriv_dir: Path, manifest_path: str, args: argparse.Namespace,
                     strategies: list[str]) -> dict[tuple[str, str], str]:
    """(run_label, strategy) -> 'yes' / 'no' from the stage-08 metrics (fmriproc.inclusion)."""
    from fmriproc import inclusion
    from fmriproc.group_report import collect_metrics, read_manifest

    table = collect_metrics(deriv_dir, read_manifest(Path(manifest_path) if manifest_path else None))
    if table.empty:
        return {}
    decision = inclusion.decide(table, inclusion.criteria_from_args(args), strategies)
    out = {}
    for row in decision.to_dict("records"):
        for strategy in strategies:
            out[(str(row["run"]), strategy)] = row.get(f"included_{strategy}", row["included"])
    return out


def mark_included(frame: pd.DataFrame, lookup: dict[tuple[str, str], str]) -> pd.Series:
    """'yes' / 'no' per row; 'n/a' when the run has no QC metrics (then it is kept)."""
    if frame.empty:
        return pd.Series([], dtype=str)
    return pd.Series([lookup.get((str(run), str(strategy)), NA)
                      for run, strategy in zip(frame["run_label"], frame["strategy"])], index=frame.index)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{path.name}")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fmriproc.compare_streams", description=__doc__.split("\n")[0])
    parser.add_argument("--deriv-dir", required=True, help="$OUT_DIR/derivatives")
    parser.add_argument("--manifest", default="", help="rawdata/manifest.tsv (site/group of every run)")
    parser.add_argument("--out-dir", required=True, help="derivatives/group")
    from fmriproc.inclusion import add_arguments
    add_arguments(parser)      # the phenotype options are accepted but only used by stage 09
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    deriv_dir, out_dir = Path(args.deriv_dir), Path(args.out_dir)
    if not deriv_dir.is_dir():
        log(f"ERROR: not a directory: {deriv_dir}")
        return 1
    manifest = load_manifest(args.manifest)
    long = gather_validation(deriv_dir, manifest)
    if long.empty:
        log(f"ERROR: no *_desc-validation.tsv below {deriv_dir}: run stages/10_validate.sh for the subjects first")
        return 1

    typicality = fc_typicality(deriv_dir, manifest)
    if not typicality.empty:
        extra = typicality.rename(columns={"fc_typicality": "value"}).assign(metric="fc_typicality")
        long = pd.concat([long, extra[LONG_COLUMNS]], ignore_index=True)
    compare_pairs, single = gather_streamcompare(deriv_dir, manifest)

    # the group statistics use the included runs only; validation_long.tsv keeps every run
    lookup = inclusion_lookup(deriv_dir, args.manifest, args, list(dict.fromkeys(long["strategy"])))
    long["included"] = mark_included(long, lookup)
    unknown = long.loc[long["included"] == NA, "run_label"].nunique()
    if unknown:
        log(f"WARNING: {unknown} run(s) without QC metrics (stage 08): inclusion not evaluated, kept")
    use = long[long["included"] != "no"]
    kept_pairs = compare_pairs[mark_included(compare_pairs, lookup) != "no"] if not compare_pairs.empty else compare_pairs
    kept_single = single[mark_included(single, lookup) != "no"] if not single.empty else single
    excluded = sorted({(r, s) for r, s, i in zip(long["run_label"], long["strategy"], long["included"]) if i == "no"})
    pairs = build_pairs(use[LONG_COLUMNS], kept_pairs)
    comparison = stream_comparison(pairs, kept_single)
    strategies = strategy_comparison(use[LONG_COLUMNS])

    write_tsv(out_dir / "validation_long.tsv", long[OUTPUT_LONG_COLUMNS])
    write_tsv(out_dir / "fc_typicality.tsv", typicality)
    write_tsv(out_dir / "stream_comparison.tsv", comparison)
    write_tsv(out_dir / "strategy_comparison.tsv", strategies)
    figure = out_dir / "stream_comparison.png"
    has_figure = plot_stream_comparison(pairs, comparison, figure)
    if not has_figure and figure.is_file():
        figure.unlink()      # belongs to an earlier configuration with a surface stream
    qcfc = sorted(p.name for p in out_dir.glob("qcfc_*.tsv"))
    write_text(out_dir / "validation_report.html",
               build_report(long, comparison, strategies, typicality, figure if has_figure else None, qcfc, excluded))
    log(f"{long['run_label'].nunique()} run(s); {pairs['run_label'].nunique() if not pairs.empty else 0} with both streams; "
        f"outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
