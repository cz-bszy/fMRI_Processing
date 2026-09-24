"""Run inclusion for the group-level steps (stages 09 and 10 --group).

Every run is judged against explicit, pre-specified criteria taken from the
per-run ``*_desc-qc_metrics.json`` (stage 08). Nothing is deleted: the decision
is written to ``derivatives/group/inclusion.tsv`` and the group statistics
(QC-FC, stream and strategy comparisons) use the included runs only.

Criteria (0 switches a numeric criterion off):

* mean FD (Power) above ``--exclude-fd-mean`` mm
* any single frame with FD above ``--exclude-fd-max`` mm
* more than ``--exclude-pct-fd-gt02`` % of frames with FD > 0.2 mm
* fewer than ``--exclude-min-retained-min`` minutes left after censoring
* per strategy: censor-aware residual DOF below ``--exclude-min-dof``
* ``--exclude-qc-fail yes``: a run-level QC flag ``fail`` or ``incomplete``
  (coregistration, normalisation, tSNR, surface reconstruction, censor
  fraction, ...; mean FD has its own criterion above)

An optional phenotype table adds a group column (for example the diagnosis), so
that exclusions and head motion can be compared between groups: in clinical
samples motion differs between groups, and exclusion can bias who remains.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from fmriproc.utils import NA

# run-level QC flags that have a dedicated criterion here and are not counted twice
OWN_CRITERION_FLAGS = {"fd_mean"}
BAD_FLAGS = ("fail", "incomplete")
MIN_GROUP_SUBJECTS = 3


@dataclass(frozen=True)
class Criteria:
    fd_mean: float = 0.5
    fd_max: float = 5.0
    pct_fd_gt02: float = 0.0
    min_retained_min: float = 4.0
    min_dof: float = 15.0
    qc_fail: bool = True

    def describe(self) -> list[str]:
        """The active criteria as short human-readable rules."""
        rules = []
        if self.fd_mean > 0:
            rules.append(f"mean FD > {self.fd_mean:g} mm")
        if self.fd_max > 0:
            rules.append(f"any frame FD > {self.fd_max:g} mm")
        if self.pct_fd_gt02 > 0:
            rules.append(f"> {self.pct_fd_gt02:g}% of frames with FD > 0.2 mm")
        if self.min_retained_min > 0:
            rules.append(f"< {self.min_retained_min:g} min retained after censoring")
        if self.min_dof > 0:
            rules.append(f"per strategy: residual DOF < {self.min_dof:g}")
        if self.qc_fail:
            rules.append("a run-level QC flag fail or incomplete")
        return rules


# ----------------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------------

def _yes_no(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"yes", "true", "1", "on"}:
        return True
    if text in {"no", "false", "0", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected yes or no: {value}")


def _non_negative(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number >= 0: {value}") from None
    if not np.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError(f"expected a number >= 0: {value}")
    return number


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The --exclude-* and --phenotype-* options shared by the group steps."""
    default = Criteria()
    group = parser.add_argument_group("run inclusion (0 = criterion off)")
    group.add_argument("--exclude-fd-mean", type=_non_negative, default=default.fd_mean, help="mm, Power FD")
    group.add_argument("--exclude-fd-max", type=_non_negative, default=default.fd_max, help="mm, any single frame")
    group.add_argument("--exclude-pct-fd-gt02", type=_non_negative, default=default.pct_fd_gt02,
                       help="%% of frames with FD > 0.2 mm")
    group.add_argument("--exclude-min-retained-min", type=_non_negative, default=default.min_retained_min,
                       help="minutes retained after censoring")
    group.add_argument("--exclude-min-dof", type=_non_negative, default=default.min_dof,
                       help="censor-aware residual DOF, per strategy")
    group.add_argument("--exclude-qc-fail", type=_yes_no, default=default.qc_fail, metavar="yes|no",
                       help="exclude runs with a run-level QC flag fail or incomplete")
    pheno = parser.add_argument_group("phenotype groups")
    pheno.add_argument("--phenotype", default="", help="TSV or CSV with one row per subject (optional)")
    pheno.add_argument("--phenotype-id-column", default="participant_id",
                       help="subject id column; 'sub-' and leading zeros are ignored")
    pheno.add_argument("--phenotype-group-column", default="group", help="for example DX_GROUP")
    pheno.add_argument("--phenotype-labels", default="", help="value=label pairs, e.g. '1=ASD 2=TD'")


def criteria_from_args(args: argparse.Namespace) -> Criteria:
    return Criteria(fd_mean=args.exclude_fd_mean, fd_max=args.exclude_fd_max, pct_fd_gt02=args.exclude_pct_fd_gt02,
                    min_retained_min=args.exclude_min_retained_min, min_dof=args.exclude_min_dof,
                    qc_fail=args.exclude_qc_fail)


# ----------------------------------------------------------------------------
# decision
# ----------------------------------------------------------------------------

def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _run_reasons(record: dict[str, Any], criteria: Criteria) -> list[str]:
    reasons: list[str] = []

    def above(key: str, limit: float, text: str, unit: str = "") -> None:
        if limit <= 0:
            return
        value = _num(record.get(key))
        if value is None:
            reasons.append(f"{text} n/a")
        elif value > limit:
            reasons.append(f"{text} {value:.3g}{unit} > {limit:g}{unit}")

    above("fd_mean", criteria.fd_mean, "mean FD", " mm")
    above("fd_max", criteria.fd_max, "max FD", " mm")
    above("fd_pct_gt_02", criteria.pct_fd_gt02, "frames with FD > 0.2 mm", "%")
    if criteria.min_retained_min > 0:
        minutes = _num(record.get("minutes_retained"))
        if minutes is None:
            reasons.append("retained minutes n/a")
        elif minutes < criteria.min_retained_min:
            reasons.append(f"retained {minutes:.3g} min < {criteria.min_retained_min:g} min")
    if criteria.qc_fail:
        bad = sorted(key[5:] for key, value in record.items()
                     if key.startswith("flag.") and "." not in key[5:] and value in BAD_FLAGS
                     and key[5:] not in OWN_CRITERION_FLAGS)
        if bad:
            reasons.append("QC " + ", ".join(f"{name} {record['flag.' + name]}" for name in bad))
        elif record.get("overall_flag") == "incomplete":
            reasons.append("QC incomplete")
    return reasons


def decide(table: pd.DataFrame, criteria: Criteria, strategies: list[str]) -> pd.DataFrame:
    """One row per run of ``table`` (the frame of group_report.collect_metrics).

    ``included`` is the run-level decision; ``included_<S>`` also applies the
    per-strategy DOF criterion. A strategy without a DOF value (not run) is not
    judged on DOF.
    """
    rows = []
    for record in table.to_dict("records"):
        reasons = _run_reasons(record, criteria)
        row: dict[str, Any] = {
            "subject": record.get("subject"), "session": record.get("session", "-"), "run": record.get("run"),
            "group": record.get("group", NA), "pheno_group": record.get("pheno_group", NA),
            "included": "no" if reasons else "yes",
        }
        extra: list[str] = []
        for strategy in strategies:
            dof = _num(record.get(f"{strategy}.dof_remaining"))
            ok = not reasons
            if criteria.min_dof > 0 and dof is not None and dof < criteria.min_dof:
                ok = False
                extra.append(f"{strategy}: DOF {dof:g} < {criteria.min_dof:g}")
            row[f"dof_{strategy}"] = dof
            row[f"included_{strategy}"] = "yes" if ok else "no"
        row["reasons"] = "; ".join(reasons + extra) if reasons or extra else "-"
        for key in ("fd_mean", "fd_max", "fd_pct_gt_02", "minutes_retained", "pct_censored", "overall_flag"):
            row[key] = record.get(key)
        rows.append(row)
    columns = ["subject", "session", "run", "group", "pheno_group", "included", "reasons"]
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=columns + [f"included_{s}" for s in strategies])
    return out[columns + [c for c in out.columns if c not in columns]]


def included_runs(decision: pd.DataFrame, strategy: str | None = None) -> set[str]:
    column = f"included_{strategy}" if strategy and f"included_{strategy}" in decision.columns else "included"
    if decision.empty or column not in decision.columns:
        return set()
    return set(decision.loc[decision[column] == "yes", "run"].astype(str))


# ----------------------------------------------------------------------------
# phenotype
# ----------------------------------------------------------------------------

def normalize_id(value: Any) -> str:
    """'sub-0028744', '0028744' and 28744 all become '28744'."""
    text = str(value).strip()
    if text.lower().startswith("sub-"):
        text = text[4:]
    if text.isdigit():
        text = str(int(text))
    return text


def parse_labels(text: str) -> dict[str, str]:
    labels = {}
    for item in str(text or "").replace(",", " ").split():
        if "=" not in item:
            raise ValueError(f"phenotype label '{item}' is not value=label")
        value, label = item.split("=", 1)
        labels[value.strip()] = label.strip()
    return labels


def read_phenotype(path: Path, id_column: str, group_column: str, labels: dict[str, str] | None = None) -> dict[str, str]:
    """normalized subject id -> group label. Conflicting duplicates are an error."""
    path = Path(path)
    separator = "," if path.suffix.lower() == ".csv" else "\t"
    frame = pd.read_csv(path, sep=separator, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = [c for c in (id_column, group_column) if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: column(s) {', '.join(missing)} not found "
                         f"(available: {', '.join(list(frame.columns)[:15])})")
    labels = labels or {}
    out: dict[str, str] = {}
    for raw_id, raw_group in zip(frame[id_column], frame[group_column]):
        key = normalize_id(raw_id)
        if not key:
            continue
        value = str(raw_group).strip()
        label = labels.get(value, value) if value else NA
        if key in out and out[key] != label:
            raise ValueError(f"{path}: subject {raw_id} has two groups: {out[key]} and {label}")
        out[key] = label
    return out


def attach_phenotype(table: pd.DataFrame, phenotype: dict[str, str]) -> pd.DataFrame:
    table = table.copy()
    subjects = table["subject"] if "subject" in table.columns else pd.Series([], dtype=str)
    table["pheno_group"] = [phenotype.get(normalize_id(s), NA) for s in subjects]
    return table


def motion_by_group(decision: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per phenotype group: subjects, exclusions and mean FD (subject means over runs).

    With exactly two groups of >= MIN_GROUP_SUBJECTS subjects: Mann-Whitney U on the
    subject mean FD of the included runs, and Fisher's exact test on the number of
    subjects without any included run.
    """
    columns = ["pheno_group", "n_subjects", "n_subjects_excluded", "pct_subjects_excluded", "n_runs", "n_runs_excluded",
               "fd_mean_median_all", "fd_mean_iqr_all", "fd_mean_median_included", "fd_mean_iqr_included"]
    tests: dict[str, Any] = {}
    if decision.empty or "pheno_group" not in decision or (decision["pheno_group"] == NA).all():
        return pd.DataFrame(columns=columns), tests
    data = decision.copy()
    data["fd"] = pd.to_numeric(data["fd_mean"], errors="coerce")
    data["kept"] = data["included"] == "yes"
    rows, fd_included, excluded_counts = [], {}, {}
    for label, block in data.groupby("pheno_group", sort=True):
        subjects = block.groupby("subject").agg(fd_all=("fd", "mean"), kept=("kept", "any"))
        fd_kept = block[block["kept"]].groupby("subject")["fd"].mean()
        n_sub, n_excl = len(subjects), int((~subjects["kept"]).sum())

        def spread(values: pd.Series) -> tuple[float, float]:
            values = values.dropna()
            if values.empty:
                return float("nan"), float("nan")
            q1, q3 = np.quantile(values, [0.25, 0.75])
            return float(values.median()), float(q3 - q1)

        med_all, iqr_all = spread(subjects["fd_all"])
        med_kept, iqr_kept = spread(fd_kept)
        rows.append((label, n_sub, n_excl, 100.0 * n_excl / n_sub if n_sub else float("nan"), len(block),
                     int((~block["kept"]).sum()), med_all, iqr_all, med_kept, iqr_kept))
        if label != NA:
            fd_included[label] = fd_kept.dropna().to_numpy()
            excluded_counts[label] = (n_excl, n_sub - n_excl)
    table = pd.DataFrame(rows, columns=columns)
    labels = sorted(fd_included)
    if len(labels) == 2 and all(len(fd_included[g]) >= MIN_GROUP_SUBJECTS for g in labels):
        a, b = labels
        tests["groups"] = f"{a} vs {b}"
        tests["mannwhitney_p_fd_included"] = float(stats.mannwhitneyu(fd_included[a], fd_included[b],
                                                                      alternative="two-sided").pvalue)
        tests["fisher_p_subjects_excluded"] = float(stats.fisher_exact([list(excluded_counts[a]),
                                                                        list(excluded_counts[b])])[1])
    return table, tests
