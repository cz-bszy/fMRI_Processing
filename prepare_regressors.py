#!/usr/bin/env python3
"""Improve numerical conditioning without changing the nuisance column space."""
import argparse
import json
from pathlib import Path

import numpy as np


def prepare(output, inputs):
    arrays = []
    labels = []
    for path in inputs:
        values = np.loadtxt(path, ndmin=2)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite nuisance regressor: {path}")
        arrays.append(values)
        labels.extend(f"{path.name}:{column}" for column in range(values.shape[1]))
    if len({len(values) for values in arrays}) != 1:
        raise ValueError("Nuisance files have different timepoint counts")

    raw = np.column_stack(arrays)
    means = raw.mean(axis=0)
    centered = raw - means
    norms = np.linalg.norm(centered, axis=0)
    retained = norms > 0
    if not retained.any():
        raise ValueError("No varying nuisance regressors")
    np.savetxt(output, centered[:, retained] / norms[retained], fmt="%.12g")
    metadata = {
        "labels": [label for label, keep in zip(labels, retained) if keep],
        "constant_columns": [label for label, keep in zip(labels, retained) if not keep],
        "original_means": means.tolist(),
        "centered_l2_norms": norms.tolist(),
        "operation": "Double-precision centering and unit L2 scaling; constants are already modeled by polort.",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    prepare(args.output, args.inputs)
