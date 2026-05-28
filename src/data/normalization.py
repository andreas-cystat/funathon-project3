"""
Compute per-band mean/std from the training tiles.

These statistics must come from the training set (not from each batch) and be
*frozen* afterwards: the same numbers are reused at inference, otherwise the
model sees inputs in a different statistical range from what it learnt on and
predictions degrade silently. The computed values are persisted alongside the
model (MLflow run params) so inference code can restore them exactly.
"""

from typing import List, Tuple
import numpy as np
import yaml

from src.data.loading import get_patchs_labels
from src.data.filter import filter_indices_from_labels


def normalization_params(nuts: str, year: str):
    """
    Load per-band mean/std for a NUTS3 / year pair from the YAML file written
    alongside the downloaded patches.
    """

    params_path = (
        f"../../data/data-preprocessed/patchs/{nuts}/{year}/metrics-normalization.yaml"
    )

    with open(params_path) as f:
        params = yaml.safe_load(f)

    return params["mean"], params["std"]


def compute_global_normalization(
    nuts_years: List[str],
    n_bands: int,
) -> Tuple[List[float], List[float]]:

    means = []
    stds = []
    weights = []

    for item in nuts_years:
        nuts, year = item.split("_")

        patches, labels = get_patchs_labels(
            from_s3=False,
            nuts=nuts,
            year=year,
        )

        indices = filter_indices_from_labels(labels, -1.0, 2.0)

        if len(indices) == 0:
            continue

        mean, std = normalization_params(nuts, year)

        means.append(mean[:n_bands])
        stds.append(std[:n_bands])
        weights.append(len(indices))

    if len(means) == 0:
        raise ValueError("No valid data found for normalization.")

    # YAML loads mean/std as Python lists; cast to numpy so element-wise math works.
    means_arr = np.asarray(means, dtype=np.float64)
    stds_arr = np.asarray(stds, dtype=np.float64)

    global_mean = np.average(means_arr, axis=0, weights=weights)
    # Approximate pooled std: sqrt of the weighted mean of variances. This is exact
    # only when the per-group means are equal; for similar tiles within a NUTS3 / year
    # group it's close enough for normalisation purposes.
    global_std = np.sqrt(np.average(stds_arr ** 2, axis=0, weights=weights))

    return global_mean.tolist(), global_std.tolist()
