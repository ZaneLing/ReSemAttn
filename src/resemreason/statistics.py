"""Question-paired bootstrap; average seed-level differences before inference."""

import numpy as np


def paired_bootstrap(a, b, metric="H@1", samples=10000, seed=12345):
    if set(a) != set(b) or not a:
        raise ValueError("Methods must have identical nonempty seed sets.")
    seeds = sorted(a)
    ids = sorted(r["question_id"] for r in a[seeds[0]])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Question IDs must be nonempty and unique.")

    def arrays(data):
        nums, dens = [], []
        for run in seeds:
            lookup = {r["question_id"]: r for r in data[run]}
            if sorted(lookup) != ids or len(lookup) != len(data[run]):
                raise ValueError("All seed runs must contain exactly the same questions.")
            rows = [lookup[i] for i in ids]
            if metric in ("Joint@1", "WPR"):
                if any(r.get("top_path_valid") is None for r in rows):
                    raise ValueError(
                        "Use a common fully annotated cohort for paired evidence tests."
                    )
                n = [
                    r["H@1"]
                    * (r["top_path_valid"] if metric == "Joint@1" else 1 - r["top_path_valid"])
                    for r in rows
                ]
                d = [1 if metric == "Joint@1" else r["H@1"] for r in rows]
            else:
                n, d = [r[metric] for r in rows], [1] * len(rows)
            nums.append(n)
            dens.append(d)
        return np.array(nums, float), np.array(dens, float)

    an, ad = arrays(a)
    bn, bd = arrays(b)

    def difference(ix):
        da, db = ad[:, ix].sum(1), bd[:, ix].sum(1)
        if np.any(da == 0) or np.any(db == 0):
            return None
        return float(np.mean(an[:, ix].sum(1) / da - bn[:, ix].sum(1) / db))

    estimate = difference(np.arange(len(ids)))
    if estimate is None:
        raise ValueError("Metric denominator is zero in at least one seed.")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        value = difference(rng.integers(0, len(ids), len(ids)))
        if value is not None:
            values.append(value)
    if not values:
        raise ValueError("No defined bootstrap replicates.")
    values = np.array(values)
    return {
        "difference": estimate,
        "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        "p_value": float(
            (1 + np.sum(np.abs(values - estimate) >= abs(estimate))) / (len(values) + 1)
        ),
        "replicates": len(values),
        "undefined_replicates": samples - len(values),
        "questions": len(ids),
        "seeds": seeds,
        "bootstrap_seed": seed,
    }


def holm_adjust(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def cohen_kappa(labels_a, labels_b):
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("Matched nonempty label vectors are required.")
    a, b = np.array(labels_a), np.array(labels_b)
    classes = set(labels_a) | set(labels_b)
    observed = float(np.mean(a == b))
    expected = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in classes)
    return (observed - expected) / (1 - expected) if expected < 1 else None


def annotation_agreement(labels_a, labels_b, samples=1000, seed=12345):
    """Determinate, paired labels only; undefined kappa replicates remain counted."""
    point = cohen_kappa(labels_a, labels_b)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(labels_a), len(labels_a))
        value = cohen_kappa([labels_a[i] for i in indices], [labels_b[i] for i in indices])
        if value is not None:
            values.append(value)
    return dict(
        agreement=float(np.mean(np.array(labels_a) == np.array(labels_b))),
        cohen_kappa=point,
        kappa_ci95=np.quantile(values, [0.025, 0.975]).tolist() if values else None,
        bootstrap_samples=samples,
        undefined_replicates=samples - len(values),
        bootstrap_seed=seed,
    )
