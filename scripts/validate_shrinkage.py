#!/usr/bin/env python3
"""Archive a deliberately simple, explicitly simulated shrinkage mechanism check.

Run from the code root:
    python scripts/validate_shrinkage.py

This is a mechanism simulation, not a TLC experiment or evidence that a fitted
empirical prior improves monitoring. The matched-prior case is intentionally
optimistic because its known Beta prior exactly matches the data generator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values, [0.05, 0.50, 0.95])
    return {"p05": float(q[0]), "p50": float(q[1]), "p95": float(q[2])}


def simulation(repetitions: int = 3000, seed: int = 20261013) -> dict:
    rng = np.random.default_rng(seed)
    shape = (repetitions, 64)
    true_rate = rng.beta(5, 45, size=shape)
    sample_size = rng.integers(5, 51, size=shape)
    events = rng.binomial(sample_size, true_rate)
    estimates = {
        "raw": events / sample_size,
        "matched_fixed_prior": (events + 50 * 0.1) / (sample_size + 50),
        "mismatched_fixed_prior": (events + 50 * 0.3) / (sample_size + 50),
    }
    mse = {name: np.mean((estimate - true_rate) ** 2, axis=1)
           for name, estimate in estimates.items()}
    rmse = {name: np.sqrt(value) for name, value in mse.items()}
    methods = {}
    for name in estimates:
        method = {
            "mean_equal_cell_mse": float(mse[name].mean()),
            "rmse_quantiles": quantiles(rmse[name]),
            "rmse_percentage_points_quantiles": quantiles(100 * rmse[name]),
        }
        if name != "raw":
            ratios = rmse[name] / rmse["raw"]
            method.update({
                "rmse_ratio_to_raw_quantiles": quantiles(ratios),
                "mean_rmse_ratio_to_raw": float(ratios.mean()),
                "fraction_of_repetitions_with_lower_mse_than_raw": float((mse[name] < mse["raw"]).mean()),
            })
        methods[name] = method
    teaching = {
        "one_sparse_cell": {
            "n": 20, "k": 8, "fixed_prior_rate": 0.1, "prior_strength": 50,
            "raw_rate": 8 / 20,
            "posterior_formula": "(8 + 50 * 0.1) / (20 + 50)",
            "posterior_rate": (8 + 50 * 0.1) / (20 + 50),
            "equivalent_beta_parameters": {"alpha": 5 + 8, "beta": 45 + 20 - 8},
            "data_weight": 20 / 70, "prior_weight": 50 / 70,
        },
        "current_versus_reference": {
            "n": 200, "k": 60, "n_ref": 600, "k_ref": 60,
            "fixed_prior_rate": 0.1, "prior_strength": 50,
            "raw_rate": 60 / 200, "reference_rate": 60 / 600,
            "posterior_rate": (60 + 50 * 0.1) / (200 + 50),
            "posterior_reference_rate": (60 + 50 * 0.1) / (600 + 50),
            "delta_pp": 100 * (60 / 200 - 60 / 600),
            "posterior_delta_pp": 100 * ((60 + 50 * 0.1) / (200 + 50) - (60 + 50 * 0.1) / (600 + 50)),
            "excess_proxy_raw": 200 * (60 / 200 - 60 / 600),
            "ci": None,
            "ci_reason": "No date-level observations are specified; no day-block interval can be computed.",
            "interpretation": "Teaching arithmetic only; excess_proxy is not a causal count of avoidable trips.",
        },
    }
    return {
        "experiment_kind": "PURE_MECHANISM_SIMULATION",
        "purpose": "Show both variance reduction under a correct prior and harm under a mismatched prior.",
        "configuration": {
            "repetitions": repetitions, "cells_per_repetition": 64, "seed": seed,
            "generator": "numpy.random.default_rng (PCG64)",
            "numpy_version": np.__version__,
            "true_rate_distribution": "Independent Beta(alpha=5, beta=45) per cell and repetition",
            "sample_size_distribution": "Independent discrete uniform integers 5 through 50 inclusive",
            "event_distribution": "Binomial(n, true_rate), independent cells",
            "matched_prior": {"rate": 0.1, "strength": 50, "beta_alpha": 5, "beta_beta": 45},
            "mismatched_prior": {"rate": 0.3, "strength": 50, "beta_alpha": 15, "beta_beta": 35},
            "mse_definition": "Within each repetition, mean((estimate - true_rate)^2) over 64 equally weighted cells.",
            "ratio_definition": "Within each repetition, method RMSE divided by raw RMSE; quantiles over repetitions.",
            "weighting": "Equal cell weights, not trip weights.",
            "units": "Rates and RMSE in probability units unless explicitly labelled percentage points.",
        },
        "methods": methods,
        "teaching_examples": teaching,
        "limitations": [
            "The correct prior is known exactly and matches the generator; this is deliberately optimistic.",
            "Real project priors are estimated from historical completed trips and can be heterogeneous or stale.",
            "Independent binomial cells contain no within-day or across-day dependence, seasonality, drift, or missingness.",
            "This checks point-estimation error, not bootstrap coverage, Top-K persistence, multiple testing, or FDR control.",
            "The simulation does not measure real TLC improvement, business benefit, fairness, or computational scale.",
            "The mismatched prior is a selected failure example, not a universal threshold where shrinkage stops helping.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/mechanism_simulation.json")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--repetitions", type=int, default=3000)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    result = simulation(args.repetitions, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "experiment_kind": result["experiment_kind"],
                      "repetitions": args.repetitions, "methods": result["methods"]},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
