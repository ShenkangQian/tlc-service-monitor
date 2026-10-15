"""Monthly service-interval analysis of completed-trip daily count panels.

The empirical-Bayes prior uses only the three reference months. Intervals are
conditional day-block bootstrap intervals with the estimated prior held fixed;
they neither control FDR nor guarantee coverage under serial dependence. No
causal, unmet-demand, regulatory, or real-time-dispatch interpretation is made.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd

KEYS = ["zone", "platform", "slot", "weekday"]
PARENT_KEYS = ["platform", "slot", "weekday"]
METHOD_COLUMNS = {
    "raw_rate": "raw_rate",
    "delta": "delta_pp",
    "posterior_delta": "posterior_delta_pp",
}


def _panel(panel: pd.DataFrame) -> pd.DataFrame:
    required = ["source_month", "request_date", *KEYS, "n", "k"]
    missing = set(required) - set(panel.columns)
    if missing:
        raise ValueError(f"Missing daily-panel columns: {sorted(missing)}")
    out = panel.loc[:, required].copy()
    if out[required].isna().any().any():
        raise ValueError("Panel keys and counts may not be missing")
    out["request_date"] = pd.to_datetime(out["request_date"])
    if out["request_date"].dt.tz is not None:
        raise ValueError("request_date must be a timezone-free local calendar date")
    if not out["request_date"].equals(out["request_date"].dt.normalize()):
        raise ValueError("request_date must contain normalized calendar dates")
    if not (out["source_month"].astype(str) == out["request_date"].dt.strftime("%Y-%m")).all():
        raise ValueError("source_month must equal the request calendar month")
    if not (out["weekday"] == out["request_date"].dt.dayofweek).all():
        raise ValueError("weekday must equal request_date.dayofweek")
    for col in ["n", "k"]:
        values = pd.to_numeric(out[col], errors="raise")
        if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f"{col} must contain finite nonnegative integer counts")
        out[col] = values.astype("int64")
    if (out["k"] > out["n"]).any():
        raise ValueError("k cannot exceed n")
    # Zero-count rows carry no observed completed trips and cannot establish support.
    out = out[out["n"] > 0]
    return (out.groupby(["source_month", "request_date", *KEYS], as_index=False,
                        sort=True, observed=True)[["n", "k"]].sum())


def _window(panel: pd.DataFrame, month: str, reference_months: int,
            require_complete_reference: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[str], bool]:
    if reference_months < 1:
        raise ValueError("reference_months must be positive")
    period = pd.Period(month, freq="M")
    reference = [str(period - offset) for offset in range(reference_months, 0, -1)]
    available = set(panel["source_month"].astype(str))
    missing = set(reference) - available
    if missing and require_complete_reference:
        raise ValueError(f"Reference window has missing months: {sorted(missing)}")
    current = panel[panel["source_month"] == str(period)]
    if current.empty:
        raise ValueError(f"No positive-count daily panel for monitoring month {period}")
    ref = panel[panel["source_month"].isin(reference)]
    return current, ref, reference, not missing


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame.groupby(KEYS, as_index=False, observed=True)
            .agg(n=("n", "sum"), k=("k", "sum"),
                 observed_days=("request_date", "nunique")))


def _rng(seed: int, key: tuple, stream: str = "bootstrap") -> np.random.Generator:
    # Stable per-group random streams: unrelated group additions do not alter a CI.
    encoded = repr((int(seed), stream, tuple(key))).encode("utf-8")
    entropy = int.from_bytes(hashlib.sha256(encoded).digest()[:16], "little")
    return np.random.default_rng(entropy)


def _bootstrap_difference(current: np.ndarray, reference: np.ndarray, prior: float,
                          strength: float, reps: int, confidence: float,
                          rng: np.random.Generator) -> tuple[float, float]:
    current_indices = rng.integers(len(current), size=(reps, len(current)))
    reference_indices = rng.integers(len(reference), size=(reps, len(reference)))
    current_sum = current[current_indices].sum(axis=1)
    reference_sum = reference[reference_indices].sum(axis=1)
    delta = ((current_sum[:, 1] + strength * prior) / (current_sum[:, 0] + strength)
             - (reference_sum[:, 1] + strength * prior) / (reference_sum[:, 0] + strength))
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(delta * 100.0, [alpha, 1.0 - alpha])
    return float(low), float(high)


def compare_month(panel: pd.DataFrame, month: str, reference_months: int = 3,
                  prior_strength: float = 50, min_n: int = 100,
                  min_n_ref: int = 300, min_days: int = 3,
                  min_ref_days: int = 6, bootstrap_reps: int = 300,
                  seed: int = 42, confidence: float = 0.95,
                  require_complete_reference: bool = True) -> pd.DataFrame:
    """Return all observed monitoring-month candidates with evidence flags.

    ``delta_pp`` is the raw current-minus-reference difference in percentage
    points. ``posterior_delta_pp`` and its bootstrap interval use a fixed
    reference-only parent rate in both current and reference posterior means.
    Missing reference months raise by default; zero valid rows cannot prove a
    month was fully ingested. Verify the separate ingestion audit as well.
    """
    if prior_strength < 0 or not np.isfinite(prior_strength):
        raise ValueError("prior_strength must be finite and nonnegative")
    if bootstrap_reps < 2 or not isinstance(bootstrap_reps, int):
        raise ValueError("bootstrap_reps must be an integer of at least 2")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if min(min_n, min_n_ref, min_days, min_ref_days) < 1:
        raise ValueError("Support thresholds must be positive")
    data = _panel(panel)
    current, reference, months, complete = _window(
        data, month, reference_months, require_complete_reference)
    result = _aggregate(current)
    ref_totals = _aggregate(reference).rename(columns={
        "n": "n_ref", "k": "k_ref", "observed_days": "reference_days"})
    result = result.merge(ref_totals, on=KEYS, how="left", validate="one_to_one")
    for col in ["n_ref", "k_ref", "reference_days"]:
        result[col] = result[col].fillna(0).astype("int64")
    parents = reference.groupby(PARENT_KEYS, as_index=False, observed=True)[["n", "k"]].sum()
    parents["prior_rate"] = parents["k"] / parents["n"]
    result = result.merge(parents[PARENT_KEYS + ["prior_rate"]], on=PARENT_KEYS,
                          how="left", validate="many_to_one")
    global_rate = reference["k"].sum() / reference["n"].sum() if reference["n"].sum() else np.nan
    result["prior_source"] = np.where(result["prior_rate"].notna(), "reference_parent", "reference_global")
    result["prior_rate"] = result["prior_rate"].fillna(global_rate)
    result.loc[result["prior_rate"].isna(), "prior_source"] = "unavailable"
    result["raw_rate"] = result["k"] / result["n"]
    result["reference_rate"] = result["k_ref"] / result["n_ref"].replace(0, np.nan)
    if prior_strength == 0:
        result["posterior_rate"] = result["raw_rate"]
        result["posterior_reference_rate"] = result["reference_rate"]
    else:
        result["posterior_rate"] = (result["k"] + prior_strength * result["prior_rate"]) / (result["n"] + prior_strength)
        result["posterior_reference_rate"] = (result["k_ref"] + prior_strength * result["prior_rate"]) / (result["n_ref"] + prior_strength)
    result["delta_pp"] = 100 * (result["raw_rate"] - result["reference_rate"])
    result["posterior_delta_pp"] = 100 * (result["posterior_rate"] - result["posterior_reference_rate"])
    result["support_eligible"] = ((result["n"] >= min_n) & (result["n_ref"] >= min_n_ref)
                                  & (result["observed_days"] >= min_days)
                                  & (result["reference_days"] >= min_ref_days))
    result["ci_low_pp"] = np.nan
    result["ci_high_pp"] = np.nan
    cur_groups = {key: group[["n", "k"]].to_numpy() for key, group in current.groupby(KEYS, sort=True, observed=True)}
    ref_groups = {key: group[["n", "k"]].to_numpy() for key, group in reference.groupby(KEYS, sort=True, observed=True)}
    for row in result[result["support_eligible"]].itertuples():
        key = tuple(getattr(row, field) for field in KEYS)
        low, high = _bootstrap_difference(cur_groups[key], ref_groups[key], row.prior_rate,
                                          prior_strength, bootstrap_reps, confidence, _rng(seed, key))
        result.loc[row.Index, ["ci_low_pp", "ci_high_pp"]] = [low, high]
    # Different n and n_ref give the same prior different weights; a positive
    # posterior difference alone can occur even when both event counts are zero.
    # Operational review therefore also requires actual raw-rate deterioration.
    result["priority_eligible"] = (result["support_eligible"]
                                   & (result["delta_pp"] > 0)
                                   & (result["posterior_delta_pp"] > 0)
                                   & (result["ci_low_pp"] > 0))
    result["evidence_status"] = np.select(
        [~result["support_eligible"], result["priority_eligible"]],
        ["evidence_insufficient", "priority"], default="supported_not_priority")
    result["excess_proxy"] = result["n"] * (result["delta_pp"] / 100).clip(lower=0)
    result["monitoring_month"] = str(pd.Period(month, freq="M"))
    result["reference_start"] = months[0]
    result["reference_end"] = months[-1]
    result["reference_complete"] = complete
    result["prior_strength"] = prior_strength
    result["interval_method"] = "day_block_bootstrap_fixed_reference_prior_conditional"
    if "dataset_kind" in panel.columns:
        kinds = sorted(set(panel["dataset_kind"].dropna().astype(str)))
        result["dataset_kind"] = "|".join(kinds)
    return result.sort_values(KEYS, kind="stable").reset_index(drop=True)


def rank_candidates(comparison: pd.DataFrame, method: str = "posterior_delta",
                    top_k: int = 20, require_priority: bool = True) -> pd.DataFrame:
    """Select at most K; never fill a short list with unsupported candidates.

    Defaults to the operational review list. For a same-pool ranking comparison,
    use ``require_priority=False`` for all three methods.
    """
    if method not in METHOD_COLUMNS:
        raise ValueError(f"method must be one of {list(METHOD_COLUMNS)}")
    if top_k < 1 or not isinstance(top_k, int):
        raise ValueError("top_k must be a positive integer")
    pool = comparison[comparison["support_eligible"].astype(bool)]
    if require_priority:
        pool = pool[pool["priority_eligible"].astype(bool)]
    selected = (pool.sort_values([METHOD_COLUMNS[method], *KEYS],
                                ascending=[False, True, True, True, True], kind="stable")
                .head(top_k).copy().reset_index(drop=True))
    selected.insert(0, "rank", np.arange(1, len(selected) + 1))
    selected["requested_k"] = top_k
    selected["effective_k"] = len(selected)
    selected.attrs.update(requested_k=top_k, effective_k=len(selected),
                          eligible_pool_size=len(pool), require_priority=require_priority)
    return selected


def evaluate_next_month(comparison: pd.DataFrame, panel: pd.DataFrame, month: str,
                        top_k: int = 20, threshold_pp: float = 2.0,
                        min_n: int = 100, min_days: int = 3) -> pd.DataFrame:
    """Historical next-observation-month persistence, not a live prediction.

    Three rankings share the same supported pool. A fourth ``priority`` row
    applies the operational confidence filter. Missing/low-support future data
    remain unknown and are excluded from hit-rate denominators. Labels compare
    future raw rates to the SAME frozen raw reference rate used at monitoring.
    """
    data = _panel(panel)
    normalized_month = str(pd.Period(month, freq="M"))
    if not comparison.empty and not (comparison["monitoring_month"] == normalized_month).all():
        raise ValueError("comparison monitoring_month does not match month")
    future_month = str(pd.Period(month, freq="M") + 1)
    future = _aggregate(data[data["source_month"] == future_month]).rename(columns={
        "n": "future_n", "k": "future_k", "observed_days": "future_days"})
    rows = []
    for name, method, operational in [
            ("raw_rate", "raw_rate", False), ("delta", "delta", False),
            ("posterior_delta", "posterior_delta", False),
            ("priority", "posterior_delta", True)]:
        selected = rank_candidates(comparison, method, top_k, require_priority=operational)
        joined = selected.merge(future, on=KEYS, how="left", validate="one_to_one")
        observable = ((joined["future_n"] >= min_n) & (joined["future_days"] >= min_days)
                      & joined["reference_rate"].notna())
        delta = 100 * (joined["future_k"] / joined["future_n"] - joined["reference_rate"])
        observed_count = int(observable.sum())
        hits = int((delta[observable] > threshold_pp).sum())
        effective = len(joined)
        rows.append({"monitoring_month": normalized_month, "future_month": future_month,
                     "method": name, "candidate_pool": "priority" if operational else "supported",
                     "requested_k": top_k, "effective_k": effective,
                     "observable_k": observed_count, "unknown_k": effective - observed_count,
                     "hits": hits, "observed_hit_rate": hits / observed_count if observed_count else np.nan,
                     "observable_fraction": observed_count / effective if effective else np.nan,
                     "threshold_pp": threshold_pp,
                     "support_pool_size": int(comparison["support_eligible"].sum()),
                     "interpretation": "historical_persistence_not_live_forecast"})
    return pd.DataFrame(rows)


def standardize_zones(panel: pd.DataFrame, month: str, reference_months: int = 3,
                      min_coverage: float = 1.0,
                      require_complete_reference: bool = True) -> pd.DataFrame:
    """Within each zone, freeze reference platform/slot/weekday composition.

    Missing current cells are never filled with a zero risk. Below min_coverage
    the standardized rates are unavailable. If the caller permits incomplete
    coverage, both rates are normalized to the same observed subset, explicitly
    labelled; this is no longer the full reference population estimand.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    data = _panel(panel)
    current, reference, _, complete = _window(data, month, reference_months, require_complete_reference)
    current_agg = _aggregate(current)
    reference_agg = _aggregate(reference)
    joined = reference_agg.merge(current_agg, on=KEYS, how="left", suffixes=("_ref", "_current"))
    rows = []
    for zone, group in joined.groupby("zone", sort=True, observed=True):
        weights = group["n_ref"] / group["n_ref"].sum()
        available = group["n_current"].notna() & (group["n_current"] > 0)
        coverage = float(weights[available].sum())
        usable = coverage + 1e-12 >= min_coverage and coverage > 0
        current_rate = reference_rate = np.nan
        if usable:
            observed_weights = weights[available] / coverage
            current_rate = float((observed_weights * group.loc[available, "k_current"] / group.loc[available, "n_current"]).sum())
            reference_rate = float((observed_weights * group.loc[available, "k_ref"] / group.loc[available, "n_ref"]).sum())
        rows.append({"zone": zone, "monitoring_month": str(pd.Period(month, freq="M")),
                     "reference_weight_coverage": coverage,
                     "standardized_current_rate": current_rate,
                     "standardized_reference_rate": reference_rate,
                     "standardized_delta_pp": (current_rate - reference_rate) * 100,
                     "coverage_eligible": usable, "reference_complete": complete,
                     "estimand": "full_reference_composition" if coverage >= 1 - 1e-12
                     else "observed_reference_subset" if usable else "unavailable_missing_cells",
                     "current_only_cells_excluded": int(
                         (~current_agg[current_agg["zone"] == zone].set_index(KEYS).index.isin(
                             reference_agg[reference_agg["zone"] == zone].set_index(KEYS).index)).sum())})
    # New zones without any reference composition are retained as unavailable.
    for zone in sorted(set(current["zone"]) - set(reference["zone"])):
        rows.append({"zone": zone, "monitoring_month": str(pd.Period(month, freq="M")),
                     "reference_weight_coverage": 0.0, "standardized_current_rate": np.nan,
                     "standardized_reference_rate": np.nan, "standardized_delta_pp": np.nan,
                     "coverage_eligible": False, "reference_complete": complete,
                     "estimand": "unavailable_no_reference_composition",
                     "current_only_cells_excluded": len(current_agg[current_agg["zone"] == zone])})
    return pd.DataFrame(rows).sort_values("zone").reset_index(drop=True)


def panel_thinning_experiment(panel: pd.DataFrame, month: str,
                              fractions: Iterable[float] = (0.1, 0.5, 1.0),
                              seed: int = 42, **compare_kwargs) -> dict:
    """SIMULATION ONLY: binomial thinning of panel event/non-event counts.

    Calendar exposure is identical at every fraction. This does not reread or
    sample raw records and cannot measure scan speed, storage, or raw-scale
    pipeline quality. Returned rows are explicitly labelled. It is one seeded
    simulated realization, not a repeated-sampling stability estimate.
    """
    data = _panel(panel)
    data = data.sort_values(["request_date", *KEYS], kind="stable").reset_index(drop=True)
    fractions = tuple(float(value) for value in fractions)
    if any(not np.isfinite(value) or value <= 0 or value > 1 for value in fractions):
        raise ValueError("fractions must lie in (0, 1]")
    summary = []
    results = {}
    # Reuse the same random uniforms through inverse binomial CDFs when scipy is
    # available would yield nested samples; this simple version does not claim nesting.
    for fraction in fractions:
        sample = data.copy()
        rng = np.random.default_rng(seed)
        sample["k"] = rng.binomial(data["k"].to_numpy(), fraction)
        sample["n"] = sample["k"] + rng.binomial((data["n"] - data["k"]).to_numpy(), fraction)
        kwargs = dict(compare_kwargs)
        kwargs.setdefault("seed", seed)
        result = compare_month(sample, month, **kwargs)
        result["experiment_kind"] = "SIMULATED_panel_count_binomial_thinning"
        result["sampling_fraction"] = fraction
        results[fraction] = result
        summary.append({"experiment_kind": "SIMULATED_panel_count_binomial_thinning",
                        "sampling_fraction": fraction, "seed": seed,
                        "calendar_start": str(data["request_date"].min().date()),
                        "calendar_end": str(data["request_date"].max().date()),
                        "input_panel_count": int(data["n"].sum()),
                        "simulated_retained_count": int(sample["n"].sum()),
                        "supported_groups": int(result["support_eligible"].sum()),
                        "priority_groups": int(result["priority_eligible"].sum()),
                        "mean_ci_width_pp": float((result["ci_high_pp"] - result["ci_low_pp"]).mean())})
    return {"summary": pd.DataFrame(summary), "results": results}


def synthetic_panel(seed: int = 42, start_month: str = "2025-01", months: int = 8) -> pd.DataFrame:
    """Build an explicitly SYNTHETIC panel for offline software demonstrations.

    Zone 1 has an injected sustained increase after three reference months;
    zone 2 has a transient increase; zone 3 is sparse. These are invented signals,
    not empirical TLC findings. One daily row is generated per zone/platform/slot.
    """
    if months < 1:
        raise ValueError("months must be positive")
    rng = np.random.default_rng(seed)
    start = pd.Period(start_month, freq="M")
    dates = pd.date_range(start.start_time, (start + months - 1).end_time.normalize(), freq="D")
    rows = []
    for date in dates:
        month_index = (date.year - start.year) * 12 + date.month - start.month
        for zone in range(1, 9):
            for platform in ["SYNTH_A", "SYNTH_B"]:
                for slot in range(6):
                    mean_n = 4 if zone == 3 else 90 + zone * 3
                    n = int(rng.poisson(mean_n))
                    if n == 0:
                        continue
                    rate = 0.09 + 0.006 * zone + 0.008 * slot + 0.012 * (date.dayofweek >= 5)
                    rate += 0.01 * (platform == "SYNTH_B")
                    if zone == 1 and month_index >= 3:
                        rate += 0.14
                    if zone == 2 and month_index == 3:
                        rate += 0.12
                    # Cell-specific daily noise creates variation across date blocks.
                    rate = float(np.clip(rate + rng.normal(0, 0.012), 0.001, 0.95))
                    k = int(rng.binomial(n, rate))
                    k1200 = int(rng.binomial(k, 0.22))
                    rows.append({"source_month": date.strftime("%Y-%m"), "request_date": date,
                                 "zone": zone, "platform": platform, "slot": slot,
                                 "weekday": date.dayofweek, "n": n, "k": k,
                                 "k1200": k1200,
                                 "wait_sum_s": float(k1200 * 1500 + (k - k1200) * 900 + (n - k) * 300),
                                 "dataset_kind": "SYNTHETIC_NOT_TLC_OBSERVATIONS"})
    result = pd.DataFrame(rows)
    result.attrs["dataset_kind"] = "SYNTHETIC_NOT_TLC_OBSERVATIONS"
    return result
