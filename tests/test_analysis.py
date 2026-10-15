"""Behavioral checks for temporal leakage, missing outcomes and fixed composition."""
import numpy as np
import pandas as pd
import pytest

from tlc_service.analysis import (compare_month, evaluate_next_month,
                                 panel_thinning_experiment, rank_candidates,
                                 standardize_zones, synthetic_panel)


def fixture_panel():
    rows = []
    for date in pd.date_range("2025-01-01", "2025-05-31", freq="D"):
        for zone, platform in [(1, "A"), (2, "B")]:
            n = 100
            k = 10 if date.month < 4 or zone == 2 else 35
            rows.append({"source_month": date.strftime("%Y-%m"), "request_date": date,
                         "zone": zone, "platform": platform, "slot": 0,
                         "weekday": date.dayofweek, "n": n, "k": k})
    return pd.DataFrame(rows)


def test_future_cannot_change_reference_prior_or_intervals():
    data = fixture_panel()
    baseline = compare_month(data, "2025-04", bootstrap_reps=100)
    changed = data.copy()
    changed.loc[changed.source_month == "2025-05", "k"] = 99
    actual = compare_month(changed, "2025-04", bootstrap_reps=100)
    pd.testing.assert_frame_equal(baseline, actual)
    monitoring_changed = data.copy()
    monitoring_changed.loc[monitoring_changed.source_month == "2025-04", "k"] = 90
    other = compare_month(monitoring_changed, "2025-04", bootstrap_reps=100)
    pd.testing.assert_series_equal(baseline.prior_rate, other.prior_rate)


def test_unrelated_parent_group_does_not_change_group_prior_or_ci():
    data = fixture_panel()
    alone = compare_month(data[data.platform == "A"], "2025-04", bootstrap_reps=100)
    together = compare_month(data, "2025-04", bootstrap_reps=100)
    pd.testing.assert_frame_equal(alone, together[together.platform == "A"].reset_index(drop=True))


def test_missing_future_is_unknown_not_a_negative_outcome():
    data = fixture_panel()
    comparison = compare_month(data, "2025-04", bootstrap_reps=20)
    no_future = data[data.source_month != "2025-05"]
    metrics = evaluate_next_month(comparison, no_future, "2025-04", top_k=5)
    assert (metrics.effective_k == 5).all()
    assert (metrics.observable_k == 0).all()
    assert (metrics.unknown_k == 5).all()
    assert metrics.observed_hit_rate.isna().all()
    assert (metrics.observable_fraction == 0).all()


def test_effective_k_and_empty_priority_do_not_divide_by_zero():
    data = fixture_panel()
    comparison = compare_month(data, "2025-04", bootstrap_reps=20)
    ranked = rank_candidates(comparison, top_k=100)
    assert len(ranked) == 7  # Seven weekday cells of the affected zone.
    assert (ranked.effective_k == 7).all()
    assert (ranked.requested_k == 100).all()
    comparison.loc[:, "priority_eligible"] = False
    empty = rank_candidates(comparison, top_k=10)
    assert empty.empty and empty.attrs["effective_k"] == 0
    metrics = evaluate_next_month(comparison, data, "2025-04", top_k=100)
    operational = metrics[metrics.method == "priority"].iloc[0]
    assert operational.effective_k == 0
    assert np.isnan(operational.observable_fraction)
    assert np.isnan(operational.observed_hit_rate)
    assert (metrics[metrics.method != "priority"].effective_k == 14).all()


def test_two_zero_event_periods_cannot_be_priority_due_to_different_shrinkage_weights():
    rows = []
    for date in pd.date_range("2024-01-01", "2024-03-31", freq="D"):
        for zone in [1, 2]:
            rows.append({"source_month": date.strftime("%Y-%m"), "request_date": date,
                         "zone": zone, "platform": "A", "slot": 0,
                         "weekday": date.dayofweek, "n": 100, "k": 0 if zone == 1 else 10})
    result = compare_month(pd.DataFrame(rows), "2024-03", reference_months=2, bootstrap_reps=50)
    zero_events = result[result.zone == 1]
    assert zero_events.support_eligible.all()
    assert (zero_events.delta_pp == 0).all()
    # A positive parent prior shrinks the smaller monitoring count more strongly.
    assert (zero_events.posterior_delta_pp > 0).all()
    assert (zero_events.ci_low_pp > 0).all()
    # That arithmetic effect is not observed deterioration.
    assert not zero_events.priority_eligible.any()


def test_fixed_reference_weights_and_missing_composition_coverage():
    data = fixture_panel()
    # Remove one current weekday stratum, preserving a positive rate elsewhere.
    incomplete = data[~((data.source_month == "2025-04") & (data.weekday == 0) & (data.zone == 1))]
    strict = standardize_zones(incomplete, "2025-04")
    row = strict[strict.zone == 1].iloc[0]
    assert 0 < row.reference_weight_coverage < 1
    assert not row.coverage_eligible
    assert np.isnan(row.standardized_current_rate)
    permitted = standardize_zones(incomplete, "2025-04", min_coverage=0.8)
    row = permitted[permitted.zone == 1].iloc[0]
    assert row.coverage_eligible
    assert row.estimand == "observed_reference_subset"
    assert np.isclose(row.standardized_current_rate, 0.35)
    assert np.isclose(row.standardized_reference_rate, 0.10)
    assert np.isclose(row.standardized_delta_pp, 25)


def test_standardization_removes_composition_only_shift():
    data = fixture_panel()
    data = data[data.zone == 1].copy()
    extra = data.copy()
    extra["slot"] = 1
    data["k"] = data["n"] * 0.1
    extra["k"] = extra["n"] * 0.4
    current = extra.source_month == "2025-04"
    extra.loc[current, ["n", "k"]] *= 5
    combined = pd.concat([data, extra], ignore_index=True)
    row = standardize_zones(combined, "2025-04").iloc[0]
    assert np.isclose(row.standardized_delta_pp, 0)
    assert np.isclose(row.standardized_current_rate, 0.25)


def test_missing_reference_month_raises_and_explicit_override_is_flagged():
    data = fixture_panel()
    data = data[data.source_month != "2025-02"]
    with pytest.raises(ValueError, match="missing months"):
        compare_month(data, "2025-04", bootstrap_reps=20)
    flagged = compare_month(data, "2025-04", bootstrap_reps=20, require_complete_reference=False)
    assert not flagged.reference_complete.any()


def test_math_example_and_zero_strength():
    data = fixture_panel()
    out = compare_month(data, "2025-04", prior_strength=50, bootstrap_reps=20)
    row = out[(out.zone == 1) & (out.weekday == 0)].iloc[0]
    expected = (row.k + 50 * 0.1) / (row.n + 50)
    assert np.isclose(row.prior_rate, 0.1)
    assert np.isclose(row.posterior_rate, expected)
    assert np.isclose(row.posterior_reference_rate, 0.1)
    assert np.isclose(row.delta_pp, 25)
    no_shrink = compare_month(data, "2025-04", prior_strength=0, bootstrap_reps=20)
    np.testing.assert_allclose(no_shrink.delta_pp, no_shrink.posterior_delta_pp)


def test_zero_count_rows_do_not_establish_support_and_bad_counts_raise():
    data = fixture_panel()
    zero = data.copy()
    zero.loc[(zero.zone == 1) & (zero.source_month == "2025-04"), ["n", "k"]] = 0
    result = compare_month(zero, "2025-04", bootstrap_reps=20)
    assert set(result.zone) == {2}
    bad = data.copy()
    bad.loc[0, "k"] = 101
    with pytest.raises(ValueError, match="k cannot exceed n"):
        compare_month(bad, "2025-04")


def test_panel_thinning_is_labeled_reproducible_and_keeps_calendar_span():
    data = fixture_panel()
    a = panel_thinning_experiment(data, "2025-04", fractions=(0.5, 1.0), bootstrap_reps=20)
    b = panel_thinning_experiment(data, "2025-04", fractions=(0.5, 1.0), bootstrap_reps=20)
    pd.testing.assert_frame_equal(a["summary"], b["summary"])
    assert a["summary"].calendar_start.nunique() == 1
    assert a["summary"].calendar_end.nunique() == 1
    assert a["summary"].experiment_kind.str.startswith("SIMULATED").all()
    full = a["summary"][a["summary"].sampling_fraction == 1.0].iloc[0]
    assert full.simulated_retained_count == full.input_panel_count


def test_synthetic_end_to_end_is_explicitly_marked():
    panel = synthetic_panel(months=5)
    assert panel.dataset_kind.str.contains("SYNTHETIC").all()
    assert (panel.k1200 <= panel.k).all() and (panel.k <= panel.n).all()
    comparison = compare_month(panel, "2025-04", bootstrap_reps=20)
    ranking = rank_candidates(comparison, top_k=20)
    metrics = evaluate_next_month(comparison, panel, "2025-04", top_k=20)
    assert not ranking.empty
    assert set(metrics.method) == {"raw_rate", "delta", "posterior_delta", "priority"}
    assert not comparison[comparison.zone == 3].support_eligible.any()
