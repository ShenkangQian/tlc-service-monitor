#!/usr/bin/env python3
"""Prespecified short-window flow-check grid; never select using future labels.

Run against the four available complete months, using a TWO-month reference:
    python scripts/run_sensitivity.py

May/June downloads were unavailable (HTTP 403), so this is a reduced-window
pipeline check, not the planned three-reference-month evaluation. The two
endpoints use the identical accepted completed-trip population. Replacing
k with k1200 changes the event definition, not the maximum-valid-interval filter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlc_service.__main__ import load_panel  # noqa: E402
from tlc_service.analysis import compare_month, evaluate_next_month  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot_forwardcheck.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]
    monitor_month = config["monitor_month"]
    reference_months = config["reference_months"]
    required_months = [str(month) for month in pd.period_range(
        config["start_month"], config["end_month"], freq="M")]
    if (required_months != ["2024-01", "2024-02", "2024-03", "2024-04"]
            or monitor_month != "2024-03" or reference_months != 2):
        parser.error("This archived short-window check is fixed to January-April 2024, monitoring March with two reference months.")
    reference_periods = [str(pd.Period(monitor_month, freq="M") - offset)
                         for offset in range(reference_months, 0, -1)]
    future_month = str(pd.Period(monitor_month, freq="M") + 1)
    absent = [month for month in required_months
              if not (Path(data_dir) / f"panel_{month}.parquet").is_file()
              or not (Path(data_dir) / f"audit_{month}.json").is_file()]
    if absent:
        parser.error(f"Wait for all full-month panels and audits; missing: {absent}")
    audits = []
    for month in required_months:
        audit_path = Path(data_dir) / f"audit_{month}.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (audit.get("coverage") != "full_month_all_row_groups"
                or audit.get("scanned_rows") != audit.get("source_rows")):
            raise ValueError(f"Full-month completed scan is required: {audit_path}")
        audits.append(audit)
    panel = load_panel(data_dir)
    if not set(required_months).issubset(set(panel.source_month)):
        raise ValueError("The required source months are not all represented in the validated panel")
    panel = panel[panel.source_month.isin(required_months)].copy()
    if "k1200" not in panel:
        raise ValueError("k1200 must be preserved in the daily panel for the 20-minute endpoint")
    if (panel.k1200 < 0).any() or (panel.k1200 > panel.k).any():
        raise ValueError("Expected nested event counts 0 <= k1200 <= k <= n")
    fixed = {"monitoring_month": monitor_month, "reference_months": reference_periods,
             "future_month": future_month, "prior_strength_grid": [0, 20, 50, 200],
             "endpoint_seconds_grid": [600, 1200], "min_n": 100, "min_n_ref": 300,
             "min_days": 3, "min_ref_days": 6, "bootstrap_reps": 300,
             "confidence": 0.95, "seed": 42, "top_k": 20,
             "future_persistence_threshold_pp": 2.0,
             "evaluation_role": "heldout_future_not_tuning",
             "selection_policy": "Report every prespecified configuration; do not select a winner or retune using held-out next-month outcomes."}
    rows = []
    configurations = []
    for endpoint, count_column in [(600, "k"), (1200, "k1200")]:
        endpoint_panel = panel.copy()
        endpoint_panel["k"] = panel[count_column]
        for strength in fixed["prior_strength_grid"]:
            started = time.perf_counter()
            result = compare_month(endpoint_panel, fixed["monitoring_month"],
                                   reference_months=reference_months,
                                   prior_strength=strength, min_n=100, min_n_ref=300,
                                   min_days=3, min_ref_days=6, bootstrap_reps=300,
                                   confidence=0.95, seed=42)
            if ((result["delta_pp"] <= 0) & result["priority_eligible"]).any():
                raise ValueError("Priority must require positive RAW change as well as positive shrunken change and interval lower bound; update analysis before running.")
            metrics = evaluate_next_month(result, endpoint_panel, fixed["monitoring_month"],
                                          top_k=20, threshold_pp=2.0, min_n=100, min_days=3)
            supported = result[result.support_eligible]
            day_counts = {str(int(days)): int(count) for days, count in
                          supported.observed_days.value_counts().sort_index().items()}
            diagnostic = {
                "endpoint_seconds": endpoint, "prior_strength": strength,
                "candidate_cells": len(result), "supported_cells": len(supported),
                "priority_cells": int(result.priority_eligible.sum()),
                "supported_monitoring_date_counts": day_counts,
                "supported_monitoring_dates_min": int(supported.observed_days.min()) if len(supported) else None,
                "supported_monitoring_dates_median": float(supported.observed_days.median()) if len(supported) else None,
                "supported_monitoring_dates_max": int(supported.observed_days.max()) if len(supported) else None,
                "median_conditional_ci_width_pp": float((supported.ci_high_pp - supported.ci_low_pp).median()) if len(supported) else None,
                "analysis_seconds": round(time.perf_counter() - started, 3),
            }
            configurations.append(diagnostic)
            for metric in metrics.to_dict(orient="records"):
                rows.append({"experiment_kind": "REAL_FOUR_MONTH_SHORT_WINDOW_FORWARD_PIPELINE_CHECK",
                             "evaluation_role": "heldout_future_not_tuning",
                             "reference_window_months": reference_months,
                             "endpoint_seconds": endpoint, "prior_strength": strength,
                             "supported_cells": len(supported),
                             "priority_cells": int(result.priority_eligible.sum()),
                             "ci_date_limitation": "At most 4-5 same-weekday monitoring dates; minimum eligibility is 3. Conditional day bootstrap only.",
                             **metric})
            print(json.dumps(diagnostic, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "sensitivity.csv", index=False)
    # Replace floating NaN with JSON null for zero-denominator outcomes.
    records = json.loads(frame.to_json(orient="records"))
    artifact = {
        "experiment_kind": "REAL_FOUR_MONTH_SHORT_WINDOW_FORWARD_PIPELINE_CHECK",
        "evaluation_role": "heldout_future_not_tuning", "configuration": fixed,
        "input_months": required_months,
        "input_panel_rows": len(panel), "accepted_trip_count": int(panel.n.sum()),
        "scanned_trip_count": int(sum(audit["scanned_rows"] for audit in audits)),
        "input_panel_sha256": {audit["source_month"]: audit["panel_sha256"] for audit in audits},
        "limitations": [
            "This is a descriptive grid recorded before examining this sensitivity run; it is not an externally registered clinical-style preregistration.",
            "Data availability forced a shorter two-month reference. This is a four-month forward pipeline check, not completion of the formal three-reference-month evaluation.",
            "Every prespecified setting is reported. April labels are held out for description, not selection, tuning, or declaring s=50 optimal.",
            "The reference is January-February, monitoring is March, and April is only the next observed month. May/June are unavailable after the source returned HTTP 403.",
            "Grouping by weekday leaves at most four or five monitoring dates per cell in March; supported cells may have only three dates.",
            "Increasing bootstrap replicates does not create more independent dates. These are coarse conditional date-block intervals, not calibrated guarantees.",
            "Prior estimation uncertainty and serial dependence across dates or weeks are not represented; intervals and selected lists do not control FDR.",
            "The operational list also requires positive raw observed change. This prevents zero-event cells entering solely because current and reference sample sizes assign different weights to the same positive prior.",
            "The 20-minute endpoint uses k1200 on the same accepted 0-to-7200-second completed-trip population and the same source-boundary exclusion.",
            "Future persistence >2 percentage points keeps the same absolute threshold for both endpoints; their base rates differ, so cross-endpoint hit rates are not interchangeable utility measures.",
            "The operational priority list can contain fewer than requested K. Unknown future outcomes are excluded from hit-rate denominators and observability is reported.",
            "One monitoring-to-follow-up pair cannot establish generalization across seasons, years, or regime changes.",
            "These results describe retrospective pilot validation, not causal effects, violations, or real-time operational impact.",
        ],
        "configurations": configurations, "results": records,
    }
    (out / "sensitivity.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(out / "sensitivity.csv"), "json": str(out / "sensitivity.json"),
                      "evaluation_rows": len(frame), "evaluation_role": "heldout_future_not_tuning"}), flush=True)


if __name__ == "__main__":
    main()
