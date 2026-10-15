"""Business invariants on TLC-shaped Parquet fixtures, with no network use."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tlc_service.data import FILTER_ORDER, month_range, official_url, probe_parquet, process_month, select_row_groups


def write_fixture(path: Path, records: list[dict], row_group_size: int = 3) -> None:
    frame = pd.DataFrame(records)
    for col in ("request_datetime", "pickup_datetime"):
        frame[col] = pd.to_datetime(frame[col]).astype("datetime64[ns]")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, row_group_size=row_group_size)


def trip(wait=0, request="2024-01-15 09:00:00", zone=1, platform="HV0003") -> dict:
    timestamp = pd.Timestamp(request) if request is not None else pd.NaT
    return {
        "request_datetime": timestamp,
        "pickup_datetime": timestamp + pd.Timedelta(seconds=wait),
        "PULocationID": zone,
        "hvfhs_license_num": platform,
    }


def test_month_validation():
    assert month_range("2023-12", "2024-02") == ["2023-12", "2024-01", "2024-02"]
    assert official_url("2024-01").endswith("/fhvhv_tripdata_2024-01.parquet")
    for start, end in [("2024-1", "2024-02"), ("2024-13", "2025-01"), ("2024-02", "2024-01")]:
        with pytest.raises(ValueError):
            month_range(start, end)


def test_thresholds_duplicates_guard_and_mutually_exclusive_ledger(tmp_path):
    records = [trip(wait) for wait in (0, 600, 601, 1200, 1201, 7200)]
    # A repeated raw row is a separate observation: no unique trip ID exists.
    records.append(trip(600))
    records += [
        trip(60, "2024-01-31 21:59:59", zone=2),  # before the 2-hour guard
        trip(0, "2024-01-31 22:00:00"),           # exact guard boundary excluded
        trip(600, "2024-01-31 23:59:00"),         # pickup next month excluded
        trip(0, "2023-12-31 23:00:00"),
        trip(0, "2024-02-01 00:00:00"),
        trip(-1),
        trip(7201),
        trip(0, zone=0),
        trip(0, zone=264),
        trip(0, zone=1.5),
        trip(0, zone=None),
        trip(0, platform=None),
        trip(0, platform="  "),
        trip(0, request=None, zone=None, platform=None),  # first failure only
        {**trip(0), "pickup_datetime": None},
    ]
    source = tmp_path / "fixture.parquet"
    write_fixture(source, records)
    result = process_month(source, "2024-01", tmp_path / "out", batch_size=2)
    panel = pd.read_parquet(result["panel_path"])
    assert result["source_rows"] == result["scanned_rows"] == 22
    assert result["accepted_rows"] == panel.n.sum() == 8
    assert result["excluded_rows"] == 14
    expected = dict(zip(FILTER_ORDER, [1, 1, 1, 3, 2, 2, 1, 1, 2, 8]))
    assert result["filter_ledger"] == expected
    assert sum(result["filter_ledger"].values()) == result["scanned_rows"]
    first = panel.loc[panel.zone == 1].iloc[0]
    assert first.n == 7
    assert first.k == 4  # strictly > 600; equality remains unflagged
    assert first.k1200 == 2  # strictly > 1200
    assert first.wait_sum_s == sum([0, 600, 601, 1200, 1201, 7200, 600])
    assert first.slot == 2 and first.weekday == 0
    assert panel.request_date.dtype == "datetime64[ns]"
    assert result["coverage"] == "full_month_all_row_groups"


def test_source_month_platform_and_time_groups_are_preserved(tmp_path):
    records = [
        trip(600, "2024-01-01 03:59:59", 263, "HV0003"),
        trip(601, "2024-01-01 04:00:00", 263, "HV0003"),
        trip(1201, "2024-01-02 04:00:00", 263, "UNKNOWN-CODE"),
    ]
    source = tmp_path / "groups.parquet"
    write_fixture(source, records)
    result = process_month(source, "2024-01", tmp_path / "out")
    panel = pd.read_parquet(result["panel_path"])
    assert len(panel) == 3 and panel.n.tolist() == [1, 1, 1]
    assert panel.slot.tolist() == [0, 1, 1]
    assert panel.weekday.tolist() == [0, 0, 1]
    assert panel.platform.tolist()[-1] == "UNKNOWN-CODE"
    assert panel.source_month.unique().tolist() == ["2024-01"]


def test_cache_checks_parameters_source_and_output(tmp_path):
    source = tmp_path / "cache.parquet"
    write_fixture(source, [trip(7200)])
    first = process_month(source, "2024-01", tmp_path / "out")
    assert first["cached"] is False
    assert process_month(source, "2024-01", tmp_path / "out")["cached"] is True
    changed = process_month(source, "2024-01", tmp_path / "out", max_wait_seconds=3600)
    assert changed["cached"] is False and changed["accepted_rows"] == 0
    write_fixture(source, [trip(600), trip(601)])
    changed_source = process_month(source, "2024-01", tmp_path / "out", max_wait_seconds=3600)
    assert changed_source["cached"] is False and changed_source["accepted_rows"] == 2
    Path(changed_source["panel_path"]).write_bytes(b"corrupted")
    repaired = process_month(source, "2024-01", tmp_path / "out", max_wait_seconds=3600)
    assert repaired["cached"] is False and repaired["accepted_rows"] == 2


def test_row_group_pilot_is_deterministic_and_not_claimed_representative(tmp_path):
    assert select_row_groups(10, 3) == [0, 4, 9]
    assert select_row_groups(10, 1) == [5]
    assert select_row_groups(2, 8) == [0, 1]
    with pytest.raises(ValueError):
        select_row_groups(3, 0)
    source = tmp_path / "pilot.parquet"
    write_fixture(source, [trip(i) for i in range(12)], row_group_size=2)
    metadata = probe_parquet(source)
    assert metadata["rows"] == 12 and metadata["row_groups"] == 6
    assert metadata["bytes"] == source.stat().st_size
    result = process_month(source, "2024-01", tmp_path / "out", sample_row_groups=2)
    assert result["selected_row_groups"] == [0, 5]
    assert result["scanned_rows"] == 4 and result["source_rows"] == 12
    assert result["coverage"] == "deterministic_row_group_pilot"
    assert "not a random or representative" in result["sampling_note"]


def test_guard_tracks_sensitivity_cap_and_requires_adequate_duration(tmp_path):
    source = tmp_path / "guard.parquet"
    write_fixture(source, [trip(0, "2024-01-31 20:00:00"), trip(3600, "2024-01-31 22:59:59")])
    result = process_month(source, "2024-01", tmp_path / "out", max_wait_seconds=3600)
    assert result["accepted_rows"] == 2
    result = process_month(source, "2024-01", tmp_path / "out", max_wait_seconds=14400)
    assert result["accepted_rows"] == 0 and result["filter_ledger"]["month_end_guard"] == 2
    with pytest.raises(ValueError, match="cover"):
        process_month(source, "2024-01", tmp_path / "out", max_wait_seconds=7200, guard_seconds=3600)
