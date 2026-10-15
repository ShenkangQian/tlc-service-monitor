"""Bounded-memory ingestion for TLC high-volume FHV monthly Parquet files.

The raw files have no unique trip identifier: repeated rows remain in the data.
Footer probes are metadata checks, not scans of all records.  A row-group pilot
is deterministic coverage of storage groups, not a probability sample of trips.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import ssl
import struct
import time
from typing import Any

import fsspec
import aiohttp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests


BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
REQUIRED_COLUMNS = (
    "request_datetime", "pickup_datetime", "PULocationID", "hvfhs_license_num"
)
PANEL_KEYS = ["source_month", "request_date", "zone", "platform", "slot", "weekday"]
PANEL_VALUES = ["n", "k", "k1200", "wait_sum_s"]
FILTER_ORDER = (
    "missing_or_invalid_request_datetime",
    "missing_or_invalid_pickup_datetime",
    "missing_zone",
    "invalid_zone",
    "missing_platform",
    "request_outside_source_month",
    "negative_interval",
    "interval_above_upper_bound",
    "month_end_guard",
    "accepted",
)
PIPELINE_VERSION = "1.0.0"


def _month_timestamp(month: str) -> pd.Timestamp:
    if not re.fullmatch(r"\d{4}-\d{2}", str(month)):
        raise ValueError(f"Expected YYYY-MM, received {month!r}")
    try:
        return pd.Timestamp(f"{month}-01")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid month: {month!r}") from exc


def month_range(start: str, end: str) -> list[str]:
    """Inclusive monthly range; invalid or reversed ranges raise ValueError."""
    first, last = _month_timestamp(start), _month_timestamp(end)
    if first > last:
        raise ValueError("start must be no later than end")
    return list(pd.period_range(first, last, freq="M").astype(str))


def official_url(month: str) -> str:
    _month_timestamp(month)
    return f"{BASE_URL}/fhvhv_tripdata_{month}.parquet"


def _remote(source: str) -> bool:
    return str(source).startswith(("https://", "http://"))


def _range_request(
    session: requests.Session,
    source: str,
    byte_range: str,
    expected_bytes: int,
    retries: int,
    timeout: float,
) -> tuple[bytes, dict[str, str]]:
    """Reject servers that ignore Range before reading a potentially huge body."""
    for attempt in range(retries):
        try:
            with session.get(
                source,
                headers={"Range": byte_range, "Accept-Encoding": "identity"},
                timeout=timeout,
                stream=True,
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Server did not honor HTTP Range ({response.status_code}): {source}"
                    )
                payload = response.raw.read(expected_bytes + 1)
                if len(payload) != expected_bytes:
                    raise RuntimeError(
                        f"Incomplete or oversized range: expected {expected_bytes}, "
                        f"received {len(payload)}"
                    )
                return payload, {k.lower(): v for k, v in response.headers.items()}
        except (requests.RequestException, OSError, RuntimeError):
            if attempt + 1 == retries:
                raise
            time.sleep(min(2**attempt, 4))
    raise AssertionError("unreachable")


def probe_parquet(source: str | Path, retries: int = 3, timeout: float = 60) -> dict[str, Any]:
    """Read only the Parquet footer and return exact file/row metadata.

    HTTP uses two bounded Range requests and refuses a full-body fallback.  For
    local files, the etag is a size/mtime-ns fingerprint.  HTTP etags are retained
    exactly as returned; absent etags disable HTTP cache reuse in process_month.
    """
    if retries < 1:
        raise ValueError("retries must be at least 1")
    source = str(source)
    last_modified = None
    if _remote(source):
        with requests.Session() as session:
            trailer, headers = _range_request(session, source, "bytes=-8", 8, retries, timeout)
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", headers.get("content-range", ""))
            if not match:
                raise RuntimeError("Range response is missing a valid Content-Range header")
            first, last, size = map(int, match.groups())
            if (first, last) != (size - 8, size - 1):
                raise RuntimeError("Unexpected suffix range positions")
            footer_size = struct.unpack("<I", trailer[:4])[0]
            if trailer[4:] != b"PAR1" or not 0 < footer_size <= size - 12:
                raise ValueError("Not an ordinary, unencrypted Parquet file")
            # A metadata footer may grow, but a bogus length must never trigger
            # an accidental multi-GB transfer while merely building a manifest.
            if footer_size > 64 * 1024 * 1024:
                raise ValueError("Footer exceeds the 64 MiB metadata-probe limit")
            footer, footer_headers = _range_request(
                session, source, f"bytes={size - 8 - footer_size}-{size - 9}",
                footer_size, retries, timeout,
            )
            etag = headers.get("etag")
            if etag and footer_headers.get("etag") != etag:
                raise RuntimeError("Source etag changed between footer requests; retry the probe")
            last_modified = headers.get("last-modified")
            meta = pq.read_metadata(pa.BufferReader(b"PAR1" + footer + trailer))
    else:
        path = Path(source)
        before = path.stat()
        size = before.st_size
        meta = pq.read_metadata(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Local source changed while its metadata was being read")
        etag = f"local-{size}-{after.st_mtime_ns}"
    return {
        "source": source,
        "rows": int(meta.num_rows),
        "bytes": int(size),
        "etag": etag,
        "last_modified": last_modified,
        "columns": list(meta.schema.names),
        "row_groups": int(meta.num_row_groups),
        "row_group_rows": [int(meta.row_group(i).num_rows) for i in range(meta.num_row_groups)],
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_kind": "footer_metadata_only",
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_manifest(start: str, end: str, output_path: str | Path | None = None) -> list[dict]:
    """Probe official monthly files; optionally persist a resumable JSON list.

    A failed probe raises rather than fabricating counts.  The output file is
    updated after each successful month, leaving a clearly partial list if a
    later month fails.  Every invocation freshly checks the requested months.
    """
    records = []
    for month in month_range(start, end):
        metadata = probe_parquet(official_url(month))
        records.append({"source_month": month, "url": official_url(month), **metadata})
        if output_path is not None:
            _write_json_atomic(Path(output_path), records)
    return records


manifest = build_manifest


def select_row_groups(total: int, sample_row_groups: int | None = None) -> list[int]:
    """Select equally spaced storage groups, never claiming random sampling."""
    if total < 0:
        raise ValueError("total row groups cannot be negative")
    if sample_row_groups is not None and (
        isinstance(sample_row_groups, bool)
        or not isinstance(sample_row_groups, (int, np.integer))
        or sample_row_groups < 1
    ):
        raise ValueError("sample_row_groups must be a positive integer or None")
    if sample_row_groups is None or sample_row_groups >= total:
        return list(range(total))
    if sample_row_groups == 1:
        return [total // 2]
    return [i * (total - 1) // (sample_row_groups - 1) for i in range(sample_row_groups)]


def _empty_panel() -> pd.DataFrame:
    return pd.DataFrame({
        "source_month": pd.Series(dtype="str"),
        "request_date": pd.Series(dtype="datetime64[ns]"),
        "zone": pd.Series(dtype="int64"),
        "platform": pd.Series(dtype="str"),
        "slot": pd.Series(dtype="int64"),
        "weekday": pd.Series(dtype="int64"),
        "n": pd.Series(dtype="int64"),
        "k": pd.Series(dtype="int64"),
        "k1200": pd.Series(dtype="int64"),
        "wait_sum_s": pd.Series(dtype="float64"),
    })


def _aggregate_batch(
    frame: pd.DataFrame,
    source_month: str,
    max_wait_seconds: int,
    guard_seconds: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    request = pd.to_datetime(frame["request_datetime"], errors="coerce")
    pickup = pd.to_datetime(frame["pickup_datetime"], errors="coerce")
    if request.dt.tz is not None or pickup.dt.tz is not None:
        raise ValueError("Expected TLC naive local timestamps; explicit timezone conversion is required")
    zones = pd.to_numeric(frame["PULocationID"], errors="coerce")
    platform = frame["hvfhs_license_num"].astype("string").str.strip()
    begin = _month_timestamp(source_month)
    end = begin + pd.offsets.MonthBegin(1)
    elapsed = (pickup - request).dt.total_seconds()
    unresolved = pd.Series(True, index=frame.index)
    ledger = {reason: 0 for reason in FILTER_ORDER}

    def exclude(reason: str, condition: pd.Series) -> None:
        chosen = unresolved & condition.fillna(False)
        ledger[reason] += int(chosen.sum())
        unresolved.loc[chosen] = False

    exclude("missing_or_invalid_request_datetime", request.isna())
    exclude("missing_or_invalid_pickup_datetime", pickup.isna())
    exclude("missing_zone", frame["PULocationID"].isna())
    exclude("invalid_zone", zones.isna() | ~zones.between(1, 263) | (zones % 1 != 0))
    exclude("missing_platform", platform.isna() | platform.eq(""))
    exclude("request_outside_source_month", (request < begin) | (request >= end))
    exclude("negative_interval", elapsed < 0)
    exclude("interval_above_upper_bound", elapsed > max_wait_seconds)
    exclude("month_end_guard", request >= end - pd.Timedelta(seconds=guard_seconds))
    ledger["accepted"] = int(unresolved.sum())
    if sum(ledger.values()) != len(frame):
        raise AssertionError("Filter ledger is not a mutually exclusive partition")
    if not ledger["accepted"]:
        return _empty_panel(), ledger
    valid_request = request.loc[unresolved]
    valid_elapsed = elapsed.loc[unresolved]
    clean = pd.DataFrame({
        "source_month": source_month,
        "request_date": valid_request.dt.normalize().astype("datetime64[ns]"),
        "zone": zones.loc[unresolved].astype("int64"),
        "platform": platform.loc[unresolved],
        "slot": (valid_request.dt.hour // 4).astype("int64"),
        "weekday": valid_request.dt.dayofweek.astype("int64"),
        "n": np.int64(1),
        "k": (valid_elapsed > 600).astype("int64"),
        "k1200": (valid_elapsed > 1200).astype("int64"),
        "wait_sum_s": valid_elapsed.astype("float64"),
    })
    return clean.groupby(PANEL_KEYS, as_index=False, observed=True)[PANEL_VALUES].sum(), ledger


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def process_month(
    source: str | Path,
    source_month: str,
    output_dir: str | Path,
    *,
    max_wait_seconds: int = 7200,
    guard_seconds: int | None = None,
    sample_row_groups: int | None = None,
    batch_size: int = 131072,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Stream a month into a daily panel and an auditable, exclusive ledger.

    Returns the JSON audit plus absolute ``panel_path``, ``audit_path`` and a
    ``cached`` flag.  HTTP reads use fsspec range access, without downloading a
    complete source first.  Only a monthly aggregate dictionary and a bounded
    Arrow batch are held in Python memory (Arrow also buffers column chunks).
    Interrupted months restart; completed months are reused only if parameters,
    source etag/size, and the generated panel's checksum all still match.
    """
    _month_timestamp(source_month)
    if max_wait_seconds < 1 or batch_size < 1:
        raise ValueError("max_wait_seconds and batch_size must be positive")
    guard_seconds = max_wait_seconds if guard_seconds is None else guard_seconds
    if guard_seconds < max_wait_seconds:
        raise ValueError("guard_seconds must cover the full allowed interval")
    if guard_seconds >= 28 * 86400:
        raise ValueError("guard_seconds must be shorter than a month")
    source = str(source)
    source = source if _remote(source) else str(Path(source).resolve())
    metadata = probe_parquet(source)
    missing = set(REQUIRED_COLUMNS) - set(metadata["columns"])
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    selected = select_row_groups(metadata["row_groups"], sample_row_groups)
    params = {
        "pipeline_version": PIPELINE_VERSION,
        "source": source,
        "source_month": source_month,
        "max_wait_seconds": max_wait_seconds,
        "guard_seconds": guard_seconds,
        "sample_row_groups": sample_row_groups,
        "selected_row_groups": selected,
        "batch_size": batch_size,
        "filter_order": list(FILTER_ORDER),
    }
    params_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    panel_path = destination / f"panel_{source_month}.parquet"
    audit_path = destination / f"audit_{source_month}.json"
    if not force and audit_path.exists() and panel_path.exists() and metadata["etag"]:
        try:
            previous = json.loads(audit_path.read_text(encoding="utf-8"))
            if (
                previous.get("params_hash") == params_hash
                and previous.get("source_metadata", {}).get("etag") == metadata["etag"]
                and previous.get("source_metadata", {}).get("bytes") == metadata["bytes"]
                and previous.get("panel_sha256") == _file_sha256(panel_path)
            ):
                return {**previous, "panel_path": str(panel_path), "audit_path": str(audit_path), "cached": True}
        except (json.JSONDecodeError, OSError):
            pass

    start_time = time.monotonic()
    ledger = Counter({reason: 0 for reason in FILTER_ORDER})
    accumulator: dict[tuple, list] = {}
    scanned = 0
    expected = sum(metadata["row_group_rows"][index] for index in selected)
    next_progress = 1_000_000
    if progress:
        print(f"[{source_month}] scan start: {expected:,} rows in {len(selected)}/{metadata['row_groups']} row groups", flush=True)
    open_options = {
        "block_size": 4 * 1024 * 1024,
        "cache_type": "readahead",
        # Bundled Python installations may not have system trust roots.  Keep
        # verification enabled and use the same CA bundle as requests.
        "ssl": ssl.create_default_context(cafile=requests.certs.where()),
        "client_kwargs": {
            "timeout": aiohttp.ClientTimeout(total=180, sock_connect=30, sock_read=60),
        },
    } if _remote(source) else {}
    with fsspec.open(source, mode="rb", **open_options) as handle:
        parquet = pq.ParquetFile(handle, pre_buffer=False)
        if parquet.metadata.num_rows != metadata["rows"] or parquet.num_row_groups != metadata["row_groups"]:
            raise RuntimeError("Source metadata changed after probing")
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            row_groups=selected,
            columns=list(REQUIRED_COLUMNS),
            use_threads=True,
        ):
            frame = batch.to_pandas()
            partial, batch_ledger = _aggregate_batch(frame, source_month, max_wait_seconds, guard_seconds)
            scanned += len(frame)
            ledger.update(batch_ledger)
            for row in partial.itertuples(index=False, name=None):
                key, values = tuple(row[:6]), row[6:]
                if key not in accumulator:
                    accumulator[key] = [int(values[0]), int(values[1]), int(values[2]), float(values[3])]
                else:
                    for index, value in enumerate(values):
                        accumulator[key][index] += value
            if progress and scanned >= next_progress:
                print(
                    f"[{source_month}] {scanned:,}/{expected:,} rows; "
                    f"accepted={ledger['accepted']:,}; elapsed={time.monotonic() - start_time:.1f}s",
                    flush=True,
                )
                next_progress = (scanned // 1_000_000 + 1) * 1_000_000

    expected = sum(metadata["row_group_rows"][index] for index in selected)
    if scanned != expected or sum(ledger.values()) != scanned:
        raise AssertionError("Scanned rows disagree with the footer or exclusive ledger")
    panel = (
        pd.DataFrame([(*key, *values) for key, values in accumulator.items()], columns=PANEL_KEYS + PANEL_VALUES)
        if accumulator else _empty_panel()
    )
    panel["request_date"] = panel["request_date"].astype("datetime64[ns]")
    for column in ["zone", "slot", "weekday", "n", "k", "k1200"]:
        panel[column] = panel[column].astype("int64")
    panel["wait_sum_s"] = panel["wait_sum_s"].astype("float64")
    panel = panel.sort_values(PANEL_KEYS).reset_index(drop=True)
    if int(panel["n"].sum()) != ledger["accepted"]:
        raise AssertionError("Panel counts do not conserve accepted records")
    if not ((panel["k1200"] <= panel["k"]) & (panel["k"] <= panel["n"])).all():
        raise AssertionError("Threshold counts are inconsistent")

    # Recheck the source before committing a finished artifact.  This catches
    # replacement during a long monthly scan, and leaves no valid cache on error.
    final_metadata = probe_parquet(source)
    if any(final_metadata[key] != metadata[key] for key in ("etag", "bytes", "rows")):
        raise RuntimeError("Source changed during ingestion; no completed output was committed")
    temporary_panel = panel_path.with_name(panel_path.name + ".tmp")
    panel.to_parquet(temporary_panel, engine="pyarrow", compression="zstd", index=False)
    temporary_panel.replace(panel_path)
    is_full = len(selected) == metadata["row_groups"]
    audit = {
        "source_month": source_month,
        "pipeline_version": PIPELINE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "params_hash": params_hash,
        "source_metadata": metadata,
        "coverage": "full_month_all_row_groups" if is_full else "deterministic_row_group_pilot",
        "sampling_note": (
            "All storage row groups scanned; no row sampling."
            if is_full else
            "Equally spaced row groups selected deterministically; this is not a random or representative trip sample."
        ),
        "selected_row_groups": selected,
        "source_rows": metadata["rows"],
        "scanned_rows": scanned,
        "accepted_rows": int(ledger["accepted"]),
        "excluded_rows": scanned - int(ledger["accepted"]),
        "coverage_fraction_rows": scanned / metadata["rows"] if metadata["rows"] else None,
        "filter_ledger": dict(ledger),
        "filter_order": list(FILTER_ORDER),
        "duplicates_policy": "No deduplication: source has no verified unique trip ID.",
        "panel_rows": len(panel),
        "panel_count_sum": int(panel["n"].sum()),
        "panel_sha256": _file_sha256(panel_path),
        "elapsed_seconds": round(time.monotonic() - start_time, 3),
    }
    _write_json_atomic(audit_path, audit)
    if progress:
        print(
            f"[{source_month}] complete: scanned={scanned:,}, accepted={ledger['accepted']:,}, "
            f"panel_rows={len(panel):,}, elapsed={audit['elapsed_seconds']:.1f}s",
            flush=True,
        )
    return {**audit, "panel_path": str(panel_path), "audit_path": str(audit_path), "cached": False}
