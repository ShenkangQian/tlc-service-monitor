"""Prevent partial or synthetic runs from being reported as real full scans."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from tlc_service.__main__ import analyze, load_panel
from tlc_service.data import month_range, official_url, process_month


def audited_month(tmp_path, month, *, official_fixture=False, sample=None):
    request = pd.Timestamp(f'{month}-15 09:00:00')
    raw = pd.DataFrame({
        'request_datetime': [request, request],
        'pickup_datetime': [request + pd.Timedelta(seconds=600), request + pd.Timedelta(seconds=601)],
        'PULocationID': [1, 1], 'hvfhs_license_num': ['HV0003', 'HV0003'],
    })
    source = tmp_path / f'raw_{month}.parquet'
    raw.to_parquet(source, index=False, row_group_size=1)
    result = process_month(source, month, tmp_path / 'panels', sample_row_groups=sample, progress=False)
    if official_fixture:
        # Test-only simulated official metadata, never written to real outputs.
        path = Path(result['audit_path'])
        audit = json.loads(path.read_text())
        audit['params']['source'] = official_url(month)
        audit['source_metadata']['source'] = official_url(month)
        audit['params_hash'] = hashlib.sha256(json.dumps(audit['params'], sort_keys=True).encode()).hexdigest()
        path.write_text(json.dumps(audit))
    return result


@pytest.fixture
def no_analysis_side_effects(monkeypatch):
    import tlc_service.analysis as statistics
    import tlc_service.reporting as reporting
    monkeypatch.setattr(statistics, 'compare_month', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(statistics, 'rank_candidates', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(statistics, 'evaluate_next_month', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(statistics, 'standardize_zones', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(reporting, 'write_report', lambda out, comparison, queue, evaluation, standardized, provenance: provenance)


def config(tmp_path, start='2024-01', end='2024-01', sample=None):
    return {'label': 'real_full_83_months', 'data_dir': str(tmp_path / 'panels'),
            'output_dir': str(tmp_path / 'results'), 'start_month': start, 'end_month': end,
            'monitor_month': end, 'sample_row_groups': sample, 'max_wait_seconds': 7200}


def test_missing_audit_and_altered_panel_are_rejected(tmp_path):
    result = audited_month(tmp_path, '2024-01')
    panel_path, audit_path = Path(result['panel_path']), Path(result['audit_path'])
    original = audit_path.read_text()
    audit_path.unlink()
    with pytest.raises(ValueError, match='Missing audit'):
        load_panel(tmp_path / 'panels')
    audit_path.write_text(original)
    panel_path.write_bytes(panel_path.read_bytes() + b'altered')
    with pytest.raises(ValueError, match='checksum'):
        load_panel(tmp_path / 'panels')


@pytest.mark.parametrize('field', ['source_rows', 'scanned_rows', 'accepted_rows', 'panel_count_sum'])
def test_all_count_layers_must_agree(tmp_path, field):
    result = audited_month(tmp_path, '2024-01')
    path = Path(result['audit_path'])
    audit = json.loads(path.read_text())
    audit[field] += 1
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        load_panel(tmp_path / 'panels')


def test_configured_months_and_sampling_must_match(tmp_path, no_analysis_side_effects):
    audited_month(tmp_path, '2024-01')
    with pytest.raises(ValueError, match='configured months'):
        analyze(config(tmp_path, end='2024-02'))
    with pytest.raises(ValueError, match='sample_row_groups'):
        analyze(config(tmp_path, sample=1))
    summary = analyze(config(tmp_path))
    assert summary['months'] == ['2024-01']
    assert summary['source_rows_in_files'] == summary['scanned_rows'] == summary['accepted_rows'] == 2
    assert summary['panel_rows'] == 1
    assert summary['coverage_kind'] == 'audited_local_or_other_sources'
    assert summary['full_83_month_processing_completed'] is False


def test_direct_synthetic_83_month_panel_never_proves_real_scan(tmp_path, no_analysis_side_effects):
    months = month_range('2019-02', '2025-12')
    panel = pd.DataFrame({'source_month': months, 'n': 1, 'zone': 1, 'platform': 'HV0003', 'slot': 1, 'weekday': 1})
    cfg = config(tmp_path, '2019-02', '2025-12')
    with pytest.raises(ValueError, match='SYNTHETIC'):
        analyze(cfg, panel=panel)
    cfg['label'] = 'SYNTHETIC_PIPELINE_DEMO_NOT_REAL_TLC'
    result = analyze(cfg, panel=panel)
    assert result['coverage_kind'] == 'synthetic_not_real'
    assert result['source_rows_in_files'] is None and result['scanned_rows'] is None
    assert result['accepted_rows'] is None
    assert result['valid_trip_count_in_panel'] == 83
    assert result['full_83_month_processing_completed'] is False


def test_full_claim_requires_all_official_months_and_no_requested_sample(tmp_path, no_analysis_side_effects):
    for month in month_range('2019-02', '2025-12'):
        audited_month(tmp_path, month, official_fixture=True)
    cfg = config(tmp_path, '2019-02', '2025-12')
    complete = analyze(cfg)
    assert complete['full_83_month_processing_completed'] is True
    assert complete['source_rows_in_files'] == complete['scanned_rows'] == complete['accepted_rows'] == 166
    assert complete['panel_rows'] == 83
    # One partial monthly partition is enough to revoke the entire full claim.
    audited_month(tmp_path, '2024-01', official_fixture=True, sample=1)
    with pytest.raises(ValueError, match='sample_row_groups'):
        analyze(cfg)
    # Matching a deliberately sampled configuration still cannot prove full scan.
    proof_panel = load_panel(tmp_path / 'panels')
    provenance = proof_panel.attrs['_tlc_ingestion_provenance']
    assert provenance['all_months_complete'] is False
    assert provenance['no_sampling_requested'] is False


def test_backtest_in_memory_panel_retains_checked_provenance(tmp_path, no_analysis_side_effects):
    audited_month(tmp_path, '2024-01')
    panel = load_panel(tmp_path / 'panels')
    result = analyze(config(tmp_path), panel=panel)
    assert result['accepted_rows'] == 2 and result['coverage_kind'] != 'synthetic_not_real'
    panel.loc[0, 'n'] += 1
    with pytest.raises(ValueError, match='changed after audit'):
        analyze(config(tmp_path), panel=panel)
