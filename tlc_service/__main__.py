import argparse
import json
from pathlib import Path
import platform
import time
import pandas as pd
from . import __version__


def load_config(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_panel(directory, expected_months=None):
    """Load only monthly panels whose source, counts and checksum are audited.

    Metadata source rows, actually scanned rows, accepted trips and aggregate
    panel rows are deliberately separate quantities.  Months with no accepted
    trips still belong to the audited coverage, although the analysis may lack
    enough observations to compare them.
    """
    import hashlib
    from .data import FILTER_ORDER, PANEL_KEYS, month_range, official_url
    files = sorted(Path(directory).rglob('*panel*.parquet'))
    if not files:
        raise FileNotFoundError(f'No daily panels in {directory}; run process first')
    frames, proofs = [], []
    for path in files:
        if not path.stem.startswith('panel_'):
            raise ValueError(f'Expected an audited monthly panel filename, received {path.name}')
        month = path.stem[len('panel_'):]
        month_range(month, month)
        audit_path = path.with_name(f'audit_{month}.json')
        if not audit_path.exists():
            raise ValueError(f'Missing audit for {path.name}; unaudited panels cannot prove real ingestion')
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        if audit.get('panel_sha256') != digest.hexdigest():
            raise ValueError(f'Panel checksum disagrees with audit: {path.name}')
        metadata, params = audit.get('source_metadata', {}), audit.get('params', {})
        if audit.get('source_month') != month or params.get('source_month') != month:
            raise ValueError(f'Audit source month disagrees with filename: {path.name}')
        if not metadata.get('source') or metadata.get('source') != params.get('source'):
            raise ValueError(f'Audit source path/URL is missing or inconsistent: {path.name}')
        params_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        if audit.get('params_hash') != params_hash:
            raise ValueError(f'Audit parameter hash is inconsistent: {path.name}')

        def count(value, name):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'Invalid nonnegative integer {name} in {audit_path.name}')
            return value

        source_rows = count(audit.get('source_rows'), 'source_rows')
        scanned = count(audit.get('scanned_rows'), 'scanned_rows')
        accepted = count(audit.get('accepted_rows'), 'accepted_rows')
        excluded = count(audit.get('excluded_rows'), 'excluded_rows')
        if source_rows != count(metadata.get('rows'), 'metadata rows'):
            raise ValueError(f'Source row count disagrees with footer audit: {path.name}')
        total_groups = count(metadata.get('row_groups'), 'row_groups')
        group_rows = metadata.get('row_group_rows', [])
        if len(group_rows) != total_groups or sum(count(n, 'row_group_rows') for n in group_rows) != source_rows:
            raise ValueError(f'Footer row-group ledger is inconsistent: {path.name}')
        selected = audit.get('selected_row_groups', [])
        if (selected != params.get('selected_row_groups')
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= total_groups for i in selected)
                or selected != sorted(set(selected))):
            raise ValueError(f'Selected row groups are inconsistent: {path.name}')
        if scanned != sum(group_rows[i] for i in selected):
            raise ValueError(f'Scanned rows disagree with selected row groups: {path.name}')
        ledger = audit.get('filter_ledger', {})
        if set(ledger) != set(FILTER_ORDER) or audit.get('filter_order') != list(FILTER_ORDER):
            raise ValueError(f'Missing or incompatible mutually exclusive filter ledger: {path.name}')
        if (sum(count(n, 'filter count') for n in ledger.values()) != scanned
                or ledger['accepted'] != accepted or accepted + excluded != scanned):
            raise ValueError(f'Filter ledger does not conserve scanned rows: {path.name}')
        frame = pd.read_parquet(path)
        if set(PANEL_KEYS + ['n']) - set(frame.columns):
            raise ValueError(f'Required daily panel columns are missing: {path.name}')
        values = pd.to_numeric(frame['n'], errors='raise')
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f'Panel n must contain finite nonnegative integer counts: {path.name}')
        if (int(values.sum()) != accepted or count(audit.get('panel_count_sum'), 'panel_count_sum') != accepted
                or count(audit.get('panel_rows'), 'panel_rows') != len(frame)):
            raise ValueError(f'Accepted count or panel rows disagree with the saved panel: {path.name}')
        if not (frame['source_month'].astype(str) == month).all():
            raise ValueError(f'Panel contains a different source month: {path.name}')
        complete = selected == list(range(total_groups)) and scanned == source_rows
        coverage = 'full_month_all_row_groups' if complete else 'deterministic_row_group_pilot'
        if audit.get('coverage') != coverage:
            raise ValueError(f'Audit coverage label disagrees with actual row groups: {path.name}')
        proofs.append({
            'month': month, 'source_rows': source_rows, 'scanned_rows': scanned,
            'accepted_rows': accepted, 'panel_rows': len(frame), 'params': params,
            'official_source': metadata['source'] == official_url(month),
            'complete_month': complete, 'no_sampling_requested': params.get('sample_row_groups') is None,
            'panel_sha256': audit['panel_sha256'],
        })
        frames.append(frame)
    months = sorted(proof['month'] for proof in proofs)
    if len(months) != len(set(months)):
        raise ValueError('Duplicate monthly outputs; do not mix alternative runs in one data directory')
    if expected_months is not None and months != sorted(expected_months):
        raise ValueError(f'Audited months {months} do not match configured months {sorted(expected_months)}')
    panel = pd.concat(frames, ignore_index=True)
    # Every month output represents a mutually exclusive source partition.
    if panel.duplicated(PANEL_KEYS).any():
        raise ValueError('Duplicate daily panel keys; do not mix alternative runs in one data directory')
    official = all(proof['official_source'] for proof in proofs)
    complete = all(proof['complete_month'] for proof in proofs)
    panel.attrs['_tlc_ingestion_provenance'] = {
        'months': months,
        'source_rows_in_files': sum(proof['source_rows'] for proof in proofs),
        'scanned_rows': sum(proof['scanned_rows'] for proof in proofs),
        'accepted_rows': sum(proof['accepted_rows'] for proof in proofs),
        'panel_rows': len(panel),
        'coverage_kind': ('full_selected_official_months' if complete else 'sampled_official_row_groups') if official else 'audited_local_or_other_sources',
        'all_official_sources': official,
        'all_months_complete': complete,
        'no_sampling_requested': all(proof['no_sampling_requested'] for proof in proofs),
        'monthly_proofs': proofs,
    }
    return panel


def analyze(config, panel=None, output_override=None):
    from .analysis import compare_month, rank_candidates, evaluate_next_month, standardize_zones
    from .reporting import write_report
    from .data import month_range
    expected_months = (month_range(config['start_month'], config['end_month'])
                       if 'start_month' in config and 'end_month' in config else None)
    if panel is None:
        if expected_months is None:
            raise ValueError('Real data analysis requires configured start_month and end_month')
        panel = load_panel(config['data_dir'], expected_months=expected_months)
    ingestion = panel.attrs.get('_tlc_ingestion_provenance')
    if ingestion is None:
        if 'synthetic' not in str(config.get('label', '')).lower():
            raise ValueError('Direct in-memory panels require a SYNTHETIC label or verified load_panel provenance')
        ingestion = {
            'months': sorted(panel['source_month'].astype(str).unique().tolist()),
            'source_rows_in_files': None, 'scanned_rows': None, 'accepted_rows': None,
            'coverage_kind': 'synthetic_not_real', 'all_official_sources': False,
            'all_months_complete': False, 'no_sampling_requested': False,
            'monthly_proofs': [],
        }
    else:
        if expected_months is None or ingestion['months'] != expected_months:
            raise ValueError('Audited months must exactly match configured start_month/end_month')
        if ingestion['accepted_rows'] != int(panel['n'].sum()) or ingestion['panel_rows'] != len(panel):
            raise ValueError('In-memory panel was changed after audit validation')
        for proof in ingestion['monthly_proofs']:
            for key in ['sample_row_groups', 'max_wait_seconds']:
                expected = config.get(key, 7200 if key == 'max_wait_seconds' else None)
                if proof['params'].get(key) != expected:
                    raise ValueError(f"Audited {key} for {proof['month']} disagrees with the analysis configuration")
    if expected_months is not None and ingestion['months'] != expected_months:
        raise ValueError('Observed/verified months do not match the configured range')
    kwargs = {k: config[k] for k in ['reference_months','prior_strength','min_n','min_n_ref','min_days','min_ref_days','bootstrap_reps','seed'] if k in config}
    month = config['monitor_month']
    started = time.perf_counter()
    comparison = compare_month(panel, month, **kwargs)
    queue = rank_candidates(comparison, method='posterior_delta', top_k=config.get('top_k',20))
    evaluation = evaluate_next_month(comparison, panel, month, top_k=config.get('top_k',20), min_n=config.get('min_n',100), min_days=config.get('min_days',3))
    standardized = standardize_zones(panel, month, reference_months=config.get('reference_months',3))
    monitor_period = pd.Period(month, freq='M')
    reference = [str(monitor_period - i) for i in range(config.get('reference_months', 3), 0, -1)]
    next_month = str(monitor_period + 1)
    reference_panel = panel[panel['source_month'].isin(reference)]
    full_completed = (
        ingestion['months'] == month_range('2019-02', '2025-12')
        and ingestion['all_official_sources'] and ingestion['all_months_complete']
        and ingestion['no_sampling_requested'] and config.get('sample_row_groups') is None
    )
    provenance = {
        'label': config['label'], 'monitor_month': month, 'version': __version__,
        'configuration': config, 'python': platform.python_version(),
        'panel_rows': len(panel), 'valid_trip_count_in_panel': int(panel['n'].sum()),
        'source_rows_in_files': ingestion['source_rows_in_files'],
        'scanned_rows': ingestion['scanned_rows'], 'accepted_rows': ingestion['accepted_rows'],
        'months': ingestion['months'], 'coverage_kind': ingestion['coverage_kind'],
        'monthly_ingestion_counts': [{k: p[k] for k in ['month', 'source_rows', 'scanned_rows', 'accepted_rows', 'panel_rows', 'official_source', 'complete_month', 'no_sampling_requested']} for p in ingestion['monthly_proofs']],
        'analysis_windows': {
            'reference_months': reference, 'monitor_month': month, 'next_observation_month': next_month,
            'reference_daily_panel_rows': len(reference_panel),
            'reference_candidate_cells': len(reference_panel[['zone', 'platform', 'slot', 'weekday']].drop_duplicates()),
            'reference_accepted_trip_count': int(reference_panel['n'].sum()),
            'monitor_accepted_trip_count': int(panel.loc[panel['source_month'] == month, 'n'].sum()),
            'next_observation_accepted_trip_count': int(panel.loc[panel['source_month'] == next_month, 'n'].sum()),
        },
        'counting_note': 'Source-file rows cover every selected source month; scanned rows were physically processed; accepted rows survived the exclusive filter ledger. Daily panel rows and reference candidate cells are aggregated statistical inputs, not additional raw trips or independent training observations. Synthetic counts are generated examples and have no real source/scanned/accepted counts.',
        'analysis_seconds': round(time.perf_counter()-started,3),
        'interpretation': 'Retrospective released-data monitoring; future-month persistence is an offline proxy, not operational or causal impact.',
        'full_83_month_processing_completed': bool(full_completed),
    }
    out = output_override or config['output_dir']
    result = write_report(out, comparison, queue, evaluation, standardized, provenance)
    print(json.dumps({'output_dir':str(out),'candidate_cells':len(comparison),'review_queue_rows':len(queue),'panel_rows':len(panel),'valid_trip_count_in_panel':provenance['valid_trip_count_in_panel']},ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='TLC completed-trip retrospective monitoring research prototype')
    sub = parser.add_subparsers(dest='command', required=True)
    m = sub.add_parser('manifest'); m.add_argument('--start',default='2019-02'); m.add_argument('--end',default='2025-12'); m.add_argument('--out',default='results/source_manifest.json')
    for command in ['process','analyze','run','backtest']:
        p = sub.add_parser(command); p.add_argument('--config',default='configs/pilot.json')
        if command == 'backtest':
            p.add_argument('--start',required=True); p.add_argument('--end',required=True)
    d = sub.add_parser('demo'); d.add_argument('--out',default='results/demo')
    args = parser.parse_args()
    if args.command == 'manifest':
        from .data import build_manifest
        records = build_manifest(args.start,args.end,output_path=args.out)
        print(json.dumps({'files':len(records),'rows':sum(r['rows'] for r in records),'bytes':sum(r['bytes'] for r in records),'manifest':args.out}),flush=True)
        return
    if args.command == 'demo':
        from .analysis import synthetic_panel
        panel = synthetic_panel()
        out = Path(args.out); out.mkdir(parents=True,exist_ok=True)
        panel.to_parquet(out/'synthetic_daily_panel.parquet',index=False)
        config = {'label':'SYNTHETIC_PIPELINE_DEMO_NOT_REAL_TLC','monitor_month':'2025-06','output_dir':str(out),'reference_months':3,'prior_strength':50,'min_n':100,'min_n_ref':300,'min_days':3,'min_ref_days':6,'bootstrap_reps':300,'seed':42,'top_k':20}
        analyze(config,panel=panel)
        return
    config = load_config(args.config)
    if args.command in ['process','run']:
        from .data import process_month,month_range,official_url
        for month in month_range(config['start_month'],config['end_month']):
            result = process_month(official_url(month),month,config['data_dir'],
                                   max_wait_seconds=config.get('max_wait_seconds',7200),
                                   sample_row_groups=config.get('sample_row_groups'),
                                   batch_size=config.get('batch_size',131072))
            print(json.dumps({'month':month,'panel_path':result['panel_path'],'audit_path':result['audit_path'],'cached':result['cached']},ensure_ascii=False),flush=True)
    if args.command in ['analyze','run']:
        analyze(config)
    if args.command == 'backtest':
        from .data import month_range
        panel = load_panel(config['data_dir'])
        rows = []
        for month in month_range(args.start,args.end):
            c = dict(config,monitor_month=month)
            summary = analyze(c,panel=panel,output_override=Path(config['output_dir'])/'rolling'/month)
            for row in summary['evaluation']:
                rows.append(dict(monitor_month=month,**row))
        dest = Path(config['output_dir'])/'rolling_evaluation.csv'
        pd.DataFrame(rows).to_csv(dest,index=False)
        print(f'rolling_evaluation={dest}',flush=True)


if __name__ == '__main__':
    main()
