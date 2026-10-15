"""Create inspectable CSVs, an offline HTML report, and publication-style PNGs."""
from pathlib import Path
from html import escape
import json
import numpy as np
import pandas as pd


def write_report(out_dir, comparison, queue, evaluation, standardized, provenance):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out / 'comparison.csv', index=False)
    queue.to_csv(out / 'review_queue.csv', index=False)
    evaluation.to_csv(out / 'evaluation.csv', index=False)
    standardized.to_csv(out / 'standardized_zones.csv', index=False)
    summary = dict(provenance)
    summary.update(candidate_cells=len(comparison), review_queue_rows=len(queue),
                   evaluation=evaluation.replace({np.nan: None}).to_dict(orient='records'))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    wanted = ['zone', 'platform', 'slot', 'weekday', 'n', 'n_ref', 'raw_rate', 'reference_rate',
              'posterior_delta_pp', 'ci_low_pp', 'ci_high_pp', 'evidence_status']
    display = comparison[[c for c in wanted if c in comparison]].copy()
    if 'posterior_delta_pp' in display:
        display = display.sort_values('posterior_delta_pp', ascending=False)
    display = display.head(1000)
    # Escape every data value; local report contains no network resources.
    table = display.to_html(index=False, classes='data', float_format=lambda x: f'{x:.4f}', escape=True)
    queue_html = queue[[c for c in wanted if c in queue]].to_html(index=False, float_format=lambda x: f'{x:.4f}', escape=True)
    eval_html = evaluation.to_html(index=False, float_format=lambda x: f'{x:.4f}', escape=True)
    title = escape(str(provenance.get('label', 'TLC monitoring')))
    meta = escape(json.dumps(provenance, ensure_ascii=False, indent=2, default=str))
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TLC Service Review</title><style>
body{font:16px/1.6 system-ui,sans-serif;color:#172a3a;background:#f4f6f8;margin:0}main{max-width:1280px;margin:40px auto;padding:32px;background:white;border-radius:12px}
h1{font-size:30px}h2{font-size:21px;margin-top:32px}.note{background:#eef3f8;padding:16px;border-left:4px solid #486d92}.scroll{overflow:auto;max-height:520px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #dfe5eb;padding:9px;text-align:right;white-space:nowrap}th{background:#edf2f6;position:sticky;top:0}input{padding:10px;border:1px solid #a9b7c6;width:300px;max-width:80%}pre{white-space:pre-wrap;font-size:12px}img{max-width:100%}</style>
<main><h1>TLC Service Interval Review</h1><p>RUN_LABEL</p>
<div class="note">Retrospective review of request-to-pickup intervals among eligible completed trips. Signals identify cells for review; they do not establish violations or causal effects. Date-block intervals are descriptive after selection, and subsequent-month persistence is an offline measure.</div>
<h2>Priority review queue</h2><p>Cells meet minimum support, have a positive estimated increase, and have a date-block interval above zero. The queue may contain fewer than the requested number of cells.</p><div class="scroll">QUEUE</div>
<h2>Persistence in the next observed month</h2><div class="scroll">EVALUATION</div>
<h2>Candidate cells</h2><p>The table shows up to 1,000 cells ranked by estimated increase. The complete table is in comparison.csv. Search by zone, platform, or evidence status.</p>
<input id="search" placeholder="Filter table"><div class="scroll" id="candidates">CANDIDATES</div>
<h2>Run provenance</h2><pre>PROVENANCE</pre></main>
<script>document.getElementById('search').addEventListener('input',function(){const q=this.value.toLowerCase();document.querySelectorAll('#candidates tbody tr').forEach(r=>{r.style.display=r.textContent.toLowerCase().includes(q)?'':'none'});});</script></html>'''
    html = html.replace('RUN_LABEL', title).replace('QUEUE', queue_html).replace('EVALUATION', eval_html).replace('CANDIDATES', table).replace('PROVENANCE', meta)
    (out / 'report.html').write_text(html, encoding='utf-8')
    make_plot(out, comparison, provenance)
    return summary


def make_plot(out, comparison, provenance):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    ok = comparison[comparison.get('support_eligible', pd.Series(False, index=comparison.index)).fillna(False)]
    if len(ok):
        axes[0].scatter(ok['n'], ok['posterior_delta_pp'], s=9, alpha=.35, color='#2f6085')
        axes[0].set_xscale('log')
        axes[0].axhline(0, color='#89949e', linewidth=.8)
        axes[0].set(xlabel='Eligible completed trips in monitor month (log)', ylabel='Shrunk excess rate (percentage points)', title='Supported cells')
        top = ok.sort_values('posterior_delta_pp', ascending=False).head(12).iloc[::-1]
        y = np.arange(len(top))
        axes[1].scatter(top['posterior_delta_pp'], y, color='#2f6085', zorder=3)
        axes[1].hlines(y, top['ci_low_pp'], top['ci_high_pp'], color='#81a7bf')
        axes[1].set_yticks(y, [f"Z{int(r.zone)} / {r.platform} / S{int(r.slot)} / D{int(r.weekday)}" for r in top.itertuples()], fontsize=7)
        axes[1].axvline(0, color='#89949e', linewidth=.8)
        axes[1].set(xlabel='Percentage points; date-block percentile intervals', title='Largest estimated changes (selected, descriptive)')
    else:
        for ax in axes:
            ax.text(.5, .5, 'No cells meet support requirements', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
    fig.suptitle(str(provenance.get('label', 'TLC')) + ' | ' + str(provenance.get('monitor_month', '')), fontsize=12)
    fig.savefig(Path(out) / 'risk_diagnostics.png', dpi=180)
    plt.close(fig)
