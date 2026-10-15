"""Read only official Parquet footers; do not claim a full data scan."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timezone
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlc_service.data import official_url, probe_parquet  # noqa: E402


def probe(month):
    # Shared probe uses stream=True, checks status 206 before reading the body,
    # and reads only the exact trailer/footer ranges with a 64 MiB footer cap.
    # Keep the existing manifest shape for documents and downstream consumers.
    url = official_url(month)
    metadata = probe_parquet(url, retries=3, timeout=45)
    return {'month': month, 'url': url, 'rows': metadata['rows'],
            'bytes': metadata['bytes'], 'row_groups': metadata['row_groups'],
            'columns': metadata['columns'], 'etag': metadata['etag'],
            'last_modified': metadata['last_modified']}


def main():
    months = [f'{y}-{m:02d}' for y in range(2019,2026) for m in range(1,13) if (y,m)>=(2019,2)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(probe, months))
    result = {'verification': 'remote_parquet_metadata_only_not_full_scan',
              'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'files': files, 'file_count': len(files),
              'total_rows': sum(x['rows'] for x in files), 'total_bytes':sum(x['bytes'] for x in files)}
    out = Path('results/source_manifest_verified.json'); out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'files'},ensure_ascii=False))


if __name__ == '__main__':
    main()
