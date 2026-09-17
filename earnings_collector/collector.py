"""Ticker resolution, filing discovery, and provenance-preserving storage."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse

from .parsing import exhibits, extract_text, is_earnings_release
from .sec import CollectionError


def now():
    return datetime.now(timezone.utc).isoformat()


def filing_rows(columns):
    for i, accession in enumerate(columns.get('accessionNumber', [])):
        yield {key: values[i] for key, values in columns.items() if isinstance(values, list) and len(values) > i}


def collect(ticker, client, output, max_earnings_filings=8):
    ticker = ticker.strip().upper()
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9.-]{0,14}', ticker):
        raise CollectionError('Enter a valid stock ticker, such as AAPL or BRK-B.')
    directory = client.json('https://www.sec.gov/files/company_tickers.json')
    company = next((c for c in directory.values() if c['ticker'].upper() == ticker), None)
    if company is None:
        raise CollectionError(f'Ticker {ticker} was not found in the SEC company directory.')
    cik = int(company['cik_str'])
    submissions_url = f'https://data.sec.gov/submissions/CIK{cik:010d}.json'
    submissions = client.json(submissions_url)
    root = Path(output) / ticker
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        'ticker': ticker, 'company_name': submissions.get('name', company['title']), 'cik': f'{cik:010d}',
        'collected_at': now(), 'submissions_url': submissions_url, 'documents': [], 'warnings': [],
        'coverage': 'Recent SEC submissions only; older submission shards are not searched.',
        'earnings_release_status': 'not_found',
    }
    (root / 'submissions.json').write_text(json.dumps(submissions, indent=2) + '\n')
    rows = sorted(filing_rows(submissions.get('filings', {}).get('recent', {})),
                  key=lambda r: (r.get('filingDate', ''), r.get('accessionNumber', '')), reverse=True)

    def base(row):
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{row['accessionNumber'].replace('-', '')}/"

    def save(row, url, kind, raw=None, extra=None):
        raw = client.get(url, immutable=True) if raw is None else raw
        name = Path(urlparse(url).path).name
        folder = root / row['accessionNumber']
        folder.mkdir(exist_ok=True)
        raw_path = folder / name
        raw_path.write_bytes(raw)
        supported = name.lower().endswith(('.htm', '.html', '.txt'))
        text_path = folder / (name + '.txt')
        text = extract_text(raw) if supported else ''
        if text:
            text_path.write_text(text, encoding='utf-8')
        record = {
            'kind': kind, 'form': row['form'], 'accession_number': row['accessionNumber'],
            'filing_date': row.get('filingDate'), 'sec_report_date': row.get('reportDate') or None,
            'reporting_period_end': row.get('reportDate') if row['form'] in {'10-K', '10-Q'} else None,
            'source_url': url, 'saved_at': now(), 'sha256': hashlib.sha256(raw).hexdigest(),
            'raw_path': str(raw_path.relative_to(root)),
            'text_path': str(text_path.relative_to(root)) if text else None,
            'extraction_status': 'complete' if text else 'unsupported_or_empty',
        }
        record.update(extra or {})
        manifest['documents'].append(record)
        return record

    for form in ('10-K', '10-Q'):
        row = next((r for r in rows if r.get('form') == form), None)
        if row is None:
            manifest['warnings'].append(f'No {form} in recent submissions.')
            continue
        try:
            save(row, base(row) + quote(row['primaryDocument'], safe=''), 'annual_report' if form == '10-K' else 'quarterly_report')
        except CollectionError as exc:
            manifest['warnings'].append(str(exc))

    # SEC item lists may include spaces after commas.
    candidates = [r for r in rows if r.get('form') == '8-K' and '2.02' in [x.strip() for x in r.get('items', '').split(',')]]
    newer_unresolved = False
    for row in candidates[:max_earnings_filings]:
        found = False
        try:
            index_url = base(row) + row['accessionNumber'] + '-index.html'
            index_raw = client.get(index_url, immutable=True)
            index_folder = root / row['accessionNumber']
            index_folder.mkdir(exist_ok=True)
            (index_folder / 'filing-index.html').write_bytes(index_raw)
            entries = exhibits(index_raw, base(row))
            save(row, base(row) + quote(row['primaryDocument'], safe=''), 'earnings_8k')
            for entry in entries:
                raw = client.get(entry['url'], immutable=True)
                supported = urlparse(entry['url']).path.lower().endswith(('.htm', '.html', '.txt'))
                matched = supported and is_earnings_release(extract_text(raw))
                save(row, entry['url'], 'earnings_release_candidate' if matched else 'earnings_exhibit', raw,
                     {'exhibit_type': entry['type'], 'description': entry['description'],
                      'classification': 'heuristic_match' if matched else 'unconfirmed'})
                found = found or matched
            if found:
                manifest['earnings_release_status'] = 'candidate_found_with_newer_gaps' if newer_unresolved else 'candidate_found'
                break
            newer_unresolved = True
        except CollectionError as exc:
            manifest['warnings'].append(str(exc))
            newer_unresolved = True
            if found:
                manifest['earnings_release_status'] = 'candidate_found_with_gaps'
                break
    if manifest['earnings_release_status'] == 'not_found':
        manifest['warnings'].append('No earnings release confidently identified in the inspected Item 2.02 filings.')
    if newer_unresolved:
        manifest['warnings'].append('Some earnings filings could not be fully classified or retrieved; latest release coverage is not verified.')
    manifest['warnings'].append('Transcripts are not collected. Earnings exhibit period ends require validation; 8-K report dates are event dates.')
    manifest['status'] = 'partial' if (manifest['earnings_release_status'] != 'candidate_found' or
                                      any(d['extraction_status'] != 'complete' for d in manifest['documents']) or
                                      not all(any(d['form'] == f for d in manifest['documents']) for f in ('10-K', '10-Q'))) else 'complete'
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return path, manifest
