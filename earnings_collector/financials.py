"""Conservative SEC financial summaries, pinned to collected filing accessions."""
import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path

from .sec import CollectionError, SecClient

METRICS = {
    'revenue': (['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet'], 'USD'),
    'net_income': (['NetIncomeLoss'], 'USD'),
    'operating_income': (['OperatingIncomeLoss'], 'USD'),
    'diluted_eps': (['EarningsPerShareDiluted'], 'USD/shares'),
    'operating_cash_flow': (['NetCashProvidedByUsedInOperatingActivities'], 'USD'),
}


class FinancialError(Exception):
    pass


class FilingMetadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.values = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'ix:nonnumeric':
            name = attrs.get('name', '').lower()
            self.stack.append((name.split(':')[-1] if name.startswith('dei:') else None, []))

    def handle_data(self, data):
        for _, buffer in self.stack:
            buffer.append(data)

    def handle_endtag(self, tag):
        if tag == 'ix:nonnumeric' and self.stack:
            field, buffer = self.stack.pop()
            if field:
                self.values.setdefault(field, set()).add(' '.join(''.join(buffer).split()))


def safe_read(root, relative):
    if not isinstance(relative, str):
        raise FinancialError('Missing document path.')
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise FinancialError('Document path escapes company directory.')
    return path.read_bytes()


def validate_period(root, doc):
    result = {'source_url': doc['source_url'], 'form': doc['form'],
              'accession_number': doc['accession_number'],
              'event_date': doc.get('sec_report_date') if doc['form'] == '8-K' else None,
              'period_end': None, 'status': 'unknown'}
    if doc['form'] in ('10-Q', '10-K'):
        raw = safe_read(root, doc['raw_path'])
        if hashlib.sha256(raw).hexdigest() != doc.get('sha256'):
            result.update(status='conflict', reason='Original document checksum mismatch.')
            return result
        parser = FilingMetadata()
        parser.feed(raw.decode('utf-8', errors='replace'))
        ends = parser.values.get('documentperiodenddate', set())
        if len(ends) != 1:
            result['reason'] = 'Missing or ambiguous inline XBRL period end.'
            return result
        end = next(iter(ends))
        try:
            try:
                end = date.fromisoformat(end).isoformat()
            except ValueError:
                end = datetime.strptime(end, '%B %d, %Y').date().isoformat()
        except ValueError:
            result['reason'] = 'Invalid inline period date.'
            return result
        if end != doc.get('reporting_period_end'):
            result.update(status='conflict', reason='Inline XBRL and manifest periods disagree.')
            return result
        result.update(status='validated', period_end=end, method='inline_dei_matches_manifest')
        for field in ('documentfiscalyearfocus', 'documentfiscalperiodfocus'):
            values = parser.values.get(field, set())
            result[field] = next(iter(values)) if len(values) == 1 else None
    elif doc.get('kind') == 'earnings_release_candidate':
        text = safe_read(root, doc['text_path']).decode('utf-8')
        # An explicit statement in the introductory release text, not an 8-K event date.
        pattern = r'(?:quarter|year)\s+ended\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})'
        matches = list(re.finditer(pattern, text[:2500]))
        ends = set()
        for match in matches:
            try:
                ends.add(datetime.strptime(match.group(1), '%B %d, %Y').date().isoformat())
            except ValueError:
                pass
        if len(ends) == 1:
            result.update(status='text_identified', period_end=next(iter(ends)), method='explicit_release_statement',
                          evidence=[{'text': m.group(), 'char_start': m.start(), 'char_end': m.end()} for m in matches])
    return result


def duration(fact):
    try:
        return (date.fromisoformat(fact['end']) - date.fromisoformat(fact['start'])).days + 1
    except (KeyError, ValueError, TypeError):
        return None


def unique_fact(facts):
    unique = {(f['start'], f['end'], str(f['val'])): f for f in facts}
    if len(unique) == 1:
        return next(iter(unique.values()))
    return None


def metric_record(fact, tag, unit, doc, facts_url):
    return {'value': fact['val'], 'unit': unit, 'currency': 'USD', 'scale': 1,
            'accounting_basis': 'US-GAAP', 'taxonomy': 'us-gaap', 'tag': tag,
            'start': fact['start'], 'end': fact['end'], 'duration_days': duration(fact),
            'accession_number': fact['accn'], 'filed': fact.get('filed'),
            'source_url': doc['source_url'], 'companyfacts_url': facts_url,
            'facts_selector': {'taxonomy': 'us-gaap', 'concept': tag, 'unit': unit,
                               'accn': fact['accn'], 'start': fact['start'], 'end': fact['end']}}


def select_metric(facts, tags, unit, doc, end, facts_url):
    bounds = (60, 120) if doc['form'] == '10-Q' else (330, 400)
    taxonomy = facts.get('facts', {}).get('us-gaap', {})
    for tag in tags:
        entries = taxonomy.get(tag, {}).get('units', {}).get(unit, [])
        eligible = [f for f in entries if f.get('accn') == doc['accession_number']
                    and f.get('form') == doc['form'] and duration(f) is not None
                    and bounds[0] <= duration(f) <= bounds[1]
                    and type(f.get('val')) in (int, float) and math.isfinite(f['val'])]
        current_candidates = [f for f in eligible if f['end'] == end]
        if not current_candidates:
            continue
        current = unique_fact(current_candidates)
        if current is None:
            return {'status': 'ambiguous', 'reason': 'Conflicting values or periods in the selected filing.'}
        prior_candidates = [f for f in eligible
                            if 350 <= (date.fromisoformat(end) - date.fromisoformat(f['end'])).days <= 380
                            and abs(duration(f) - duration(current)) <= 10]
        prior = unique_fact(prior_candidates)
        growth = None
        reason = 'No unambiguous same-tag, same-unit prior-year comparison in this filing.'
        if prior:
            if prior['val'] > 0:
                growth = float((Decimal(str(current['val'])) / Decimal(str(prior['val'])) - 1) * 100)
                reason = None
            else:
                reason = 'Growth percentage withheld for a zero or negative prior-year base.'
        return {'status': 'available', 'current': metric_record(current, tag, unit, doc, facts_url),
                'prior_year': metric_record(prior, tag, unit, doc, facts_url) if prior else None,
                'year_over_year_percent': growth, 'comparison_note': reason}
    return {'status': 'unavailable', 'reason': 'No matching US-GAAP fact for this accession, unit, and duration. Year-to-date values are not substituted for quarters.'}


def build_summary(root, manifest, facts):
    if str(facts.get('cik', '')).zfill(10) != str(manifest['cik']).zfill(10):
        raise FinancialError('Company facts CIK does not match the collected company.')
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(manifest['cik']):010d}.json"
    periods, reports = [], []
    for doc in manifest['documents']:
        try:
            period = validate_period(root, doc)
        except (FinancialError, OSError, UnicodeError) as exc:
            period = {'source_url': doc.get('source_url'), 'status': 'unknown', 'reason': str(exc), 'period_end': None}
        periods.append(period)
        if doc['form'] not in ('10-Q', '10-K') or period['status'] != 'validated':
            continue
        metrics = {name: select_metric(facts, tags, unit, doc, period['period_end'], url)
                   for name, (tags, unit) in METRICS.items()}
        reports.append({'period_type': 'quarter' if doc['form'] == '10-Q' else 'annual',
                        'period_end': period['period_end'], 'fiscal_year': period.get('documentfiscalyearfocus'),
                        'fiscal_period': period.get('documentfiscalperiodfocus'), 'source_url': doc['source_url'],
                        'metrics': metrics})
    validated_ends = {p['period_end'] for p in periods if p['status'] == 'validated'}
    for period in periods:
        if period['status'] == 'text_identified':
            period['matches_collected_filing_period'] = period['period_end'] in validated_ends
    return {'ticker': manifest['ticker'], 'company_name': manifest['company_name'],
            'cik': manifest['cik'], 'companyfacts_url': url, 'document_periods': periods, 'reports': reports,
            'policy': 'As presented in collected filings. No later-restatement merging, adjusted metrics, inferred Q4, or derived quarterly cash flow.',
            'warnings': ['US-GAAP USD metrics only; unsupported tags and missing values stay unavailable.',
                         'Release dates identified from text are evidence, not an automatic validation of every figure in the release.']}


def markdown(summary):
    lines = [f"# {summary['ticker']} financial summary", '', summary['policy'], '']
    for report in summary['reports']:
        lines += [f"## {report['period_type'].title()} ended {report['period_end']}", '',
                  '| Metric | Reported value | Prior year | YoY change |', '| --- | --- | --- | --- |']
        for name, metric in report['metrics'].items():
            if metric['status'] != 'available':
                lines.append(f"| {name} | {metric['status']} | | |")
                continue
            cur, prior = metric['current'], metric['prior_year']
            growth = metric['year_over_year_percent']
            lines.append(f"| {name} | [{cur['value']:,} {cur['unit']}]({cur['source_url']}) | {str(prior['value']) if prior else 'Unavailable'} | {f'{growth:.2f}%' if growth is not None else 'Unavailable'} |")
        lines += ['', 'Values are in base units, not millions. Detailed periods, tags, and missing-data explanations are in financial-summary.json.', '']
    return '\n'.join(lines) + '\n'


def prepare_financials(ticker, data_dir=Path('data'), refresh=False, client=None):
    ticker = ticker.strip().upper()
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9.-]{0,14}', ticker):
        raise FinancialError('Invalid ticker.')
    root = data_dir / ticker
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('ticker') != ticker:
        raise FinancialError('Manifest ticker mismatch.')
    path = root / 'companyfacts.json'
    if refresh or not path.exists():
        agent = os.environ.get('SEC_USER_AGENT', '')
        if client is None and '@' not in agent:
            raise FinancialError('Export SEC_USER_AGENT with your contact email for SEC downloads.')
        client = client or SecClient(agent, data_dir / '.cache')
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(manifest['cik']):010d}.json"
        raw = client.get(url)
        facts = json.loads(raw)
        if int(facts['cik']) != int(manifest['cik']):
            raise FinancialError('SEC response CIK mismatch.')
        path.write_bytes(raw)
        (root / 'companyfacts-source.json').write_text(json.dumps({'source_url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat(), 'sha256': hashlib.sha256(raw).hexdigest()}, indent=2)+'\n')
    raw = path.read_bytes()
    result = build_summary(root, manifest, json.loads(raw))
    result['companyfacts_sha256'] = hashlib.sha256(raw).hexdigest()
    result['generated_at'] = datetime.now(timezone.utc).isoformat()
    (root / 'financial-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    (root / 'financial-summary.md').write_text(markdown(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ticker')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--refresh', action='store_true', help='Refresh SEC company facts; otherwise reuse the saved snapshot')
    args = parser.parse_args()
    try:
        result = prepare_financials(args.ticker, args.data_dir, args.refresh)
        ticker = result['ticker']
        root = args.data_dir / ticker
        print(f"{ticker}: saved {len(result['reports'])} period summaries to {root.resolve() / 'financial-summary.json'}")
        return 0 if result['reports'] else 2
    except (FinancialError, CollectionError, OSError, ValueError, KeyError) as exc:
        print(f'Financial summary failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
