import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from earnings_collector.collector import collect
from earnings_collector.parsing import exhibits, extract_text, is_earnings_release
from earnings_collector.sec import CollectionError, SecClient


BASE = 'https://www.sec.gov/Archives/edgar/data/320193/000032019326000003/'
INDEX = b'''<table><tr><td>1</td><td>Earnings release</td><td><a href="release.htm">release.htm</a></td><td>EX-99.1</td></tr>
<tr><td>2</td><td>External</td><td><a href="https://evil.test/x">x</a></td><td>EX-99.2</td></tr></table>'''
RELEASE = b'<h1>Apple reports third quarter results</h1><p>Revenue was $100 billion.</p>'


class FakeClient:
    def __init__(self, fail_release=False, missing=False):
        self.fail_release = fail_release
        self.missing = missing

    def json(self, url):
        if url.endswith('company_tickers.json'):
            return {'0': {'ticker': 'AAPL', 'cik_str': 320193, 'title': 'Apple'}}
        return {'name': 'Apple Inc.', 'filings': {'recent': {
            'accessionNumber': ['0000320193-26-000003', '0000320193-26-000002', '0000320193-25-000001'],
            'form': ['8-K', '10-Q', '10-K'], 'items': ['2.02, 9.01', '', ''],
            'filingDate': ['2026-07-30', '2026-05-01', '2025-10-31'],
            'reportDate': ['2026-07-30', '2026-03-28', '2025-09-27'],
            'primaryDocument': ['event.htm', 'quarter.htm', 'annual.htm'],
        }}}

    def get(self, url, immutable=False):
        if url.endswith('-index.html'):
            return b'<table></table>' if self.missing else INDEX
        if url.endswith('release.htm'):
            if self.fail_release:
                raise CollectionError('Simulated provider failure')
            return RELEASE
        return b'<p>A filing with financial information.</p>'


class ParsingTests(unittest.TestCase):
    def test_text_preserves_tables_and_hides_ix_metadata(self):
        text = extract_text(b'<head><title>Hidden</title></head><ix:hidden>999</ix:hidden><h1>Results</h1><table><tr><td>Revenue</td><td>$10 &amp; more</td></tr></table><script>bad()</script>')
        self.assertEqual(text, 'Results\nRevenue\t$10 & more')

    def test_exhibits_only_accept_this_sec_filing(self):
        result = exhibits(INDEX, BASE)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['url'], BASE + 'release.htm')

    def test_release_requires_financial_announcement(self):
        self.assertTrue(is_earnings_release(extract_text(RELEASE)))
        self.assertFalse(is_earnings_release('Quarterly dividend announced. Revenue unchanged.'))


class PipelineTests(unittest.TestCase):
    def test_collection_provenance_and_periods(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, result = collect('aapl', FakeClient(), tmp)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(result['documents']), 4)
            self.assertEqual(json.loads(path.read_text())['ticker'], 'AAPL')
            release = next(d for d in result['documents'] if d['kind'] == 'earnings_release_candidate')
            self.assertIsNone(release['reporting_period_end'])
            self.assertEqual(release['sec_report_date'], '2026-07-30')
            self.assertTrue((path.parent / release['raw_path']).exists())
            self.assertIn('Revenue', (path.parent / release['text_path']).read_text())
            quarter = next(d for d in result['documents'] if d['form'] == '10-Q')
            self.assertEqual(quarter['reporting_period_end'], '2026-03-28')

    def test_missing_release_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, result = collect('AAPL', FakeClient(missing=True), tmp)
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['earnings_release_status'], 'not_found')

    def test_failed_download_keeps_other_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, result = collect('AAPL', FakeClient(fail_release=True), tmp)
            self.assertEqual(result['status'], 'partial')
            self.assertTrue(path.exists())
            self.assertIn('Simulated provider failure', result['warnings'])
            self.assertEqual(len(result['documents']), 3)

    def test_unknown_or_unsafe_ticker(self):
        with tempfile.TemporaryDirectory() as tmp:
            for ticker in ['UNKNOWN', '../../elsewhere']:
                with self.assertRaises(CollectionError):
                    collect(ticker, FakeClient(), tmp)

    def test_older_release_does_not_hide_newer_gap(self):
        class HistoricalClient(FakeClient):
            def json(self, url):
                data = super().json(url)
                if 'filings' in data:
                    columns = data['filings']['recent']
                    older = {
                        'accessionNumber': '0000320193-26-000001', 'form': '8-K',
                        'items': '2.02', 'filingDate': '2026-04-30',
                        'reportDate': '2026-04-30', 'primaryDocument': 'older.htm',
                    }
                    for key, value in older.items():
                        columns[key].append(value)
                return data

            def get(self, url, immutable=False):
                if url == BASE + '0000320193-26-000003-index.html':
                    raise CollectionError('Newer filing unavailable')
                return super().get(url, immutable)

        with tempfile.TemporaryDirectory() as tmp:
            _, result = collect('AAPL', HistoricalClient(), tmp)
            self.assertEqual(result['earnings_release_status'], 'candidate_found_with_newer_gaps')
            self.assertEqual(result['status'], 'partial')

    def test_unsupported_exhibit_is_saved_and_flagged(self):
        class PdfClient(FakeClient):
            def get(self, url, immutable=False):
                if url.endswith('-index.html'):
                    return INDEX.replace(b'release.htm', b'release.pdf')
                if url.endswith('.pdf'):
                    return b'%PDF-1.7 synthetic'
                return super().get(url, immutable)

        with tempfile.TemporaryDirectory() as tmp:
            path, result = collect('AAPL', PdfClient(), tmp)
            exhibit = next(d for d in result['documents'] if d['kind'] == 'earnings_exhibit')
            self.assertIsNone(exhibit['text_path'])
            self.assertTrue((path.parent / exhibit['raw_path']).exists())
            self.assertEqual(result['status'], 'partial')


class ClientTests(unittest.TestCase):
    def test_immutable_download_is_cached(self):
        with tempfile.TemporaryDirectory() as tmp, patch('earnings_collector.sec.urlopen') as opener:
            opener.return_value.__enter__.return_value.read.return_value = b'document'
            client = SecClient('Test test@example.com', tmp)
            for _ in range(2):
                self.assertEqual(client.get(BASE + 'release.htm', immutable=True), b'document')
            self.assertEqual(opener.call_count, 1)

    def test_retries_transient_errors(self):
        with tempfile.TemporaryDirectory() as tmp, patch('earnings_collector.sec.urlopen') as opener, patch('earnings_collector.sec.time.sleep'):
            opener.side_effect = HTTPError(BASE, 503, 'Unavailable', {}, None)
            with self.assertRaises(CollectionError):
                SecClient('Test test@example.com', tmp).get(BASE)
            self.assertEqual(opener.call_count, 3)

    def test_does_not_retry_access_denial(self):
        with tempfile.TemporaryDirectory() as tmp, patch('earnings_collector.sec.urlopen') as opener:
            opener.side_effect = HTTPError(BASE, 403, 'Forbidden', {}, None)
            with self.assertRaises(CollectionError):
                SecClient('Test test@example.com', tmp).get(BASE)
            self.assertEqual(opener.call_count, 1)


if __name__ == '__main__':
    unittest.main()
