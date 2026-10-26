import hashlib
import tempfile
import unittest
from pathlib import Path
from earnings_collector.financials import select_metric, validate_period, build_summary, FinancialError


class FinancialTests(unittest.TestCase):
    def setUp(self):
        self.doc = {'form': '10-Q', 'accession_number': 'current', 'source_url': 'https://www.sec.gov/report',
                    'reporting_period_end': '2026-06-27'}
        self.current = {'accn': 'current', 'form': '10-Q', 'start': '2026-03-29', 'end': '2026-06-27', 'val': 120}
        self.prior = {**self.current, 'start': '2025-03-30', 'end': '2025-06-28', 'val': 100}

    def select(self, entries, unit='USD', doc=None):
        facts = {'facts': {'us-gaap': {'Revenues': {'units': {unit: entries}}}}}
        return select_metric(facts, ['Revenues'], 'USD', doc or self.doc, '2026-06-27', 'facts-url')

    def test_quarter_growth_and_provenance(self):
        result = self.select([self.current, self.prior])
        self.assertEqual(result['year_over_year_percent'], 20)
        self.assertEqual(result['current']['accession_number'], 'current')
        self.assertEqual(result['current']['duration_days'], 91)
        self.assertEqual(result['current']['scale'], 1)

    def test_ytd_wrong_accession_and_currency_excluded(self):
        bad = [{**self.current, 'start': '2025-09-28', 'val': 300},
               {**self.current, 'accn': 'later-restatement', 'val': 999}]
        self.assertEqual(self.select(bad)['status'], 'unavailable')
        self.assertEqual(self.select([self.current], unit='EUR')['status'], 'unavailable')

    def test_conflicts_are_not_arbitrarily_selected(self):
        result = self.select([self.current, {**self.current, 'val': 121}])
        self.assertEqual(result['status'], 'ambiguous')
        self.assertEqual(self.select([self.current, self.current])['status'], 'available')

    def test_nonpositive_base_and_missing_prior(self):
        for value in [0, -10]:
            result = self.select([self.current, {**self.prior, 'val': value}])
            self.assertIsNone(result['year_over_year_percent'])
            self.assertIn('withheld', result['comparison_note'])
        self.assertIsNone(self.select([self.current])['prior_year'])

    def test_annual_excludes_quarters(self):
        doc = {**self.doc, 'form': '10-K'}
        annual = {**self.current, 'form': '10-K', 'start': '2025-06-29'}
        result = self.select([annual, {**self.current, 'form': '10-K'}], doc=doc)
        self.assertEqual(result['current']['start'], '2025-06-29')

    def test_inline_period_and_mismatch(self):
        raw = b'<ix:nonNumeric name="dei:DocumentPeriodEndDate">2026-06-27</ix:nonNumeric>'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'report.htm').write_bytes(raw)
            doc = {**self.doc, 'raw_path': 'report.htm', 'sha256': hashlib.sha256(raw).hexdigest()}
            self.assertEqual(validate_period(root, doc)['status'], 'validated')
            doc['reporting_period_end'] = '2026-07-30'
            self.assertEqual(validate_period(root, doc)['status'], 'conflict')

    def test_release_date_is_not_event_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'release.txt').write_text('Results for its third quarter ended June 27, 2026.')
            doc = {**self.doc, 'form': '8-K', 'kind': 'earnings_release_candidate',
                   'text_path': 'release.txt', 'sec_report_date': '2026-07-30'}
            result = validate_period(root, doc)
            self.assertEqual(result['period_end'], '2026-06-27')
            self.assertEqual(result['event_date'], '2026-07-30')
            self.assertEqual(result['status'], 'text_identified')

    def test_wrong_company_rejected(self):
        with self.assertRaises(FinancialError):
            build_summary(Path('.'), {'cik': '123'}, {'cik': 456})

    def test_nested_transformed_inline_date(self):
        raw = b'<ix:nonNumeric name="dei:DocumentPeriodEndDate"><ix:nonNumeric name="dei:CurrentFiscalYearEndDate">June&#160;27</ix:nonNumeric>, 2026</ix:nonNumeric>'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'report.htm').write_bytes(raw)
            doc = {**self.doc, 'raw_path': 'report.htm', 'sha256': hashlib.sha256(raw).hexdigest()}
            self.assertEqual(validate_period(root, doc)['period_end'], '2026-06-27')

    def test_ambiguous_prior_withholds_growth(self):
        result = self.select([self.current, self.prior, {**self.prior, 'val': 90}])
        self.assertIsNone(result['prior_year'])
        self.assertIsNone(result['year_over_year_percent'])

    def test_52_and_53_week_years_are_comparable(self):
        doc = {**self.doc, 'form': '10-K'}
        current = {**self.current, 'form': '10-K', 'start': '2025-06-22'}
        prior = {**self.prior, 'form': '10-K', 'start': '2024-06-30'}
        self.assertEqual(self.select([current, prior], doc=doc)['year_over_year_percent'], 20)
