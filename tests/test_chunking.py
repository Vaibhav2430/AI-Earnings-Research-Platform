import json
import tempfile
import unittest
from pathlib import Path

from earnings_collector.chunking import ChunkingError, chunk_company, spans


class ChunkTests(unittest.TestCase):
    def test_lossless_bounded_unicode_and_crlf(self):
        text = 'ITEM 1. BUSINESS\r\n' + ('Revenue\t$100\t$90\r\n' * 40) + 'é' * 450
        parts = list(spans(text, 200))
        self.assertEqual(''.join(text[a:b] for a, b, _ in parts), text)
        self.assertTrue(all(0 < b - a <= 200 for a, b, _ in parts))
        self.assertTrue(any(split for _, _, split in parts))

    def test_table_rows_remain_whole(self):
        text = 'Revenue\t100\t90\n' * 40
        for a, b, split in spans(text, 200):
            self.assertTrue(text[a:b].endswith('\n'))
            self.assertFalse(split)

    def test_heading_stays_with_following_line(self):
        text = 'a' * 175 + '\nITEM 2. RESULTS\nRevenue grew.\n'
        chunks = [text[a:b] for a, b, _ in spans(text, 200)]
        self.assertEqual(chunks[1], 'ITEM 2. RESULTS\nRevenue grew.\n')

    def test_invalid_size(self):
        with self.assertRaises(ChunkingError):
            list(spans('text', 0))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'AAPL'
        self.root.mkdir()
        self.text = 'ITEM 1. RESULTS\n' + 'Revenue\t100\t90\n' * 40
        (self.root / 'release.txt').write_text(self.text)
        self.doc = {'text_path': 'release.txt', 'source_url': 'https://www.sec.gov/example.htm',
                    'accession_number': '123', 'form': '8-K', 'sec_report_date': '2026-07-30',
                    'reporting_period_end': None, 'sha256': 'original-checksum'}
        self.manifest = {'ticker': 'AAPL', 'company_name': 'Apple', 'cik': '0000320193',
                         'status': 'partial', 'warnings': ['No transcript'], 'documents': [self.doc]}
        self.write_manifest()

    def write_manifest(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))

    def run_chunker(self):
        return chunk_company('aapl', Path(self.tmp.name), 200)

    def test_provenance_offsets_and_repeat_stability(self):
        path, result = self.run_chunker()
        first_bytes = path.read_bytes()
        self.assertEqual(result['collection_status'], 'partial')
        self.assertEqual(result['collection_warnings'], ['No transcript'])
        for chunk in result['chunks']:
            location = chunk['location']
            self.assertEqual(chunk['text'], self.text[location['char_start']:location['char_end']])
            self.assertIsNone(chunk['source']['reporting_period_end'])
            self.assertEqual(chunk['source']['sha256'], 'original-checksum')
            self.assertEqual(chunk['source']['source_url'], self.doc['source_url'])
        self.run_chunker()
        self.assertEqual(first_bytes, path.read_bytes())

    def test_changed_document_replaces_stale_chunks(self):
        _, old = self.run_chunker()
        (self.root / 'release.txt').write_text('New results')
        _, new = self.run_chunker()
        self.assertEqual(new['chunk_count'], 1)
        self.assertNotEqual(old['chunks'][0]['id'], new['chunks'][0]['id'])
        self.assertEqual(old['chunks'][0]['document_id'], new['chunks'][0]['document_id'])

    def test_missing_and_duplicate_sources(self):
        self.manifest['documents'] += [self.doc.copy(), {**self.doc, 'text_path': 'missing.txt'}]
        self.write_manifest()
        _, result = self.run_chunker()
        self.assertEqual(result['documents_processed'], 1)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(len(result['warnings']), 2)

    def test_rejects_path_escape(self):
        outside = Path(self.tmp.name) / 'outside.txt'
        outside.write_text('Must not read')
        self.doc['text_path'] = '../outside.txt'
        self.write_manifest()
        _, result = self.run_chunker()
        self.assertEqual(result['chunk_count'], 0)
        self.assertEqual(result['status'], 'empty')

    def test_missing_manifest_and_ticker_mismatch(self):
        with self.assertRaises(ChunkingError):
            chunk_company('MSFT', Path(self.tmp.name))
        self.manifest['ticker'] = 'MSFT'
        self.write_manifest()
        with self.assertRaises(ChunkingError):
            self.run_chunker()

    def test_missing_metadata_empty_text_and_unsupported_documents(self):
        (self.root / 'release.txt').write_text('  \n')
        self.manifest['documents'] += [{'text_path': None}, {'text_path': 'release.txt'}]
        self.write_manifest()
        _, result = self.run_chunker()
        self.assertEqual(result['status'], 'empty')
        self.assertEqual(len(result['warnings']), 3)


if __name__ == '__main__':
    unittest.main()
