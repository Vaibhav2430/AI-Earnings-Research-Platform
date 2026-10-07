import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from earnings_collector.embeddings import IndexError, OpenAIEmbedder, unit_vector
from earnings_collector.indexing import index_company, search_company


class FakeEmbedder:
    provider = 'test'
    model = 'test-vectors'
    dimensions = 2

    def __init__(self):
        self.calls = []
        self.fail = False

    def embed(self, texts):
        self.calls.append(texts)
        if self.fail:
            raise IndexError('Simulated failure')
        return [[1., 0.] if 'revenue' in text.lower() else [0., 1.] for text in texts]


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        self.root = self.data / 'AAPL'
        self.root.mkdir()
        self.payload = {'schema_version': 1, 'ticker': 'AAPL', 'status': 'complete',
                        'collection_status': 'partial', 'collection_warnings': ['No transcripts'],
                        'chunks': [self.chunk('a', 'Revenue increased.', '10-Q'),
                                   self.chunk('b', 'Risks include competition.', '10-K')]}
        self.write()
        self.embedder = FakeEmbedder()

    def chunk(self, id, text, form):
        return {'id': id, 'text': text, 'location': {'char_start': 0, 'char_end': len(text)},
                'source': {'ticker': 'AAPL', 'source_url': 'https://www.sec.gov/example',
                           'form': form, 'reporting_period_end': None}}

    def write(self):
        (self.root / 'chunks.json').write_text(json.dumps(self.payload))

    def index(self, **kwargs):
        return index_company('AAPL', self.data, self.embedder, **kwargs)

    def search(self, **kwargs):
        return search_company('AAPL', 'revenue growth', self.data, embedder=self.embedder, **kwargs)

    def test_persistent_search_preserves_citations(self):
        self.index()
        result = self.search(top_k=1)
        self.assertEqual(result['results'][0]['id'], 'a')
        self.assertEqual(result['results'][0]['score'], 1)
        self.assertEqual(result['results'][0]['source'], self.payload['chunks'][0]['source'])
        self.assertEqual(result['collection_warnings'], ['No transcripts'])

    def test_rerun_reuses_vectors_and_removes_stale_chunks(self):
        self.index()
        self.embedder.calls.clear()
        self.assertEqual(self.index()['texts_to_embed'], 0)
        self.assertEqual(self.embedder.calls, [])
        self.payload['chunks'] = [self.chunk('new', 'Revenue increased.', '8-K')]
        self.write()
        self.index()
        self.assertEqual(self.embedder.calls, [])
        self.assertEqual([c['id'] for c in self.search()['results']], ['new'])
        self.assertEqual(self.search()['results'][0]['source']['form'], '8-K')

    def test_filters_and_unknown_period(self):
        self.index()
        self.assertEqual(self.search(form='10-K')['results'][0]['id'], 'b')
        self.embedder.calls.clear()
        self.assertEqual(self.search(period_end='2026-06-27')['results'], [])
        self.assertEqual(self.embedder.calls, [])

    def test_changed_chunks_require_reindexing(self):
        self.index()
        self.payload['chunks'][0]['text'] = 'Changed text'
        self.write()
        with self.assertRaisesRegex(IndexError, 'stale'):
            self.search()

    def test_failed_refresh_keeps_previous_snapshot(self):
        self.index()
        self.payload['chunks'][0]['text'] = 'Changed revenue'
        self.write()
        self.embedder.fail = True
        with self.assertRaises(IndexError):
            self.index()
        with sqlite3.connect(self.root / 'index.sqlite3') as db:
            stored = json.loads(db.execute('SELECT payload FROM chunks WHERE id="a"').fetchone()[0])
        self.assertEqual(stored['text'], 'Revenue increased.')

    def test_model_mismatch_and_new_model_cache(self):
        self.index()
        self.embedder.model = 'different'
        with self.assertRaisesRegex(IndexError, 'configuration'):
            self.search()
        self.assertEqual(self.index()['texts_to_embed'], 2)

    def test_dry_run_has_no_writes_or_calls(self):
        result = self.index(dry_run=True)
        self.assertEqual(result['texts_to_embed'], 2)
        self.assertFalse((self.root / 'index.sqlite3').exists())
        self.assertEqual(self.embedder.calls, [])

    def test_invalid_input_is_rejected_before_requests(self):
        for change in ('duplicate', 'oversize', 'company', 'empty'):
            with self.subTest(change=change):
                original = json.loads(json.dumps(self.payload))
                if change == 'duplicate':
                    self.payload['chunks'].append(self.payload['chunks'][0])
                elif change == 'oversize':
                    self.payload['chunks'][0]['text'] = 'a' * 8001
                elif change == 'company':
                    self.payload['chunks'][0]['source']['ticker'] = 'MSFT'
                else:
                    self.payload['chunks'] = []
                self.write()
                with self.assertRaises(IndexError):
                    self.index()
                self.payload = original
        self.assertEqual(self.embedder.calls, [])

    def test_partial_batch_is_cached_for_resume(self):
        self.payload['chunks'] = [self.chunk(str(i), f'Revenue {i}', '10-Q') for i in range(20)]
        self.write()
        original_embed = self.embedder.embed
        count = 0
        def fail_second(texts):
            nonlocal count
            count += 1
            if count == 2:
                raise IndexError('Failure on second batch')
            return original_embed(texts)
        with patch.object(self.embedder, 'embed', side_effect=fail_second):
            with self.assertRaises(IndexError):
                self.index()
        self.assertEqual(self.index()['texts_to_embed'], 4)
        self.assertEqual(len(self.search(top_k=50)['results']), 20)


class TransportTests(unittest.TestCase):
    def client(self):
        client = OpenAIEmbedder('test-secret')
        client.dimensions = 2
        return client

    def response(self, data):
        return io.BytesIO(json.dumps({'model': 'text-embedding-3-small', 'data': data}).encode())

    def test_response_order_and_payload(self):
        response = self.response([{'index': 1, 'embedding': [0, 2]}, {'index': 0, 'embedding': [2, 0]}])
        with patch('earnings_collector.embeddings.urlopen', return_value=response) as opener:
            self.assertEqual(self.client().embed(['one', 'two']), [[1, 0], [0, 1]])
            request = opener.call_args.args[0]
            self.assertEqual(json.loads(request.data)['encoding_format'], 'float')

    def test_bad_vectors_and_duplicate_indices(self):
        for vector in ([0, 0], [float('nan'), 1], [1], ['1', 2]):
            with self.assertRaises(IndexError):
                unit_vector(vector, 2)
        with patch('earnings_collector.embeddings.urlopen', return_value=self.response([
            {'index': 0, 'embedding': [1, 0]}, {'index': 0, 'embedding': [0, 1]}])):
            with self.assertRaises(IndexError):
                self.client().embed(['one', 'two'])

    def test_missing_key_never_calls_api(self):
        with patch('earnings_collector.embeddings.urlopen') as opener:
            with self.assertRaisesRegex(IndexError, 'OPENAI_API_KEY'):
                OpenAIEmbedder('').embed(['one'])
            opener.assert_not_called()

    def test_retries_transient_errors_but_not_auth(self):
        for status, calls in [(429, 3), (503, 3), (401, 1)]:
            with patch('earnings_collector.embeddings.urlopen', side_effect=HTTPError('url', status, 'failure', {}, None)) as opener, patch('earnings_collector.embeddings.time.sleep'):
                with self.assertRaises(IndexError) as error:
                    self.client().embed(['one'])
                self.assertNotIn('test-secret', str(error.exception))
                self.assertEqual(opener.call_count, calls)


if __name__ == '__main__':
    unittest.main()
