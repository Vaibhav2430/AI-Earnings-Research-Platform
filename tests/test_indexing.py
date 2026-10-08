import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import MagicMock
from types import SimpleNamespace

from earnings_collector.embeddings import IndexError, LocalEmbedder, unit_vector, token_windows
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


class LocalEmbeddingTests(unittest.TestCase):
    def test_inference_loads_cached_model_only(self):
        encoder = MagicMock()
        encoder.get_embedding_dimension.return_value = 384
        constructor = MagicMock(return_value=encoder)
        with patch.dict('sys.modules', {'sentence_transformers': SimpleNamespace(SentenceTransformer=constructor)}):
            client = LocalEmbedder()
            self.assertIs(client.load(), encoder)
            self.assertIs(client.load(), encoder)
            self.assertEqual(constructor.call_count, 1)
            self.assertTrue(constructor.call_args.kwargs['local_files_only'])
            self.assertFalse(constructor.call_args.kwargs['trust_remote_code'])

    def test_download_is_explicit(self):
        encoder = MagicMock()
        encoder.get_embedding_dimension.return_value = 384
        constructor = MagicMock(return_value=encoder)
        with patch.dict('sys.modules', {'sentence_transformers': SimpleNamespace(SentenceTransformer=constructor)}):
            LocalEmbedder().load(download=True)
            self.assertFalse(constructor.call_args.kwargs['local_files_only'])

    def test_windows_cover_all_tokens(self):
        ids = list(range(700))
        windows = token_windows(ids, 254)
        self.assertEqual([x for w in windows for x in w], ids)
        self.assertEqual([len(w) for w in windows], [254, 254, 192])

    def test_invalid_vectors(self):
        for vector in ([0, 0], [float('nan'), 1], [1], ['1', 2]):
            with self.assertRaises(IndexError):
                unit_vector(vector, 2)

    def test_default_provider_is_local(self):
        self.assertEqual(LocalEmbedder().provider, 'local-sentence-transformers')
        self.assertEqual(LocalEmbedder().dimensions, 384)

    def test_input_validation_precedes_model_load(self):
        with patch.object(LocalEmbedder, 'load') as load:
            for texts in ([], [''], ['x' * 8001]):
                with self.assertRaises(IndexError):
                    LocalEmbedder().embed(texts)
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
