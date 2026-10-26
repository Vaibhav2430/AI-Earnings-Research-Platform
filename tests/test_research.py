import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
from earnings_collector.research import research, cached_manifest, load_local_settings, ResearchError


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        self.root = self.data / 'TEST'
        self.root.mkdir()
        self.manifest = {'ticker': 'TEST', 'documents': [{'form': '10-Q', 'raw_path': 'raw.html',
            'text_path': 'text.txt', 'sha256': hashlib.sha256(b'raw').hexdigest()}]}
        self.save(self.root)

    def save(self, root):
        root.mkdir(exist_ok=True)
        (root / 'raw.html').write_bytes(b'raw')
        (root / 'text.txt').write_text('text')
        (root / 'manifest.json').write_text(json.dumps(self.manifest))
        (root / 'companyfacts.json').write_text('{}')

    def mocks(self, stack):
        values = {'prepare_financials': {'reports': [{'period_type': 'quarter', 'period_end': '2026-06-30'}]},
                  'chunk_company': (None, {'chunks': ['one']}), 'ensure_embedding_model': None,
                  'index_company': {'cached_texts': 1, 'texts_to_embed': 0},
                  'answer_question': {'status': 'answered', 'claims': []}}
        return {name: stack.enter_context(patch('earnings_collector.research.' + name, return_value=value))
                for name, value in values.items()}

    def test_cached_workflow_skips_collection(self):
        with ExitStack() as stack:
            mocks = self.mocks(stack)
            collect = stack.enter_context(patch('earnings_collector.research.collect'))
            result = research('test', 'Revenue?', self.data, offline=True)
            collect.assert_not_called()
            mocks['prepare_financials'].assert_called_once_with('TEST', self.data, refresh=False)
            self.assertTrue(result['workflow']['documents_reused'])
            self.assertEqual(json.loads((self.root / 'last-answer.json').read_text()), result)

    def test_new_company_prepares_every_stage(self):
        (self.root / 'manifest.json').unlink()
        def collect(ticker, client, staging):
            self.save(staging / ticker)
            return None, self.manifest
        with ExitStack() as stack:
            mocks = self.mocks(stack)
            stack.enter_context(patch.dict(os.environ, {'SEC_USER_AGENT': 'test@example.com'}))
            collector = stack.enter_context(patch('earnings_collector.research.collect', side_effect=collect))
            result = research('TEST', 'Revenue?', self.data)
            collector.assert_called_once()
            mocks['prepare_financials'].assert_called_once_with('TEST', self.data, refresh=True)
            self.assertFalse(result['workflow']['documents_reused'])

    def test_failed_refresh_preserves_existing_manifest_and_answer(self):
        original = (self.root / 'manifest.json').read_bytes()
        (self.root / 'last-answer.json').write_text('previous')
        with patch.dict(os.environ, {'SEC_USER_AGENT': 'test@example.com'}), patch(
                'earnings_collector.research.collect', return_value=(None, {'documents': []})):
            with self.assertRaisesRegex(ResearchError, 'usable 10-Q'):
                research('TEST', 'Revenue?', self.data, refresh=True)
        self.assertEqual((self.root / 'manifest.json').read_bytes(), original)
        self.assertEqual((self.root / 'last-answer.json').read_text(), 'previous')

    def test_offline_missing_facts_fails_before_preparation(self):
        (self.root / 'companyfacts.json').unlink()
        with patch('earnings_collector.research.prepare_financials') as prepare:
            with self.assertRaisesRegex(ResearchError, 'facts are missing'):
                research('TEST', 'Revenue?', self.data, offline=True)
            prepare.assert_not_called()

    def test_tampered_or_escaping_document_invalidates_cache(self):
        (self.root / 'raw.html').write_bytes(b'changed')
        self.assertIsNone(cached_manifest(self.root, 'TEST'))
        self.manifest['documents'][0]['raw_path'] = '../outside'
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))
        self.assertIsNone(cached_manifest(self.root, 'TEST'))

    def test_settings_allowlist_and_shell_precedence(self):
        env = self.data / '.env'
        env.write_text('export SEC_USER_AGENT="file@example.com"\nSSL_CERT_FILE="/tmp/cert"\nUNRELATED=value\n')
        with patch.dict(os.environ, {'SEC_USER_AGENT': 'shell@example.com'}, clear=True):
            load_local_settings(env)
            self.assertEqual(os.environ['SEC_USER_AGENT'], 'shell@example.com')
            self.assertEqual(os.environ['SSL_CERT_FILE'], '/tmp/cert')
            self.assertNotIn('UNRELATED', os.environ)

    def test_invalid_inputs_never_collect(self):
        with patch('earnings_collector.research.collect') as collect:
            for ticker, question in [('../bad', 'Revenue?'), ('TEST', '')]:
                with self.assertRaises(ResearchError):
                    research(ticker, question, self.data)
            collect.assert_not_called()
