import json
import unittest
from unittest.mock import Mock, patch
from earnings_collector.answers import validate_answer, answer_question, render
from earnings_collector.generation import GenerationError


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.evidence = {'M_revenue': {'text': 'Revenue was 109417000000 USD. Prior year revenue was 94036000000 USD.',
                                     'source': {'source_url': 'https://www.sec.gov/filing'}}}
        self.report = {'period_end': '2026-06-27'}

    def test_cited_numbers_pass(self):
        raw = json.dumps({'answerable': True, 'claims': [{'text': 'Revenue was 109417000000 USD.', 'source_ids': ['M_revenue']}]})
        self.assertEqual(len(validate_answer(raw, self.evidence)), 1)

    def test_unknown_citation_missing_citation_and_invented_number(self):
        for ids, text in [(['P99'], 'Revenue grew.'), ([], 'Revenue grew.'), (['M_revenue'], 'Revenue was 999 USD.')]:
            with self.assertRaises(GenerationError):
                validate_answer(json.dumps({'answerable': True, 'claims': [{'text': text, 'source_ids': ids}]}), self.evidence)

    def test_malformed_output_and_false_answer(self):
        with self.assertRaises(GenerationError):
            validate_answer('not JSON', self.evidence)
        self.assertEqual(validate_answer('{"answerable": false, "claims": []}', self.evidence), [])
        with self.assertRaises(GenerationError):
            validate_answer('{"answerable": "yes", "claims": []}', self.evidence)

    def test_pipeline_and_render_sources(self):
        generator = Mock()
        generator.generate.return_value = '{"selections": ["M_revenue:0"]}'
        with patch('earnings_collector.answers.prepare_evidence', return_value=(self.report, self.evidence, [])):
            result = answer_question('aapl', 'Revenue?', generator=generator)
        self.assertEqual(result['status'], 'answered')
        self.assertIn('https://www.sec.gov/filing', render(result))
        sent = json.loads(generator.generate.call_args.args[0][1]['content'])
        self.assertEqual(sent['period_end'], '2026-06-27')

    def test_rejected_output_is_not_exposed_as_answer(self):
        generator = Mock()
        generator.generate.return_value = '{"answerable": true, "claims": [{"text": "Revenue was 999 USD.", "source_ids": ["M_revenue"]}]}'
        with patch('earnings_collector.answers.prepare_evidence', return_value=(self.report, self.evidence, [])):
            result = answer_question('AAPL', 'Revenue?', generator=generator)
        self.assertEqual(result['status'], 'withheld')
        self.assertEqual(result['claims'], [])
        self.assertNotIn('999', render(result))

    def test_no_evidence_skips_generation(self):
        generator = Mock()
        with patch('earnings_collector.answers.prepare_evidence', return_value=(self.report, {}, [])):
            result = answer_question('AAPL', 'Question', generator=generator)
        self.assertEqual(result['status'], 'insufficient_evidence')
        generator.generate.assert_not_called()


class ExtractiveValidationTests(unittest.TestCase):
    def test_plausible_but_unsupported_paraphrase_rejected(self):
        evidence = {'P1': {'text': 'Company growth came from iPhone, Services and Mac.'}}
        raw = json.dumps({'answerable': True, 'claims': [{'text': 'iPhone growth came from Services and Mac.', 'source_ids': ['P1']}]})
        with self.assertRaises(GenerationError):
            validate_answer(raw, evidence)

    def test_whitespace_changes_allowed(self):
        evidence = {'P1': {'text': 'Sales increased\nprimarily due to Pro models.'}}
        raw = json.dumps({'answerable': True, 'claims': [{'text': 'Sales increased primarily due to Pro models.', 'source_ids': ['P1']}]})
        self.assertEqual(len(validate_answer(raw, evidence)), 1)

    def test_invalid_output_retried_once(self):
        generator = Mock()
        generator.generate.side_effect = ['invalid', '{"selections": []}']
        with patch('earnings_collector.answers.prepare_evidence', return_value=({'period_end': '2026-06-27'}, {'P1': {'text': 'Sales.', 'source': {}}}, [])):
            result = answer_question('AAPL', 'Question', generator=generator)
        self.assertEqual(result['status'], 'insufficient_evidence')
        self.assertEqual(generator.generate.call_count, 2)


class SelectionTests(unittest.TestCase):
    def test_selections_cannot_invent_text_or_citations(self):
        from earnings_collector.answers import evidence_choices, decode_selections
        evidence = {'P1': {'text': 'Pro models drove iPhone sales growth.'}}
        choices = evidence_choices(evidence)
        content = json.loads(decode_selections('{"selections":["P1:0"]}', choices))
        self.assertEqual(content['claims'][0]['text'], evidence['P1']['text'])
        with self.assertRaises(GenerationError):
            decode_selections('{"selections":["P999:0"]}', choices)
        self.assertFalse(json.loads(decode_selections('{"selections":[]}', choices))['answerable'])

    def test_runtime_load_is_offline(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from earnings_collector.generation import LocalGenerator, MODEL_ID
        with tempfile.TemporaryDirectory() as tmp:
            generator = LocalGenerator(Path(tmp))
            generator.cache.mkdir()
            generator.lock.write_text(json.dumps({'model_id': MODEL_ID, 'revision': 'fixed'}))
            snapshot = Mock(return_value=tmp)
            load = Mock(return_value=(object(), object()))
            with patch.dict('sys.modules', {'huggingface_hub': SimpleNamespace(snapshot_download=snapshot), 'mlx_lm': SimpleNamespace(load=load)}), patch('platform.system', return_value='Darwin'), patch('platform.machine', return_value='arm64'):
                generator.load()
            self.assertTrue(snapshot.call_args.kwargs['local_files_only'])
            self.assertEqual(snapshot.call_args.kwargs['revision'], 'fixed')
            self.assertFalse(load.call_args.kwargs['tokenizer_config']['trust_remote_code'])


class RelevanceTests(unittest.TestCase):
    def test_irrelevant_valid_metric_selection_rejected(self):
        from earnings_collector.answers import decode_selections
        choices = {'M_revenue:0': {'text': 'Revenue was 100 USD.', 'source_ids': ['M_revenue']}}
        with self.assertRaises(GenerationError):
            decode_selections('{"selections":["M_revenue:0"]}', choices, "What is the CEO's shoe size?")
        self.assertTrue(json.loads(decode_selections('{"selections":["M_revenue:0"]}', choices, 'What was revenue?'))['answerable'])

    def test_table_rows_excluded_and_us_abbreviation_preserved(self):
        from earnings_collector.answers import evidence_choices
        evidence = {'P1': {'text': 'Revenue\t100\t90\nThe U.S. dollar affected revenue growth.\niPhone sales increased due to Pro models.'}}
        texts = [v['text'] for v in evidence_choices(evidence).values()]
        self.assertEqual(texts, ['The U.S. dollar affected revenue growth.', 'iPhone sales increased due to Pro models.'])


class DirectMetricTests(unittest.TestCase):
    def test_current_values_only(self):
        from earnings_collector.answers import direct_metric_choices
        choices = {'M_revenue:0': {'text': 'Revenue was 100 USD.', 'source_ids': ['M_revenue']},
                   'M_diluted_eps:0': {'text': 'Diluted EPS was 2 USD/shares.', 'source_ids': ['M_diluted_eps']}}
        self.assertEqual(len(direct_metric_choices('What was revenue and diluted EPS?', choices)), 2)
        self.assertIsNone(direct_metric_choices('What was revenue in 2025?', choices))
        self.assertIsNone(direct_metric_choices('Why did revenue grow?', choices))
        self.assertEqual(direct_metric_choices('What was net income?', choices), [])
