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
        with patch('earnings_collector.answers.prepare_evidence', return_value=({'period_end': '2026-06-27'}, {'P1': {'text': 'Revenue increased due to demand.', 'source': {}}}, [])):
            result = answer_question('AAPL', 'Why did revenue grow?', generator=generator)
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


class NarrativeRegressionTests(unittest.TestCase):
    def setUp(self):
        from earnings_collector.answers import evidence_choices
        self.evidence = {'P1': {'text': 'iPhone net sales increased primarily due to higher net sales of Pro models.'},
                         'P2': {'text': 'Mac net sales increased due to higher net sales of laptops.'},
                         'M_revenue': {'text': 'Revenue was 100 USD.'}}
        self.choices = evidence_choices(self.evidence)

    def test_sales_increased_matches_revenue_growth(self):
        from earnings_collector.answers import decode_selections
        raw = decode_selections('{"selections":["P1:0"]}', self.choices, 'What drove iPhone revenue growth?')
        self.assertEqual(validate_answer(raw, self.evidence)[0]['text'], self.evidence['P1']['text'])

    def test_wrong_product_and_generic_metric_cannot_answer_why(self):
        from earnings_collector.answers import decode_selections
        for id in ['P2:0', 'M_revenue:0']:
            with self.assertRaises(GenerationError):
                decode_selections(json.dumps({'selections': [id]}), self.choices, 'What drove iPhone revenue growth?')

    def test_filter_keeps_original_text_and_removes_distractors(self):
        from earnings_collector.answers import select_candidate_choices
        choices = select_candidate_choices('What drove iPhone revenue growth?', self.choices)
        self.assertEqual(list(choices), ['P1:0'])
        self.assertEqual(choices['P1:0'], self.choices['P1:0'])
        self.assertEqual(select_candidate_choices("What is the CEO's shoe size?", self.choices), {})

    def test_trace_records_rejection_and_model_cannot_select_hidden_choice(self):
        generator = Mock()
        generator.generate.side_effect = ['{"selections":["P2:0"]}', '{"selections":["P1:0"]}']
        with patch('earnings_collector.answers.prepare_evidence', return_value=({'period_end': '2026-06-27'}, self.evidence, [])):
            result = answer_question('AAPL', 'What drove iPhone revenue growth?', generator=generator)
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['candidate_choice_ids'], ['P1:0'])
        self.assertIn('validation_error', result['selection_attempts'][0])
        self.assertEqual(len(result['claims']), 1)

    def test_cause_subject_is_not_confused_with_effect_subject(self):
        from earnings_collector.answers import relevant_choice
        choice = {'text': 'Intelligent Cloud revenue increased driven by Azure.'}
        self.assertFalse(relevant_choice('What drove Azure revenue growth?', 'P1:0', choice))
        self.assertTrue(relevant_choice('What drove Intelligent Cloud revenue growth?', 'P1:0', choice))
        other = {'text': 'Revenue increased driven by growth in Microsoft Cloud.'}
        self.assertFalse(relevant_choice('What drove Intelligent Cloud revenue growth?', 'P1:1', other))

    def test_generic_business_word_does_not_admit_unrelated_risk(self):
        from earnings_collector.answers import relevant_choice
        choice = {'text': 'Legal proceedings could negatively impact the business and its reputation.'}
        self.assertFalse(relevant_choice('How could tariffs affect the business?', 'P1:0', choice))

    def test_exact_unique_offered_text_resolves_to_existing_citation(self):
        from earnings_collector.answers import decode_selections
        text = self.choices['P1:0']['text']
        content = json.loads(decode_selections(json.dumps({'selections': [text]}), self.choices,
                                             'What drove iPhone revenue growth?'))
        self.assertEqual(content['claims'], [self.choices['P1:0']])
        with self.assertRaises(GenerationError):
            decode_selections(json.dumps({'selections': [text.replace('Pro', 'standard')]}), self.choices)
        duplicate = {**self.choices, 'P9:0': self.choices['P1:0']}
        with self.assertRaises(GenerationError):
            decode_selections(json.dumps({'selections': [text]}), duplicate)
