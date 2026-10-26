import unittest
from earnings_collector.narrative_evaluation import score_case


class NarrativeEvaluationTests(unittest.TestCase):
    def test_distinguishes_retrieval_candidate_and_answer_failures(self):
        case = {'expected_excerpt': 'higher net sales of Pro models'}
        result = {'status': 'insufficient_evidence', 'claims': [],
                  'sources': {'P1': {'text': 'iPhone growth reflected higher net sales of Pro models.'}},
                  'candidate_choice_ids': []}
        score = score_case(case, result)
        self.assertTrue(score['retrieved_expected'])
        self.assertFalse(score['candidate_expected'])
        self.assertFalse(score['passed'])
        result['candidate_choice_ids'] = ['P1:0']
        self.assertTrue(score_case(case, result)['candidate_expected'])
        result.update(status='answered', claims=[{'text': 'Mac sales increased.'}])
        self.assertFalse(score_case(case, result)['passed'])
        result['claims'] = [{'text': result['sources']['P1']['text']}]
        self.assertTrue(score_case(case, result)['passed'])

    def test_unanswerable_requires_clean_abstention(self):
        case = {'expected_excerpt': None}
        self.assertTrue(score_case(case, {'status': 'insufficient_evidence'})['passed'])
        self.assertFalse(score_case(case, {'status': 'answered'})['passed'])
        self.assertFalse(score_case(case, {'status': 'withheld'})['passed'])
