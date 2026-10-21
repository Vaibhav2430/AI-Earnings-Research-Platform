import unittest
from earnings_collector.ranking import bm25, rank_passages
from earnings_collector.evaluation import metrics


class RankingTests(unittest.TestCase):
    def test_rare_exact_terms_find_evidence(self):
        results = [{'id': str(i), 'text': 'General business discussion.', 'score': 1 - i / 100} for i in range(10)]
        results.append({'id': 'dividend', 'text': 'The dividend is payable on August 13.', 'score': 0.1,
                        'source': {'source_url': 'source'}, 'location': {'char_start': 0}})
        ranked = rank_passages('When is the dividend payable?', results)
        self.assertEqual(ranked[0]['id'], 'dividend')
        self.assertEqual(ranked[0]['source'], results[-1]['source'])
        self.assertEqual(ranked[0]['location'], results[-1]['location'])

    def test_no_keyword_match_preserves_dense_order(self):
        docs = [{'id': 'b', 'text': 'sales', 'score': 0.5}, {'id': 'a', 'text': 'profits', 'score': 0.7}]
        self.assertEqual([x['id'] for x in rank_passages('unknown', docs)], ['a', 'b'])
        self.assertTrue(all(x['lexical_rank'] is None for x in rank_passages('unknown', docs)))

    def test_case_punctuation_and_stopwords(self):
        self.assertEqual(bm25('THE DIVIDEND?', ['dividend']), bm25('dividend', ['dividend']))
        self.assertEqual(bm25('the', ['the']), [0.0])
        self.assertEqual(bm25('anything', []), [])

    def test_deterministic_ties_and_dense_baseline(self):
        docs = [{'id': 'b', 'text': 'same', 'score': .5}, {'id': 'a', 'text': 'same', 'score': .5}]
        self.assertEqual(rank_passages('same', docs), rank_passages('same', list(reversed(docs))))
        self.assertEqual(rank_passages('same', docs, 'dense')[0]['score'], .5)
        with self.assertRaises(ValueError):
            rank_passages('same', docs, 'invalid')

    def test_metrics_distinguish_hit_recall_and_rank(self):
        result = metrics(['wrong', 'right', 'other'], ['right', 'missing'])
        self.assertEqual(result, {'hit_at_1': 0., 'hit_at_5': 1., 'recall_at_5': .5, 'mrr_at_5': .5})
        self.assertEqual(metrics([], ['right'])['hit_at_5'], 0.)


class EvaluationValidationTests(unittest.TestCase):
    def test_corpus_change_rejected_before_model_load(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from earnings_collector.evaluation import evaluate
        from earnings_collector.embeddings import IndexError
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'suite.json'
            path.write_text(json.dumps({'ticker': 'AAPL', 'chunks_sha256': 'old'}))
            with patch('earnings_collector.evaluation.load_chunks', return_value=(None, {}, 'new')):
                with self.assertRaisesRegex(IndexError, 'corpus changed'):
                    evaluate(path)

    def test_missing_evidence_rejected(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from earnings_collector.evaluation import evaluate
        from earnings_collector.embeddings import IndexError
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'suite.json'
            path.write_text(json.dumps({'ticker': 'AAPL', 'chunks_sha256': 'same',
                                        'cases': [{'relevant_ids': ['absent']}]}))
            with patch('earnings_collector.evaluation.load_chunks', return_value=(None, {'chunks': []}, 'same')):
                with self.assertRaisesRegex(IndexError, 'missing evidence'):
                    evaluate(path)
