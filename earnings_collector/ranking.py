"""Local BM25 and reciprocal-rank fusion, without extra dependencies."""
import math
import re
from collections import Counter

STOP_WORDS = set('a an the of in on at for to and or is are was were be been did does do what why how when which its it from by with as'.split())


def terms(text):
    return [t for t in re.findall(r'[a-z0-9]+', text.lower()) if t not in STOP_WORDS]


def bm25(query, texts):
    documents = [Counter(terms(text)) for text in texts]
    lengths = [sum(doc.values()) for doc in documents]
    average = sum(lengths) / len(lengths) if lengths else 0
    scores = [0.0] * len(documents)
    if not average:
        return scores
    for term in set(terms(query)):
        frequency = sum(term in doc for doc in documents)
        idf = math.log(1 + (len(documents) - frequency + 0.5) / (frequency + 0.5))
        for i, doc in enumerate(documents):
            tf = doc[term]
            if tf:
                scores[i] += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * lengths[i] / average))
    return scores


def rank_passages(query, dense_results, mode='hybrid'):
    if mode not in ('dense', 'hybrid'):
        raise ValueError('Ranking mode must be dense or hybrid.')
    dense = sorted(dense_results, key=lambda c: (-c['score'], c['id']))
    if mode == 'dense':
        return [{**c, 'semantic_score': c['score']} for c in dense]
    lexical = bm25(query, [c['text'] for c in dense])
    lexical_order = sorted((i for i in range(len(dense)) if lexical[i] > 0),
                           key=lambda i: (-lexical[i], dense[i]['id']))
    lexical_ranks = {index: rank for rank, index in enumerate(lexical_order, 1)}
    results = []
    for i, chunk in enumerate(dense):
        lexical_rank = lexical_ranks.get(i)
        score = 1 / (20 + i + 1)
        if lexical_rank is not None:
            score += 1 / (20 + lexical_rank)
        results.append({**chunk, 'semantic_score': chunk['score'], 'lexical_score': lexical[i],
                        'score': score, 'semantic_rank': i + 1, 'lexical_rank': lexical_rank})
    return sorted(results, key=lambda c: (-c['score'], c['id']))
