"""Evaluate retrieval against frozen, source-backed passage judgments."""
import argparse
import json
import sys
from pathlib import Path

from .embeddings import IndexError, LocalEmbedder
from .indexing import load_chunks, search_company


def metrics(ids, relevant, k=5):
    ranks = [i for i, id in enumerate(ids[:k], 1) if id in relevant]
    return {'hit_at_1': float(bool(ids) and ids[0] in relevant),
            'hit_at_5': float(bool(ranks)),
            'recall_at_5': len(set(ids[:k]) & set(relevant)) / len(set(relevant)),
            'mrr_at_5': 1 / min(ranks) if ranks else 0.0}


def evaluate(dataset, data_dir=Path('data')):
    suite = json.loads(Path(dataset).read_text())
    _, payload, fingerprint = load_chunks(suite['ticker'], data_dir)
    if fingerprint != suite['chunks_sha256']:
        raise IndexError('Evaluation corpus changed. Restore the frozen corpus or review and update the judgments.')
    known = {c['id'] for c in payload['chunks']}
    for case in suite['cases']:
        if not case['relevant_ids'] or not set(case['relevant_ids']) <= known:
            raise IndexError('Evaluation contains missing evidence IDs.')
    embedder = LocalEmbedder(Path(data_dir) / '.models')
    rows = []
    for case in suite['cases']:
        # Both modes search the complete frozen corpus with identical filters.
        dense = search_company(suite['ticker'], case['query'], data_dir, top_k=5, embedder=embedder, mode='dense')
        hybrid = search_company(suite['ticker'], case['query'], data_dir, top_k=5, embedder=embedder, mode='hybrid')
        row = {'id': case['id'], 'query': case['query'], 'split': case['split']}
        for mode, result in [('dense', dense), ('hybrid', hybrid)]:
            ids = [c['id'] for c in result['results']]
            row[mode] = {**metrics(ids, case['relevant_ids']), 'retrieved_ids': ids}
        rows.append(row)
    summary = {}
    for split in ('development', 'held_out', 'all'):
        subset = [r for r in rows if split == 'all' or r['split'] == split]
        summary[split] = {'cases': len(subset)}
        for mode in ('dense', 'hybrid'):
            summary[split][mode] = {metric: sum(r[mode][metric] for r in subset) / len(subset)
                                   for metric in ('hit_at_1', 'hit_at_5', 'recall_at_5', 'mrr_at_5')} if subset else {}
    return {'ticker': suite['ticker'], 'chunks_sha256': fingerprint, 'model': embedder.model,
            'summary': summary, 'cases': rows,
            'limitations': 'One-company, ten-question smoke benchmark with known relevant passages, not exhaustive labels. Does not test answer generation or unanswerable-query abstention.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('evaluations/aapl.json'))
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('data/retrieval-evaluation.json'))
    args = parser.parse_args()
    try:
        result = evaluate(args.dataset, args.data_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    except (OSError, ValueError, IndexError) as exc:
        print(f'Evaluation failed: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(result['summary'], indent=2))
    print(f'Details: {args.output.resolve()}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
