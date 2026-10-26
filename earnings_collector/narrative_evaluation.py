"""Offline smoke evaluation of retrieval, candidate selection, and cited narrative answers."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def score_case(case, result):
    expected = case['expected_excerpt']
    if expected is None:
        return {'passed': result['status'] == 'insufficient_evidence',
                'retrieved_expected': None, 'candidate_expected': None, 'answered_expected': None}
    from .answers import evidence_choices
    contains = lambda text: expected.casefold() in text.casefold()
    choices = evidence_choices(result['sources'])
    answered = result['status'] == 'answered' and any(contains(c['text']) for c in result['claims'])
    return {'passed': answered,
            'retrieved_expected': any(contains(item['text']) for item in result['sources'].values()),
            'candidate_expected': any(contains(choices[id]['text']) for id in result.get('candidate_choice_ids', []) if id in choices),
            'answered_expected': answered}


def evaluate(dataset, data_dir, output):
    from .answers import answer_question
    from .generation import LocalGenerator
    suite = json.loads(dataset.read_text())
    for ticker, expected in suite['corpora'].items():
        chunks = data_dir / ticker / 'chunks.json'
        if hashlib.sha256(chunks.read_bytes()).hexdigest() != expected:
            raise ValueError(f'{ticker} corpus changed. Review the source judgments before updating the dataset.')
        text = '\n'.join(c['text'] for c in json.loads(chunks.read_text())['chunks'])
        for case in suite['cases']:
            if case['ticker'] == ticker and case['expected_excerpt'] and case['expected_excerpt'] not in text:
                raise ValueError(f"Expected evidence missing for {case['id']}.")
    generator = LocalGenerator(data_dir)
    results = []
    for case in suite['cases']:
        result = answer_question(case['ticker'], case['question'], data_dir, generator=generator)
        row = {'id': case['id'], **score_case(case, result), 'result': result}
        results.append(row)
        print(f"{case['id']}: {'PASS' if row['passed'] else 'FAIL'} ({result['status']})", flush=True)
    report = {'description': suite['description'], 'corpora': suite['corpora'],
              'passed': sum(row['passed'] for row in results), 'total': len(results), 'cases': results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('evaluations/narrative.json'))
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('data/narrative-evaluation.json'))
    args = parser.parse_args()
    # Set before importing model libraries. Evaluation never downloads data or models.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    report = evaluate(args.dataset, args.data_dir, args.output)
    print(f"{report['passed']}/{report['total']} cases passed. Details: {args.output}")
    return 0 if report['passed'] == report['total'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
