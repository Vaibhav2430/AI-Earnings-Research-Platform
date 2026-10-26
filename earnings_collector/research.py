"""One-command collection, preparation, and local cited answers."""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
from pathlib import Path

from .collector import collect
from .chunking import chunk_company
from .financials import prepare_financials, safe_read, FinancialError
from .indexing import index_company
from .answers import answer_question, render
from .sec import SecClient
from .embeddings import LocalEmbedder, MODEL_ID, MODEL_REVISION
from .generation import LocalGenerator


class ResearchError(Exception):
    pass


def load_local_settings(path=Path('.env')):
    """Read only SEC transport settings. Never execute shell contents."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        match = re.match(r'^\s*(?:export\s+)?(SEC_USER_AGENT|SSL_CERT_FILE)\s*=\s*(.*)$', line)
        if match and match[1] not in os.environ:
            parts = shlex.split(match[2], comments=True)
            if len(parts) == 1:
                os.environ[match[1]] = parts[0]


def cached_manifest(root, ticker):
    try:
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest['ticker'] != ticker or not manifest['documents']:
            return None
        for doc in manifest['documents']:
            raw = safe_read(root, doc['raw_path'])
            if hashlib.sha256(raw).hexdigest() != doc['sha256']:
                return None
            if doc.get('text_path'):
                safe_read(root, doc['text_path']).decode('utf-8')
        return manifest
    except (OSError, ValueError, KeyError, TypeError, FinancialError):
        return None


def ensure_embedding_model(data_dir, offline):
    # Resolve the pinned snapshot without loading the encoder or using the network.
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(data_dir / '.models'), local_files_only=True)
    except OSError:
        if offline:
            raise ResearchError('Embedding model is not cached. Run indexing download-model while online.') from None
        LocalEmbedder(data_dir / '.models').load(download=True)


class AutoGenerator(LocalGenerator):
    def __init__(self, data_dir, offline, progress):
        super().__init__(data_dir)
        self.offline = offline
        self.progress = progress

    def generate(self, messages):
        if not self.lock.exists():
            if self.offline:
                raise ResearchError('Answer model is not cached. Run answers download-model while online.')
            self.progress('Downloading the local answer model for first use.')
            self.download()
        return super().generate(messages)


def research(ticker, question, data_dir=Path('data'), refresh=False, offline=False,
             period_end=None, period_type='quarter', top_k=3, progress=lambda message: None):
    ticker = ticker.strip().upper()
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9.-]{0,14}', ticker):
        raise ResearchError('Enter a valid ticker, such as AAPL or MSFT.')
    if not question.strip() or len(question.encode()) > 2000:
        raise ResearchError('Question must contain 1 to 2,000 UTF-8 bytes.')
    if refresh and offline:
        raise ResearchError('--refresh cannot be combined with --offline.')
    if period_type not in ('quarter', 'annual') or not 1 <= top_k <= 5:
        raise ResearchError('Invalid period type or top-k (must be 1 to 5).')
    if period_end:
        from datetime import date
        date.fromisoformat(period_end)
    data_dir = Path(data_dir)
    root = data_dir / ticker
    step = 'collection'
    try:
        manifest = cached_manifest(root, ticker)
        needs_collection = refresh or manifest is None
        required_form = '10-Q' if period_type == 'quarter' else '10-K'
        if manifest and not any(d['form'] == required_form and d.get('text_path') for d in manifest['documents']):
            needs_collection = True
        if needs_collection:
            if offline:
                raise ResearchError('Required company documents are missing or damaged; collect them while online.')
            agent = os.environ.get('SEC_USER_AGENT', '')
            if '@' not in agent or '\n' in agent or '\r' in agent:
                raise ResearchError('Set SEC_USER_AGENT in your local .env or shell with a contact email for SEC downloads.')
            progress(f'Collecting {ticker} filings from the SEC.')
            client = SecClient(agent, data_dir / '.cache', refresh=refresh)
            data_dir.mkdir(parents=True, exist_ok=True)
            # Do not replace usable documents when a network failure produces no requested filing.
            with tempfile.TemporaryDirectory(prefix='.research-', dir=data_dir) as staging:
                _, new_manifest = collect(ticker, client, Path(staging))
                if not any(d['form'] == required_form and d.get('text_path') for d in new_manifest['documents']):
                    raise ResearchError(f'Collection did not obtain a usable {required_form}. ' + ' '.join(new_manifest.get('warnings', [])))
                shutil.copytree(Path(staging) / ticker, root, dirs_exist_ok=True)
                manifest = new_manifest
        else:
            progress(f"Using saved {ticker} documents collected {manifest.get('collected_at', 'at an unknown time')}. Use --refresh to check for new filings.")
        step = 'financial preparation'
        facts_refresh = needs_collection or not (root / 'companyfacts.json').exists()
        if facts_refresh and offline:
            raise ResearchError('SEC financial facts are missing; run online once.')
        progress('Validating reporting periods and preparing financial metrics.')
        summary = prepare_financials(ticker, data_dir, refresh=facts_refresh)
        if not any(r['period_type'] == period_type and (period_end is None or r['period_end'] == period_end) for r in summary['reports']):
            raise ResearchError('No validated filing matches the requested period. No answer was generated.')
        step = 'chunking'
        progress('Preparing source-linked passages.')
        _, chunked = chunk_company(ticker, data_dir)
        if not chunked['chunks']:
            raise ResearchError('No usable text passages were produced.')
        step = 'indexing'
        progress('Preparing the local search index; unchanged embeddings are reused.')
        ensure_embedding_model(data_dir, offline)
        indexed = index_company(ticker, data_dir)
        step = 'answering'
        progress('Finding evidence and preparing a cited answer locally.')
        result = answer_question(ticker, question, data_dir, period_end, period_type, top_k,
                                 generator=AutoGenerator(data_dir, offline, progress))
        result['workflow'] = {'documents_reused': not needs_collection, 'collected_at': manifest.get('collected_at'),
                              'collection_status': manifest.get('status'), 'chunks': len(chunked['chunks']),
                              'embeddings_reused': indexed['cached_texts'], 'embeddings_created': indexed['texts_to_embed'],
                              'offline': offline}
        # Save successful or withheld outcomes, but never replace the last answer on a preparation failure.
        with tempfile.NamedTemporaryFile(mode='w', dir=root, encoding='utf-8', delete=False) as handle:
            json.dump(result, handle, indent=2)
            handle.write('\n')
            temp = Path(handle.name)
        temp.replace(root / 'last-answer.json')
        return result
    except Exception as exc:
        raise ResearchError(f'{step}: {exc}') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ticker')
    parser.add_argument('question')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--refresh', action='store_true')
    group.add_argument('--offline', action='store_true')
    parser.add_argument('--period-end')
    parser.add_argument('--period-type', choices=['quarter', 'annual'], default='quarter')
    parser.add_argument('--top-k', type=int, choices=range(1, 6), default=3)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    try:
        if args.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        load_local_settings()
        result = research(args.ticker, args.question, args.data_dir, args.refresh, args.offline,
                          args.period_end, args.period_type, args.top_k,
                          progress=lambda message: print(message, file=sys.stderr, flush=True))
        print(json.dumps(result, indent=2) if args.json else render(result))
        return 0 if result['status'] == 'answered' else 2
    except (ResearchError, OSError, ValueError) as exc:
        print(f'Research failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
