"""Persistent per-company SQLite vector storage and exact cosine retrieval."""
import argparse
import hashlib
import json
import re
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from .embeddings import IndexError, OpenAIEmbedder, unit_vector, validate_text


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_chunks(ticker, data_dir):
    ticker = ticker.strip().upper()
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9.-]{0,14}', ticker):
        raise IndexError('Enter a valid ticker, such as AAPL.')
    root = Path(data_dir) / ticker
    try:
        raw = (root / 'chunks.json').read_bytes()
        payload = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise IndexError(f'Cannot read chunks.json for {ticker}. Run the chunker first: {exc}') from exc
    if not isinstance(payload, dict) or payload.get('schema_version') != 1 or payload.get('ticker') != ticker:
        raise IndexError('Chunk file has an unsupported schema or mismatched ticker.')
    chunks = payload.get('chunks')
    if not isinstance(chunks, list) or not chunks:
        raise IndexError('No chunks to index. Existing index is unchanged.')
    seen = set()
    for chunk in chunks:
        if not isinstance(chunk, dict) or not isinstance(chunk.get('id'), str) or not chunk['id']:
            raise IndexError('Chunk IDs must be nonempty strings.')
        if chunk['id'] in seen:
            raise IndexError('Duplicate chunk IDs. Run the chunker again.')
        seen.add(chunk['id'])
        validate_text(chunk.get('text'))
        source = chunk.get('source')
        if (not isinstance(source, dict) or source.get('ticker') != ticker
                or not source.get('source_url') or not isinstance(chunk.get('location'), dict)):
            raise IndexError('Chunk has incomplete or mismatched source metadata.')
    return root, payload, sha(raw)


def initialize(db):
    version = db.execute('PRAGMA user_version').fetchone()[0]
    if version not in (0, 1):
        raise IndexError('Unsupported index version.')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS embeddings (
            provider TEXT, model TEXT, dimensions INTEGER, text_hash TEXT, vector TEXT NOT NULL,
            PRIMARY KEY (provider, model, dimensions, text_hash));
        CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, payload TEXT NOT NULL, vector TEXT NOT NULL);
        PRAGMA user_version = 1;
    ''')


def index_company(ticker, data_dir=Path('data'), embedder=None, dry_run=False):
    root, payload, fingerprint = load_chunks(ticker, data_dir)
    embedder = embedder or OpenAIEmbedder()
    chunks = payload['chunks']
    path = root / 'index.sqlite3'
    settings = {'provider': embedder.provider, 'model': embedder.model, 'dimensions': embedder.dimensions}
    texts = {sha(c['text'].encode()): c['text'] for c in chunks}
    vectors = {}
    if path.exists():
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
            if db.execute('PRAGMA user_version').fetchone()[0] != 1:
                raise IndexError('Unsupported index version.')
            for key, vector in db.execute('SELECT text_hash, vector FROM embeddings WHERE provider=? AND model=? AND dimensions=?',
                                         (embedder.provider, embedder.model, embedder.dimensions)):
                if key in texts:
                    vectors[key] = unit_vector(json.loads(vector), embedder.dimensions)
    missing = [key for key in texts if key not in vectors]
    summary = {**settings, 'ticker': payload['ticker'], 'chunks': len(chunks),
               'unique_texts': len(texts), 'cached_texts': len(vectors), 'texts_to_embed': len(missing),
               'input_bytes_to_embed': sum(len(texts[key].encode()) for key in missing),
               'index_path': str(path.resolve()), 'dry_run': dry_run}
    if dry_run:
        return summary
    # Successful batches are cached separately. A later failed batch leaves the
    # previous searchable snapshot intact and can resume without paying twice.
    with closing(sqlite3.connect(path)) as db:
        initialize(db)
        for start in range(0, len(missing), 16):
            keys = missing[start:start + 16]
            batch = embedder.embed([texts[key] for key in keys])
            if len(batch) != len(keys):
                raise IndexError('Embedding provider returned an incomplete batch.')
            normalized = [unit_vector(v, embedder.dimensions) for v in batch]
            with db:
                for key, vector in zip(keys, normalized):
                    db.execute('INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?)',
                               (embedder.provider, embedder.model, embedder.dimensions, key, json.dumps(vector)))
                    vectors[key] = vector
        # Refresh metadata and remove obsolete passages together, after all vectors exist.
        with db:
            db.execute('DELETE FROM chunks')
            db.executemany('INSERT INTO chunks VALUES (?, ?, ?)', [
                (c['id'], json.dumps(c), json.dumps(vectors[sha(c['text'].encode())])) for c in chunks])
            metadata = {**settings, 'ticker': payload['ticker'], 'chunks_sha256': fingerprint,
                        'chunk_status': payload.get('status'), 'collection_status': payload.get('collection_status'),
                        'warnings': payload.get('warnings', []), 'collection_warnings': payload.get('collection_warnings', [])}
            db.execute('DELETE FROM metadata')
            db.executemany('INSERT INTO metadata VALUES (?, ?)', [(key, json.dumps(value)) for key, value in metadata.items()])
    return summary


def search_company(ticker, query, data_dir=Path('data'), top_k=5, form=None, period_end=None, embedder=None):
    if not 1 <= top_k <= 50:
        raise IndexError('top_k must be between 1 and 50.')
    validate_text(query)
    root, _, fingerprint = load_chunks(ticker, data_dir)
    path = root / 'index.sqlite3'
    if not path.exists():
        raise IndexError('No index exists for this company. Run the index command first.')
    embedder = embedder or OpenAIEmbedder()
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        # Read metadata and passages from one consistent SQLite snapshot.
        db.execute('BEGIN')
        if db.execute('PRAGMA user_version').fetchone()[0] != 1:
            raise IndexError('Unsupported index version.')
        meta = {key: json.loads(value) for key, value in db.execute('SELECT key, value FROM metadata')}
        if meta.get('chunks_sha256') != fingerprint:
            raise IndexError('Index is missing or stale. Run the index command again.')
        if any(meta.get(k) != getattr(embedder, k) for k in ('provider', 'model', 'dimensions')):
            raise IndexError('Query embedding configuration does not match the stored index. Reindex first.')
        candidates = []
        for payload, vector in db.execute('SELECT payload, vector FROM chunks'):
            chunk = json.loads(payload)
            source = chunk['source']
            if form and source.get('form') != form:
                continue
            if period_end and source.get('reporting_period_end') != period_end:
                continue
            candidates.append((chunk, unit_vector(json.loads(vector), embedder.dimensions)))
    result = {'ticker': meta['ticker'], 'query': query, 'model': meta['model'],
              'chunk_status': meta['chunk_status'], 'collection_status': meta['collection_status'],
              'warnings': meta['warnings'], 'collection_warnings': meta['collection_warnings'], 'results': []}
    if not candidates:
        return result
    query_vectors = embedder.embed([query])
    if len(query_vectors) != 1:
        raise IndexError('Embedding provider returned an invalid query vector count.')
    query_vector = unit_vector(query_vectors[0], embedder.dimensions)
    ranked = []
    for chunk, vector in candidates:
        score = sum(a * b for a, b in zip(query_vector, vector))
        ranked.append({**chunk, 'score': max(-1.0, min(1.0, score))})
    result['results'] = sorted(ranked, key=lambda c: (-c['score'], c['id']))[:top_k]
    return result


def main():
    parser = argparse.ArgumentParser(description='Embed passages and search a local SQLite index.')
    commands = parser.add_subparsers(dest='command', required=True)
    index = commands.add_parser('index', help='Create or refresh a company index')
    index.add_argument('--dry-run', action='store_true', help='Validate input and show uncached work without API calls or database writes')
    search = commands.add_parser('search', help='Return relevant passages, not a generated answer')
    search.add_argument('--top-k', type=int, default=5)
    search.add_argument('--form', choices=['10-K', '10-Q', '8-K'])
    search.add_argument('--period-end', help='Exact reporting period end, YYYY-MM-DD; unknown periods will not match')
    for command in (index, search):
        command.add_argument('ticker')
        command.add_argument('--data-dir', type=Path, default=Path('data'))
    search.add_argument('query')
    args = parser.parse_args()
    try:
        if args.command == 'index':
            result = index_company(args.ticker, args.data_dir, dry_run=args.dry_run)
        else:
            result = search_company(args.ticker, args.query, args.data_dir, args.top_k, args.form, args.period_end)
    except (IndexError, OSError, sqlite3.Error, ValueError) as exc:
        print(f'Index operation failed: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
